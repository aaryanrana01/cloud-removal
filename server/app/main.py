"""FastAPI app: JSON API under /api, sample images under /samples, React site at /.

Run locally:  uvicorn server.app.main:app --reload --port 7860
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cloudremoval import __version__

from .imaging import BadImage, change_overlay, changed_fraction, decode_upload, to_png_data_url
from .model_service import ModelService
from .rate_limit import RateLimiter
from .settings import Settings, get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ARCHITECTURE = {
    "name": "Swin Transformer U-Net",
    "summary": "Shifted-window self-attention encoder-decoder with skip connections that "
               "predicts a residual correction for the cloudy image.",
    "input": "RGB satellite image, any size (large images are processed in blended 512px tiles)",
    "loss": "L1 + 0.5 x (1 - SSIM) + 0.05 x VGG16 perceptual",
    "training_data": "RICE1 (500 Google Earth image pairs, 512x512)",
}


def create_app(settings: Settings | None = None, load_model: bool = True) -> FastAPI:
    settings = settings or get_settings()
    service = ModelService(settings)
    limiter = RateLimiter(settings.rate_limit_per_minute)
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_model:
            service.load()
        yield

    app = FastAPI(title=f"{settings.app_name} API", version=__version__, lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.service = service
    app.state.limiter = limiter

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    # ------------------------------------------------------------------ API
    @app.get("/api/health")
    def health():
        p = service.predictor
        return {
            "status": "ok" if service.ready else "degraded",
            "model_loaded": service.ready,
            "model_error": service.error,
            "parameters": p.num_parameters if p else None,
            "version": __version__,
        }

    @app.get("/api/model")
    def model_info():
        metrics = None
        if settings.results_path.exists():
            metrics = json.loads(settings.results_path.read_text())
        return {
            "app_name": settings.app_name,
            "architecture": ARCHITECTURE,
            "parameters": service.predictor.num_parameters if service.ready else None,
            "metrics": metrics,
            "limits": {"max_upload_mb": settings.max_upload_mb, "max_side": settings.max_side},
        }

    @app.get("/api/samples")
    def samples():
        index = settings.samples_dir / "index.json"
        if not index.exists():
            return []
        items = json.loads(index.read_text())
        files = ("cloudy", "clear", "cleaned", "changes")
        return [{**it, **{f"{k}_url": f"/samples/{it[k]}" for k in files if k in it}} for it in items]

    @app.post("/api/predict")
    async def predict(request: Request, file: UploadFile = File(...)):
        wait = limiter.check(limiter.client_ip(request))
        if wait:
            raise HTTPException(429, f"Too many requests. Try again in {int(wait) + 1} seconds.",
                                headers={"Retry-After": str(int(wait) + 1)})
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes + 64 * 1024:
            raise HTTPException(413, f"File too large (max {settings.max_upload_mb:g} MB).")
        if not service.ready:
            raise HTTPException(503, "The model is not loaded yet. Please try again in a minute.")

        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"File too large (max {settings.max_upload_mb:g} MB).")
        if not data:
            raise HTTPException(400, "Empty file.")
        try:
            img, (orig_w, orig_h), resized = decode_upload(data, settings.max_side)
        except BadImage as e:
            raise HTTPException(415, str(e)) from e

        out, ms = await service.predict(img)
        h, w = out.shape[:2]
        return {
            "input": to_png_data_url(img),          # browsers can't show TIFF, so send back a PNG
            "result": to_png_data_url(out),
            "changes": to_png_data_url(change_overlay(img, out)),
            "changed_fraction": round(changed_fraction(img, out), 4),
            "width": w,
            "height": h,
            "original_width": orig_w,
            "original_height": orig_h,
            "resized": resized,
            "inference_ms": round(ms, 1),
        }

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)

    # ------------------------------------------------------------------ static files
    if settings.samples_dir.exists():
        app.mount("/samples", StaticFiles(directory=settings.samples_dir), name="samples")

    dist = settings.static_dir
    if dist.exists():
        if (dist / "assets").exists():
            app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404, "Not found")
            target = (dist / path).resolve()
            if path and target.is_file() and dist.resolve() in target.parents:
                return FileResponse(target)
            return FileResponse(dist / "index.html")

    return app


app = create_app()
