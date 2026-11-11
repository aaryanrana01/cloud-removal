import io
import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from cloudremoval.inference import Predictor
from cloudremoval.model import build_model
from server.app.main import create_app
from server.app.settings import Settings


def png_bytes(w=96, h=80, mode="RGB"):
    buf = io.BytesIO()
    Image.fromarray((np.random.rand(h, w, 3) * 255).astype(np.uint8)).convert(mode).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def make_client(tmp_path, tiny_cfg):
    def make(loaded=True, **overrides):
        samples = tmp_path / "samples"
        samples.mkdir(exist_ok=True)
        (samples / "index.json").write_text(json.dumps([{"id": "7", "cloudy": "7_cloudy.jpg", "clear": "7_clear.jpg"}]))
        settings = Settings(model_path=tmp_path / "missing.pt", model_repo="", results_path=tmp_path / "none.json",
                            samples_dir=samples, static_dir=tmp_path / "no-dist", **overrides)
        app = create_app(settings, load_model=False)
        if loaded:
            model = build_model(tiny_cfg)
            torch.nn.init.normal_(model.tail.weight, std=0.05)
            app.state.service.use(Predictor(model))
        return TestClient(app)
    return make


def test_health_reports_missing_model(make_client):
    body = make_client(loaded=False).get("/api/health").json()
    assert body["status"] == "degraded" and body["model_loaded"] is False


def test_predict_returns_images_and_stats(make_client):
    r = make_client().post("/api/predict", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ("input", "result", "changes"):
        assert body[key].startswith("data:image/png;base64,")
    assert (body["width"], body["height"], body["resized"]) == (96, 80, False)
    assert 0 <= body["changed_fraction"] <= 1


def test_large_image_is_scaled_down(make_client):
    r = make_client(max_side=64).post("/api/predict", files={"file": ("a.png", png_bytes(200, 100), "image/png")})
    body = r.json()
    assert body["resized"] and max(body["width"], body["height"]) == 64 and body["original_width"] == 200


def test_rgba_and_greyscale_uploads_work(make_client):
    client = make_client()
    for mode in ("RGBA", "L"):
        assert client.post("/api/predict", files={"file": ("a.png", png_bytes(mode=mode), "image/png")}).status_code == 200


@pytest.mark.parametrize("payload,code", [(b"not an image", 415), (b"", 400)])
def test_bad_uploads_are_rejected_clearly(make_client, payload, code):
    r = make_client().post("/api/predict", files={"file": ("x.png", payload, "image/png")})
    assert r.status_code == code and r.json()["detail"]


def test_file_size_limit(make_client):
    r = make_client(max_upload_mb=0.001).post("/api/predict", files={"file": ("a.png", png_bytes(300, 300), "image/png")})
    assert r.status_code == 413


def test_model_not_loaded_returns_503(make_client):
    r = make_client(loaded=False).post("/api/predict", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 503


def test_rate_limit(make_client):
    client = make_client(rate_limit_per_minute=2)
    codes = [client.post("/api/predict", files={"file": ("a.png", png_bytes(), "image/png")}).status_code
             for _ in range(3)]
    assert codes == [200, 200, 429]


def test_samples_and_model_info(make_client):
    client = make_client()
    s = client.get("/api/samples").json()
    assert s[0]["cloudy_url"] == "/samples/7_cloudy.jpg"
    info = client.get("/api/model").json()
    assert info["metrics"] is None and info["parameters"] > 0
