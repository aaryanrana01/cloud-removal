"""Loads the model once at startup and runs it, one request at a time."""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np
import torch
from starlette.concurrency import run_in_threadpool

from cloudremoval.inference import Predictor

from .settings import Settings

log = logging.getLogger("cloud-removal")


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.predictor: Predictor | None = None
        self.error: str | None = None
        # The free CPU has 2 cores; running two images at once would make both slow.
        # Requests wait their turn instead.
        self._lock = asyncio.Semaphore(1)

    @property
    def ready(self) -> bool:
        return self.predictor is not None

    def load(self) -> None:
        torch.set_num_threads(self.settings.torch_threads)
        try:
            path = self.settings.model_path
            if not path.exists():
                if not self.settings.model_repo:
                    raise FileNotFoundError(f"No weights at {path} and MODEL_REPO is not set")
                from huggingface_hub import hf_hub_download
                log.info("Downloading %s from %s", self.settings.model_file, self.settings.model_repo)
                path = Path(hf_hub_download(self.settings.model_repo, self.settings.model_file))
            self.predictor = Predictor.from_checkpoint(path, "cpu")
            self._warm_up()
            log.info("Model ready: %s parameters", f"{self.predictor.num_parameters:,}")
        except Exception as e:  # keep the server up so /api/health can report the problem
            self.error = f"{type(e).__name__}: {e}"
            log.error("Model failed to load: %s", self.error)

    def use(self, predictor: Predictor) -> None:
        """Inject a predictor directly (used by tests)."""
        self.predictor, self.error = predictor, None

    def _warm_up(self) -> None:
        # The first call is slow (memory allocation); do it now, not on the first visitor.
        self.predictor(np.zeros((128, 128, 3), np.float32))

    async def predict(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        assert self.predictor is not None
        async with self._lock:
            t0 = time.perf_counter()
            out = await run_in_threadpool(self.predictor, img)
            return out, (time.perf_counter() - t0) * 1000
