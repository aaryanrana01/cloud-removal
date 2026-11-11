"""Load a trained checkpoint and clean images of any size.

Big images are processed in overlapping tiles and blended with soft edges, so
memory use stays flat no matter how large the upload is, and there are no
visible seams between tiles.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .model import SwinUNet, build_model, count_parameters

CHECKPOINT_VERSION = 1


def save_checkpoint(path, model: SwinUNet, extra: dict | None = None):
    """Small, portable checkpoint: weights + model config + optional metadata."""
    payload = {
        "version": CHECKPOINT_VERSION,
        "model_config": vars(model.cfg).copy(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "meta": extra or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, device="cpu") -> tuple[SwinUNet, dict]:
    # weights_only=True refuses to run arbitrary pickled code from the file.
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path} is not a checkpoint saved by this project")
    cfg = ModelConfig(**payload["model_config"])
    cfg.grad_checkpoint = False
    model = build_model(cfg)
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload.get("meta", {})


def _blend_window(h: int, w: int, overlap: int) -> np.ndarray:
    """1 in the middle, fading linearly to ~0 over `overlap` pixels at each edge."""
    def ramp(n):
        r = np.ones(n, np.float32)
        if overlap > 0 and n > 2 * overlap:
            edge = np.linspace(1.0 / overlap, 1.0, overlap, dtype=np.float32)
            r[:overlap], r[-overlap:] = edge, edge[::-1]
        return r
    return np.outer(ramp(h), ramp(w))[..., None]


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    s = list(range(0, length - tile, stride))
    return s + [length - tile]


class Predictor:
    def __init__(self, model: SwinUNet, device="cpu", tile: int = 512, overlap: int = 48):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.tile, self.overlap = tile, overlap

    @classmethod
    def from_checkpoint(cls, path, device="cpu", **kw) -> Predictor:
        model, meta = load_checkpoint(path, device)
        p = cls(model, device, **kw)
        p.meta = meta
        return p

    @property
    def num_parameters(self) -> int:
        return count_parameters(self.model)

    @torch.inference_mode()
    def _run(self, img: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None].to(self.device)
        y = self.model(x).clamp_(0, 1)
        return y[0].permute(1, 2, 0).cpu().numpy()

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """img: float32 (H, W, 3) in [0, 1] -> cloud-free image of the same shape."""
        H, W = img.shape[:2]
        t = self.tile
        if H <= t and W <= t:
            return self._run(img)

        out = np.zeros_like(img, dtype=np.float32)
        weight = np.zeros((H, W, 1), np.float32)
        stride = t - self.overlap
        for y in _starts(H, t, stride):
            for x in _starts(W, t, stride):
                patch = img[y:y + t, x:x + t]
                ph, pw = patch.shape[:2]
                win = _blend_window(ph, pw, self.overlap)
                out[y:y + ph, x:x + pw] += self._run(patch) * win
                weight[y:y + ph, x:x + pw] += win
        return out / np.maximum(weight, 1e-8)
