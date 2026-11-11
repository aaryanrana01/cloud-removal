"""Shared helper for the 'what changed' map shown on the website."""
from __future__ import annotations

import numpy as np

CHANGE_COLOR = (0.949, 0.718, 0.188)   # amber #F2B730


def change_overlay(before: np.ndarray, after: np.ndarray, color=CHANGE_COLOR) -> np.ndarray:
    """Transparent RGBA layer, opaque where the model changed the image most.
    Drawn on top of the input, it shows *where* clouds were removed."""
    diff = np.abs(after - before).mean(axis=2)
    alpha = np.clip(diff / 0.35, 0, 1) ** 0.7               # spread small changes so they are visible
    rgba = np.empty((*diff.shape, 4), np.float32)
    rgba[..., :3] = color
    rgba[..., 3] = alpha * 0.9
    return rgba


def changed_fraction(before: np.ndarray, after: np.ndarray, threshold: float = 0.08) -> float:
    """Share of pixels the model changed noticeably (a rough 'cloud cover' estimate)."""
    return float((np.abs(after - before).mean(axis=2) > threshold).mean())
