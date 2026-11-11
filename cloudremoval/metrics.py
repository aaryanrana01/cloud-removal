"""Evaluation metrics, computed on the real RGB image only.

  * PSNR (dB)  - how close the pixel values are. Higher is better; +1 dB is a
                 clearly visible improvement at these levels.
  * SSIM (0-1) - how similar the structure (edges, textures) looks. Higher is better.
  * CC (-1..1) - Pearson correlation between predicted and true pixel values.

The original code averaged these over 4 channels where the 4th was all zeros.
A zero channel is trivially easy to 'predict', which inflated PSNR by several
dB and dragged CC down by 25%. Here every number comes from the 3 real channels.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def image_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """pred, gt: float arrays (H, W, 3) in [0, 1]."""
    pred = np.clip(pred, 0.0, 1.0).astype(np.float64)
    gt = gt.astype(np.float64)
    mse = float(np.mean((pred - gt) ** 2))
    psnr = 100.0 if mse == 0 else float(peak_signal_noise_ratio(gt, pred, data_range=1.0))
    ssim = float(structural_similarity(gt, pred, data_range=1.0, channel_axis=2))
    p, g = pred.ravel(), gt.ravel()
    cc = float(np.corrcoef(p, g)[0, 1]) if p.std() > 0 and g.std() > 0 else 0.0
    return {"psnr": psnr, "ssim": ssim, "cc": cc}


def summarise(rows: list[dict]) -> dict:
    """Mean and standard deviation of each metric over many images."""
    keys = rows[0].keys()
    return {
        k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
        for k in keys
    }
