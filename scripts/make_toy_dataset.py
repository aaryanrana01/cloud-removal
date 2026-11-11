"""Create a tiny fake cloud dataset (cloud/ + label/) for tests and CI.

Real training uses RICE1; this only checks that the pipeline runs end to end
without downloading anything.

    python scripts/make_toy_dataset.py --out /tmp/toy --n 12 --size 128
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def smooth_noise(rng, size, blur):
    img = Image.fromarray((rng.random((size, size)) * 255).astype(np.uint8))
    arr = np.asarray(img.filter(ImageFilter.GaussianBlur(blur)), np.float32)
    return (arr - arr.min()) / (np.ptp(arr) + 1e-6)


def make_pair(rng, size):
    terrain = np.stack([smooth_noise(rng, size, 3) * s + o for s, o in ((0.5, 0.2), (0.6, 0.25), (0.4, 0.15))], -1)
    cloud = np.clip((smooth_noise(rng, size, size / 10) - 0.45) * 2.5, 0, 1)[..., None]
    cloudy = terrain * (1 - cloud) + 0.95 * cloud
    to8 = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return to8(cloudy), to8(terrain)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    root = Path(a.out) / "RICE1"
    (root / "cloud").mkdir(parents=True, exist_ok=True)
    (root / "label").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for i in range(a.n):
        c, lbl = make_pair(rng, a.size)
        Image.fromarray(c).save(root / "cloud" / f"{i}.png")
        Image.fromarray(lbl).save(root / "label" / f"{i}.png")
    print(f"wrote {a.n} pairs to {root}")


if __name__ == "__main__":
    main()
