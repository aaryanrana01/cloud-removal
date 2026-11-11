"""Dataset discovery, a fixed train/val/test split, and the PyTorch Dataset.

Changes from the original notebook:
  * Images are loaded as real RGB (3 channels). No fake zero 'NIR' band.
  * Pixels are scaled with /255 (or /65535 for 16-bit), the standard for 8-bit
    images, so there is no stats file that can go stale.
  * Training uses *random* 256x256 crops of the 512x512 images, so over many
    epochs the model sees every part of every image. Val/test use full images.
  * The split is saved to splits.json so test images are never trained on,
    even if you rerun training later.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def find_dataset_root(search_root: str | Path, prefer: str = "RICE1") -> Path:
    """Find a folder that has both `cloud/` and `label/` sub-folders.
    Prefers one named RICE1 (case-insensitive) because Kaggle zips nest folders."""
    search_root = Path(search_root)
    candidates = []
    for d in [search_root, *search_root.rglob("*")]:
        if d.is_dir() and (d / "cloud").is_dir() and (d / "label").is_dir():
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError(f"No folder with cloud/ and label/ found under {search_root}")
    for d in candidates:
        if d.name.upper() == prefer.upper():
            return d
    return candidates[0]


def collect_pairs(root: str | Path) -> list[tuple[Path, Path]]:
    """Match cloud/<name> with label/<name> by file name (extension may differ)."""
    root = Path(root)
    cloud_dir, label_dir = root / "cloud", root / "label"
    labels = {p.stem: p for p in label_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    pairs = [
        (p, labels[p.stem])
        for p in sorted(cloud_dir.iterdir())
        if p.suffix.lower() in IMG_EXTS and p.stem in labels
    ]
    if not pairs:
        raise FileNotFoundError(f"No matching image pairs in {cloud_dir} and {label_dir}")
    return pairs


def make_split(names: list[str], val_fraction: float, test_fraction: float, seed: int) -> dict:
    names = sorted(names)
    rng = random.Random(seed)
    rng.shuffle(names)
    n_test = round(len(names) * test_fraction)
    n_val = round(len(names) * val_fraction)
    return {
        "seed": seed,
        "test": sorted(names[:n_test]),
        "val": sorted(names[n_test:n_test + n_val]),
        "train": sorted(names[n_test + n_val:]),
    }


def load_or_create_split(pairs, split_path: str | Path, val_fraction, test_fraction, seed) -> dict:
    """Reuse an existing split file if present; otherwise create and save one."""
    split_path = Path(split_path)
    names = [c.stem for c, _ in pairs]
    if split_path.exists():
        split = json.loads(split_path.read_text())
        missing = set(split["train"] + split["val"] + split["test"]) - set(names)
        if missing:
            raise ValueError(f"{split_path} lists {len(missing)} images that are not in the dataset")
        return split
    split = make_split(names, val_fraction, test_fraction, seed)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2))
    return split


def select(pairs, names: list[str]):
    wanted = set(names)
    return [p for p in pairs if p[0].stem in wanted]


def load_rgb(path: str | Path) -> np.ndarray:
    """Load any common image as float32 (H, W, 3) in [0, 1]."""
    with Image.open(path) as im:
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.asarray(im, dtype=np.float32) / 65535.0
            arr = np.repeat(arr[..., None], 3, axis=2)
        else:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


class PairDataset(Dataset):
    """Returns (cloudy, clear) tensors of shape (3, H, W) in [0, 1]."""

    def __init__(self, pairs, crop_size: int | None = None, augment: bool = False, repeats: int = 1):
        self.pairs = list(pairs)
        self.crop_size = crop_size
        self.augment = augment
        self.repeats = max(1, repeats)

    def __len__(self):
        return len(self.pairs) * self.repeats

    def __getitem__(self, idx):
        cloudy_path, clear_path = self.pairs[idx % len(self.pairs)]
        cloudy, clear = load_rgb(cloudy_path), load_rgb(clear_path)
        if cloudy.shape != clear.shape:
            raise ValueError(f"Size mismatch: {cloudy_path.name} {cloudy.shape} vs {clear.shape}")

        if self.crop_size:
            cloudy, clear = self._random_crop(cloudy, clear, self.crop_size)
        if self.augment:
            cloudy, clear = self._flip_rotate(cloudy, clear)
        return to_tensor(cloudy), to_tensor(clear)

    @staticmethod
    def _random_crop(a, b, size):
        H, W = a.shape[:2]
        if H < size or W < size:  # small image: pad by reflecting edges
            ph, pw = max(0, size - H), max(0, size - W)
            pad = ((0, ph), (0, pw), (0, 0))
            a, b = np.pad(a, pad, mode="reflect"), np.pad(b, pad, mode="reflect")
            H, W = a.shape[:2]
        y = np.random.randint(0, H - size + 1)
        x = np.random.randint(0, W - size + 1)
        return a[y:y + size, x:x + size], b[y:y + size, x:x + size]

    @staticmethod
    def _flip_rotate(a, b):
        # Satellite images have no 'up', so flips and 90-degree turns are free extra data.
        if np.random.rand() < 0.5:
            a, b = a[:, ::-1], b[:, ::-1]
        if np.random.rand() < 0.5:
            a, b = a[::-1], b[::-1]
        k = np.random.randint(4)
        return np.rot90(a, k), np.rot90(b, k)
