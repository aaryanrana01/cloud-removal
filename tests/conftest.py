import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cloudremoval.config import ModelConfig  # noqa: E402

TINY = dict(embed_dim=16, depths=[1, 1], num_heads=[2, 4], bottleneck_depth=1, window_size=4, mlp_ratio=2.0)


@pytest.fixture
def tiny_cfg():
    return ModelConfig(**TINY)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture
def toy_dataset(tmp_path):
    """A nested folder like Kaggle's zip: <tmp>/extracted/RICE_DATASET/RICE1/{cloud,label}."""
    from make_toy_dataset import make_pair

    root = tmp_path / "extracted" / "RICE_DATASET" / "RICE1"
    (root / "cloud").mkdir(parents=True)
    (root / "label").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(10):
        c, lbl = make_pair(rng, 64)
        Image.fromarray(c).save(root / "cloud" / f"{i}.png")
        Image.fromarray(lbl).save(root / "label" / f"{i}.png")
    Image.fromarray(c).save(root / "cloud" / "orphan.png")      # no matching label -> ignored
    return tmp_path / "extracted"
