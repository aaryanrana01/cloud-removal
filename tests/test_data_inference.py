import json

import numpy as np
import torch

from cloudremoval.data import PairDataset, collect_pairs, find_dataset_root, load_or_create_split, select
from cloudremoval.inference import Predictor
from cloudremoval.model import build_model


def test_finds_nested_rice1_and_pairs_by_name(toy_dataset):
    root = find_dataset_root(toy_dataset)
    assert root.name == "RICE1"
    pairs = collect_pairs(root)
    assert len(pairs) == 10                                    # the orphan cloudy image is skipped
    assert all(c.name == lbl.name for c, lbl in pairs)


def test_split_is_disjoint_saved_and_reused(toy_dataset, tmp_path):
    pairs = collect_pairs(find_dataset_root(toy_dataset))
    path = tmp_path / "splits.json"
    split = load_or_create_split(pairs, path, 0.2, 0.2, seed=1)
    sets = [set(split[k]) for k in ("train", "val", "test")]
    assert sum(map(len, sets)) == 10 and not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert load_or_create_split(pairs, path, 0.5, 0.5, seed=99) == json.loads(path.read_text())   # reused


def test_dataset_crops_are_aligned_rgb(toy_dataset):
    pairs = collect_pairs(find_dataset_root(toy_dataset))
    ds = PairDataset(select(pairs, [p[0].stem for p in pairs]), crop_size=32, augment=True, repeats=3)
    assert len(ds) == 30
    cloudy, clear = ds[0]
    assert cloudy.shape == clear.shape == (3, 32, 32)
    assert cloudy.dtype == torch.float32 and 0 <= cloudy.min() and cloudy.max() <= 1


def test_tiled_inference_has_no_seams(tiny_cfg):
    # An untrained model is the identity, so tiled output must equal the input exactly
    # if the overlapping tiles are blended correctly.
    predictor = Predictor(build_model(tiny_cfg), tile=64, overlap=16)
    img = np.random.rand(150, 210, 3).astype(np.float32)
    assert np.allclose(predictor(img), img, atol=1e-5)
