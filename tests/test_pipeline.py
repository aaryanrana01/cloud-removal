"""End-to-end: train for a few steps on fake data, resume, evaluate. Catches broken wiring
between the scripts, which unit tests alone would miss."""
import json
import subprocess
import sys

import yaml
from conftest import ROOT, TINY


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


def test_train_resume_evaluate(toy_dataset, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({
        "model": TINY,
        "data": {"crop_size": 32, "repeats": 1, "val_fraction": 0.2, "test_fraction": 0.2},
        "train": {"epochs": 1, "batch_size": 2, "num_workers": 0, "warmup_epochs": 1},
    }))
    out = tmp_path / "run"
    common = ["--config", cfg, "--data-root", toy_dataset, "--out-dir", out, "--no-pretrained-vgg", "--max-steps", "2"]
    run("scripts/train.py", *common)
    assert (out / "best.pt").exists() and (out / "last.pt").exists()

    log = run("scripts/train.py", *common, "--epochs", "2")
    assert "Resuming from epoch 1" in log

    results, samples = tmp_path / "results", tmp_path / "samples"
    run("scripts/evaluate.py", "--checkpoint", out / "best.pt", "--data-root", toy_dataset,
        "--split", out / "splits.json", "--out", results, "--samples-dir", samples, "--num-samples", "2")
    metrics = json.loads((results / "metrics.json").read_text())
    assert metrics["test_images"] == 2 and "no_processing" in metrics
    index = json.loads((samples / "index.json").read_text())
    assert all((samples / item[k]).exists() for item in index for k in ("cloudy", "clear", "cleaned", "changes"))
