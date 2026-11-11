"""Measure the trained model on the held-out test images.

    python scripts/evaluate.py --checkpoint runs/rice1/best.pt --data-root /content/data \
        --split runs/rice1/splits.json

Writes:
  results/metrics.json   numbers shown on the website and in the README
  results/results.md     the same as a Markdown table
  results/examples/*.png cloudy | model | ground truth for a few test images
  server/samples/        a few test images for the website's "try an example" gallery

Two references are reported next to the model so the numbers mean something:
  * "No processing" = score of the cloudy image itself vs the clear one.
    Any useful model must beat this by a wide margin.
  * The model's average improvement over that baseline, per image.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudremoval.data import collect_pairs, find_dataset_root, load_rgb, select  # noqa: E402
from cloudremoval.inference import Predictor  # noqa: E402
from cloudremoval.metrics import image_metrics, summarise  # noqa: E402
from cloudremoval.visualize import change_overlay, changed_fraction  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def to_uint8(img):
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", required=True, help="splits.json written by train.py")
    p.add_argument("--out", default=str(REPO / "results"))
    p.add_argument("--samples-dir", default=str(REPO / "server" / "samples"))
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--num-examples", type=int, default=8)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = Predictor.from_checkpoint(args.checkpoint, device)
    root = find_dataset_root(args.data_root)
    split = json.loads(Path(args.split).read_text())
    test_pairs = select(collect_pairs(root), split["test"])
    if not test_pairs:
        sys.exit("No test images found - check --data-root and --split")
    print(f"Evaluating {len(test_pairs)} test images from {root} on {device}")

    model_rows, base_rows, per_image, times = [], [], [], []
    for i, (cloudy_path, clear_path) in enumerate(test_pairs):
        cloudy, clear = load_rgb(cloudy_path), load_rgb(clear_path)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = predictor(cloudy)
        if device == "cuda":
            torch.cuda.synchronize()
        if i > 0:                                   # first call includes warm-up
            times.append((time.perf_counter() - t0) * 1000)
        m, b = image_metrics(pred, clear), image_metrics(cloudy, clear)
        model_rows.append(m)
        base_rows.append(b)
        per_image.append({"name": cloudy_path.stem, "model": m, "no_processing": b})

    model_s, base_s = summarise(model_rows), summarise(base_rows)
    gain = [m["psnr"] - b["psnr"] for m, b in zip(model_rows, base_rows)]
    result = {
        "dataset": root.name,
        "test_images": len(test_pairs),
        "image_size": list(load_rgb(test_pairs[0][0]).shape[:2]),
        "parameters": predictor.num_parameters,
        "checkpoint_epoch": getattr(predictor, "meta", {}).get("epoch"),
        "device": device if device == "cpu" else torch.cuda.get_device_name(0),
        "ms_per_image": round(float(np.mean(times)), 1) if times else None,
        "model": model_s,
        "no_processing": base_s,
        "psnr_gain_db": {"mean": float(np.mean(gain)), "min": float(np.min(gain)), "max": float(np.max(gain))},
        "images_improved_pct": round(100 * float(np.mean([g > 0 for g in gain])), 1),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    (out / "per_image.json").write_text(json.dumps(per_image, indent=2))

    fmt = lambda s, k, d: f"{s[k]['mean']:.{d}f} ± {s[k]['std']:.{d}f}"
    table = (
        f"| Method | PSNR (dB) ↑ | SSIM ↑ | CC ↑ |\n|---|---|---|---|\n"
        f"| No processing (cloudy input) | {fmt(base_s, 'psnr', 2)} | {fmt(base_s, 'ssim', 4)} | {fmt(base_s, 'cc', 4)} |\n"
        f"| **Swin U-Net (ours)** | **{fmt(model_s, 'psnr', 2)}** | **{fmt(model_s, 'ssim', 4)}** | **{fmt(model_s, 'cc', 4)}** |\n"
    )
    (out / "results.md").write_text(
        f"Test set: {len(test_pairs)} held-out {root.name} images at full resolution "
        f"({result['image_size'][0]}x{result['image_size'][1]}).\n\n{table}\n"
        f"Average PSNR gain over the cloudy input: {result['psnr_gain_db']['mean']:+.2f} dB; "
        f"{result['images_improved_pct']}% of test images improved.\n"
    )
    print("\n" + table)
    print(f"Mean PSNR gain {result['psnr_gain_db']['mean']:+.2f} dB | {result['ms_per_image']} ms/image on {result['device']}")

    # Examples and website samples, picked evenly across the score range
    # (best, median, worst...) so the gallery is representative, not cherry-picked.
    order = np.argsort([r["psnr"] for r in model_rows])[::-1]
    def spread(k):
        k = min(k, len(order))
        return [order[int(round(j))] for j in np.linspace(0, len(order) - 1, k)] if k else []

    ex_dir = out / "examples"
    ex_dir.mkdir(exist_ok=True)
    for j, idx in enumerate(spread(args.num_examples)):
        c, g = load_rgb(test_pairs[idx][0]), load_rgb(test_pairs[idx][1])
        pred = predictor(c)
        gap = np.ones((c.shape[0], 6, 3), np.float32)
        Image.fromarray(to_uint8(np.concatenate([c, gap, pred, gap, g], axis=1))).save(
            ex_dir / f"{j:02d}_{test_pairs[idx][0].stem}.png")

    sd = Path(args.samples_dir)
    sd.mkdir(parents=True, exist_ok=True)
    index = []
    for idx in spread(args.num_samples):
        # The model's output is saved too, so the gallery loads instantly without using the server.
        name = test_pairs[idx][0].stem
        c, g = load_rgb(test_pairs[idx][0]), load_rgb(test_pairs[idx][1])
        pred = predictor(c)
        Image.fromarray(to_uint8(c)).save(sd / f"{name}_cloudy.jpg", quality=92)
        Image.fromarray(to_uint8(g)).save(sd / f"{name}_clear.jpg", quality=92)
        Image.fromarray(to_uint8(pred)).save(sd / f"{name}_cleaned.jpg", quality=92)
        Image.fromarray(to_uint8(change_overlay(c, pred))).save(sd / f"{name}_changes.png")
        index.append({"id": name, "cloudy": f"{name}_cloudy.jpg", "clear": f"{name}_clear.jpg",
                      "cleaned": f"{name}_cleaned.jpg", "changes": f"{name}_changes.png",
                      "changed_fraction": round(changed_fraction(c, pred), 4),
                      "psnr_model": round(model_rows[idx]["psnr"], 2),
                      "psnr_input": round(base_rows[idx]["psnr"], 2),
                      "ssim_model": round(model_rows[idx]["ssim"], 4),
                      "ssim_input": round(base_rows[idx]["ssim"], 4)})
    (sd / "index.json").write_text(json.dumps(index, indent=2))
    print(f"Wrote {out / 'metrics.json'}, {len(index)} website samples to {sd}")


if __name__ == "__main__":
    main()
