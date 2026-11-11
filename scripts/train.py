"""Train the Swin U-Net on RICE1.

    python scripts/train.py --config configs/rice1.yaml --data-root /content/data

Everything needed to continue is saved in <out_dir>/last.pt after every epoch.
If Colab disconnects, run the same command again and it picks up where it stopped.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudremoval.config import Config  # noqa: E402
from cloudremoval.data import PairDataset, collect_pairs, find_dataset_root, load_or_create_split, select  # noqa: E402
from cloudremoval.inference import save_checkpoint  # noqa: E402
from cloudremoval.losses import RestorationLoss  # noqa: E402
from cloudremoval.metrics import image_metrics  # noqa: E402
from cloudremoval.model import build_model, count_parameters  # noqa: E402

CSV_FIELDS = ["epoch", "train_loss", "l1", "ssim_loss", "perceptual", "val_psnr", "val_ssim", "val_cc", "lr", "seconds"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/rice1.yaml")
    p.add_argument("--data-root", help="folder containing the dataset (searched for cloud/ + label/)")
    p.add_argument("--out-dir", help="override train.out_dir (point this at Google Drive in Colab)")
    p.add_argument("--epochs", type=int, help="override train.epochs")
    p.add_argument("--batch-size", type=int, help="override train.batch_size")
    p.add_argument("--max-steps", type=int, default=0, help="stop each epoch after N steps (smoke tests)")
    p.add_argument("--no-pretrained-vgg", action="store_true", help="skip the VGG download (tests only)")
    p.add_argument("--fresh", action="store_true", help="ignore last.pt and start over")
    return p.parse_args()


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id: int):
    # Without this every DataLoader worker would produce the SAME random crops.
    np.random.seed((torch.initial_seed() + worker_id) % 2 ** 32)


def lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warm-up, then cosine decay to 1% of the peak learning rate."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))
    return f


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    rows = []
    for cloudy, clear in loader:
        pred = model(cloudy.to(device)).clamp(0, 1).float().cpu().numpy()
        gt = clear.numpy()
        for p, g in zip(pred, gt):
            rows.append(image_metrics(p.transpose(1, 2, 0), g.transpose(1, 2, 0)))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


@torch.no_grad()
def save_preview(model, dataset, device, path, n=3):
    """cloudy | prediction | ground truth, one row per image."""
    model.eval()
    rows = []
    for i in range(min(n, len(dataset))):
        cloudy, clear = dataset[i]
        pred = model(cloudy[None].to(device)).clamp(0, 1)[0].float().cpu()
        row = torch.cat([cloudy, pred, clear], dim=2)
        rows.append(row)
    grid = (torch.cat(rows, dim=1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(path)


def plot_curves(csv_path: Path, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for a, key, title in zip(ax, ["train_loss", "val_psnr", "val_ssim"], ["Training loss", "Validation PSNR (dB)", "Validation SSIM"]):
        a.plot(ep, [float(r[key]) for r in rows], lw=1.8)
        a.set_title(title)
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    if args.data_root:
        cfg.data.root = args.data_root
    if args.out_dir:
        cfg.train.out_dir = args.out_dir
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if not cfg.data.root:
        sys.exit("Set --data-root (or data.root in the config)")

    tc = cfg.train
    out = Path(tc.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(tc.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = tc.amp and device.type == "cuda"
    print(f"device={device}  mixed_precision={use_amp}")

    # ---- data
    root = find_dataset_root(cfg.data.root)
    pairs = collect_pairs(root)
    split = load_or_create_split(pairs, out / "splits.json", cfg.data.val_fraction, cfg.data.test_fraction, cfg.data.seed)
    train_pairs, val_pairs = select(pairs, split["train"]), select(pairs, split["val"])
    print(f"dataset={root}  pairs={len(pairs)}  train={len(train_pairs)} val={len(val_pairs)} test={len(split['test'])}")

    train_ds = PairDataset(train_pairs, crop_size=cfg.data.crop_size, augment=True, repeats=cfg.data.repeats)
    val_ds = PairDataset(val_pairs)
    nw = tc.num_workers
    train_dl = DataLoader(train_ds, tc.batch_size, shuffle=True, drop_last=True, num_workers=nw,
                          pin_memory=device.type == "cuda", worker_init_fn=worker_init, persistent_workers=nw > 0)
    val_dl = DataLoader(val_ds, 1, shuffle=False, num_workers=nw)

    # ---- model + optimisation
    model = build_model(cfg.model).to(device)
    print(f"model parameters: {count_parameters(model):,}")
    criterion = RestorationLoss(tc.w_l1, tc.w_ssim, tc.w_perceptual, pretrained_vgg=not args.no_pretrained_vgg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    steps_per_epoch = min(len(train_dl), args.max_steps) if args.max_steps else len(train_dl)
    total_steps = steps_per_epoch * tc.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(steps_per_epoch * tc.warmup_epochs, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_psnr, bad_epochs = 1, -1.0, 0
    last_path = out / "last.pt"
    if last_path.exists() and not args.fresh:
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_psnr, bad_epochs = state["epoch"] + 1, state["best_psnr"], state["bad_epochs"]
        print(f"Resuming from epoch {state['epoch']} (best val PSNR so far {best_psnr:.2f} dB)")
    else:
        print("Starting fresh")
        with open(out / "metrics.csv", "w", newline="") as f:
            csv.writer(f).writerow(CSV_FIELDS)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    for epoch in range(start_epoch, tc.epochs + 1):
        if bad_epochs >= tc.patience:
            print(f"Early stopping: no improvement for {tc.patience} epochs")
            break
        model.train()
        criterion.train()
        t0 = time.time()
        sums = {"loss": 0.0, "l1": 0.0, "ssim_loss": 0.0, "perceptual": 0.0}
        for step, (cloudy, clear) in enumerate(train_dl):
            if args.max_steps and step >= args.max_steps:
                break
            cloudy, clear = cloudy.to(device, non_blocking=True), clear.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(cloudy)
            loss, parts = criterion(pred, clear)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            sums["loss"] += loss.item()
            for k, v in parts.items():
                sums[k] += float(v)
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step:4d}/{steps_per_epoch}  loss {loss.item():.4f}")

        n = max(1, steps_per_epoch)
        val = validate(model, val_dl, device)
        lr = optimizer.param_groups[0]["lr"]
        secs = time.time() - t0
        print(f"epoch {epoch:3d} | loss {sums['loss'] / n:.4f} | val PSNR {val['psnr']:.2f} dB "
              f"SSIM {val['ssim']:.4f} CC {val['cc']:.4f} | lr {lr:.2e} | {secs:.0f}s")
        with open(out / "metrics.csv", "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(sums["loss"] / n, 6), round(sums["l1"] / n, 6),
                                    round(sums["ssim_loss"] / n, 6), round(sums["perceptual"] / n, 6),
                                    round(val["psnr"], 4), round(val["ssim"], 6), round(val["cc"], 6),
                                    f"{lr:.3e}", round(secs, 1)])

        if val["psnr"] > best_psnr:
            best_psnr, bad_epochs = val["psnr"], 0
            save_checkpoint(out / "best.pt", model, {"epoch": epoch, "val": val, "dataset": root.name})
            print(f"  new best model saved (val PSNR {best_psnr:.2f} dB)")
        else:
            bad_epochs += 1

        if epoch % tc.save_every == 0 or epoch == 1:
            save_preview(model, val_ds, device, out / "previews" / f"epoch_{epoch:03d}.png")

        torch.save({
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "best_psnr": best_psnr, "bad_epochs": bad_epochs,
        }, out / "last.pt.tmp")
        (out / "last.pt.tmp").replace(last_path)   # atomic: a disconnect can't leave a half-written file

    plot_curves(out / "metrics.csv", out / "training_curves.png")
    print(f"Done. Best val PSNR {best_psnr:.2f} dB -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
