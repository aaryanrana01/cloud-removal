# Cloud Removal from Satellite Images

[![CI](https://github.com/aaryanrana01/cloud-removal/actions/workflows/ci.yml/badge.svg)](https://github.com/aaryanrana01/cloud-removal/actions/workflows/ci.yml)

A Swin Transformer U-Net that removes clouds from satellite images, served as a web app where you
upload an image and drag a before/after slider.

**Live demo:** https://huggingface.co/spaces/YOUR-HF-USERNAME/cloud-removal &nbsp;·&nbsp;
**Model weights:** https://huggingface.co/YOUR-HF-USERNAME/cloud-removal-swin

<!-- After deploying, add a screenshot: save it as docs/demo.png and uncomment the next line. -->
<!-- ![Demo](docs/demo.png) -->

## Results

Test set: 100 RICE1 image pairs held out from training and validation, evaluated at full 512 × 512 size.
"No processing" scores the cloudy image itself against the clear one, which is the baseline any model must beat.

<!-- Replace this table with the contents of results/results.md after running the Colab notebook. -->
| Method | PSNR (dB) ↑ | SSIM ↑ | CC ↑ |
|---|---|---|---|
| No processing (cloudy input) | _run evaluate.py_ | | |
| **Swin U-Net (this repo)** | _run evaluate.py_ | | |

All metrics are computed on the three real RGB channels. Numbers from papers use different splits,
so they are not directly comparable.

## How it works

```
cloudy RGB ─ conv ─> 24ch @ full res ─────────────────────────────────────────┐ detail path
                └─ 2×2 patch embed ─> 48ch @ ½ ── Swin stage ──┐ skip          │
                                     merge ↓                    │               │
                                   96ch @ ¼ ── Swin stage ──────┤ skip          │
                                     merge ↓                    │               │
                                  192ch @ ⅛ ── Swin stage ──────┤ skip          │
                                     merge ↓                    │               │
                                  384ch @ 1/16  bottleneck      │               │
                    expand ↑ + skip ─> Swin stage  (×3, mirrors the encoder)    │
                    expand ↑ to full res + detail path <────────────────────────┘
                    conv ─> residual          output = input + residual
```

| Choice | Why |
|---|---|
| **Shifted-window attention** (Swin) | Every token attends within an 8×8 window; alternate layers shift windows so information crosses them. Cost grows linearly with image size, unlike full self-attention. |
| **U-Net encoder–decoder with skips** | The encoder sees large structures (cloud banks); skip connections bring back fine detail (roads, field edges). |
| **Attention from ½ resolution, conv path at full resolution** | Running attention at full resolution needed ~1.7 GB per training image. Starting at ½ cut memory and CPU inference time ~4×, while a cheap full-res conv path keeps detail. |
| **Residual output, last layer initialised to zero** | The model predicts a *correction* to the input. Untrained, it returns the input unchanged, so training starts from the "no processing" baseline and cloud-free areas are left alone. |
| **Loss: L1 + 0.5·(1−SSIM) + 0.05·VGG perceptual** | L1 gets colours right, SSIM keeps structure, perceptual loss avoids blurry output. |
| **Random 256 crops + flips/rotations** | Uses every part of each 512×512 image across epochs; satellite views have no "up", so rotations are free extra data. |
| **Tiled inference with feathered blending** | Any image size runs in constant memory with no visible seams. |

**Model size:** ~7.1 M parameters (~28 MB). Cleans a 512×512 image in about 1.2 s on a 2-core CPU.

## Changes from the first prototype

The first version (a single Colab notebook) ran without errors but had problems that made its numbers unreliable.
Each one has a regression test in `tests/`.

| Problem | Effect | Fix |
|---|---|---|
| A fake 4th "NIR" channel of zeros was added to RGB images | Averaging metrics over a trivially predictable channel inflated PSNR (26.2 → 34.7 dB in a test) and cut CC by 25% | Real 3-channel RGB everywhere |
| Perceptual loss computed under `torch.no_grad()` | It sent no gradient: the VGG term never trained anything | Only target features skip gradients; ImageNet normalisation added |
| 67 M parameters for ~350 training images | Overfitting risk, too heavy to host for free | 7 M parameters, attention from ½ resolution |
| Centre crop 512 → 256 | 75% of every image never seen | Random crops for training, full images for evaluation |
| SSIM loss computed in float16 | Variance terms can go negative | Losses computed in float32 |
| DataLoader workers shared one NumPy seed | Identical "random" augmentations across workers | Per-worker seeding |

## Project layout

```
cloudremoval/      model, data pipeline, losses, metrics, tiled inference (shared by training and server)
scripts/           train.py, evaluate.py, make_toy_dataset.py
configs/rice1.yaml every training setting in one file
notebooks/         train_colab.ipynb (one-click training on a free Colab GPU)
server/app/        FastAPI: /api/predict, /api/health, /api/model, /api/samples; serves the React site
web/               React + Vite + Tailwind front end
tests/             pytest: model, losses, metrics, data, API, end-to-end train → resume → evaluate
results/           metrics.json and examples produced by evaluate.py
Dockerfile         single container for Hugging Face Spaces
.github/workflows  CI (lint, tests, web build, Docker build) and auto-deploy to the Space
```

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

# put a trained checkpoint at weights/best.pt (download it from the Hugging Face model repo)
uvicorn server.app.main:app --port 7860 --reload           # API on :7860

cd web && npm install && npm run dev                        # site on :5173, forwards /api to :7860
```

Or run the production container: `docker build -t cloud-removal . && docker run -p 7860:7860 -e MODEL_REPO=YOUR-HF-USERNAME/cloud-removal-swin cloud-removal`

## Train and evaluate

Open `notebooks/train_colab.ipynb` in Colab with a T4 GPU and run all cells (about 1–1.5 hours). It resumes
automatically after a disconnect. Or from a terminal:

```bash
python scripts/train.py --config configs/rice1.yaml --data-root /path/to/RICE --out-dir runs/rice1
python scripts/evaluate.py --checkpoint runs/rice1/best.pt --data-root /path/to/RICE --split runs/rice1/splits.json
```

## API

```bash
curl -F file=@cloudy.png https://YOUR-SPACE-URL/api/predict
```

Returns JSON with `result` (cleaned PNG as a data URL), `changes` (overlay of what changed), `inference_ms`,
`changed_fraction` and the output size. Interactive docs are at `/api/docs`. Uploads are limited to 10 MB, images
larger than 1024 px are scaled down, and each IP can make 12 requests per minute.

## Tests

```bash
pytest        # 28 tests, ~25 s on CPU, no downloads needed
ruff check .
```

## Limitations

- Under thick cloud the ground is not visible, so the model reconstructs it from the surroundings. Treat those
  regions as a plausible guess, not a measurement.
- Trained on RICE1 (Google Earth RGB imagery, 500 pairs). Results on other sensors or multispectral data will be weaker.

## Credits

- Dataset: D. Lin et al., *A Remote Sensing Image Dataset for Cloud Removal*, arXiv:1901.00600, 2019
  ([RICE](https://github.com/BUPTLdy/RICE_DATASET)).
- Architecture ideas: Swin Transformer (Liu et al., 2021), SwinIR (Liang et al., 2021), Swin-Unet (Cao et al., 2021).

MIT licensed.
