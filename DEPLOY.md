# From this folder to a live website, step by step

Each step says **what** to do and **why**. Total time is about 2 hours, most of it waiting for training.

---

## Step 1: Put the code on your own GitHub

```bash
cd cloud-removal
git init
git add .
git commit -m "Cloud removal: Swin U-Net, FastAPI, React"
git branch -M main
git remote add origin https://github.com/aaryanrana01/cloud-removal.git
git push -u origin main
```

First create an empty repo named `cloud-removal` on github.com (no README, so the push doesn't conflict).

**Why:** the Colab notebook downloads the code from GitHub, and the website is deployed from GitHub. A repo under
your own account is also what recruiters will open from your resume. The old repo can stay where it is.

**What happens next:** GitHub Actions runs the **CI** workflow automatically: lint, 28 tests, website build and
Docker build. Check the *Actions* tab; all three jobs should go green. The **Deploy** workflow will fail
for now. That's expected until Step 5.

---

## Step 2: Train the model on Colab

1. Open `notebooks/train_colab.ipynb` from your repo in Colab
   (in Colab: *File → Open notebook → GitHub*, paste your repo URL).
2. *Runtime → Change runtime type → T4 GPU.*
3. *Runtime → Run all.* Upload `kaggle.json` when asked (or put it in the top of your Google Drive once).

**Why Colab:** training needs a GPU and Colab's T4 is free.
**Why checkpoints go to Google Drive:** free Colab sessions can disconnect. If that happens, run all cells again
and training continues from the last finished epoch instead of starting over.

**What to check:** validation PSNR should rise fast for ~10 epochs, then slowly. The preview image in
cell 6 should show clouds fading. If PSNR is stuck or falls, stop and ask before going further.

---

## Step 3: Add the results to the repo

The notebook downloads `results_bundle.zip`. Unzip it **inside your repo folder**. It fills `results/` and
`server/samples/`. Then:

```bash
git add results server/samples
git commit -m "Add test results and demo samples"
git push
```

Also copy the table from `results/results.md` into the **Results** section of `README.md`.

**Why:** the website reads the real test numbers from `results/metrics.json` and shows the test images from
`server/samples/`. These are small files, so they belong in git. The model weights don't (next step).

---

## Step 4: Upload the model weights to Hugging Face

Done by cell 9 of the notebook. Before running it:
1. Sign up at huggingface.co.
2. *Settings → Access Tokens → Create new token*, type **Write**. Copy it.
3. In the notebook set `HF_USERNAME`, run the cell, paste the token.

**Why:** GitHub isn't meant for model files (they bloat the repo forever). A Hugging Face model repo stores
weights properly, shows your results as a model card, and the website downloads the weights from it at startup.

---

## Step 5: Connect GitHub to Hugging Face for automatic deploys

In your GitHub repo: *Settings → Secrets and variables → Actions*.

| Type | Name | Value |
|---|---|---|
| Secret | `HF_TOKEN` | the write token from Step 4 |
| Variable | `HF_SPACE` | `your-hf-username/cloud-removal` |
| Variable | `HF_MODEL_REPO` | `your-hf-username/cloud-removal-swin` |

Then *Actions → Deploy to Hugging Face → Run workflow*.

**Why a secret:** the token can write to your Hugging Face account, so GitHub keeps it hidden, even from logs.
**Why automatic deploys:** from now on every push to `main` that passes CI updates the live site by itself. You
never upload files to Hugging Face by hand, and broken code (failing tests) never gets deployed.

**What happens:** the workflow creates a Docker **Space**, copies the app into it, and tells it where the weights
are. Hugging Face then builds the Docker image (about 5–8 minutes the first time; watch the *Logs* tab on the
Space page).

---

## Step 6: Check the live site

- Open `https://huggingface.co/spaces/your-hf-username/cloud-removal`.
- Visit `/api/health` on the Space's direct URL (*⋮ menu → Embed this Space* shows it). It should say
  `"model_loaded": true`.
- Try a test image and upload one of your own.

Then put the real links in `README.md` (replace `YOUR-HF-USERNAME`) and add a screenshot as `docs/demo.png`.

**Good to know:** free Spaces go to sleep after 48 hours without visitors. The first visit after that takes about
a minute to wake up. Open it yourself before an interview.

---

## Renaming the project later

The visible name comes from one environment variable. In the Space: *Settings → Variables*, add
`APP_NAME = YourName`. For the repo, Space and model repo names, rename them on GitHub/Hugging Face and update
`HF_SPACE`, `HF_MODEL_REPO`, `web/src/config.js` and the links in `README.md`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Deploy workflow: "Set the HF_TOKEN secret" | Step 5 not done, or the variable names are misspelled. |
| Site loads, but "The model is not loaded yet" | `HF_MODEL_REPO` is wrong or `best.pt` wasn't uploaded. `/api/health` shows the exact error. |
| Colab: `CUDA out of memory` | In `configs/rice1.yaml` set `batch_size: 4` (or `grad_checkpoint: true`) and rerun. |
| Colab: "No folder with cloud/ and label/" | The Kaggle zip layout changed. Run `!find /content/data -maxdepth 3 -type d` and pass the RICE1 folder's parent as `--data-root`. |
