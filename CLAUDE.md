# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

All commands must be run from the repo root with `PYTHONPATH=src`. See `RUNBOOK.md` for the full phased sequence with exact best HPO parameters for all 3 models.

```bash
# Crop black backgrounds from all images (run once before build-manifest)
PYTHONPATH=src python -m retinai preprocess --config configs/base.yaml

# Build image manifest with stratified 80/20 train/test split
PYTHONPATH=src python -m retinai build-manifest --config configs/base.yaml

# Hyperparameter optimisation (50 trials × 5-fold CV)
PYTHONPATH=src python -m retinai hpo --config configs/base.yaml --model-name efficientnet_b0

# Repeated CV (10 × 5-fold = 50 runs) with best HPO params
PYTHONPATH=src python -m retinai repeat-cv --config configs/base.yaml --model-name efficientnet_b0 --lr 5.6e-4 --weight-decay 1.3e-6 --batch-size 32

# Train final model on 100% training pool, 100 epochs, no early stopping
PYTHONPATH=src python -m retinai train-final --config configs/base.yaml --model-name efficientnet_b0 --lr 5.6e-4 --weight-decay 1.3e-6 --batch-size 32 --epochs 100 --out-dir artifacts/final/efficientnet_b0

# Evaluate on held-out test split (read from manifest split==test)
PYTHONPATH=src python -m retinai evaluate --config configs/base.yaml --checkpoint artifacts/final/efficientnet_b0/best.pt --model-name efficientnet_b0

# Statistical comparison across all 3 models (Wilcoxon + Holm-Bonferroni + effect r)
PYTHONPATH=src python -m retinai compare-models --config configs/base.yaml

# Grad-CAM localisation (comma-separated images)
PYTHONPATH=src python -m retinai localize --config configs/base.yaml --checkpoint artifacts/final/efficientnet_b0/best.pt --model-name efficientnet_b0 --images path/to/img.png --output-dir artifacts/gradcam/efficientnet_b0

# Bayesian prior correction for label shift (saves before/after metrics)
PYTHONPATH=src python -m retinai prior-correct --config configs/base.yaml --checkpoint artifacts/final/efficientnet_b0/best.pt --model-name efficientnet_b0
```

## Architecture

### Config system
`configs/base.yaml` is the single source of truth. Model-specific yaml files (`configs/efficientnet_b0.yaml` etc.) only override `train.model_name`. `load_config()` in `src/retinai/config.py` merges them into a typed `AppConfig` dataclass. All pipeline code receives `cfg: AppConfig` — never raw dicts.

### Data flow
1. `preprocess/cropper.py` scans `train/` and `test/`, crops black backgrounds (grayscale threshold > 10), saves to `data/preprocessed/{class}/` with `train_`/`test_` filename prefix to avoid collisions
2. `data/manifest.py` scans `data/preprocessed/` → performs stratified 80/20 split (seed=42) → writes `artifacts/manifest.csv` with a `split` column (`train` or `test`)
3. `data/splits.py` filters manifest to `split == 'train'`, then assigns stratified folds (HPO) or repeated stratified folds (repeat-cv) — both use `seed=42` so splits are **paired across models**
4. `data/datasets.py` (`RetinalDataset`) applies torchvision transforms: resize 224, RandomHorizontalFlip + RandomRotation(10°) for train, ToTensor only for val/test. **No ImageNet normalisation** (intentional — omitted for consistency across all 3 models)

### Training engine
`train/engine.py:run_training()` is the single training function used by HPO, repeat-cv, and train-final. Key behaviours:
- `valid_ds=None` → trains all epochs with no early stopping (used by train-final)
- `epoch_callback(epoch, metric)` → used by Optuna pruner during HPO
- Returns `training_seconds` in result dict — consumed by repeat-cv and train-final for reporting

### HPO
`hpo/optuna_runner.py` creates one Optuna study per model in `artifacts/optuna.db` (SQLite). `load_if_exists=True` + counting completed trials before calling `study.optimize()` ensures exactly `n_trials` completed trials regardless of restarts. Pruner: `MedianPruner(n_startup_trials=5, n_warmup_steps=10)`. Batch size search spaces differ by model: EfficientNet `[8,16,32]`, DeiT/ResNet18 `[16,32,64,128]`.

### Evaluation
- `eval/repeated_cv.py` → saves `scores.csv` after **every run** (resume-safe: re-running the command skips already-completed runs by reading existing `scores.csv`). Also saves `cv_summary.json` per model.
- `eval/test_evaluator.py` → full test-set metrics, bootstrap CI (1000 resamples), ROC Type 1 plot (per-model, 3 class curves), PR-AUC, FLOPs via `eval/efficiency.py`. Reads test images from `manifest.csv` where `split == 'test'`. Handles missing classes gracefully (sets AUC to NaN for absent classes, macro AUC = mean of available classes).
- `eval/compare.py` → reads all models' `scores.csv`, computes 95% CI, pairwise Wilcoxon signed-rank, Holm-Bonferroni correction, effect size r, ROC Type 2 plots (per-disease, 3 model curves), AUC comparison table

### Supported models
Defined in `models/factory.py:SUPPORTED_MODELS`. Adding a new model requires: entry in `SUPPORTED_MODELS`, batch size choices in `optuna_runner.py`, and adding the model name to all `choices=` lists in `cli.py`.

### Artifact layout
```
artifacts/
  manifest.csv                         # image_path, class_name, class_id, split (train/test)
  optuna.db                            # all HPO studies
  repeated_cv/{model}/
    scores.csv                         # macro_f1 + training_seconds × 50 runs
    cv_summary.json                    # mean±SD F1, mean±SD time, params
    run_{i}/history.csv                # per-epoch train_loss, val_loss, val_macro_f1
  final/{model}/
    best.pt                            # final model weights
    history.csv
    loss_curve.png                     # epoch vs train_loss, auto-generated by train-final
    train_summary.json                 # training time
  eval/{model}/
    classification_report.csv
    confusion_matrix.csv
    auc_scores.csv / pr_auc_scores.csv
    bootstrap_ci.json
    roc_per_model.png                  # ROC Type 1
    summary.json                       # FLOPs + all macro metrics
  comparison/
    model_ci_summary.csv
    wilcoxon_pairwise.csv
    auc_comparison_table.csv
    roc_disease_{class}.png            # ROC Type 2
```

### Final model training
`train-final` uses **100 epochs with no early stopping** (`valid_ds=None`, no `--val-split`). After training completes, a `loss_curve.png` (epoch vs. train_loss) is automatically saved to the model's output directory.

### Prior correction
`eval/prior_correction.py` applies Bayesian label-shift correction: `p_corrected ∝ p_model × (test_prior / train_prior)`. Computes train prior from `manifest.csv` (split==train rows), test prior from manifest split==test rows. Saves `prior_correction_summary.json` with before/after macro_f1, macro_auc, per-class metrics, and confusion matrices to `artifacts/prior_correction/{model}/`.

### scores.csv resume caveat
`eval/repeated_cv.py` saves `scores.csv` after every run. If `training_seconds` is corrupted (e.g. manually edited to 0), recover exact values from `history.csv` — each run's `history.csv` has a `seconds` column (cumulative elapsed time); the last row = total training time for that run:
```python
import pandas as pd, os
from pathlib import Path
for model in ["efficientnet_b0", "deit_tiny", "resnet18"]:
    base = Path(f"artifacts/repeated_cv/{model}")
    scores_df = pd.read_csv(base / "scores.csv")
    for i in range(len(scores_df)):
        hist_path = base / f"run_{i}/history.csv"
        if hist_path.exists():
            scores_df.at[i, "training_seconds"] = round(
                float(pd.read_csv(hist_path)["seconds"].iloc[-1]), 1
            )
    scores_df.to_csv(base / "scores.csv", index=False)
```

### compare-models timing
Run `compare-models` only after **all 50 runs are complete for every model** — it uses `min(len(s1), len(s2))` for pairing, so incomplete `scores.csv` files silently reduce the paired sample size and invalidate the Wilcoxon test.

### Localization modules
`localization/gradcam_runner.py` is the active method. `localization/pylon_runner.py` also exists (PYLON-style localization) but is not used in the final analysis and has no CLI entry point.

### DeiT Grad-CAM reshape
`localization/gradcam_runner.py:_reshape_transform_deit()` computes `n_skip = tensor.size(1) - 14*14` dynamically. Older timm includes a distillation token (skip 2); newer timm (1.0+) has only CLS token (skip 1). The dynamic computation handles both. Do not hardcode `tensor[:, 2:, :]`.

### Grad-CAM image aspect ratio
`_load_image()` returns `orig_size` (original PIL dimensions). After generating the 224×224 overlay, `generate_gradcam_maps()` resizes back to `orig_size` with `Image.LANCZOS` before saving, so landscape retinal images are not squished.

## Frontend & API

### Running the API backend
Run from repo root (checkpoints must exist at `artifacts/final/{model}/best.pt`):
```bash
PYTHONPATH=src uvicorn retinai.api.server:app --reload --host 0.0.0.0 --port 8000
```
Check health: `curl http://localhost:8000/health`

On Colab, expose the port with `ngrok` or Colab port forwarding, then update `frontend/.env`:
```
REACT_APP_API_URL=https://<your-ngrok-url>
```

### Running the React frontend
```bash
cd frontend
npm install
npm start      # dev server on http://localhost:3000
npm run build  # production build → frontend/build/
```

The frontend shows 4 panels: original scan + one card per model (EfficientNet-B0, DeiT-Tiny, ResNet18). Each model card displays: Grad-CAM overlay (toggleable), diagnosis badge, confidence score, and per-class probability bars.

### API response shape
`POST /predict` accepts `multipart/form-data` with a `file` field.
Returns:
```json
{
  "models": {
    "efficientnet_b0": {"diagnosis": "normal", "confidence": 0.95, "probabilities": [...], "gradcam": "data:image/png;base64,...", "label": "EfficientNet-B0"},
    "deit_tiny": { ... },
    "resnet18": { ... }
  },
  "class_names": ["normal", "amd", "dr"]
}
```

## HF Spaces + Vercel deployment

`spaces/` contains a self-contained Docker deployment for Hugging Face Spaces.
- `spaces/app.py` — FastAPI server, loads all 3 checkpoints from HF model repo `janetkupt-22/ReinAI_Multiclass` on startup, serves `POST /predict` on port 7860
- `spaces/Dockerfile` — CPU-only torch to keep image size manageable
- Space URL: `https://janetkupt-22-retinai-compare.hf.space`

Upload models to HF Hub (run once from Colab after training):
```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("ReinAI_Multiclass", repo_type="model", exist_ok=True)
for model in ["efficientnet_b0", "deit_tiny", "resnet18"]:
    api.upload_file(path_or_fileobj=f"artifacts/final/{model}/best.pt",
                    path_in_repo=f"{model}/best.pt",
                    repo_id="janetkupt-22/ReinAI_Multiclass")
```

Frontend (`frontend/`) deploys to Vercel from GitHub. Set env var in Vercel dashboard:
`REACT_APP_API_URL=https://janetkupt-22-retinai-compare.hf.space`

## Google Colab notes

All data and outputs live on Google Drive — nothing is stored locally on the Colab VM across sessions.

**Drive folders:**
- `MyDrive/RetinAI_Revised_Data/` — source images (read-only): `train/`, `test/`
- `MyDrive/RetinAI_Revised2_Output/` — all outputs (read/write): `preprocessed/`, `artifacts/`

**Session setup** (see RUNBOOK.md Colab Setup block):
```python
!ln -sf /content/drive/MyDrive/RetinAI_Revised_Data/train train
!ln -sf /content/drive/MyDrive/RetinAI_Revised_Data/test test
!ln -sf /content/drive/MyDrive/RetinAI_Revised2_Output/preprocessed data/preprocessed
!ln -sf /content/drive/MyDrive/RetinAI_Revised2_Output/artifacts artifacts
```

All writes to `data/preprocessed/` and `artifacts/` persist directly to Drive via symlink — no manual backup step needed.

**`num_workers` by GPU:**
- T4 / L4: set to `2` (system max)
- A100: set to `4`

**Critical: copy images to local disk before training** — reading from Drive via symlink is slow and leaves the GPU underutilised (GPU RAM stays near 0). After setup, copy once per session:
```python
!cp -r /content/drive/MyDrive/RetinAI_Revised2_Output/preprocessed /content/preprocessed
!ln -sfn /content/preprocessed data/preprocessed
```
This re-points `data/preprocessed` to local SSD. The `artifacts/` symlink still goes to Drive so all outputs are persisted.

**`data/` directory must exist before symlinking** — `ln -sf` on `data/preprocessed` fails if `data/` doesn't exist yet. Always run `mkdir -p data` before the symlink line.

**Check HPO progress while it's running** (from Colab Terminal or a second notebook):
```python
import optuna
s = optuna.load_study(study_name="efficientnet_b0_hpo_224", storage="sqlite:///artifacts/optuna.db")
print(f"Complete: {len([t for t in s.trials if t.state.name=='COMPLETE'])}")
print(f"Pruned:   {len([t for t in s.trials if t.state.name=='PRUNED'])}")
if s.best_trial: print(f"Best: {s.best_value:.4f}  params: {s.best_params}")
```
Study names follow the pattern `{model_name}_hpo_224`.
