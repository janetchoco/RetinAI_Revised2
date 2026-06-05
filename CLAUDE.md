# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the pipeline

All commands must be run from the repo root with `PYTHONPATH=src`. See `RUNBOOK.md` for the full phased sequence with exact best HPO parameters for all 3 models.

```bash
# Build image manifest from folder-labelled train/ directory
PYTHONPATH=src python -m retinai build-manifest --config configs/base.yaml

# Hyperparameter optimisation (50 trials × 5-fold CV)
PYTHONPATH=src python -m retinai hpo --config configs/base.yaml --model-name efficientnet_b0

# Repeated CV (10 × 5-fold = 50 runs) with best HPO params
PYTHONPATH=src python -m retinai repeat-cv --config configs/base.yaml --model-name efficientnet_b0 --lr 5.6e-4 --weight-decay 1.3e-6 --batch-size 32

# Train final model on 100% training data (--epochs overrides config; --out-dir saves to custom path)
PYTHONPATH=src python -m retinai train-final --config configs/base.yaml --model-name efficientnet_b0 --lr 5.6e-4 --weight-decay 1.3e-6 --batch-size 32 --epochs 30 --out-dir artifacts/final2/efficientnet_b0

# Train final with early stopping via val split (10% held out for stopping only)
PYTHONPATH=src python -m retinai train-final --config configs/base.yaml --model-name efficientnet_b0 --lr 5.6e-4 --weight-decay 1.3e-6 --batch-size 32 --val-split 0.1 --out-dir artifacts/final2/efficientnet_b0

# Evaluate on held-out test set (--out-dir saves to custom path)
PYTHONPATH=src python -m retinai evaluate --config configs/base.yaml --checkpoint artifacts/final2/efficientnet_b0/best.pt --model-name efficientnet_b0 --out-dir artifacts/eval2/efficientnet_b0

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
1. `data/manifest.py` scans `train/` by folder name → strips `train_`/`test_` prefix to match `class_names` → writes `artifacts/manifest.csv`
2. `data/splits.py` assigns stratified folds (HPO) or repeated stratified folds (repeat-cv) — both use `seed=42` so splits are **paired across models**
3. `data/datasets.py` (`RetinalDataset`) applies torchvision transforms: resize 224, RandomHorizontalFlip + RandomRotation(10°) for train, ToTensor only for val/test. **No ImageNet normalisation** (intentional — omitted for consistency across all 3 models)

### Training engine
`train/engine.py:run_training()` is the single training function used by HPO, repeat-cv, and train-final. Key behaviours:
- `valid_ds=None` → trains all epochs with no early stopping (used by train-final)
- `epoch_callback(epoch, metric)` → used by Optuna pruner during HPO
- Returns `training_seconds` in result dict — consumed by repeat-cv and train-final for reporting

### HPO
`hpo/optuna_runner.py` creates one Optuna study per model in `artifacts/optuna.db` (SQLite). `load_if_exists=True` + counting completed trials before calling `study.optimize()` ensures exactly `n_trials` completed trials regardless of restarts. Pruner: `MedianPruner(n_startup_trials=5, n_warmup_steps=10)`. Batch size search spaces differ by model: EfficientNet `[8,16,32]`, DeiT/ResNet18 `[16,32,64,128]`.

### Evaluation
- `eval/repeated_cv.py` → saves `scores.csv` after **every run** (resume-safe: re-running the command skips already-completed runs by reading existing `scores.csv`). Also saves `cv_summary.json` per model.
- `eval/test_evaluator.py` → full test-set metrics, bootstrap CI (1000 resamples), ROC Type 1 plot (per-model, 3 class curves), PR-AUC, FLOPs via `eval/efficiency.py`. Handles missing classes in test set gracefully (sets AUC to NaN for absent classes, macro AUC = mean of available classes).
- `eval/compare.py` → reads all models' `scores.csv`, computes 95% CI, pairwise Wilcoxon signed-rank, Holm-Bonferroni correction, effect size r, ROC Type 2 plots (per-disease, 3 model curves), AUC comparison table

### Supported models
Defined in `models/factory.py:SUPPORTED_MODELS`. Adding a new model requires: entry in `SUPPORTED_MODELS`, batch size choices in `optuna_runner.py`, and adding the model name to all `choices=` lists in `cli.py`.

### Artifact layout
```
artifacts/
  manifest.csv
  optuna.db                          # all HPO studies
  repeated_cv/{model}/
    scores.csv                       # macro_f1 + training_seconds × 50 runs
    cv_summary.json                  # mean±SD F1, mean±SD time, params
    run_{i}/history.csv              # per-epoch train_loss, val_loss, val_macro_f1
  final/{model}/
    best.pt                          # final model weights
    history.csv
    train_summary.json               # training time
  eval/{model}/
    classification_report.csv
    confusion_matrix.csv
    auc_scores.csv / pr_auc_scores.csv
    bootstrap_ci.json
    roc_per_model.png                # ROC Type 1
    summary.json                     # FLOPs + all macro metrics
  final2/{model}/                      # retrained with fixed epochs (avg CV stopping epoch)
    best.pt
    train_summary.json
  eval2/{model}/                       # evaluation of final2 checkpoints
    classification_report.csv / confusion_matrix.csv
    auc_scores.csv / pr_auc_scores.csv
    bootstrap_ci.json / summary.json
    roc_per_model.png
  eval_threshold/{model}/              # post-hoc threshold-adjusted evaluation (not used in final analysis)
    classification_report_threshold.csv
    confusion_matrix_threshold.csv
    summary_threshold.json             # optimal thresholds + adjusted metrics
    roc_threshold.png
  comparison/
    model_ci_summary.csv
    wilcoxon_pairwise.csv
    auc_comparison_table.csv
    roc_disease_{class}.png          # ROC Type 2
```

### Final model epoch selection
Training 100 epochs with no validation causes AMD-biased collapse on external test set. Use `--epochs` set to the average early-stopping epoch from CV history:
```python
# Find average stopping epoch per model
import pandas as pd, glob
for model in ['efficientnet_b0', 'deit_tiny', 'resnet18']:
    epochs = [len(pd.read_csv(f)) for f in glob.glob(f'artifacts/repeated_cv/{model}/run_*/history.csv')]
    print(f"{model}: avg={sum(epochs)/len(epochs):.0f}")
# Results: efficientnet_b0=30, deit_tiny=25, resnet18=35
```

### Prior correction
`eval/prior_correction.py` applies Bayesian label-shift correction: `p_corrected ∝ p_model × (test_prior / train_prior)`. Computes train prior from `manifest.csv`, test prior from the test directory. Saves `prior_correction_summary.json` with before/after macro_f1, macro_auc, per-class metrics, and confusion matrices to `artifacts/prior_correction/{model}/`. Has CLI entry point `prior-correct` — see commands above.

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
`localization/gradcam_runner.py` is the active method used in the IS. `localization/pylon_runner.py` also exists (PYLON-style localization) but is not used in the final analysis and has no CLI entry point.

### DeiT Grad-CAM reshape
`localization/gradcam_runner.py:_reshape_transform_deit()` computes `n_skip = tensor.size(1) - 14*14` dynamically. Older timm includes a distillation token (skip 2); newer timm (1.0+) has only CLS token (skip 1). The dynamic computation handles both. Do not hardcode `tensor[:, 2:, :]`.

### Grad-CAM image aspect ratio
`_load_image()` returns `orig_size` (original PIL dimensions). After generating the 224×224 overlay, `generate_gradcam_maps()` resizes back to `orig_size` with `Image.LANCZOS` before saving, so landscape retinal images are not squished.

### External test set findings
Test set (n=1,401: Normal=1,000, DR=301, AMD=100) uses images from different sources than training — **source-class coupled**: Normal images all from Source C, DR+AMD all from Source B. This means cross-source generalization and disease classification ability cannot be separated.

CV macro F1 (~0.93) reflects within-dataset performance. External test macro F1 before prior correction: EfficientNet=0.431, DeiT=0.406, ResNet18=0.452. After Bayesian Prior Correction: EfficientNet=0.456, DeiT=0.457, ResNet18=0.506. Macro AUC ~0.84 for all models — AUC is the more reliable metric under prior shift.

AMD over-prediction is systematic across all models (AMD recall 0.94–1.00, precision 0.11–0.13) due to prior shift: AMD is ~28% of train but only ~7% of test. Threshold adjustment was evaluated but removed from the final methodology — scores.csv contains only macro_f1 per run (no per-image probabilities), making proper OOF-based threshold tuning impossible without re-running CV.

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

`spaces/` contains a self-contained Docker deployment for Hugging Face Spaces. `hf_space_tmp/` is a scratch directory (untracked, safe to ignore).
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

Data lives in Google Drive, not git. Symlink pattern used at runtime:
```python
!ln -sf /content/drive/MyDrive/RetinAI_Revised_Data/train train
!ln -sf /content/drive/MyDrive/RetinAI_Revised_Data/test test
```
`artifacts/` is restored from `/content/drive/MyDrive/RetinAI/artifacts/` at session start (not `RetinAI_data` — that only contains HPO outputs). The `optuna.db` contains all HPO results and must be restored before running any subsequent phase. `num_workers` in `base.yaml` should be set to `2` on T4/L4 Colab runtimes (system max).
