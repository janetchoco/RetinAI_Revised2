# Design: Black-Background Cropping + Dataset Re-split

**Date:** 2026-06-05
**Status:** Approved

---

## Source reference

The new project (`RetinAI_Revised2`) is built on top of the previous project at `/Users/j.kupt/Documents/RetinAI_Revised/`. All existing code (`src/`, `configs/`, `spaces/`, `frontend/`, `requirements.txt`, etc.) is copied from there as the starting point. The frontend must look **exactly the same** as `RetinAI_Revised/frontend/` — no visual or functional changes.

---

## Overview

Two changes are added to the RetinAI pipeline before any model training:

1. **Preprocessing** — crop black backgrounds off all images (from both `train/` and `test/`) and save the cropped versions to Google Drive at `data/preprocessed/`.
2. **Re-split** — combine the full image pool and do a stratified 80/20 split into a training pool (80%) and a held-out external test set (20%). The split is encoded as a `split` column in `artifacts/manifest.csv`.

All downstream phases (HPO, repeated CV, final training, evaluation, Grad-CAM, prior correction, API, frontend) are unchanged.

---

## Pipeline Sequence

```
Phase 0 (NEW):  retinai preprocess       → data/preprocessed/{class}/   (on Google Drive)
Phase 1:        retinai build-manifest   → artifacts/manifest.csv        (now includes split column)
Phase 2–6:      HPO → repeat-cv → compare-models → train-final → evaluate → localize
                                          (unchanged; read split from manifest transparently)
```

---

## Phase 0: `retinai preprocess`

### Purpose
Crop the black border from every retinal image and save results to a preprocessed directory. Run once before `build-manifest`. Idempotent — existing output files are skipped.

### Input directories
- `train/` (symlinked to Google Drive at runtime)
- `test/` (symlinked to Google Drive at runtime)

### Output directory
- `data/preprocessed/{class}/` (symlinked to Google Drive at runtime)
- Folder names strip the `train_`/`test_` prefix to match `class_names`, consistent with the current `build-manifest` behaviour.

### Cropping algorithm
1. Load image (PIL)
2. Convert to grayscale
3. Build binary mask: pixels with intensity > `crop_threshold` (default 10) are signal
4. Find bounding box of signal pixels (min/max row and col across the mask)
5. Crop the original colour image to that bounding box
6. Save as PNG with filename prefixed by source directory: `train_<original>` or `test_<original>` — prevents collisions when the same filename exists in both `train/` and `test/` for the same class

A threshold of 10 handles slight noise in nominally-black borders without clipping real retinal tissue.

### New files
- `src/retinai/preprocess/cropper.py` — cropping logic
- `src/retinai/preprocess/__init__.py`
- CLI entry point `preprocess` registered in `src/retinai/cli.py`

---

## Phase 1: Modified `build-manifest`

### Changes
- Scans `data/preprocessed/` (via `data.preprocessed_dir` config key) instead of `train/`
- After building the full image list, performs a **stratified 80/20 split** using `sklearn.model_selection.train_test_split` with `stratify=labels` and `random_state=split_seed` (default 42)
- Adds a `split` column to `artifacts/manifest.csv` with values `train` or `test`

### New manifest columns
| column | values |
|---|---|
| `image_path` | path relative to repo root |
| `label` | class name string |
| `split` | `train` or `test` |

### Reproducibility
`split_seed=42` matches the seed used throughout the rest of the pipeline. The same image pool always produces the same split.

---

## Downstream Changes

### `data/splits.py`
- Filter manifest to `split == 'train'` before building k-folds
- 80% training pool is then divided into 5 stratified folds exactly as today
- `seed=42`, cross-model pairing: unchanged

### `eval/test_evaluator.py`
- Read image paths from manifest where `split == 'test'` instead of walking `test/` directory
- Labels from the manifest `label` column
- All metrics (macro F1, AUC, bootstrap CI, ROC plots): unchanged

### `train-final`
- Uses **100 epochs with no early stopping** (`valid_ds=None`, no `--val-split`) — the average CV stopping-epoch approach is dropped
- After training completes, generates a **train-loss curve** PNG from `history.csv` and saves it to `artifacts/final2/{model}/loss_curve.png`
- The plot is a simple epoch vs. train_loss line chart; no val_loss line (none recorded without a val split)

### Everything else
No changes to: `hpo/optuna_runner.py`, `eval/repeated_cv.py`, `eval/compare.py`, `train/engine.py`, `eval/prior_correction.py`, `localization/`, API, frontend, HF Spaces.

---

## Config Changes (`configs/base.yaml`)

```yaml
data:
  train_dir: train                  # symlinked to MyDrive/RetinAI_Revised_Data/train
  test_dir: test                    # symlinked to MyDrive/RetinAI_Revised_Data/test
  preprocessed_dir: data/preprocessed  # symlinked to MyDrive/RetinAI_Revised2_Output/preprocessed
  crop_threshold: 10
  test_split_ratio: 0.20
  split_seed: 42
```

`artifacts/` is symlinked to `MyDrive/RetinAI_Revised2_Output/artifacts` — no config key needed since `artifacts/` is the hardcoded output root used throughout the pipeline.

---

## Google Drive Integration

Two Drive folders are used:

| Drive folder | Purpose | Access |
|---|---|---|
| `MyDrive/RetinAI_Revised_Data` | Source images (`train/`, `test/`) | Read only |
| `MyDrive/RetinAI_Revised2_Output` | All outputs: preprocessed images + all artifacts | Read/Write |

All paths in the pipeline (code, config) use local symlinked names — nothing in the codebase references Drive paths directly.

The old manual artifacts copy/restore step is eliminated: since `artifacts/` is symlinked directly to Drive, writes persist automatically.

---

## RUNBOOK Changes

### Colab setup block — full symlink block
```python
# Source data (read-only)
!ln -sf /content/drive/MyDrive/RetinAI_Revised_Data/train train
!ln -sf /content/drive/MyDrive/RetinAI_Revised_Data/test test

# Outputs (read/write) — create folders on Drive first if they don't exist
!mkdir -p /content/drive/MyDrive/RetinAI_Revised2_Output/preprocessed
!mkdir -p /content/drive/MyDrive/RetinAI_Revised2_Output/artifacts
!ln -sf /content/drive/MyDrive/RetinAI_Revised2_Output/preprocessed data/preprocessed
!ln -sf /content/drive/MyDrive/RetinAI_Revised2_Output/artifacts artifacts
```

### New Phase 0
```
## Phase 0 — Preprocessing (run once, then skip on subsequent sessions)

### 0) Crop black backgrounds and save to Drive
PYTHONPATH=src python -m retinai preprocess --config configs/base.yaml

### 1) Build manifest with 80/20 stratified split
PYTHONPATH=src python -m retinai build-manifest --config configs/base.yaml
```

All subsequent phases (HPO through Grad-CAM) are renumbered +1. Final training commands use `--epochs 100` with no `--val-split` flag and no average-stopping-epoch calculation. Example:

```
PYTHONPATH=src python -m retinai train-final --config configs/base.yaml --model-name efficientnet_b0 \
  --lr 0.0005618447545488722 --weight-decay 1.3129046436784662e-06 --batch-size 32 --epochs 100 \
  --out-dir artifacts/final2/efficientnet_b0
```

A `loss_curve.png` is automatically saved alongside `best.pt` after each `train-final` run.
