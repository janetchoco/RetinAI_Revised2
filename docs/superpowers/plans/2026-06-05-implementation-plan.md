# Implementation Plan: Preprocessing + Dataset Re-split

**Spec:** `docs/superpowers/specs/2026-06-05-preprocessing-resplit-design.md`
**Date:** 2026-06-05

---

## Step 1 — Bootstrap project from RetinAI_Revised

Copy the following from `/Users/j.kupt/Documents/RetinAI_Revised/` into `RetinAI_Revised2/`:

```
src/
configs/
frontend/
spaces/
requirements.txt
pyproject.toml
RUNBOOK.md   (will be edited in Step 11)
CLAUDE.md    (will be edited in Step 12)
```

Do NOT copy `train/`, `test/`, `artifacts/`, `hf_space_tmp/`, or any `.docx` files.

---

## Step 2 — Set up GitHub repository

### 2a — Create `.gitignore`

Create `.gitignore` in the repo root. Start from the one in `RetinAI_Revised/` and add two new entries:

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/

# Data and model artifacts (too large for GitHub — store in Google Drive)
/train/
/test/
/data/
/artifacts/

# Optuna database
*.db

# Office documents
*.docx
*.doc
*.xlsx
*.pptx

# macOS
.DS_Store

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Environment
.env
venv/
.venv/

# Scratch
hf_space_tmp/
```

Key additions vs. the old project: `/data/` (covers `data/preprocessed/` if run locally) and `hf_space_tmp/`.

### 2b — Initialise git and make first commit

```bash
cd /Users/j.kupt/Documents/RetinAI_Revised2
git init
git add .
git commit -m "Initial commit: bootstrap from RetinAI_Revised"
```

### 2c — Create GitHub repo and push

Using the GitHub CLI (`gh`):

```bash
gh repo create RetinAI_Revised2 --private --source=. --remote=origin --push
```

This creates the repo, sets the remote, and pushes in one command. If you prefer to create it manually on github.com first, then:

```bash
git remote add origin https://github.com/<your-username>/RetinAI_Revised2.git
git branch -M main
git push -u origin main
```

### 2d — Connect Vercel to the new repo

1. Go to [vercel.com](https://vercel.com) → Add New Project → Import from GitHub → select `RetinAI_Revised2`
2. Set **Root Directory** to `frontend`
3. Set environment variable: `REACT_APP_API_URL=https://janetkupt-22-retinai-compare.hf.space`
4. Deploy

Vercel will auto-deploy on every push to `main` from this point on.

---

## Step 3 — Add new config keys to `DataConfig` and `base.yaml`

**File:** `src/retinai/config.py`

Add four fields to `DataConfig`:

```python
@dataclass
class DataConfig:
    train_dir: str = "train"           # already exists
    test_dir: str = "test"             # already exists
    manifest_path: str = "artifacts/manifest.csv"  # already exists
    image_size: int = 224              # already exists
    class_names: list[str] = field(default_factory=lambda: ["normal", "amd", "dr"])  # already exists
    patient_id_regex: str | None = None  # already exists
    # --- new fields ---
    preprocessed_dir: str = "data/preprocessed"
    crop_threshold: int = 10
    test_split_ratio: float = 0.20
    split_seed: int = 42
```

**File:** `configs/base.yaml`

Add under the `data:` section:

```yaml
data:
  preprocessed_dir: data/preprocessed
  crop_threshold: 10
  test_split_ratio: 0.20
  split_seed: 42
```

(`train_dir` and `test_dir` already exist in the config — leave them as-is.)

---

## Step 4 — Create `src/retinai/preprocess/` module

### `src/retinai/preprocess/__init__.py`
Empty file.

### `src/retinai/preprocess/cropper.py`

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _crop_black(img: Image.Image, threshold: int) -> Image.Image:
    gray = np.array(img.convert("L"))
    mask = gray > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img  # all black — return as-is
    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return img.crop((cmin, rmin, cmax + 1, rmax + 1))


def run_preprocess(
    src_dirs: list[tuple[Path, str]],   # (directory, prefix)  e.g. (Path("train"), "train")
    out_dir: Path,
    class_names: list[str],
    threshold: int = 10,
) -> None:
    """
    Crop black backgrounds from all images in src_dirs and save to out_dir/{class}/.
    Output filenames are prefixed with the source directory name to avoid collisions.
    Idempotent: existing output files are skipped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for class_name in class_names:
        (out_dir / class_name).mkdir(parents=True, exist_ok=True)

    total = skipped = saved = 0
    for src_root, prefix in src_dirs:
        for img_path in sorted(src_root.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = img_path.parent.name.lower().removeprefix("train_").removeprefix("test_")
            if label not in class_names:
                continue
            total += 1
            out_name = f"{prefix}_{img_path.name}"
            out_path = out_dir / label / out_name
            if out_path.exists():
                skipped += 1
                continue
            with Image.open(img_path) as img:
                cropped = _crop_black(img.convert("RGB"), threshold)
                cropped.save(out_path)
            saved += 1

    print(f"preprocess: total={total}  saved={saved}  skipped(already exist)={skipped}")
```

---

## Step 5 — Register `preprocess` command in `cli.py`

**File:** `src/retinai/cli.py`

### Add import at the top
```python
from retinai.preprocess.cropper import run_preprocess
```

### Add handler function
```python
def cmd_preprocess(args):
    cfg = load_config(args.config)
    src_dirs = [
        (Path(cfg.data.train_dir), "train"),
        (Path(cfg.data.test_dir), "test"),
    ]
    run_preprocess(
        src_dirs=src_dirs,
        out_dir=Path(cfg.data.preprocessed_dir),
        class_names=cfg.data.class_names,
        threshold=cfg.data.crop_threshold,
    )
```

### Add subparser (inside `build_parser()`, before the `return parser` line)
```python
p_preprocess = sub.add_parser("preprocess")
p_preprocess.add_argument("--config", required=True)
p_preprocess.set_defaults(func=cmd_preprocess)
```

---

## Step 6 — Modify `build_manifest` to scan preprocessed dir + add split column

**File:** `src/retinai/data/manifest.py`

Add import at top:
```python
from sklearn.model_selection import train_test_split
```

Replace the `build_manifest` function:

```python
def build_manifest(
    preprocessed_dir: str,
    class_names: list[str],
    test_split_ratio: float = 0.20,
    split_seed: int = 42,
) -> pd.DataFrame:
    root = Path(preprocessed_dir)
    rows: list[dict] = []
    class_to_id = {name: i for i, name in enumerate(class_names)}
    for image_path in _iter_images(root):
        label = image_path.parent.name.lower()
        if label not in class_to_id:
            continue
        rows.append({
            "image_path": str(image_path),
            "class_name": label,
            "class_id": class_to_id[label],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No labeled images found under {preprocessed_dir}.")

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_split_ratio,
        stratify=df["class_id"],
        random_state=split_seed,
    )
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    return df
```

Note: images in `preprocessed_dir` are already in `{class}/` subdirectories with no `train_`/`test_` prefix, so the label is just the folder name directly.

---

## Step 7 — Update `cmd_build_manifest` in `cli.py`

**File:** `src/retinai/cli.py`

Replace the existing `cmd_build_manifest` function:

```python
def cmd_build_manifest(args):
    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)
    df = build_manifest(
        preprocessed_dir=cfg.data.preprocessed_dir,
        class_names=cfg.data.class_names,
        test_split_ratio=cfg.data.test_split_ratio,
        split_seed=cfg.data.split_seed,
    )
    valid_df, bad_df = validate_manifest_images(df)
    Path(cfg.data.manifest_path).parent.mkdir(parents=True, exist_ok=True)
    valid_df.to_csv(cfg.data.manifest_path, index=False)
    bad_path = str(Path(cfg.data.manifest_path).with_name("quarantine_bad_images.csv"))
    bad_df.to_csv(bad_path, index=False)
    train_n = (valid_df["split"] == "train").sum()
    test_n = (valid_df["split"] == "test").sum()
    print(f"manifest={cfg.data.manifest_path} train={train_n} test={test_n} bad={len(bad_df)}")
```

---

## Step 8 — Filter `splits.py` to training pool only

**File:** `src/retinai/data/splits.py`

In `assign_stratified_folds`, add one filter line at the start:

```python
def assign_stratified_folds(df: pd.DataFrame, n_splits: int, seed: int) -> pd.DataFrame:
    df = df[df["split"] == "train"].copy()   # <-- add this line
    out = df.copy()
    out["fold"] = -1
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(splitter.split(out["image_path"], out["class_id"])):
        out.loc[valid_idx, "fold"] = fold
    return out
```

For `repeated_folds_indices` — it receives a `y` array directly, so the filter must happen at the call site. In `eval/repeated_cv.py` and `hpo/optuna_runner.py`: wherever they pass `manifest` into splitting functions, add `manifest = manifest[manifest["split"] == "train"]` before that call.

---

## Step 9 — Update `cmd_evaluate` to read test split from manifest

**File:** `src/retinai/cli.py`

Replace the existing `cmd_evaluate` function:

```python
def cmd_evaluate(args):
    cfg = load_config(args.config)
    manifest = pd.read_csv(cfg.data.manifest_path)
    test_df_raw = manifest[manifest["split"] == "test"].copy()
    test_df, bad_df = validate_manifest_images(test_df_raw)
    if bad_df is not None and len(bad_df):
        print(f"Skipping {len(bad_df)} corrupt test images.")
    out_dir = args.out_dir if args.out_dir else f"{cfg.runtime.artifact_root}/eval/{args.model_name}"
    evaluate_on_test(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        test_df=test_df,
        class_names=cfg.data.class_names,
        output_dir=out_dir,
        image_size=cfg.data.image_size,
        device=cfg.runtime.device,
        n_bootstrap=args.n_bootstrap,
    )
    print(f"Results saved to {out_dir}")
```

Also update `cmd_prior_correct` the same way — replace its `build_manifest(cfg.data.test_dir, ...)` call with the manifest-filter pattern above.

---

## Step 10 — Add loss curve plot to `cmd_train_final`

**File:** `src/retinai/cli.py`

At the end of `cmd_train_final`, after writing `train_summary.json`, add:

```python
    # Generate loss curve from history.csv
    history_path = Path(out_dir) / "history.csv"
    if history_path.exists():
        hist = pd.read_csv(history_path)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(hist["epoch"] if "epoch" in hist.columns else range(1, len(hist) + 1),
                hist["train_loss"], label="train_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss — {args.model_name}")
        ax.legend()
        fig.tight_layout()
        loss_curve_path = Path(out_dir) / "loss_curve.png"
        fig.savefig(loss_curve_path, dpi=150)
        plt.close(fig)
        print(f"loss_curve={loss_curve_path}")
```

Add `import matplotlib.pyplot as plt` to the top of `cli.py` if not already present.

---

## Step 11 — Update `RUNBOOK.md`

Replace the content of `RUNBOOK.md` with the updated version:

- Add **Colab session setup block** at the very top:
  ```python
  # Mount Drive
  from google.colab import drive
  drive.mount('/content/drive')

  # Clone or pull repo
  !git clone https://github.com/<your-username>/RetinAI_Revised2.git
  %cd RetinAI_Revised2

  # Install dependencies
  !pip install -r requirements.txt

  # Symlink source data (read-only)
  !ln -sf /content/drive/MyDrive/RetinAI_Revised_Data/train train
  !ln -sf /content/drive/MyDrive/RetinAI_Revised_Data/test test

  # Symlink outputs (read/write) — mkdir creates folders on Drive if missing
  !mkdir -p /content/drive/MyDrive/RetinAI_Revised2_Output/preprocessed
  !mkdir -p /content/drive/MyDrive/RetinAI_Revised2_Output/artifacts
  !ln -sf /content/drive/MyDrive/RetinAI_Revised2_Output/preprocessed data/preprocessed
  !ln -sf /content/drive/MyDrive/RetinAI_Revised2_Output/artifacts artifacts
  ```
- Add **Phase 0** (preprocess + build-manifest) before existing Phase 1
- Change all `train-final` commands to use `--epochs 100` with no `--val-split`
- Note that `loss_curve.png` is auto-saved after each `train-final`
- Remove the "find average stopping epoch" Python snippet (no longer needed)

---

## Step 12 — Update `CLAUDE.md`

- Update the "Data flow" section: Step 1 now reads from `data/preprocessed/` (not `train/`), manifest now has a `split` column
- Update the "Final model epoch selection" section: remove the average-stopping-epoch calculation; replace with "100 epochs, no early stopping"
- Update the "Google Colab notes" section: replace the old Drive restore pattern with the new four-symlink block
- Update the artifact layout to include `loss_curve.png` under `final2/{model}/`

---

## Step 13 — Smoke test + final push

### Smoke test (local, no GPU needed)
```bash
# Confirm preprocess command is registered
PYTHONPATH=src python -m retinai preprocess --help

# Confirm build-manifest command is registered
PYTHONPATH=src python -m retinai build-manifest --help

# If source data is available locally, run end-to-end:
PYTHONPATH=src python -m retinai preprocess --config configs/base.yaml
PYTHONPATH=src python -m retinai build-manifest --config configs/base.yaml
python -c "import pandas as pd; df=pd.read_csv('artifacts/manifest.csv'); print(df['split'].value_counts())"
```

### Push to GitHub
```bash
git add .
git commit -m "Add preprocessing + dataset re-split pipeline"
git push
```

Full pipeline validation (HPO → repeated CV → train-final → evaluate) happens on Colab where GPU is available.

---

## Execution order summary

| Step | File(s) changed | Action |
|---|---|---|
| 1 | — | Copy source from RetinAI_Revised |
| 2 | `.gitignore` | Create .gitignore, init git, create GitHub repo, connect Vercel |
| 3 | `config.py`, `base.yaml` | Add 4 new config fields |
| 4 | `src/retinai/preprocess/` (new) | Create cropper module |
| 5 | `cli.py` | Register `preprocess` command |
| 6 | `data/manifest.py` | Scan preprocessed dir + add split column |
| 7 | `cli.py` | Update `cmd_build_manifest` |
| 8 | `data/splits.py`, `eval/repeated_cv.py`, `hpo/optuna_runner.py` | Filter to train split |
| 9 | `cli.py` | Update `cmd_evaluate` + `cmd_prior_correct` |
| 10 | `cli.py` | Add loss curve plot to `cmd_train_final` |
| 11 | `RUNBOOK.md` | Full rewrite with new phases + Colab setup block |
| 12 | `CLAUDE.md` | Update data flow + Drive sections |
| 13 | — | Smoke test + push to GitHub |
