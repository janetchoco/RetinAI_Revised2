# RetinAI Runbook

## Colab Session Setup (run at the start of every session)

```python
from google.colab import drive
drive.mount('/content/drive')

# Clone repo (first session) or pull latest code (subsequent sessions)
import os
if not os.path.exists('RetinAI_Revised2'):
    !git clone https://github.com/<your-username>/RetinAI_Revised2.git
%cd RetinAI_Revised2
!git pull

# Install dependencies
!pip install -r requirements.txt

# Symlink source data (read-only)
!ln -sf /content/drive/MyDrive/RetinAI_Revised_Data/train train
!ln -sf /content/drive/MyDrive/RetinAI_Revised_Data/test test

# Symlink outputs to Drive (create folders if first time)
!mkdir -p /content/drive/MyDrive/RetinAI_Revised2_Output/preprocessed
!mkdir -p /content/drive/MyDrive/RetinAI_Revised2_Output/artifacts
!ln -sf /content/drive/MyDrive/RetinAI_Revised2_Output/preprocessed data/preprocessed
!ln -sf /content/drive/MyDrive/RetinAI_Revised2_Output/artifacts artifacts
```

---

## Phase 0 — Preprocessing (run once, then skip on subsequent sessions)

### 0) Crop black backgrounds and save to Drive
`PYTHONPATH=src python -m retinai preprocess --config configs/base.yaml`

### 1) Build manifest with 80/20 stratified split
`PYTHONPATH=src python -m retinai build-manifest --config configs/base.yaml`

---

## Phase 1 — Setup

### 2) Install dependencies
`pip install -r requirements.txt`

---

## Phase 2 — HPO (5-fold CV × 50 trials per model)

### 3) HPO — EfficientNet-B0
`PYTHONPATH=src python -m retinai hpo --config configs/base.yaml --model-name efficientnet_b0`

### 4) HPO — DeiT-Tiny
`PYTHONPATH=src python -m retinai hpo --config configs/base.yaml --model-name deit_tiny`

### 5) HPO — ResNet18
`PYTHONPATH=src python -m retinai hpo --config configs/base.yaml --model-name resnet18`

---

## Phase 3 — Model Comparison (Repeated 5-fold × 10 = 50 runs per model)
> Replace --lr, --weight-decay, --batch-size with best params from each model's HPO output.

### 6) Repeated CV — EfficientNet-B0
`PYTHONPATH=src python -m retinai repeat-cv --config configs/base.yaml --model-name efficientnet_b0 --lr <best_lr> --weight-decay <best_wd> --batch-size <best_bs>`
> Output: `artifacts/repeated_cv/efficientnet_b0/scores.csv` (50 macro F1 scores)

### 7) Repeated CV — DeiT-Tiny
`PYTHONPATH=src python -m retinai repeat-cv --config configs/base.yaml --model-name deit_tiny --lr <best_lr> --weight-decay <best_wd> --batch-size <best_bs>`

### 8) Repeated CV — ResNet18
`PYTHONPATH=src python -m retinai repeat-cv --config configs/base.yaml --model-name resnet18 --lr <best_lr> --weight-decay <best_wd> --batch-size <best_bs>`

### 9) Statistical comparison across models
`PYTHONPATH=src python -m retinai compare-models --config configs/base.yaml`
> Outputs:
> - `artifacts/comparison/model_ci_summary.csv`
> - `artifacts/comparison/wilcoxon_pairwise.csv`
> - `artifacts/comparison/auc_comparison_table.csv`
> - `artifacts/comparison/roc_disease_{class}.png`

---

## Phase 4 — Final Training (100% training pool, 100 epochs, no early stopping)
> Replace --lr, --weight-decay, --batch-size with best HPO params per model.
> loss_curve.png is automatically saved alongside best.pt after each run.

### 10) Train final — EfficientNet-B0
`PYTHONPATH=src python -m retinai train-final --config configs/base.yaml --model-name efficientnet_b0 --lr <best_lr> --weight-decay <best_wd> --batch-size <best_bs> --epochs 100 --out-dir artifacts/final/efficientnet_b0`

### 11) Train final — DeiT-Tiny
`PYTHONPATH=src python -m retinai train-final --config configs/base.yaml --model-name deit_tiny --lr <best_lr> --weight-decay <best_wd> --batch-size <best_bs> --epochs 100 --out-dir artifacts/final/deit_tiny`

### 12) Train final — ResNet18
`PYTHONPATH=src python -m retinai train-final --config configs/base.yaml --model-name resnet18 --lr <best_lr> --weight-decay <best_wd> --batch-size <best_bs> --epochs 100 --out-dir artifacts/final/resnet18`

---

## Phase 5 — External Test Set Evaluation (all 3 models)
> Outputs per model: classification_report.csv, confusion_matrix.csv, auc_scores.csv,
> pr_auc_scores.csv, bootstrap_ci.json, roc_per_model.png, summary.json

### 13) Evaluate — EfficientNet-B0
`PYTHONPATH=src python -m retinai evaluate --config configs/base.yaml --checkpoint artifacts/final/efficientnet_b0/best.pt --model-name efficientnet_b0`

### 14) Evaluate — DeiT-Tiny
`PYTHONPATH=src python -m retinai evaluate --config configs/base.yaml --checkpoint artifacts/final/deit_tiny/best.pt --model-name deit_tiny`

### 15) Evaluate — ResNet18
`PYTHONPATH=src python -m retinai evaluate --config configs/base.yaml --checkpoint artifacts/final/resnet18/best.pt --model-name resnet18`

### 16) Re-run compare-models after evaluation
`PYTHONPATH=src python -m retinai compare-models --config configs/base.yaml`

---

## Phase 6 — Localization (Grad-CAM)

### 17) Grad-CAM maps — replace model-name and image paths as needed
`PYTHONPATH=src python -m retinai localize --config configs/base.yaml --checkpoint artifacts/final/efficientnet_b0/best.pt --model-name efficientnet_b0 --images path/to/img1.png,path/to/img2.png --output-dir artifacts/gradcam/efficientnet_b0`
> Output: `artifacts/gradcam/{model}/` — heatmap PNG per image
