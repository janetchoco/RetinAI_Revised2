# RetinAI Runbook

## Phase 1 — Setup
### 1) Install dependencies
`pip install -r requirements.txt`

### 2) Build manifest from folder labels
`PYTHONPATH=src python -m retinai build-manifest --config configs/base.yaml`

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

### 6) Repeated CV — EfficientNet-B0 (HPO best: macro_f1=0.9190)
`PYTHONPATH=src python -m retinai repeat-cv --config configs/base.yaml --model-name efficientnet_b0 --lr 0.0005618447545488722 --weight-decay 1.3129046436784662e-06 --batch-size 32`
> Output: `artifacts/repeated_cv/efficientnet_b0/scores.csv` (50 macro F1 scores)

### 7) Repeated CV — DeiT-Tiny (HPO best: macro_f1=0.9129)
`PYTHONPATH=src python -m retinai repeat-cv --config configs/base.yaml --model-name deit_tiny --lr 7.054516705607733e-05 --weight-decay 4.1372877625975084e-05 --batch-size 128`
> Output: `artifacts/repeated_cv/deit_tiny/scores.csv`

### 8) Repeated CV — ResNet18 (HPO best: macro_f1=0.9242)
`PYTHONPATH=src python -m retinai repeat-cv --config configs/base.yaml --model-name resnet18 --lr 0.00013751516691064744 --weight-decay 2.3436685827211785e-06 --batch-size 32`
> Output: `artifacts/repeated_cv/resnet18/scores.csv`

### 9) Statistical comparison across models
`PYTHONPATH=src python -m retinai compare-models --config configs/base.yaml`
> Outputs:
> - `artifacts/comparison/model_ci_summary.csv` — mean ± SD + 95% CI per model
> - `artifacts/comparison/wilcoxon_pairwise.csv` — Wilcoxon signed-rank + Holm-Bonferroni p-value + effect size r
> - `artifacts/comparison/auc_comparison_table.csv` — AUC per class per model
> - `artifacts/comparison/roc_disease_{class}.png` — ROC Type 2: per-disease, all 3 models on one plot

---

## Phase 4 — Final Training (100% training data, no validation, all 3 models)
> Replace --lr, --weight-decay, --batch-size with best HPO params per model.

### 10) Train final — EfficientNet-B0
`PYTHONPATH=src python -m retinai train-final --config configs/base.yaml --model-name efficientnet_b0 --lr 0.0005618447545488722 --weight-decay 1.3129046436784662e-06 --batch-size 32`

### 11) Train final — DeiT-Tiny
`PYTHONPATH=src python -m retinai train-final --config configs/base.yaml --model-name deit_tiny --lr 7.054516705607733e-05 --weight-decay 4.1372877625975084e-05 --batch-size 128`

### 12) Train final — ResNet18
`PYTHONPATH=src python -m retinai train-final --config configs/base.yaml --model-name resnet18 --lr 0.00013751516691064744 --weight-decay 2.3436685827211785e-06 --batch-size 32`

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

### 16) Re-run compare-models after evaluation (generates ROC Type 2 + AUC table)
`PYTHONPATH=src python -m retinai compare-models --config configs/base.yaml`

---

## Phase 6 — Localization (Grad-CAM)
### 17) Grad-CAM maps — replace model-name and image paths as needed
`PYTHONPATH=src python -m retinai localize --config configs/base.yaml --checkpoint artifacts/final/efficientnet_b0/best.pt --model-name efficientnet_b0 --images path/to/img1.png,path/to/img2.png --output-dir artifacts/gradcam/efficientnet_b0`
> Output: `artifacts/gradcam/{model}/` — heatmap PNG per image
