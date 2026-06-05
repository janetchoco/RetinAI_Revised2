from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import torch
from torch.utils.data import DataLoader

from retinai.data.datasets import RetinalDataset
from retinai.eval.efficiency import estimate_flops
from retinai.models.factory import create_model


def _bootstrap_ci(y_true, y_prob, y_pred, class_names, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metrics = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        report = classification_report(yt, yp, target_names=class_names, output_dict=True, zero_division=0)
        row = {
            "macro_f1": report["macro avg"]["f1-score"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
        }
        # Per-class metrics
        for cls in class_names:
            if cls in report:
                row[f"{cls}_precision"] = report[cls]["precision"]
                row[f"{cls}_recall"] = report[cls]["recall"]
                row[f"{cls}_f1"] = report[cls]["f1-score"]
        try:
            row["macro_auc"] = float(roc_auc_score(yt, ypr, multi_class="ovr", average="macro"))
            for i, cls in enumerate(class_names):
                y_bin = (yt == i).astype(int)
                if y_bin.sum() > 0:
                    row[f"{cls}_auc"] = float(roc_auc_score(y_bin, ypr[:, i]))
        except ValueError:
            row["macro_auc"] = float("nan")
        metrics.append(row)
    arr = pd.DataFrame(metrics)
    ci = {}
    for col in arr.columns:
        lo, hi = float(np.nanpercentile(arr[col], 2.5)), float(np.nanpercentile(arr[col], 97.5))
        ci[col] = {"lo": lo, "hi": hi}
    return ci


def evaluate_on_test(
    checkpoint_path: str,
    model_name: str,
    test_df: pd.DataFrame,
    class_names: list[str],
    output_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    device: str = "cpu",
    n_bootstrap: int = 1000,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = create_model(model_name=model_name, num_classes=len(class_names), pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)

    ds = RetinalDataset(test_df, image_size=image_size, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_true, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_true.extend(y.numpy().tolist())
            all_pred.extend(preds.tolist())
            all_prob.extend(probs.tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_prob = np.array(all_prob)

    # Per-class + macro metrics
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(out / "classification_report.csv")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out / "confusion_matrix.csv")

    # Per-class AUC (One-vs-Rest) + Macro AUC + PR-AUC
    auc_scores = {}
    pr_auc_scores = {}
    roc_data = {}
    pr_data = {}

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            auc_scores[cls] = float("nan")
            pr_auc_scores[cls] = float("nan")
            roc_data[cls] = {"fpr": [], "tpr": [], "auc": float("nan")}
            pr_data[cls] = {"precision": [], "recall": [], "pr_auc": float("nan")}
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        auc_val = float(roc_auc_score(y_bin, y_prob[:, i]))
        auc_scores[cls] = auc_val
        roc_data[cls] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc_val}
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc_val:.3f})")

        prec, rec, _ = precision_recall_curve(y_bin, y_prob[:, i])
        pr_auc_val = float(average_precision_score(y_bin, y_prob[:, i]))
        pr_auc_scores[cls] = pr_auc_val
        pr_data[cls] = {"precision": prec.tolist(), "recall": rec.tolist(), "pr_auc": pr_auc_val}

    valid_aucs = [v for v in auc_scores.values() if not np.isnan(v)]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    macro_pr_auc = float(np.mean(list(pr_auc_scores.values())))
    auc_scores["macro"] = macro_auc
    pr_auc_scores["macro"] = macro_pr_auc

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc_per_model.png", dpi=150)
    plt.close(fig)

    pd.DataFrame([auc_scores]).to_csv(out / "auc_scores.csv", index=False)
    pd.DataFrame([pr_auc_scores]).to_csv(out / "pr_auc_scores.csv", index=False)

    with open(out / "roc_data.json", "w", encoding="utf-8") as f:
        json.dump(roc_data, f)
    with open(out / "pr_data.json", "w", encoding="utf-8") as f:
        json.dump(pr_data, f)

    # FLOPs
    flops = estimate_flops(model, input_size=image_size)

    # Bootstrap 95% CI
    ci = _bootstrap_ci(y_true, y_prob, y_pred, class_names, n_bootstrap=n_bootstrap)
    with open(out / "bootstrap_ci.json", "w", encoding="utf-8") as f:
        json.dump(ci, f, indent=2)

    summary = {
        "model": model_name,
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_auc": macro_auc,
        "macro_pr_auc": macro_pr_auc,
        "per_class_auc": auc_scores,
        "per_class_pr_auc": pr_auc_scores,
        "flops": flops,
        "bootstrap_ci": ci,
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluation complete → {output_dir}")
    print(f"  Macro F1   : {summary['macro_f1']:.4f}")
    print(f"  Macro AUC  : {macro_auc:.4f}")
    print(f"  Macro PR-AUC: {macro_pr_auc:.4f}")
    if flops:
        print(f"  FLOPs      : {flops:,}")
    return summary
