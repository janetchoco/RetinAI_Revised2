"""Bayesian prior correction for label shift between training and external test set.

Formula: p_corrected(y|x) ∝ p_model(y|x) × (test_prior[y] / train_prior[y])
Then normalise to sum = 1 and take argmax for final prediction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torch
from torch.utils.data import DataLoader

from retinai.config import AppConfig
from retinai.data.datasets import RetinalDataset
from retinai.models.factory import create_model


def _compute_prior(paths_or_df, class_names: list[str]) -> np.ndarray:
    """Return class proportions as numpy array, ordered by class_names."""
    if isinstance(paths_or_df, pd.DataFrame):
        counts = paths_or_df["class_id"].value_counts().sort_index()
    else:
        # paths_or_df is a directory Path; count files per class folder
        counts_dict = {}
        for i, cls in enumerate(class_names):
            # folder may be named cls or with train_/test_ prefix
            for pattern in [cls, f"train_{cls}", f"test_{cls}"]:
                folder = Path(paths_or_df) / pattern
                if folder.exists():
                    counts_dict[i] = len(list(folder.glob("*")))
                    break
            else:
                counts_dict[i] = 0
        counts = pd.Series(counts_dict).sort_index()
    total = counts.sum()
    return (counts / total).to_numpy(dtype=float)


def _run_inference(model, test_df: pd.DataFrame, cfg: AppConfig, device: str):
    ds = RetinalDataset(test_df, image_size=cfg.data.image_size, train=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    all_true, all_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
            all_true.extend(y.numpy().tolist())
            all_prob.extend(probs.tolist())
    return np.array(all_true), np.array(all_prob)


def _metrics(y_true, y_pred, y_prob, class_names) -> dict:
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    try:
        macro_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        macro_auc = float("nan")
    return {
        "macro_f1":        report["macro avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall":    report["macro avg"]["recall"],
        "macro_auc":       macro_auc,
        "per_class":       {c: report[c] for c in class_names if c in report},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def run_prior_correction(
    cfg: AppConfig,
    checkpoint_path: str,
    model_name: str,
    test_df: pd.DataFrame,
    output_dir: str,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    class_names = cfg.data.class_names
    device = cfg.runtime.device

    # ── load model ──────────────────────────────────────────────────────────
    model = create_model(model_name=model_name, num_classes=len(class_names), pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    # ── run inference ────────────────────────────────────────────────────────
    y_true, y_prob = _run_inference(model, test_df, cfg, device)

    # ── compute priors ───────────────────────────────────────────────────────
    manifest = pd.read_csv(cfg.data.manifest_path)
    train_prior = _compute_prior(manifest, class_names)
    test_prior  = _compute_prior(test_df, class_names)

    print(f"\nClass priors ({class_names}):")
    for i, cls in enumerate(class_names):
        ratio = test_prior[i] / train_prior[i] if train_prior[i] > 0 else 0
        print(f"  {cls:10s}  train={train_prior[i]:.3f}  test={test_prior[i]:.3f}  ratio={ratio:.3f}")

    # ── before correction ────────────────────────────────────────────────────
    y_pred_before = np.argmax(y_prob, axis=1)
    before = _metrics(y_true, y_pred_before, y_prob, class_names)

    # ── apply prior correction ───────────────────────────────────────────────
    correction = test_prior / np.where(train_prior > 0, train_prior, 1e-9)
    y_prob_corrected = y_prob * correction
    y_prob_corrected /= y_prob_corrected.sum(axis=1, keepdims=True)
    y_pred_after = np.argmax(y_prob_corrected, axis=1)
    after = _metrics(y_true, y_pred_after, y_prob_corrected, class_names)

    # ── save ─────────────────────────────────────────────────────────────────
    result = {
        "model":        model_name,
        "train_prior":  dict(zip(class_names, train_prior.tolist())),
        "test_prior":   dict(zip(class_names, test_prior.tolist())),
        "correction_factor": dict(zip(class_names, correction.tolist())),
        "before":       before,
        "after":        after,
    }
    with open(out / "prior_correction_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    pd.DataFrame(before["confusion_matrix"],
                 index=class_names, columns=class_names).to_csv(out / "confusion_before.csv")
    pd.DataFrame(after["confusion_matrix"],
                 index=class_names, columns=class_names).to_csv(out / "confusion_after.csv")

    # ── print comparison ─────────────────────────────────────────────────────
    print(f"\n{'Metric':<22} {'Before':>10} {'After':>10}")
    print("-" * 44)
    for key in ["macro_f1", "macro_precision", "macro_recall", "macro_auc"]:
        print(f"  {key:<20} {before[key]:>10.4f} {after[key]:>10.4f}")
    print()
    for cls in class_names:
        b, a = before["per_class"][cls], after["per_class"][cls]
        print(f"  {cls} precision:     {b['precision']:>10.4f} {a['precision']:>10.4f}")
        print(f"  {cls} recall:        {b['recall']:>10.4f} {a['recall']:>10.4f}")
        print(f"  {cls} f1:            {b['f1-score']:>10.4f} {a['f1-score']:>10.4f}")

    print(f"\nResults saved to {output_dir}")
    return result
