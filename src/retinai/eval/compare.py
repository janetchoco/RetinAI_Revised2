from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def _wilcoxon_effect_r(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    diff = x - y
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    result = stats.wilcoxon(diff)
    z = stats.norm.ppf(result.pvalue / 2)
    return float(abs(z) / np.sqrt(len(diff)))


def run_model_comparison(scores_dir: str, output_dir: str) -> pd.DataFrame:
    """
    Load per-model scores.csv files, compute 95% CI, pairwise Wilcoxon
    signed-rank with Holm-Bonferroni correction, and effect size r.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_scores: dict[str, np.ndarray] = {}
    for csv_path in Path(scores_dir).glob("*/scores.csv"):
        model_name = csv_path.parent.name
        model_scores[model_name] = pd.read_csv(csv_path)["macro_f1"].to_numpy()

    if len(model_scores) < 2:
        raise ValueError(f"Need at least 2 models in {scores_dir}, found {len(model_scores)}.")

    # 95% CI per model (percentile bootstrap on observed scores)
    ci_rows = []
    for model, scores in model_scores.items():
        lo, hi = float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))
        ci_rows.append({
            "model": model,
            "n": len(scores),
            "mean_macro_f1": float(scores.mean()),
            "sd_macro_f1": float(scores.std(ddof=1)),
            "ci95_lo": lo,
            "ci95_hi": hi,
        })
    ci_df = pd.DataFrame(ci_rows).sort_values("mean_macro_f1", ascending=False)
    ci_df.to_csv(out / "model_ci_summary.csv", index=False)

    # Pairwise Wilcoxon signed-rank (paired — same folds, same seed)
    model_names = list(model_scores.keys())
    pairs = list(combinations(model_names, 2))
    raw_pvalues = []
    pair_results = []

    for m1, m2 in pairs:
        s1, s2 = model_scores[m1], model_scores[m2]
        n = min(len(s1), len(s2))
        s1, s2 = s1[:n], s2[:n]
        result = stats.wilcoxon(s1, s2, alternative="two-sided")
        r = _wilcoxon_effect_r(s1, s2)
        raw_pvalues.append(result.pvalue)
        pair_results.append({
            "model_a": m1,
            "model_b": m2,
            "statistic": float(result.statistic),
            "p_raw": float(result.pvalue),
            "effect_r": r,
        })

    # Holm-Bonferroni correction
    k = len(raw_pvalues)
    sorted_idx = np.argsort(raw_pvalues)
    adjusted = [None] * k
    for rank, idx in enumerate(sorted_idx):
        adjusted[idx] = min(1.0, raw_pvalues[idx] * (k - rank))
    # Enforce monotonicity
    for i in range(len(adjusted) - 1, 0, -1):
        adjusted[i - 1] = min(adjusted[i - 1], adjusted[i])

    for i, row in enumerate(pair_results):
        row["p_holm"] = float(adjusted[i])
        row["significant"] = bool(adjusted[i] < 0.05)

    wilcoxon_df = pd.DataFrame(pair_results)
    wilcoxon_df.to_csv(out / "wilcoxon_pairwise.csv", index=False)

    # AUC comparison table across models
    auc_rows = []
    for model_name in model_names:
        auc_path = Path(scores_dir).parent / "eval" / model_name / "auc_scores.csv"
        if auc_path.exists():
            row = pd.read_csv(auc_path).iloc[0].to_dict()
            row["model"] = model_name
            auc_rows.append(row)
    if auc_rows:
        auc_table = pd.DataFrame(auc_rows).set_index("model")
        auc_table.to_csv(out / "auc_comparison_table.csv")

    # ROC Type 2 — per-disease plots (one plot per class, all models on same axes)
    roc_paths = {
        m: Path(scores_dir).parent / "eval" / m / "roc_data.json"
        for m in model_names
    }
    all_roc = {m: json.load(open(p)) for m, p in roc_paths.items() if p.exists()}
    if all_roc:
        sample_model = next(iter(all_roc.values()))
        class_names = list(sample_model.keys())
        for cls in class_names:
            fig, ax = plt.subplots(figsize=(7, 6))
            for model_name, roc in all_roc.items():
                if cls in roc:
                    fpr = roc[cls]["fpr"]
                    tpr = roc[cls]["tpr"]
                    auc = roc[cls]["auc"]
                    ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve — {cls} (One-vs-Rest)")
            ax.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(out / f"roc_disease_{cls}.png", dpi=150)
            plt.close(fig)

    summary = {"ci_summary": ci_df.to_dict(orient="records"),
               "wilcoxon_pairwise": pair_results}
    with open(out / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Model Comparison ===")
    print(ci_df.to_string(index=False))
    print(f"\n=== Pairwise Wilcoxon (Holm-Bonferroni corrected) ===")
    print(wilcoxon_df.to_string(index=False))
    return ci_df
