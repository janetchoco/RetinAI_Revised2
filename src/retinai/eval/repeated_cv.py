from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold

from retinai.config import AppConfig
from retinai.data.datasets import RetinalDataset
from retinai.models.factory import create_model
from retinai.train.engine import run_training


def run_repeated_cv(cfg: AppConfig, manifest: pd.DataFrame, model_name: str, params: dict) -> dict:
    out_dir = Path(cfg.runtime.artifact_root) / "repeated_cv" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores.csv"

    # Resume: load already-completed runs
    if scores_path.exists():
        done_df = pd.read_csv(scores_path)
        scores = done_df["macro_f1"].tolist()
        training_times = done_df["training_seconds"].tolist()
        n_done = len(scores)
        print(f"Resuming {model_name}: {n_done} runs already done, continuing from run_{n_done}")
    else:
        scores = []
        training_times = []
        n_done = 0

    splitter = RepeatedStratifiedKFold(
        n_splits=cfg.cv.n_splits,
        n_repeats=cfg.cv.n_repeats,
        random_state=cfg.runtime.seed,
    )
    y = manifest["class_id"].to_numpy()
    all_splits = list(splitter.split(manifest["image_path"], y))
    total_runs = len(all_splits)

    for run_idx, (train_idx, valid_idx) in enumerate(all_splits):
        if run_idx < n_done:
            continue  # already done
        train_df = manifest.iloc[train_idx]
        valid_df = manifest.iloc[valid_idx]
        train_ds = RetinalDataset(train_df, image_size=cfg.data.image_size, train=True)
        valid_ds = RetinalDataset(valid_df, image_size=cfg.data.image_size, train=False)
        model = create_model(model_name=model_name, num_classes=len(cfg.data.class_names), pretrained=True)
        result = run_training(
            model=model,
            train_ds=train_ds,
            valid_ds=valid_ds,
            device=cfg.runtime.device,
            out_dir=str(out_dir / f"run_{run_idx}"),
            epochs=cfg.train.epochs,
            batch_size=int(params.get("batch_size", cfg.train.batch_size)),
            lr=float(params.get("lr", cfg.train.lr)),
            weight_decay=float(params.get("weight_decay", cfg.train.weight_decay)),
            mixed_precision=cfg.train.mixed_precision,
            patience=cfg.train.patience,
            num_workers=cfg.runtime.num_workers,
        )
        scores.append(result["best_macro_f1"])
        training_times.append(result["training_seconds"])
        # Save after every run so a disconnect loses at most 1 run
        pd.DataFrame({"macro_f1": scores, "training_seconds": training_times}).to_csv(
            scores_path, index=False
        )
        print(f"run_{run_idx}/{total_runs - 1} done — macro_f1={result['best_macro_f1']:.4f} — saved")

    arr = np.array(scores, dtype=float)
    times = np.array(training_times, dtype=float)
    summary = {
        "mean_macro_f1": float(arr.mean()),
        "sd_macro_f1": float(arr.std(ddof=1)),
        "n_runs": int(len(arr)),
        "mean_training_seconds": float(times.mean()),
        "sd_training_seconds": float(times.std(ddof=1)),
    }
    return summary

