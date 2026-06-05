from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from retinai.config import AppConfig
from retinai.data.datasets import RetinalDataset
from retinai.data.splits import assign_stratified_folds
from retinai.models.factory import create_model
from retinai.train.engine import run_training


def run_hpo(cfg: AppConfig, manifest: pd.DataFrame, model_name: str) -> optuna.Study:
    fold_df = assign_stratified_folds(manifest, n_splits=cfg.cv.n_splits, seed=cfg.runtime.seed)
    Path(cfg.runtime.artifact_root).mkdir(parents=True, exist_ok=True)
    storage_uri = f"sqlite:///{cfg.runtime.study_db}"
    study_name = f"{model_name}_hpo_224"

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_choices = {
            "efficientnet_b0": [16, 32, 64, 128],
            "deit_tiny": [16, 32, 64, 128],
            "resnet18": [16, 32, 64, 128],
        }.get(model_name, [16, 32, 64, 128])
        batch_size = trial.suggest_categorical("batch_size", batch_choices)
        fold_scores = []
        for fold in range(cfg.cv.n_splits):
            train_df = fold_df[fold_df["fold"] != fold]
            valid_df = fold_df[fold_df["fold"] == fold]
            train_ds = RetinalDataset(train_df, image_size=cfg.data.image_size, train=True)
            valid_ds = RetinalDataset(valid_df, image_size=cfg.data.image_size, train=False)
            model = create_model(model_name=model_name, num_classes=len(cfg.data.class_names), pretrained=True)
            out_dir = f"{cfg.runtime.artifact_root}/hpo/{model_name}/trial_{trial.number}/fold_{fold}"

            def make_callback(t, f):
                def callback(epoch, metric):
                    t.report(metric, epoch * cfg.cv.n_splits + f)
                    if t.should_prune():
                        raise optuna.TrialPruned()
                return callback

            result = run_training(
                model=model,
                train_ds=train_ds,
                valid_ds=valid_ds,
                device=cfg.runtime.device,
                out_dir=out_dir,
                epochs=cfg.hpo.epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=wd,
                mixed_precision=cfg.train.mixed_precision,
                patience=cfg.train.patience,
                num_workers=cfg.runtime.num_workers,
                epoch_callback=make_callback(trial, fold),
            )
            fold_scores.append(result["best_macro_f1"])
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_uri,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, cfg.hpo.n_trials - completed)
    study.optimize(objective, n_trials=remaining)
    return study

