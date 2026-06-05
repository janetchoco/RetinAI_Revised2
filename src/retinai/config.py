from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    artifact_root: str = "artifacts"
    study_db: str = "artifacts/optuna.db"


@dataclass
class DataConfig:
    train_dir: str = "train"
    test_dir: str = "test"
    manifest_path: str = "artifacts/manifest.csv"
    image_size: int = 224
    class_names: list[str] = field(default_factory=lambda: ["normal", "amd", "dr"])
    patient_id_regex: str | None = None


@dataclass
class TrainConfig:
    model_name: str = "efficientnet_b0"
    pretrained: bool = True
    pretrained_source: str = "imagenet1k"
    epochs: int = 20
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 8
    mixed_precision: bool = True
    metric_for_best: str = "macro_f1"


@dataclass
class CVConfig:
    n_splits: int = 5
    n_repeats: int = 10
    shuffle: bool = True


@dataclass
class HPOConfig:
    n_trials: int = 50
    metric_to_optimize: str = "macro_f1"
    epochs: int = 30


@dataclass
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)


def load_config(config_path: str) -> AppConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        data=DataConfig(**raw.get("data", {})),
        train=TrainConfig(**raw.get("train", {})),
        cv=CVConfig(**raw.get("cv", {})),
        hpo=HPOConfig(**raw.get("hpo", {})),
    )


def merge_model_override(base: AppConfig, model_overrides: dict[str, Any]) -> AppConfig:
    for k, v in model_overrides.items():
        if hasattr(base.train, k):
            setattr(base.train, k, v)
    return base


def ensure_artifact_dirs(cfg: AppConfig) -> None:
    Path(cfg.runtime.artifact_root).mkdir(parents=True, exist_ok=True)

