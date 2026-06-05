from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


def assign_stratified_folds(
    df: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    out = df.copy()
    out["fold"] = -1
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(splitter.split(out["image_path"], out["class_id"])):
        out.loc[valid_idx, "fold"] = fold
    return out


def repeated_folds_indices(
    y: np.ndarray,
    n_splits: int,
    n_repeats: int,
    seed: int,
):
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    x_dummy = np.zeros_like(y)
    return list(rkf.split(x_dummy, y))

