from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            yield p


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
        rows.append(
            {
                "image_path": str(image_path),
                "class_name": label,
                "class_id": class_to_id[label],
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No labeled images found under {preprocessed_dir}.")
    _, test_idx = train_test_split(
        df.index,
        test_size=test_split_ratio,
        stratify=df["class_id"],
        random_state=split_seed,
    )
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    return df


def validate_manifest_images(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    good_rows: list[dict] = []
    bad_rows: list[dict] = []
    for row in df.to_dict(orient="records"):
        try:
            with Image.open(row["image_path"]) as img:
                img.verify()
            good_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            bad_rows.append({**row, "error": str(exc)})
    return pd.DataFrame(good_rows), pd.DataFrame(bad_rows)

