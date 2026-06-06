from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _crop_black(img: Image.Image, threshold: int) -> Image.Image:
    gray = np.array(img.convert("L"))
    mask = gray > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img  # all black — return as-is
    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return img.crop((cmin, rmin, cmax + 1, rmax + 1))


def run_preprocess(
    src_dirs: list[tuple[Path, str]],
    out_dir: Path,
    class_names: list[str],
    threshold: int = 10,
    image_size: int | None = None,
) -> None:
    """
    Crop black backgrounds from all images in src_dirs and save to out_dir/{class}/.
    Output filenames are prefixed with the source directory name to avoid collisions.
    If image_size is set, images are resized to (image_size, image_size) after cropping.
    Idempotent: existing output files are skipped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for class_name in class_names:
        (out_dir / class_name).mkdir(parents=True, exist_ok=True)

    total = skipped = saved = 0
    for src_root, prefix in src_dirs:
        for img_path in sorted(src_root.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = img_path.parent.name.lower().removeprefix("train_").removeprefix("test_")
            if label not in class_names:
                continue
            total += 1
            out_name = f"{prefix}_{img_path.name}"
            out_path = out_dir / label / out_name
            if out_path.exists():
                skipped += 1
                continue
            with Image.open(img_path) as img:
                cropped = _crop_black(img.convert("RGB"), threshold)
                if image_size is not None:
                    cropped = cropped.resize((image_size, image_size), Image.LANCZOS)
                cropped.save(out_path)
            saved += 1

    print(f"preprocess: total={total}  saved={saved}  skipped(already exist)={skipped}")
