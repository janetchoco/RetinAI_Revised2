from __future__ import annotations

from pathlib import Path


def generate_pylon_maps(checkpoint_path: str, image_paths: list[str], output_dir: str) -> None:
    """
    Hook for PYLON localization.
    Integrate your preferred PYLON library/API implementation here.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "README_PYLON.txt"
    marker.write_text(
        "PYLON integration placeholder.\n"
        f"Checkpoint: {checkpoint_path}\n"
        f"Requested images: {len(image_paths)}\n"
        "Replace this function body with actual PYLON invocation and image export.\n",
        encoding="utf-8",
    )

