from __future__ import annotations

import logging
import math
from pathlib import Path

from PIL import Image

logger = logging.getLogger("llamacap")

_HAS_ALPHA_MODES = {"RGBA", "LA", "PA"}


def resize_for_analysis(image_path: Path, target_megapixels: float, tmp_dir: Path) -> Path:
    """Scales image_path (preserving aspect ratio, upscaling allowed) so that
    width*height is close to target_megapixels * 1_000_000, saving into tmp_dir.
    Returns image_path unchanged if already within 1% of the target."""
    target_px = target_megapixels * 1_000_000

    with Image.open(image_path) as img:
        width, height = img.size
        current_px = width * height

        if current_px == 0 or abs(current_px - target_px) / target_px < 0.01:
            return image_path

        scale = math.sqrt(target_px / current_px)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))

        resized = img.resize(new_size, Image.LANCZOS)

        has_alpha = img.mode in _HAS_ALPHA_MODES or (
            img.mode == "P" and "transparency" in img.info
        )
        if has_alpha:
            out_path = tmp_dir / f"{image_path.stem}.png"
            resized = resized.convert("RGBA")
            resized.save(out_path, format="PNG")
        else:
            out_path = tmp_dir / f"{image_path.stem}.jpg"
            resized = resized.convert("RGB")
            resized.save(out_path, format="JPEG", quality=95)

        logger.debug(
            "resized %s: %dx%d -> %dx%d", image_path.name, width, height, *new_size
        )
        return out_path
