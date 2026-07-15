from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    images = [
        p for p in iterator
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def validate_image(path: Path) -> str | None:
    """Return None if the image is valid, else an error message."""
    try:
        with Image.open(path) as img:
            img.verify()
        return None
    except (UnidentifiedImageError, OSError, ValueError) as e:
        return f"invalid image: {e}"
