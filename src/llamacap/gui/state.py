"""Pure helpers for GUI preferences, validation, and dataset summaries."""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from llamacap.image_utils import list_images


def preferences_path() -> Path:
    base = Path(os.environ.get("APPDATA") or Path.home() / ".config")
    return base / "llamacap" / "gui.json"


@dataclass
class GuiPreferences:
    input_dir: str = ""
    output_dir: str = ""
    profile: str = ""
    recursive: bool = False
    theme: str = "System"
    geometry: str = "1100x800"
    advanced_open: bool = False
    details_open: bool = False

    @classmethod
    def load(cls, path: Path | None = None) -> "GuiPreferences":
        path = path or preferences_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            allowed = cls.__dataclass_fields__
            return cls(**{key: value for key, value in data.items() if key in allowed})
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: Path | None = None) -> None:
        path = path or preferences_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


@dataclass(frozen=True)
class DatasetSummary:
    images: tuple[Path, ...]
    existing: int

    @property
    def pending(self) -> int:
        return len(self.images) - self.existing


def summarize_dataset(
    folder: Path,
    recursive: bool,
    suffix: str = ".txt",
    output_dir: Path | None = None,
) -> DatasetSummary:
    images = tuple(list_images(folder, recursive))
    existing = sum(
        (
            image.with_suffix(suffix)
            if output_dir is None
            else (output_dir / image.relative_to(folder)).with_suffix(suffix)
        ).is_file()
        for image in images
    )
    return DatasetSummary(images, existing)


def validate_number(value: str, label: str, *, integer: bool, allow_zero: bool) -> str | None:
    if not value.strip():
        return None
    try:
        parsed = int(value) if integer else float(value)
    except ValueError:
        return f"{label} must be {'an integer' if integer else 'a number'}."
    if isinstance(parsed, float) and not math.isfinite(parsed):
        return f"{label} must be finite."
    if parsed < 0 or (not allow_zero and parsed == 0):
        return f"{label} must be {'zero or greater' if allow_zero else 'greater than zero'}."
    return None


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"
