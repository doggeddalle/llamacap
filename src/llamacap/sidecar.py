from __future__ import annotations

from pathlib import Path


def sidecar_path_for(
    image_path: Path,
    input_dir: Path,
    output_dir: Path | None,
    suffix: str,
) -> Path:
    if output_dir is None:
        return image_path.with_suffix(suffix)

    relative = image_path.relative_to(input_dir)
    target = output_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.with_suffix(suffix)


def should_skip(sidecar_path: Path, overwrite: bool) -> bool:
    return sidecar_path.exists() and not overwrite


def write_caption(sidecar_path: Path, caption: str) -> None:
    sidecar_path.write_text(caption + "\n", encoding="utf-8")
