from __future__ import annotations

import os
import tempfile
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
    return target.with_suffix(suffix)


def should_skip(sidecar_path: Path, overwrite: bool) -> bool:
    return sidecar_path.exists() and not overwrite


def write_caption(sidecar_path: Path, caption: str) -> None:
    """Atomically replace a caption so interruptions cannot leave a partial file."""
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{sidecar_path.name}.",
            suffix=".tmp",
            dir=sidecar_path.parent,
            delete=False,
        ) as temp_file:
            temp_file.write(caption + "\n")
            temp_path = Path(temp_file.name)
        os.replace(temp_path, sidecar_path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
