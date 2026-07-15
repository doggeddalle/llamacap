from __future__ import annotations

import sys
from pathlib import Path

from llamacap.errors import ModelNotFoundError


def resolve_model_dir(model_dir: Path, interactive: bool = False) -> tuple[Path, Path]:
    """Returns (gguf_path, mmproj_path) for a directory containing exactly one
    non-mmproj .gguf and one mmproj-named (case-insensitive substring) .gguf.

    If interactive=True and either bucket has more than one candidate, prompts
    on stdin to disambiguate instead of raising (still raises if a bucket is
    empty, or if stdin isn't a real terminal)."""
    if not model_dir.is_dir():
        raise ModelNotFoundError(f"--model is not a directory: {model_dir}")

    gguf_files = sorted(model_dir.glob("*.gguf"))
    mmproj_files = [p for p in gguf_files if "mmproj" in p.name.lower()]
    main_files = [p for p in gguf_files if p not in mmproj_files]

    if len(main_files) == 1 and len(mmproj_files) == 1:
        return main_files[0], mmproj_files[0]

    if interactive and main_files and mmproj_files:
        return _prompt_model_selection(main_files, mmproj_files)

    listing = "\n".join(
        f"  - {p.name} ({'mmproj' if p in mmproj_files else 'main'})" for p in gguf_files
    ) or "  (no .gguf files found)"
    raise ModelNotFoundError(
        f"--model {model_dir} must contain exactly one main .gguf and one "
        f"mmproj-named .gguf (found {len(main_files)} main, {len(mmproj_files)} mmproj):\n"
        f"{listing}"
    )


def _prompt_model_selection(
    main_files: list[Path], mmproj_files: list[Path]
) -> tuple[Path, Path]:
    if not sys.stdin.isatty():
        raise ModelNotFoundError(
            "Model resolution is ambiguous and --interactive requires a real "
            "terminal (stdin is not a TTY); pass --model with an unambiguous "
            "directory instead, or run interactively."
        )
    return (
        _select_one("main model", main_files),
        _select_one("mmproj file", mmproj_files),
    )


def _select_one(label: str, candidates: list[Path]) -> Path:
    if len(candidates) == 1:
        return candidates[0]

    print(f"Multiple {label} candidates found:")
    for i, p in enumerate(candidates, start=1):
        print(f"  {i}. {p.name}")

    for _ in range(3):
        try:
            choice = input(f"Select {label} [1-{len(candidates)}]: ").strip()
        except EOFError:
            raise ModelNotFoundError(
                f"No input available while selecting {label} "
                "(stdin closed/non-interactive)."
            ) from None
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return candidates[int(choice) - 1]
        print("Invalid selection, try again.")

    raise ModelNotFoundError(f"No valid {label} selection after 3 attempts.")
