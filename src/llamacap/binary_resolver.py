from __future__ import annotations

import logging
import shutil
from pathlib import Path

from llamacap.config import GlobalConfig
from llamacap.errors import BinaryNotFoundError

BINARY_NAME = "llama-server"
BINARY_EXE = "llama-server.exe"

logger = logging.getLogger("llamacap")


def _resolve_override(override: str) -> Path:
    p = Path(override)
    if p.is_dir():
        candidate = p / BINARY_EXE
        if candidate.is_file():
            return candidate
        raise BinaryNotFoundError(
            f"llama_cpp.binary_path_override is a directory but does not contain "
            f"{BINARY_EXE}: {p}"
        )
    if p.is_file():
        return p
    raise BinaryNotFoundError(
        f"llama_cpp.binary_path_override is set but does not exist: {p}"
    )


def resolve_llama_server(config: GlobalConfig, cli_override: str | None = None) -> Path:
    checked: list[str] = []

    if cli_override:
        checked.append(f"--llama-bin {cli_override}")
        path = _resolve_override(cli_override)
        logger.info("Resolved llama-server via --llama-bin: %s", path)
        return path

    if config.llama_cpp.binary_path_override:
        checked.append(f"config.toml binary_path_override = {config.llama_cpp.binary_path_override}")
        path = _resolve_override(config.llama_cpp.binary_path_override)
        logger.info("Resolved llama-server via config.toml override: %s", path)
        return path

    on_path = shutil.which(BINARY_NAME)
    checked.append("PATH (shutil.which)")
    if on_path:
        logger.info("Resolved llama-server via PATH: %s", on_path)
        return Path(on_path)

    local_bin = config.project_root / config.llama_cpp.bin_subdir / BINARY_EXE
    checked.append(str(local_bin))
    if local_bin.is_file():
        logger.info("Resolved llama-server via local bin/ subdir: %s", local_bin)
        return local_bin

    raise BinaryNotFoundError(
        "Could not find llama-server. Checked:\n  - "
        + "\n  - ".join(checked)
        + "\n\nInstall it with `winget install ggml.llamacpp`, place it in "
        f"{config.project_root / config.llama_cpp.bin_subdir}, or set "
        "[llama_cpp].binary_path_override in config.toml."
    )
