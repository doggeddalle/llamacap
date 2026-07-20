from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from llamacap.errors import LlamacapError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"


@dataclass
class LlamaCppConfig:
    binary_path_override: str = ""
    bin_subdir: str = "bin"


@dataclass
class ModelsConfig:
    default_dir: str = "models"


@dataclass
class OutputConfig:
    default_mode: str = "in_place"
    overwrite: bool = False
    recursive: bool = False


@dataclass
class GenerationConfig:
    timeout_seconds: int = 300
    server_startup_timeout_seconds: int = 120


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class PreprocessingConfig:
    resize_megapixels: float = 0.0


@dataclass
class GlobalConfig:
    llama_cpp: LlamaCppConfig
    models: ModelsConfig
    output: OutputConfig
    generation: GenerationConfig
    logging: LoggingConfig
    preprocessing: PreprocessingConfig
    project_root: Path = PROJECT_ROOT


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> GlobalConfig:
    if not path.exists():
        raise LlamacapError(f"Global config not found: {path}")

    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise LlamacapError(f"Invalid TOML in config ({path}): {e}") from e

    try:
        return GlobalConfig(
            llama_cpp=LlamaCppConfig(**data.get("llama_cpp", {})),
            models=ModelsConfig(**data.get("models", {})),
            output=OutputConfig(**data.get("output", {})),
            generation=GenerationConfig(**data.get("generation", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            preprocessing=PreprocessingConfig(**data.get("preprocessing", {})),
        )
    except TypeError as e:
        raise LlamacapError(f"Malformed config.toml ({path}): {e}") from e
