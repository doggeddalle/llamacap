from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from llamacap.config import GlobalConfig
from llamacap.errors import ModelNotFoundError, ProfileError
from llamacap.model_resolver import resolve_model_dir

PROFILES_DIR_NAME = "profiles"


@dataclass
class TriggerWordConfig:
    value: str = ""
    placement: str = "prefix_comma"


@dataclass
class GenerationParams:
    ctx_size: int = 8192
    n_predict: int = 220
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    ngl: int = 99
    image_min_tokens: int = 1024
    seed: int = 1337
    no_warmup: bool = True
    extra_args: list[str] = field(default_factory=list)


@dataclass
class Profile:
    name: str
    description: str
    gguf_path: Path
    mmproj_path: Path
    prompt_text: str
    trigger_word: TriggerWordConfig
    generation: GenerationParams
    output_suffix: str


def list_profile_names(config: GlobalConfig) -> list[str]:
    profiles_dir = config.project_root / PROFILES_DIR_NAME
    return sorted(p.stem for p in profiles_dir.glob("*.toml"))


def _resolve_model_file(
    profile_name: str,
    kind: str,
    path_field: str,
    file_field: str,
    config: GlobalConfig,
) -> Path:
    if path_field:
        resolved = Path(path_field)
    elif file_field:
        resolved = config.project_root / config.models.default_dir / file_field
    else:
        raise ProfileError(
            f"Profile '{profile_name}': [model] must set either "
            f"{kind}_path (absolute) or {kind}_file (filename under "
            f"{config.models.default_dir}/)."
        )

    if not resolved.is_file():
        raise ModelNotFoundError(
            f"Profile '{profile_name}': {kind} file not found: {resolved}"
        )
    return resolved


def load_profile(
    name: str,
    config: GlobalConfig,
    model_override: tuple[Path, Path] | None = None,
    interactive: bool = False,
) -> Profile:
    path = config.project_root / PROFILES_DIR_NAME / f"{name}.toml"
    if not path.exists():
        available = list_profile_names(config)
        raise ProfileError(
            f"Profile '{name}' not found ({path}). Available profiles: "
            f"{', '.join(available) if available else '(none)'}"
        )

    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ProfileError(f"Profile '{name}' is not valid TOML: {e}") from e

    try:
        profile_meta = data["profile"]
        model = data["model"]
        prompt = data["prompt"]
    except KeyError as e:
        raise ProfileError(f"Profile '{name}' is missing required section: {e}") from e

    # Optional sections fall back to sensible defaults (krea2's values), so a
    # minimal profile only needs [profile], [model], and [prompt].
    trigger = data.get("trigger_word", {})
    generation = data.get("generation", {})
    output = data.get("output", {})

    model_fields = ("gguf_path", "mmproj_path", "gguf_file", "mmproj_file")
    if model_override is not None:
        # --model was passed: bypass the profile's [model] section entirely,
        # including not requiring its configured files to exist on disk.
        gguf_path, mmproj_path = model_override
    elif interactive and not any(model.get(f, "") for f in model_fields):
        # Nothing configured in the profile: fall back to scanning
        # [models].default_dir and prompting if it's ambiguous.
        gguf_path, mmproj_path = resolve_model_dir(
            config.project_root / config.models.default_dir, interactive=True
        )
    else:
        gguf_path = _resolve_model_file(
            name, "gguf", model.get("gguf_path", ""), model.get("gguf_file", ""), config
        )
        mmproj_path = _resolve_model_file(
            name, "mmproj", model.get("mmproj_path", ""), model.get("mmproj_file", ""), config
        )

    prompt_text = prompt.get("text", "")
    if not prompt_text:
        prompt_file = prompt.get("file", "")
        if not prompt_file:
            raise ProfileError(
                f"Profile '{name}': [prompt] must set either 'text' or 'file'."
            )
        prompt_path = config.project_root / prompt_file
        if not prompt_path.is_file():
            raise ProfileError(f"Profile '{name}': prompt file not found: {prompt_path}")
        prompt_text = prompt_path.read_text(encoding="utf-8").strip()

    try:
        return Profile(
            name=profile_meta["name"],
            description=profile_meta.get("description", ""),
            gguf_path=gguf_path,
            mmproj_path=mmproj_path,
            prompt_text=prompt_text,
            trigger_word=TriggerWordConfig(**trigger),
            generation=GenerationParams(**generation),
            output_suffix=output.get("suffix", ".txt"),
        )
    except TypeError as e:
        raise ProfileError(f"Profile '{name}' is malformed: {e}") from e
