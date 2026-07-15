from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from llamacap.errors import CaptionGenerationError
from llamacap.profiles import Profile

logger = logging.getLogger("llamacap")


def build_argv(binary: Path, profile: Profile, image_path: Path) -> list[str]:
    gen = profile.generation
    argv = [
        str(binary),
        "-m", str(profile.gguf_path),
        "--mmproj", str(profile.mmproj_path),
        "--image", str(image_path),
        "-p", profile.prompt_text,
        "-ngl", str(gen.ngl),
        "-c", str(gen.ctx_size),
        "-n", str(gen.n_predict),
        "--temp", str(gen.temperature),
        "--top-p", str(gen.top_p),
        "--top-k", str(gen.top_k),
        "--repeat-penalty", str(gen.repeat_penalty),
        "--image-min-tokens", str(gen.image_min_tokens),
        "--seed", str(gen.seed),
    ]
    if gen.no_warmup:
        argv.append("--no-warmup")
    argv.extend(gen.extra_args)
    return argv


def generate_caption(
    binary: Path,
    profile: Profile,
    image_path: Path,
    timeout_seconds: int,
) -> str:
    argv = build_argv(binary, profile, image_path)
    logger.debug("argv: %s", argv)
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        raise CaptionGenerationError(
            f"timed out after {timeout_seconds}s"
        ) from e

    if result.returncode != 0:
        stderr_excerpt = result.stderr.strip()[-500:]
        raise CaptionGenerationError(
            f"llama-mtmd-cli exited with code {result.returncode}: {stderr_excerpt}"
        )

    caption = result.stdout.strip()
    if not caption:
        raise CaptionGenerationError("model produced no output")

    return caption
