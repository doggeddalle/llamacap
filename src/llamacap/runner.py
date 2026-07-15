from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from llamacap.binary_resolver import resolve_llama_mtmd_cli
from llamacap.captioner import generate_caption
from llamacap.config import GlobalConfig
from llamacap.errors import CaptionGenerationError
from llamacap.image_utils import list_images, validate_image
from llamacap.model_resolver import resolve_model_dir
from llamacap.postprocess import apply_trigger_word
from llamacap.profiles import Profile, TriggerWordConfig, load_profile
from llamacap.resize import resize_for_analysis
from llamacap.sidecar import should_skip, sidecar_path_for, write_caption

logger = logging.getLogger("llamacap")


@dataclass
class BatchOptions:
    profile_name: str
    input_dir: Path
    output_dir: Path | None
    overwrite: bool
    recursive: bool
    limit: int | None
    llama_bin_override: str | None
    trigger_override: str | None = None
    model_dir: Path | None = None
    resize_megapixels: float = 0.0
    prompt_override: str | None = None
    seed_override: int | None = None
    dry_run: bool = False
    interactive: bool = False


@dataclass
class BatchResult:
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0
    failures: list[tuple[Path, str]] | None = None

    def __post_init__(self):
        if self.failures is None:
            self.failures = []


def run_batch(config: GlobalConfig, options: BatchOptions) -> BatchResult:
    binary = resolve_llama_mtmd_cli(config, options.llama_bin_override)

    model_override = (
        resolve_model_dir(options.model_dir, interactive=options.interactive)
        if options.model_dir is not None
        else None
    )
    profile: Profile = load_profile(
        options.profile_name, config, model_override, interactive=options.interactive
    )

    if options.trigger_override is not None:
        profile.trigger_word = TriggerWordConfig(
            options.trigger_override, profile.trigger_word.placement
        )
    if options.prompt_override is not None:
        profile.prompt_text = options.prompt_override
    if options.seed_override is not None:
        profile.generation.seed = options.seed_override

    resize_megapixels = options.resize_megapixels or config.preprocessing.resize_megapixels

    logger.info("Profile: %s (%s)", profile.name, profile.description)
    logger.info("gguf: %s", profile.gguf_path)
    logger.info("mmproj: %s", profile.mmproj_path)
    logger.info(
        "Trigger word: %s",
        profile.trigger_word.value if profile.trigger_word.value else "(disabled)",
    )
    if resize_megapixels:
        logger.info("Resize target: %.2f MP", resize_megapixels)

    images = list_images(options.input_dir, options.recursive)
    if options.limit is not None:
        images = images[: options.limit]

    if not images:
        logger.warning("No images found in %s", options.input_dir)
        return BatchResult()

    if options.dry_run:
        return _dry_run(profile, images, options)

    result = BatchResult()

    tmp_dir: Path | None = None
    if resize_megapixels:
        tmp_dir = Path(tempfile.mkdtemp(prefix="llamacap_resize_"))

    try:
        for image_path in tqdm(images, desc="Captioning", unit="img"):
            sidecar_path = sidecar_path_for(
                image_path, options.input_dir, options.output_dir, profile.output_suffix
            )

            if should_skip(sidecar_path, options.overwrite):
                result.skipped += 1
                continue

            error = validate_image(image_path)
            if error:
                result.failed += 1
                result.failures.append((image_path, error))
                continue

            analysis_path = image_path
            if tmp_dir is not None:
                try:
                    analysis_path = resize_for_analysis(image_path, resize_megapixels, tmp_dir)
                except Exception as e:
                    result.failed += 1
                    result.failures.append((image_path, f"resize failed: {e}"))
                    continue

            try:
                raw_caption = generate_caption(
                    binary, profile, analysis_path, config.generation.timeout_seconds
                )
            except CaptionGenerationError as e:
                result.failed += 1
                result.failures.append((image_path, str(e)))
                continue

            final_caption = apply_trigger_word(raw_caption, profile.trigger_word)
            write_caption(sidecar_path, final_caption)
            result.succeeded += 1
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    _report(result, options.input_dir)
    return result


def _dry_run(profile: Profile, images: list[Path], options: BatchOptions) -> BatchResult:
    logger.info("Prompt preview: %s", profile.prompt_text[:80])
    result = BatchResult()
    invalid = 0

    for image_path in images:
        sidecar_path = sidecar_path_for(
            image_path, options.input_dir, options.output_dir, profile.output_suffix
        )
        if should_skip(sidecar_path, options.overwrite):
            result.skipped += 1
            continue

        error = validate_image(image_path)
        if error:
            invalid += 1
            result.failures.append((image_path, error))
            continue

        result.succeeded += 1

    print(
        f"\n[dry-run] Would process {result.succeeded}, skip {result.skipped}, "
        f"invalid {invalid} (of {len(images)} images)."
    )
    result.failed = invalid
    return result


def _report(result: BatchResult, input_dir: Path) -> None:
    print(
        f"\nDone. {result.succeeded} captioned, {result.skipped} skipped, "
        f"{result.failed} failed."
    )
    if result.failures:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = input_dir / f"llamacap_failures_{timestamp}.txt"
        lines = [f"{path}\t{reason}" for path, reason in result.failures]
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Failure details written to {report_path}")
