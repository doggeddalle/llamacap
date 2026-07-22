from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from llamacap.binary_resolver import resolve_llama_server
from llamacap.captioner import generate_caption
from llamacap.config import GlobalConfig
from llamacap.errors import CaptionGenerationError, LlamacapError
from llamacap.image_utils import list_images, validate_image
from llamacap.model_resolver import resolve_model_dir
from llamacap.postprocess import apply_trigger_word
from llamacap.profiles import Profile, TriggerWordConfig, load_profile
from llamacap.resize import resize_for_analysis
from llamacap.server import LlamaServer
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
    single_image: Path | None = None
    trigger_override: str | None = None
    model_dir: Path | None = None
    model_files: tuple[Path, Path] | None = None
    # None = fall back to config.toml; an explicit 0 disables resizing.
    resize_megapixels: float | None = None
    prompt_override: str | None = None
    seed_override: int | None = None
    dry_run: bool = False
    interactive: bool = False
    progress_json: bool = False


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
    binary = resolve_llama_server(config, options.llama_bin_override)

    model_override = options.model_files
    if model_override is None and options.model_dir is not None:
        model_override = resolve_model_dir(
            options.model_dir, interactive=options.interactive
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

    resize_megapixels = (
        options.resize_megapixels
        if options.resize_megapixels is not None
        else config.preprocessing.resize_megapixels
    )

    logger.info("Profile: %s (%s)", profile.name, profile.description)
    logger.info("gguf: %s", profile.gguf_path)
    logger.info("mmproj: %s", profile.mmproj_path)
    logger.info(
        "Trigger word: %s",
        profile.trigger_word.value if profile.trigger_word.value else "(disabled)",
    )
    if resize_megapixels:
        logger.info("Resize target: %.2f MP", resize_megapixels)

    images = (
        [options.single_image]
        if options.single_image is not None
        else list_images(options.input_dir, options.recursive)
    )
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

    server = LlamaServer(binary, profile, config.generation.server_startup_timeout_seconds)

    if options.progress_json:
        _emit_progress("phase", phase="loading", total=len(images))
        bar = None
        iterator = images
    else:
        bar = tqdm(images, desc="Captioning", unit="img")
        iterator = bar

    try:
        server.start()
        if options.progress_json:
            _emit_progress("start", total=len(images))

        for done, image_path in enumerate(iterator, start=1):
            _process_image(
                image_path, server, profile, config, options, tmp_dir,
                resize_megapixels, result,
            )
            if bar is not None:
                bar.set_postfix(
                    ok=result.succeeded, skip=result.skipped, fail=result.failed
                )
            if options.progress_json:
                _emit_progress(
                    "image",
                    done=done,
                    total=len(images),
                    ok=result.succeeded,
                    skip=result.skipped,
                    fail=result.failed,
                    current=image_path.name,
                )
    finally:
        if bar is not None:
            bar.close()
        server.stop()
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if options.progress_json:
        _emit_progress(
            "end", ok=result.succeeded, skip=result.skipped, fail=result.failed
        )
    _report(result, options.input_dir)
    return result


def _process_image(
    image_path: Path,
    server: LlamaServer,
    profile: Profile,
    config: GlobalConfig,
    options: BatchOptions,
    tmp_dir: Path | None,
    resize_megapixels: float,
    result: BatchResult,
) -> None:
    sidecar_path = sidecar_path_for(
        image_path, options.input_dir, options.output_dir, profile.output_suffix
    )

    if should_skip(sidecar_path, options.overwrite):
        result.skipped += 1
        return

    error = validate_image(image_path)
    if error:
        result.failed += 1
        result.failures.append((image_path, error))
        return

    analysis_path = image_path
    if tmp_dir is not None:
        try:
            analysis_path = resize_for_analysis(image_path, resize_megapixels, tmp_dir)
        except Exception as e:
            result.failed += 1
            result.failures.append((image_path, f"resize failed: {e}"))
            return

    try:
        raw_caption = generate_caption(
            server, profile, analysis_path, config.generation.timeout_seconds
        )
    except CaptionGenerationError as e:
        if not server.is_running:
            # The server itself died (crash, OOM, killed): abort the batch with
            # its log instead of failing every remaining image one by one.
            raise LlamacapError(
                f"llama-server terminated unexpectedly while captioning "
                f"{image_path.name} ({result.succeeded} captioned before the crash).\n"
                f"Log tail:\n{server.log_tail()}"
            ) from e
        result.failed += 1
        result.failures.append((image_path, str(e)))
        return

    final_caption = apply_trigger_word(raw_caption, profile.trigger_word)
    try:
        write_caption(sidecar_path, final_caption)
    except OSError as e:
        result.failed += 1
        result.failures.append((image_path, f"could not write sidecar: {e}"))
        return
    result.succeeded += 1


def _emit_progress(event: str, **fields) -> None:
    print("@@LLAMACAP@@ " + json.dumps({"event": event, **fields}), flush=True)


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
