from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

from llamacap.config import load_config
from llamacap.errors import LlamacapError
from llamacap.profiles import list_profile_names
from llamacap.runner import BatchOptions, run_batch


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite number zero or greater")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llamacap",
        description="Batch-caption images with a local llama.cpp GGUF vision model, for LoRA training data prep.",
    )
    parser.add_argument("--profile", help="Profile name (see --list-profiles)")
    parser.add_argument("--input", type=Path, help="Directory of images to caption")
    parser.add_argument("--image", type=Path, default=None, help="Caption one specific image from --input")
    parser.add_argument("--output-dir", type=Path, default=None, help="Write sidecars here instead of in-place")
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=None,
        help="Regenerate captions even if a sidecar exists (or override config with --no-overwrite)",
    )
    parser.add_argument(
        "--recursive", action=argparse.BooleanOptionalAction, default=None,
        help="Recurse into subdirectories (or override config with --no-recursive)",
    )
    parser.add_argument("--limit", type=_positive_int, default=None, help="Only process the first N images")
    parser.add_argument("--llama-bin", default=None, help="Override the resolved llama-server path for this run")
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles and exit")
    parser.add_argument("--trigger", default=None, help='Override the profile trigger word for this run (use "" to disable it)')
    parser.add_argument("--model", type=Path, default=None, help="Directory with exactly one .gguf and one *mmproj*.gguf; overrides the profile's [model]")
    parser.add_argument("--model-gguf", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model-mmproj", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--size", type=_non_negative_float, default=None, help="Resize images to this many target megapixels before captioning (0 disables resizing even if config.toml sets a default)")
    parser.add_argument("--prompt", default=None, help="Override the profile prompt text for this run")
    parser.add_argument("--seed", type=int, default=None, help="Override the profile generation seed for this run")
    parser.add_argument("--dry-run", action="store_true", help="Resolve everything and report what would happen, without captioning or writing files")
    parser.add_argument("--interactive", action="store_true", help="Prompt to pick a model when resolution is ambiguous, instead of failing fast (requires a real terminal)")
    parser.add_argument("--config", type=Path, default=None, help="Path to an alternate global config.toml")
    parser.add_argument("--progress-json", action="store_true", help="Emit machine-readable progress lines instead of a progress bar (for front-ends)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        config = load_config(args.config) if args.config else load_config()

        if not args.verbose:
            # -v always wins; otherwise honor [logging].level from config.toml.
            level = getattr(logging, config.logging.level.upper(), logging.INFO)
            logging.getLogger().setLevel(level)

        if args.list_profiles:
            names = list_profile_names(config)
            if names:
                print("\n".join(names))
            else:
                print("(no profiles found)")
            return 0

        if not args.profile or not args.input:
            parser.error("--profile and --input are required (unless using --list-profiles)")

        if not args.input.is_dir():
            raise LlamacapError(f"--input is not a directory: {args.input}")
        if args.image is not None:
            if not args.image.is_file():
                raise LlamacapError(f"--image is not a file: {args.image}")
            try:
                args.image.resolve().relative_to(args.input.resolve())
            except ValueError as e:
                raise LlamacapError("--image must be inside --input") from e

        exact_model = args.model_gguf is not None or args.model_mmproj is not None
        if exact_model and (args.model_gguf is None or args.model_mmproj is None):
            raise LlamacapError("--model-gguf and --model-mmproj must be supplied together")
        if exact_model and args.model is not None:
            raise LlamacapError("Use either --model or the exact model file pair, not both")
        if exact_model:
            if not args.model_gguf.is_file():
                raise LlamacapError(f"GGUF model file not found: {args.model_gguf}")
            if not args.model_mmproj.is_file():
                raise LlamacapError(f"mmproj model file not found: {args.model_mmproj}")

        options = BatchOptions(
            profile_name=args.profile,
            input_dir=args.input,
            single_image=args.image,
            output_dir=(
                args.output_dir
                if args.output_dir is not None
                else config.project_root / config.output.default_dir
                if config.output.default_mode == "output_dir"
                else None
            ),
            overwrite=config.output.overwrite if args.overwrite is None else args.overwrite,
            recursive=config.output.recursive if args.recursive is None else args.recursive,
            limit=args.limit,
            llama_bin_override=args.llama_bin,
            trigger_override=args.trigger,
            model_dir=args.model,
            model_files=(args.model_gguf, args.model_mmproj) if exact_model else None,
            resize_megapixels=args.size,
            prompt_override=args.prompt,
            seed_override=args.seed,
            dry_run=args.dry_run,
            interactive=args.interactive,
            progress_json=args.progress_json,
        )

        result = run_batch(config, options)
        return 2 if result.failed else 0

    except LlamacapError as e:
        logging.error(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
