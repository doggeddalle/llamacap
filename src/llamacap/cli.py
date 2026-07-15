from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from llamacap.config import load_config
from llamacap.errors import LlamacapError
from llamacap.profiles import list_profile_names
from llamacap.runner import BatchOptions, run_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llamacap",
        description="Batch-caption images with a local llama.cpp GGUF vision model, for LoRA training data prep.",
    )
    parser.add_argument("--profile", help="Profile name (see --list-profiles)")
    parser.add_argument("--input", type=Path, help="Directory of images to caption")
    parser.add_argument("--output-dir", type=Path, default=None, help="Write sidecars here instead of in-place")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate captions even if a sidecar already exists")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images")
    parser.add_argument("--llama-bin", default=None, help="Override the resolved llama-mtmd-cli path for this run")
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles and exit")
    parser.add_argument("--trigger", default=None, help='Override the profile trigger word for this run (use "" to disable it)')
    parser.add_argument("--model", type=Path, default=None, help="Directory with exactly one .gguf and one *mmproj*.gguf; overrides the profile's [model]")
    parser.add_argument("--size", type=float, default=None, help="Resize images to this many target megapixels before captioning")
    parser.add_argument("--prompt", default=None, help="Override the profile prompt text for this run")
    parser.add_argument("--seed", type=int, default=None, help="Override the profile generation seed for this run")
    parser.add_argument("--dry-run", action="store_true", help="Resolve everything and report what would happen, without captioning or writing files")
    parser.add_argument("--interactive", action="store_true", help="Prompt to pick a model when resolution is ambiguous, instead of failing fast (requires a real terminal)")
    parser.add_argument("--config", type=Path, default=None, help="Path to an alternate global config.toml")
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

        options = BatchOptions(
            profile_name=args.profile,
            input_dir=args.input,
            output_dir=args.output_dir,
            overwrite=args.overwrite or config.output.overwrite,
            recursive=args.recursive or config.output.recursive,
            limit=args.limit,
            llama_bin_override=args.llama_bin,
            trigger_override=args.trigger,
            model_dir=args.model,
            resize_megapixels=args.size or 0.0,
            prompt_override=args.prompt,
            seed_override=args.seed,
            dry_run=args.dry_run,
            interactive=args.interactive,
        )

        result = run_batch(config, options)
        return 2 if result.failed else 0

    except LlamacapError as e:
        logging.error(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
