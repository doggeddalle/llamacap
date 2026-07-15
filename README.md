# llamacap

Batch-captions images with a local llama.cpp GGUF vision-language model, producing
per-image `.txt` sidecar captions for LoRA training data (Kohya-ss convention).

## Setup

```
uv sync
```

Requires `llama-mtmd-cli` on PATH (e.g. `winget install ggml.llamacpp`), or set
`[llama_cpp].binary_path_override` in `config.toml`, or drop the binary in `bin/`.

## Usage

```
uv run scripts/caption.py --list-profiles
uv run scripts/caption.py --profile krea2 --input C:\path\to\images
```

Options: `--output-dir DIR`, `--overwrite`, `--recursive`, `--limit N`,
`--llama-bin PATH`, `-v/--verbose`.

### Per-run overrides

- `--trigger TEXT` — override the profile's trigger word for this run. Pass
  `--trigger ""` to disable a trigger word the profile has configured.
- `--model DIR` — use the model pair found in `DIR` instead of the profile's
  configured `[model]` paths. `DIR` must contain exactly one main `.gguf` and
  one `.gguf` with "mmproj" in its filename (case-insensitive); subdirectories
  are not searched.
- `--size FLOAT` — resize images to this many target megapixels (aspect ratio
  preserved, upscaling allowed) before captioning. Overrides
  `config.toml [preprocessing].resize_megapixels`.
- `--prompt TEXT` — override the profile's prompt text for this run.
- `--seed INT` — override the profile's generation seed for this run.
- `--dry-run` — resolve the binary, profile, model, and prompt, and report
  how many images would be captioned/skipped/rejected, without running any
  captioning or writing any files.
- `--config PATH` — load an alternate global config file instead of the
  project-root `config.toml`.
- `--interactive` — when model resolution is ambiguous, prompt on the
  terminal to pick instead of failing fast. See "Setting the model" below.

## Profiles

Profiles live in `profiles/*.toml` and configure the GGUF/mmproj model pair,
the captioning prompt, an optional trigger/activation word, and generation
parameters. See `profiles/krea2.toml` for the reference profile.

## Setting the model

Profiles ship with an empty `[model]` section — you must point them at a
GGUF + mmproj pair before running, using one of two methods:

1. **Edit the profile's `[model]` section** (persists across runs):
   - `gguf_path` / `mmproj_path` — absolute paths to the two files, or
   - `gguf_file` / `mmproj_file` — bare filenames resolved under
     `[models].default_dir` in `config.toml` (defaults to `models/`).

   Only one of the two pairs is needed; `*_path` takes precedence if both
   are set.

2. **Pass `--model DIR` at the command line** (per-run, no TOML edits):
   `DIR` must contain exactly one main `.gguf` and one `.gguf` with
   "mmproj" in its filename (case-insensitive); subdirectories aren't
   searched. This fully overrides the profile's `[model]` section for that
   run, including bypassing validation of whatever the profile has
   configured.

If neither is set, running the profile fails fast with a clear error
naming the missing setting — unless `--interactive` is passed, in which case
llamacap scans `[models].default_dir` and, if it finds more than one
candidate `.gguf`/mmproj file, prompts you to pick from a numbered list
(requires a real terminal; falls back to the same fail-fast error if stdin
isn't a TTY). The same prompt is used if `--model DIR` itself points at an
ambiguous directory with `--interactive` set. Off by default, so unattended
or scripted runs never hang waiting on input.
