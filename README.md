# llamacap

Batch-captions images with a local llama.cpp GGUF vision-language model, producing
per-image `.txt` sidecar captions for LoRA training data (Kohya-ss convention).

Includes both a command-line interface and a Tkinter GUI. The initial captioning
scope profile is Krea 2; the roadmap is to add more soon. Profiles and configs are
entirely customizable and you can create your own.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ea0896e2-fff1-4b11-8bad-9c93b25a7c46" alt="Screenshot 2026-07-20 130520" width="50%">
</p>

> New here? See **[GUIDE.md](GUIDE.md)** for a step-by-step walkthrough covering
> setup, the GUI, custom profiles, and troubleshooting.

## Setup

```
uv sync
```

Requires `llama-server` on PATH (e.g. `winget install ggml.llamacpp`), or set
`[llama_cpp].binary_path_override` in `config.toml`, or drop the binary in `bin/`.
llamacap starts one `llama-server` process per run and keeps the model resident
in memory for the whole batch, rather than reloading it per image.

## Models

The `models/` folder ships **empty** — GGUF weights are not distributed with the
project. The bundled `krea2` profile expects this pair placed in `models/`:

- `Qwen3-VL-4B-Instruct-Unredacted-MAX.Q8_0.gguf` (main model, ~4 GB)
- `Qwen3-VL-4B-Instruct-Unredacted-MAX.mmproj-q8_0.gguf` (vision projector, ~433 MB)

Alternatively, point at any GGUF + mmproj pair with `--model DIR` at run time, or
edit the profile's `[model]` section (see "Setting the model" below).

## GUI

```
llamacap-gui.bat
```

Double-click the launcher (or run `uv run python scripts/gui.py`). The task-focused
two-tab workspace includes:

- first-run checks for llama-server, model files, images, and output writability
- a resizable image preview and caption editor with atomic saves and one-image
  regeneration
- exact GGUF/mmproj selection, inline validation, overwrite forecasting, and a
  safe preview-run action
- phase, filename, elapsed time, ETA, color-coded counts, and a collapsible log
- persistent folders/profile/theme/layout with System, Light, and Dark themes
- profile creation/editing, drag-and-drop, keyboard shortcuts, and clean Stop
  behavior for the complete process tree

Details in [GUIDE.md](GUIDE.md).

## CLI usage

```
uv run scripts/caption.py --list-profiles
uv run scripts/caption.py --profile krea2 --input C:\path\to\images
```

Options: `--output-dir DIR`, `--overwrite` / `--no-overwrite`,
`--recursive` / `--no-recursive`, `--limit N`,
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
  `config.toml [preprocessing].resize_megapixels`; `--size 0` disables
  resizing even when the config sets a default.
- `--prompt TEXT` — override the profile's prompt text for this run.
- `--seed INT` — override the profile's generation seed for this run.
- `--dry-run` — resolve the binary, profile, model, and prompt, and report
  how many images would be captioned/skipped/rejected, without running any
  captioning or writing any files.
- `--config PATH` — load an alternate global config file instead of the
  project-root `config.toml`.
- `--interactive` — when model resolution is ambiguous, prompt on the
  terminal to pick instead of failing fast. See "Setting the model" below.
- `--progress-json` — emit machine-readable progress lines
  (`@@LLAMACAP@@ {...}`) instead of the tqdm bar, for driving front-ends;
  this is what the GUI uses for its progress bar.

## Profiles

Profiles live in `profiles/*.toml` and configure the GGUF/mmproj model pair,
the captioning prompt, an optional trigger/activation word, and generation
parameters. See `profiles/krea2.toml` for the reference profile, or use the
GUI's profile editor (Edit… / New… next to the profile dropdown).

Only `[profile]`, `[model]`, and `[prompt]` are required — `[trigger_word]`,
`[generation]`, and `[output]` fall back to sensible defaults (krea2's values)
when omitted, so a minimal custom profile is just a few lines.

## Setting the model

Point a profile at a GGUF + mmproj pair using one of two methods:

1. **Edit the profile's `[model]` section** (persists across runs):
   - `gguf_path` / `mmproj_path` — absolute paths to the two files, or
   - `gguf_file` / `mmproj_file` — bare filenames resolved under
     `[models].default_dir` in `config.toml` (defaults to `models/`).

   Only one of the two pairs is needed; `*_path` takes precedence if both
   are set. The bundled `krea2` profile uses the bare-filename form, so it
   works as soon as the model pair is dropped into `models/`.

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

## Project structure

```
llamacap-gui.bat        GUI launcher (Windows)
config.toml             Global settings (binary/model dirs, output, timeouts, ...)
bin/                    Optional: drop llama-server.exe here
models/                 GGUF model files go here (empty by default)
profiles/               Captioning profiles (*.toml)
prompts/                Prompt text files referenced by profiles
scripts/caption.py      CLI entry point
scripts/gui.py          GUI launcher
src/llamacap/           The Python package (CLI, runner, server manager, ...)
src/llamacap/gui/       The GUI application (app, profile editor, widgets)
```
