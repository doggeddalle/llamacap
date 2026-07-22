# llamacap User Guide

A step-by-step walkthrough for setting up and using llamacap. For the compact
option reference, see [README.md](README.md).

## Contents

1. [What llamacap does](#what-llamacap-does)
2. [Prerequisites](#prerequisites)
3. [First-time setup](#first-time-setup)
4. [Using the GUI](#using-the-gui)
5. [Using the CLI](#using-the-cli)
6. [Creating a custom profile](#creating-a-custom-profile)
7. [config.toml reference](#configtoml-reference)
8. [Troubleshooting](#troubleshooting)

---

## What llamacap does

llamacap points a local vision-language model (a llama.cpp GGUF such as
Qwen3-VL-4B-Instruct) at a folder of images and writes one caption per image as a
`.txt` sidecar file next to it — the Kohya-ss convention used for LoRA /
diffusion-model training datasets.

For each run it starts a single `llama-server` process, waits for the model to
load, then captions the whole batch against that one resident server — so the
multi-gigabyte model is loaded once per run, not once per image. A tqdm progress
bar tracks the batch, and any per-image failures are collected into a
`llamacap_failures_<timestamp>.txt` report instead of aborting the run.

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

## Prerequisites

- **Windows** (the GUI launcher and DPI handling target Windows; the CLI itself
  is plain Python).
- **Python 3.11+** and **[uv](https://docs.astral.sh/uv/)** — `uv sync` creates
  the project virtualenv and installs the (small) dependency set: Pillow, tqdm,
  and the GUI extras (darkdetect, tkinterdnd2, tomli-w).
- **llama.cpp's `llama-server`** — the one external tool llamacap drives. Any of
  these works; llamacap resolves them in this order:
  1. `--llama-bin PATH` on the command line
  2. `[llama_cpp].binary_path_override` in `config.toml` (directory or full exe path)
  3. `llama-server` on your PATH — easiest install: `winget install ggml.llamacpp`
  4. `llama-server.exe` dropped into the project's `bin\` folder

  For GPU acceleration make sure you install a llama.cpp build matching your
  hardware (e.g. a CUDA build for NVIDIA cards).

## First-time setup

1. **Install dependencies** (from the project root):

   ```
   uv sync
   ```

2. **Get the model.** The `models\` folder is empty on purpose — model weights
   are not part of the project. The bundled `krea2` profile expects this pair
   (from the Qwen3-VL-4B-Instruct GGUF release you use) placed directly in
   `models\`:

   - `Qwen3-VL-4B-Instruct-Unredacted-MAX.Q8_0.gguf` — the main model (~4 GB)
   - `Qwen3-VL-4B-Instruct-Unredacted-MAX.mmproj-q8_0.gguf` — the vision
     projector (~433 MB)

   Both files are required: the mmproj file is what lets the text model see
   images. If you use a different model pair, either rename nothing and instead
   pass `--model DIR` at run time, or edit `profiles\krea2.toml` `[model]` to
   your filenames.

3. **Sanity check** — this should print `krea2`:

   ```
   uv run scripts/caption.py --list-profiles
   ```

4. **Dry run** against a folder of images (resolves the binary, profile, model
   and prompt, and reports what would happen — writes nothing):

   ```
   uv run scripts/caption.py --profile krea2 --input C:\path\to\images --dry-run
   ```

## Using the GUI

Launch with **`llamacap-gui.bat`** (double-click; it uses `pythonw` so no console
window lingers), or from a terminal:

```
uv run python scripts/gui.py
```

The GUI uses a flat Windows 11-style theme and offers System, Light, and Dark
appearance modes. You can also **drag and drop a folder** from Explorer anywhere
onto the window to set the input path.

The window has two tabs: **Caption** for everyday dataset work and **Profiles &
Settings** for profile maintenance, appearance, alternate configuration, and
keyboard help. The Caption tab uses a resizable split view: setup and readiness
on the left, image/caption review and run status on the right. Folders, profile,
theme, window geometry, recursion choice, and disclosure-panel states are
remembered. Overwrite and prompt overrides deliberately are not persisted.

### Profile & paths

- **Profile** — dropdown of everything in `profiles\`. **Edit…** opens the
  selected profile in the profile editor; **New…** creates a profile using the
  selected one as a template (see "Creating a custom profile" below).
  **Refresh** re-scans the folder.
- **Input folder** — the folder of images to caption. Required. Once set, a
  small "≈N images found" count appears beneath it, and **Open** opens the
  folder in Explorer.
- **Output folder** — optional. Leave blank to write `.txt` sidecars in place
  next to each image (the usual choice for training datasets).

### Model override (optional)

- **GGUF model** / **mmproj file** — pick a main `.gguf` and its `*mmproj*.gguf`
  to use instead of the profile's `[model]` settings for this run. The GUI passes
  this exact pair, so other GGUF files in the same folder do not cause ambiguity.
  Leave both blank to use what the profile configures.

### Per-run overrides (optional)

- **Override trigger word** — check the box to take control of the trigger
  (activation) word for this run. With the box checked and the field left
  empty, the run explicitly *disables* any trigger word the profile sets;
  with text entered, that text is prefixed to every caption.
- **Resize (megapixels)** — downscale/upscale images to this many megapixels
  (aspect ratio preserved) before captioning. `0` disables resizing.
- **Seed** — override the profile's generation seed.
- **Limit (first N images)** — caption only the first N images; handy for
  testing a prompt on a few images before committing to a full batch.
- **Prompt** — free-text prompt override; blank means use the profile's prompt.
- **Config file** — load an alternate global config instead of the project's
  `config.toml`.

### Options

- **Overwrite existing sidecars** — by default images that already have a
  `.txt` sidecar are skipped; check this to re-caption them.
- **Recurse into subfolders** — include images in subdirectories.
- **Preview run** — resolve everything and report counts without captioning or
  writing files.
- **Verbose diagnostic logging** — available on Profiles & Settings; adds debug
  output to Technical details.

### Running

**Run captioning** starts the batch as a subprocess. While the model loads the
progress bar animates with "Starting llama-server / loading model…", then
switches to a real per-image bar with a live status line
(`12/120 — 11 ok · 1 skipped · 0 failed`). Full output still streams into the
collapsible **Technical details** pane. The status card also reports the current
phase and filename, elapsed time, ETA, and color-coded result counts.

**Preview run** validates and reports the batch without writing captions.
Before an overwrite run, the GUI reports how many captions would be replaced
and asks for confirmation. The setup-readiness card checks llama-server, model
files, supported images, and output writability before a run.

The preview pane displays each image with EXIF orientation applied and loads its
current sidecar. Use the arrow buttons (or Left/Right keys) to browse, **Save
caption** to edit a sidecar atomically, and **Caption this image** to regenerate
only the displayed item using the exact selected GGUF/mmproj files.

Keyboard shortcuts: Ctrl+O chooses the input folder, Ctrl+Enter runs the batch,
Ctrl+Shift+Enter previews it, Ctrl+S saves the displayed caption, Left/Right
navigate images, and Escape stops an active run.

**Stop** first asks the process tree to exit cleanly and force-kills it after
3 seconds if needed — either way the `llama-server` child dies too, so no model
stays loaded in VRAM. **Clear log** empties the pane. If a run had failures,
**Open failure report** lights up and opens the `llamacap_failures_*.txt`
report directly.

## Using the CLI

Everything the GUI does maps to `scripts/caption.py` flags. Common recipes:

```
# Basic in-place captioning
uv run scripts/caption.py --profile krea2 --input C:\data\myset

# Preview what would happen (no files written)
uv run scripts/caption.py --profile krea2 --input C:\data\myset --dry-run

# Re-caption everything, including images that already have sidecars
uv run scripts/caption.py --profile krea2 --input C:\data\myset --overwrite

# Include subfolders, write captions to a separate folder
uv run scripts/caption.py --profile krea2 --input C:\data\myset --recursive --output-dir C:\data\captions

# Add a trigger word for this run (prefixed to every caption)
uv run scripts/caption.py --profile krea2 --input C:\data\myset --trigger "myloraname"

# Resize to ~1 megapixel before captioning, test on the first 5 images
uv run scripts/caption.py --profile krea2 --input C:\data\myset --size 1.0 --limit 5

# Use a different model pair without editing any TOML
uv run scripts/caption.py --profile krea2 --input C:\data\myset --model D:\gguf\other-model

# Ambiguous models folder? Pick interactively from a numbered list
uv run scripts/caption.py --profile krea2 --input C:\data\myset --interactive

# Force-disable resizing even if config.toml sets a default
uv run scripts/caption.py --profile krea2 --input C:\data\myset --size 0

# Machine-readable progress (what the GUI uses) instead of the tqdm bar
uv run scripts/caption.py --profile krea2 --input C:\data\myset --progress-json
```

Exit behavior: images that fail (corrupt file, request timeout, server error)
don't abort the batch — they're logged and listed in
`llamacap_failures_<timestamp>.txt` in the project root.

## Creating a custom profile

The easiest way is the GUI's profile editor: select a profile to base yours on,
click **New…**, adjust the fields (model, prompt — editable right in the
dialog, trigger word, generation parameters), give it a name, and Save. The
editor writes the TOML for you and, in file mode, saves your prompt edits back
to the `prompts\` file.

By hand:

1. Copy `profiles\krea2.toml` to `profiles\myprofile.toml` — or start minimal:
   only `[profile]`, `[model]`, and `[prompt]` are required. `[trigger_word]`,
   `[generation]`, and `[output]` fall back to krea2-equivalent defaults when
   omitted, so this is a complete, valid profile:

   ```toml
   [profile]
   name = "myprofile"
   description = "My captioning profile."

   [model]
   gguf_file = "my-model.Q8_0.gguf"
   mmproj_file = "my-model.mmproj-q8_0.gguf"

   [prompt]
   text = "Describe this image for training data."
   ```

2. Edit the sections:

   | Section | What it controls |
   |---|---|
   | `[profile]` | `name` (must match how you call it) and a human description. |
   | `[model]` | Which GGUF + mmproj pair to load. Use `gguf_file`/`mmproj_file` for filenames under `models\`, or `gguf_path`/`mmproj_path` for absolute paths (paths win if both are set). |
   | `[prompt]` | `file` — a text file under `prompts\` (path relative to project root) — or inline `text`, which overrides `file` when non-empty. |
   | `[trigger_word]` | `value` (empty = disabled) and `placement`: `prefix_comma` (`word, caption`), `prefix_period` (`word. caption`), or `none`. |
   | `[generation]` | llama-server / sampling parameters: `ctx_size`, `n_predict` (max caption tokens), `temperature`, `top_p`, `top_k`, `repeat_penalty`, `ngl` (GPU layers; 99 = all), `image_min_tokens`, `seed`, `no_warmup`, and `extra_args` for any raw llama-server flags. |
   | `[output]` | Sidecar `suffix` (default `.txt`). |

3. For the prompt, add a new file under `prompts\` (see
   `prompts\krea2_caption.txt` for the style: describe how the model should
   caption — subject, style, composition, what to avoid).
4. Verify it loads, then test on a few images:

   ```
   uv run scripts/caption.py --list-profiles
   uv run scripts/caption.py --profile myprofile --input C:\data\myset --limit 3 --overwrite
   ```

## config.toml reference

Global settings that apply to every profile/run (override per-run with `--config`):

- `[llama_cpp]`
  - `binary_path_override` — if set, wins over PATH and `bin\`. Directory or
    direct exe path.
  - `bin_subdir` — project-relative folder checked for `llama-server.exe`
    (default `bin`).
- `[models]`
  - `default_dir` — where bare `gguf_file`/`mmproj_file` names and
    `--interactive` scanning resolve (default `models`).
- `[output]`
  - `default_mode` — `in_place` (sidecars next to images) or `output_dir`.
  - `default_dir` — destination used in `output_dir` mode when `--output-dir`
    is omitted (default `captions`, relative to the project root).
  - `overwrite`, `recursive` — defaults for the corresponding flags.
- `[generation]`
  - `timeout_seconds` — per-image request safety net (default 300).
  - `server_startup_timeout_seconds` — how long to wait for llama-server to
    load the model and report healthy (default 120; raise for very large GGUFs
    or slow disks).
- `[logging]`
  - `level` — e.g. `INFO`, `DEBUG`.
- `[preprocessing]`
  - `resize_megapixels` — default resize target; `0.0` disables. `--size`
    overrides per run.

## Troubleshooting

**"llama-server not found"** — install llama.cpp (`winget install ggml.llamacpp`),
or set `[llama_cpp].binary_path_override` in `config.toml`, or drop
`llama-server.exe` (with its DLLs) into `bin\`. Verify with `llama-server --version`
in a fresh terminal.

**"gguf file not found: ...models\...gguf"** — the profile points at a model that
isn't in `models\` yet. Place the GGUF + mmproj pair there (see
[First-time setup](#first-time-setup)), or pass `--model DIR` at the folder that
has them.

**"exactly one main .gguf" errors with `--model DIR`** — the folder must contain
exactly one main `.gguf` and one `*mmproj*.gguf`, no more, no less, and
subfolders aren't searched. Either clean up the folder or add `--interactive`
to pick from a list.

**Server startup timeout** — big models on slow disks can take longer than the
default 120 s to load. Raise `[generation].server_startup_timeout_seconds` in
`config.toml`. Also check the log (`Verbose logging` / `-v`) for llama-server's
own error output — an out-of-VRAM failure looks like a timeout from the outside.

**"llama-server terminated unexpectedly while captioning ..."** — the server
process crashed mid-batch (most often VRAM exhaustion or a driver reset). The
batch aborts immediately with the tail of the server's log in the error, and
the full log file (`%TEMP%\llamacap_server_*.log`) is kept for inspection —
already-written captions are preserved, and re-running skips them. On clean
runs these temp logs are deleted automatically.

**Per-image timeouts / very slow captions** — check that llama.cpp is actually
using your GPU (`ngl = 99` in the profile, CUDA build installed). CPU-only
inference of a 4 GB vision model is slow; raising
`[generation].timeout_seconds` helps, but fixing GPU offload helps more.

**Some images failed** — the run finishes anyway and writes
`llamacap_failures_<timestamp>.txt` in the project root listing each failed image
and why (corrupt file, timeout, server error). Fix or remove those images and
re-run; already-captioned images are skipped unless `--overwrite` is set.

**GUI opens then a run does nothing** — check the Output pane; the most common
causes are the model-not-found and llama-server-not-found errors above, which
appear there exactly as they would in a terminal.
