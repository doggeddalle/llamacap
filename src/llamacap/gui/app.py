"""Task-focused Tkinter front-end for llamacap."""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageOps, ImageTk

try:
    import darkdetect
except ImportError:  # pragma: no cover
    darkdetect = None

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:  # pragma: no cover
    DND_FILES = None
    TkinterDnD = None

from llamacap.binary_resolver import resolve_llama_server
from llamacap.config import PROJECT_ROOT, load_config
from llamacap.gui.profile_editor import ProfileEditor
from llamacap.gui.state import (
    GuiPreferences,
    format_duration,
    summarize_dataset,
    validate_number,
)
from llamacap.gui.theme import apply_theme
from llamacap.gui.widgets import Collapsible, Tooltip, parse_dnd_paths
from llamacap.profiles import list_profile_names
from llamacap.sidecar import sidecar_path_for, write_caption

PROGRESS_SENTINEL = "@@LLAMACAP@@ "
FAILURE_REPORT_RE = re.compile(r"Failure details written to (.+)")


def enable_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        from ctypes import windll
        try:
            windll.user32.SetProcessDpiAwarenessContext(-4)
        except Exception:
            try:
                windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def apply_scaling(root: tk.Tk) -> float:
    dpi = root.winfo_fpixels("1i")
    root.tk.call("tk", "scaling", dpi / 72.0)
    return dpi / 96.0


class LlamacapGUI:
    def __init__(self, root: tk.Tk, scale: float) -> None:
        self.root, self.scale = root, scale
        self.proc: subprocess.Popen | None = None
        self.out_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.preferences = GuiPreferences.load()
        self.images: tuple[Path, ...] = ()
        self.image_index = 0
        self.preview_photo = None
        self._preview_source: Image.Image | None = None
        self._preview_path: Path | None = None
        self._preview_size: tuple[int, int] | None = None
        self._preview_after: str | None = None
        self._preview_generation = 0
        self._report_path: str | None = None
        self._refresh_generation = 0
        self._started_at = 0.0
        self._last_done = 0
        self._last_total = 0
        self._stop_escalation: str | None = None

        root.title("llamacap — local dataset captioning")
        root.geometry(self.preferences.geometry)
        root.minsize(self.px(900), self.px(620))

        self.theme_var = tk.StringVar(value=self.preferences.theme)
        self.theme = self._resolved_theme()
        self.palette = apply_theme(root, self.theme)

        self.profile_var = tk.StringVar(value=self.preferences.profile)
        self.input_var = tk.StringVar(value=self.preferences.input_dir)
        self.output_var = tk.StringVar(value=self.preferences.output_dir)
        self.gguf_var, self.mmproj_var = tk.StringVar(), tk.StringVar()
        self.trigger_enabled, self.trigger_var = tk.BooleanVar(), tk.StringVar()
        self.size_var, self.seed_var, self.limit_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.config_var = tk.StringVar()
        self.overwrite_var = tk.BooleanVar(value=False)
        self.recursive_var = tk.BooleanVar(value=self.preferences.recursive)
        self.verbose_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")
        self.count_var = tk.StringVar(value="Choose an image folder to begin.")
        self.validation_var = tk.StringVar()
        self.warning_var = tk.StringVar()
        self.profile_summary_var = tk.StringVar()
        self.readiness_var = tk.StringVar(value="Checking setup…")
        self.ok_var, self.skipped_var, self.failed_var = tk.StringVar(value="0 complete"), tk.StringVar(value="0 skipped"), tk.StringVar(value="0 failed")
        self.time_var = tk.StringVar(value="Elapsed 0:00")
        self.current_var = tk.StringVar(value="No image selected")

        self._build_ui()
        self._bind_shortcuts()
        self._setup_dnd()
        self._load_profiles()
        self._install_traces()
        self._refresh_dataset()
        self._refresh_readiness()
        root.after(60, self._drain_output)

    def px(self, n: int) -> int:
        return round(n * self.scale)

    def _resolved_theme(self) -> str:
        choice = self.theme_var.get().lower()
        if choice in {"light", "dark"}:
            return choice
        detected = (darkdetect.theme() or "").lower() if darkdetect else ""
        return detected if detected in {"light", "dark"} else "light"

    def _style_text_widget(self, widget: tk.Text) -> None:
        widget.configure(
            background=self.palette["text_bg"], foreground=self.palette["text_fg"],
            insertbackground=self.palette["text_fg"], relief="flat", borderwidth=1,
            highlightthickness=1, highlightbackground=self.palette["border"],
            highlightcolor=self.palette["accent"],
        )

    def _build_ui(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)
        self.caption_tab = ttk.Frame(self.notebook, padding=self.px(10))
        self.settings_tab = ttk.Frame(self.notebook, padding=self.px(10))
        self.notebook.add(self.caption_tab, text="  Caption  ")
        self.notebook.add(self.settings_tab, text="  Profiles & Settings  ")
        self._build_caption_tab()
        self._build_settings_tab()

    def _build_caption_tab(self) -> None:
        tab = self.caption_tab
        tab.rowconfigure(0, weight=1); tab.columnconfigure(0, weight=1)
        pane = ttk.Panedwindow(tab, orient="horizontal")
        pane.grid(row=0, column=0, sticky="nsew")
        left, right = ttk.Frame(pane, padding=(0, 0, 8, 0)), ttk.Frame(pane)
        pane.add(left, weight=2); pane.add(right, weight=3)
        left.columnconfigure(0, weight=1)
        right.columnconfigure(0, weight=1); right.rowconfigure(1, weight=1)

        setup = ttk.LabelFrame(left, text="Caption setup", padding=8)
        setup.grid(row=0, column=0, sticky="ew"); setup.columnconfigure(1, weight=1)
        ttk.Label(setup, text="Profile").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        row = ttk.Frame(setup); row.grid(row=0, column=1, sticky="ew", padx=4, pady=4); row.columnconfigure(0, weight=1)
        self.profile_combo = ttk.Combobox(row, textvariable=self.profile_var, state="readonly")
        self.profile_combo.grid(row=0, column=0, sticky="ew")
        ttk.Button(row, text="⋯", width=3, command=lambda: self.notebook.select(self.settings_tab)).grid(row=0, column=1, padx=(5, 0))
        self.input_entry = self._path_row(setup, 1, "Images", self.input_var, self._pick_input)
        self.output_entry = self._path_row(setup, 2, "Captions", self.output_var, self._pick_output)
        ttk.Label(setup, textvariable=self.count_var, style="Muted.TLabel", wraplength=self.px(360)).grid(row=3, column=1, sticky="w", padx=4)
        ttk.Checkbutton(setup, text="Include subfolders", variable=self.recursive_var).grid(row=4, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 2))
        ttk.Checkbutton(setup, text="Replace existing captions", variable=self.overwrite_var).grid(row=5, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        ttk.Label(setup, textvariable=self.warning_var, style="Warning.TLabel", wraplength=self.px(370)).grid(row=6, column=0, columnspan=2, sticky="w", padx=4, pady=(2, 5))

        self.advanced = Collapsible(left, "Advanced run options", open=self.preferences.advanced_open)
        self.advanced.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self._build_advanced(self.advanced.content)

        readiness = ttk.LabelFrame(left, text="Setup readiness", padding=8)
        readiness.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(readiness, textvariable=self.readiness_var, justify="left", wraplength=self.px(380)).pack(anchor="w")
        ttk.Button(readiness, text="Check again", command=self._refresh_readiness).pack(anchor="w", pady=(6, 0))
        ttk.Label(left, textvariable=self.validation_var, style="Error.TLabel", wraplength=self.px(390)).grid(row=3, column=0, sticky="w", pady=(6, 0))

        actions = ttk.Frame(left); actions.grid(row=4, column=0, sticky="ew", pady=(10, 0)); actions.columnconfigure(0, weight=1)
        self.run_btn = ttk.Button(actions, text="Run batch", style="Accent.TButton", command=self._run)
        self.run_btn.grid(row=0, column=0, sticky="ew")
        ttk.Button(actions, text="Preview run", command=lambda: self._run(dry_run=True)).grid(row=0, column=1, padx=(6, 0))
        self.stop_btn = ttk.Button(actions, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.grid(row=0, column=2, padx=(6, 0))

        self._build_preview(right)
        self._build_status(right)

    def _path_row(self, parent, row: int, label: str, variable, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(parent, text="Browse…", command=command).grid(row=row, column=2, padx=4, pady=4)
        return entry

    def _build_advanced(self, parent) -> None:
        parent.columnconfigure(1, weight=1)
        self._path_row(parent, 0, "GGUF", self.gguf_var, self._pick_gguf)
        self._path_row(parent, 1, "mmproj", self.mmproj_var, self._pick_mmproj)
        ttk.Checkbutton(parent, text="Override trigger", variable=self.trigger_enabled, command=self._sync_trigger).grid(row=2, column=0, sticky="w", padx=4, pady=4)
        self.trigger_entry = ttk.Entry(parent, textvariable=self.trigger_var)
        self.trigger_entry.grid(row=2, column=1, columnspan=2, sticky="ew", padx=4, pady=4)
        self._sync_trigger()
        ttk.Label(parent, text="Resize MP").grid(row=3, column=0, sticky="w", padx=4, pady=4)
        ttk.Spinbox(parent, from_=0, to=100, increment=.1, textvariable=self.size_var, width=10).grid(row=3, column=1, sticky="w", padx=4)
        ttk.Label(parent, text="Seed").grid(row=4, column=0, sticky="w", padx=4, pady=4)
        ttk.Spinbox(parent, from_=0, to=2147483647, textvariable=self.seed_var, width=12).grid(row=4, column=1, sticky="w", padx=4)
        ttk.Label(parent, text="Limit").grid(row=5, column=0, sticky="w", padx=4, pady=4)
        ttk.Spinbox(parent, from_=1, to=1000000, textvariable=self.limit_var, width=12).grid(row=5, column=1, sticky="w", padx=4)
        ttk.Label(parent, text="Prompt override").grid(row=6, column=0, sticky="nw", padx=4, pady=4)
        self.prompt_text = tk.Text(parent, height=4, wrap="word")
        self._style_text_widget(self.prompt_text)
        self.prompt_text.grid(row=6, column=1, columnspan=2, sticky="ew", padx=4, pady=4)
        Tooltip(self.prompt_text, "Leave blank to use the selected profile's prompt.")

    def _build_preview(self, parent) -> None:
        header = ttk.Frame(parent); header.grid(row=0, column=0, sticky="ew"); header.columnconfigure(1, weight=1)
        ttk.Button(header, text="‹", width=3, command=lambda: self._move_preview(-1)).grid(row=0, column=0)
        ttk.Label(header, textvariable=self.current_var, anchor="center").grid(row=0, column=1, sticky="ew")
        ttk.Button(header, text="›", width=3, command=lambda: self._move_preview(1)).grid(row=0, column=2)
        body = ttk.Panedwindow(parent, orient="vertical"); body.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        image_frame = ttk.LabelFrame(body, text="Image preview", padding=6)
        caption_frame = ttk.LabelFrame(body, text="Caption", padding=6)
        body.add(image_frame, weight=3); body.add(caption_frame, weight=2)
        self.preview_label = ttk.Label(image_frame, text="Select an image folder", anchor="center")
        self.preview_label.pack(fill="both", expand=True)
        # Configure fires continuously while a window or sash is dragged.
        # Coalesce those events instead of decoding/resizing on every pixel.
        self.preview_label.bind("<Configure>", self._schedule_preview_render)
        caption_frame.rowconfigure(0, weight=1); caption_frame.columnconfigure(0, weight=1)
        self.caption_editor = tk.Text(caption_frame, wrap="word", undo=True)
        self._style_text_widget(self.caption_editor)
        self.caption_editor.grid(row=0, column=0, columnspan=3, sticky="nsew")
        ttk.Button(caption_frame, text="Save caption", command=self._save_caption).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Button(caption_frame, text="Caption this image", command=self._run_single).grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))
        ttk.Button(caption_frame, text="Open folder", command=self._open_input).grid(row=1, column=2, sticky="e", pady=(6, 0))

    def _build_status(self, parent) -> None:
        card = ttk.LabelFrame(parent, text="Run status", padding=8)
        card.grid(row=2, column=0, sticky="ew", pady=(8, 0)); card.columnconfigure(0, weight=1)
        ttk.Label(card, textvariable=self.status_var, style="Status.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(card, textvariable=self.time_var, style="Muted.TLabel").grid(row=0, column=1, sticky="e")
        self.progress = ttk.Progressbar(card, mode="determinate")
        self.progress.grid(row=1, column=0, columnspan=2, sticky="ew", pady=6)
        counters = ttk.Frame(card); counters.grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Label(counters, textvariable=self.ok_var, style="Success.TLabel").pack(side="left")
        ttk.Label(counters, text="  ·  ", style="Muted.TLabel").pack(side="left")
        ttk.Label(counters, textvariable=self.skipped_var, style="Muted.TLabel").pack(side="left")
        ttk.Label(counters, text="  ·  ", style="Muted.TLabel").pack(side="left")
        ttk.Label(counters, textvariable=self.failed_var, style="Error.TLabel").pack(side="left")
        self.details = Collapsible(card, "Technical details", open=self.preferences.details_open)
        self.details.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.log = ScrolledText(self.details.content, height=7, wrap="word", state="disabled", font=("Consolas", 9))
        self._style_text_widget(self.log); self.log.pack(fill="both", expand=True)
        bar = ttk.Frame(self.details.content); bar.pack(fill="x", pady=(4, 0))
        ttk.Button(bar, text="Clear", command=self._clear_log).pack(side="left")
        self.report_btn = ttk.Button(bar, text="Open failure report", command=self._open_report, state="disabled")
        self.report_btn.pack(side="right")

    def _build_settings_tab(self) -> None:
        tab = self.settings_tab; tab.columnconfigure(0, weight=1); tab.rowconfigure(2, weight=1)
        profiles = ttk.LabelFrame(tab, text="Profiles", padding=8); profiles.grid(row=0, column=0, sticky="ew"); profiles.columnconfigure(1, weight=1)
        ttk.Label(profiles, text="Selected profile").grid(row=0, column=0, sticky="w", padx=4)
        self.settings_profile_combo = ttk.Combobox(profiles, textvariable=self.profile_var, state="readonly")
        self.settings_profile_combo.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(profiles, text="Edit…", command=self._edit_profile).grid(row=0, column=2, padx=4)
        ttk.Button(profiles, text="New…", command=self._new_profile).grid(row=0, column=3, padx=4)
        ttk.Button(profiles, text="Refresh", command=self._load_profiles).grid(row=0, column=4, padx=4)
        ttk.Label(profiles, textvariable=self.profile_summary_var, justify="left", style="Muted.TLabel", wraplength=self.px(850)).grid(row=1, column=0, columnspan=5, sticky="w", padx=4, pady=(8, 0))

        appearance = ttk.LabelFrame(tab, text="Appearance & behavior", padding=8); appearance.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(appearance, text="Theme").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Combobox(appearance, textvariable=self.theme_var, values=("System", "Light", "Dark"), state="readonly", width=12).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Button(appearance, text="Apply", command=self._apply_theme_choice).grid(row=0, column=2, padx=4)
        ttk.Checkbutton(appearance, text="Verbose diagnostic logging", variable=self.verbose_var).grid(row=0, column=3, padx=18)
        ttk.Label(appearance, text="Config file").grid(row=1, column=0, sticky="w", padx=4, pady=(8, 0))
        ttk.Entry(appearance, textvariable=self.config_var).grid(row=1, column=1, columnspan=2, sticky="ew", padx=4, pady=(8, 0))
        ttk.Button(appearance, text="Browse…", command=self._pick_config).grid(row=1, column=3, sticky="w", padx=4, pady=(8, 0))
        appearance.columnconfigure(2, weight=1)
        ttk.Label(appearance, text="Folders, profile, theme, window size, and panel states are remembered. Overwrite and prompt overrides are intentionally not saved.", style="Muted.TLabel", wraplength=self.px(850)).grid(row=2, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 0))

        help_box = ttk.LabelFrame(tab, text="Keyboard & workflow", padding=8); help_box.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        ttk.Label(help_box, justify="left", text="Ctrl+O  Choose image folder\nCtrl+Enter  Run batch\nCtrl+Shift+Enter  Preview run\nCtrl+S  Save displayed caption\nLeft / Right  Previous / next image\nEscape  Stop an active run\n\nTip: drag a folder anywhere onto the Caption tab.").pack(anchor="nw")

    def _install_traces(self) -> None:
        for var in (self.input_var, self.output_var, self.recursive_var, self.overwrite_var):
            var.trace_add("write", lambda *_: self._refresh_dataset())
        for var in (self.size_var, self.seed_var, self.limit_var, self.gguf_var, self.mmproj_var):
            var.trace_add("write", lambda *_: self._validate_form())
        self.config_var.trace_add("write", lambda *_: self._refresh_readiness())
        self.profile_var.trace_add("write", lambda *_: (self._refresh_profile_summary(), self._refresh_readiness()))

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-o>", lambda _e: self._pick_input())
        self.root.bind("<Control-Return>", lambda _e: self._run())
        self.root.bind("<Control-Shift-Return>", lambda _e: self._run(dry_run=True))
        self.root.bind("<Control-s>", lambda _e: self._save_caption())
        self.root.bind("<Left>", lambda _e: self._move_preview(-1))
        self.root.bind("<Right>", lambda _e: self._move_preview(1))
        self.root.bind("<Escape>", lambda _e: self._stop())

    def _setup_dnd(self) -> None:
        if DND_FILES is None or not hasattr(self.root, "drop_target_register"):
            return
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", self._on_drop)

    def _on_drop(self, event) -> None:
        paths = parse_dnd_paths(self.root, event.data)
        if paths:
            path = Path(paths[0]); self.input_var.set(str(path.parent if path.is_file() else path))

    def _pick_input(self) -> None:
        value = filedialog.askdirectory(title="Choose image folder", initialdir=self.input_var.get() or None)
        if value: self.input_var.set(value)

    def _pick_output(self) -> None:
        value = filedialog.askdirectory(title="Choose caption folder", initialdir=self.output_var.get() or None)
        if value: self.output_var.set(value)

    def _pick_gguf(self) -> None:
        value = filedialog.askopenfilename(title="Choose main GGUF", filetypes=[("GGUF", "*.gguf")])
        if value:
            self.gguf_var.set(value)
            matches = [p for p in Path(value).parent.glob("*.gguf") if "mmproj" in p.name.lower()]
            if len(matches) == 1 and not self.mmproj_var.get(): self.mmproj_var.set(str(matches[0]))

    def _pick_mmproj(self) -> None:
        value = filedialog.askopenfilename(title="Choose mmproj GGUF", filetypes=[("GGUF", "*.gguf")])
        if value: self.mmproj_var.set(value)

    def _pick_config(self) -> None:
        value = filedialog.askopenfilename(
            title="Choose config.toml",
            filetypes=[("TOML", "*.toml"), ("All files", "*.*")],
        )
        if value:
            self.config_var.set(value)
            self._refresh_readiness()

    def _sync_trigger(self) -> None:
        if hasattr(self, "trigger_entry"):
            self.trigger_entry.configure(state="normal" if self.trigger_enabled.get() else "disabled")

    def _load_profiles(self, select: str | None = None) -> None:
        try: names = list_profile_names(load_config())
        except Exception as exc:
            names = []; self._append_line(f"Could not load profiles: {exc}")
        self.profile_combo["values"] = names; self.settings_profile_combo["values"] = names
        wanted = select or self.profile_var.get()
        if wanted in names: self.profile_var.set(wanted)
        elif names: self.profile_var.set(names[0])
        self._refresh_profile_summary()

    def _refresh_profile_summary(self) -> None:
        name = self.profile_var.get().strip()
        try:
            with (PROJECT_ROOT / "profiles" / f"{name}.toml").open("rb") as file:
                data = tomllib.load(file)
            model, meta, trigger, generation = data.get("model", {}), data.get("profile", {}), data.get("trigger_word", {}), data.get("generation", {})
            gguf = model.get("gguf_path") or model.get("gguf_file") or "not configured"
            mmproj = model.get("mmproj_path") or model.get("mmproj_file") or "not configured"
            self.profile_summary_var.set(f"{meta.get('description', 'No description')}\nModel: {gguf}\nVision: {mmproj}\nTrigger: {trigger.get('value') or 'disabled'}  ·  Context: {generation.get('ctx_size', 'default')}  ·  Max tokens: {generation.get('n_predict', 'default')}")
        except (OSError, ValueError): self.profile_summary_var.set("Select a profile to see its configuration.")

    def _edit_profile(self) -> None:
        if self.profile_var.get(): ProfileEditor(self.root, self, existing_name=self.profile_var.get())

    def _new_profile(self) -> None:
        ProfileEditor(self.root, self, template_name=self.profile_var.get() or None)

    def _refresh_dataset(self) -> None:
        self._refresh_generation += 1; generation = self._refresh_generation
        folder = Path(self.input_var.get()) if self.input_var.get().strip() else None
        if folder is None or not folder.is_dir():
            self.images = (); self.count_var.set("Choose an existing image folder."); self.warning_var.set(""); self._show_image(); return
        recursive = self.recursive_var.get()
        def work():
            try:
                output = self._effective_output_dir()
                summary = summarize_dataset(folder, recursive, output_dir=output)
            except OSError: return
            def finish():
                if generation != self._refresh_generation: return
                self.images = summary.images; self.image_index = min(self.image_index, max(0, len(self.images) - 1))
                self.count_var.set(f"{len(summary.images)} images  ·  {summary.pending} need captions  ·  {summary.existing} already captioned")
                self.warning_var.set(f"Warning: running now will replace {summary.existing} existing caption{'s' if summary.existing != 1 else ''}." if self.overwrite_var.get() and summary.existing else "")
                self._show_image(); self._refresh_readiness()
            self.root.after(0, finish)
        threading.Thread(target=work, daemon=True).start()

    def _move_preview(self, delta: int) -> None:
        if self.images:
            self.image_index = (self.image_index + delta) % len(self.images); self._show_image()

    def _show_image(self) -> None:
        if not self.images:
            self._preview_generation += 1
            self._preview_path = None
            self._preview_source = None
            self._preview_size = None
            self.preview_photo = None
            if self._preview_after is not None:
                self.root.after_cancel(self._preview_after)
                self._preview_after = None
            self.current_var.set("No image selected")
            if hasattr(self, "preview_label"): self.preview_label.configure(image="", text="No supported images found")
            if hasattr(self, "caption_editor"): self.caption_editor.delete("1.0", "end")
            return
        path = self.images[self.image_index]
        self.current_var.set(f"{self.image_index + 1} / {len(self.images)}  ·  {path.name}")
        self._load_preview(path)
        sidecar = sidecar_path_for(path, Path(self.input_var.get()), self._effective_output_dir(), ".txt")
        try: text = sidecar.read_text(encoding="utf-8").rstrip()
        except OSError: text = ""
        self.caption_editor.delete("1.0", "end"); self.caption_editor.insert("1.0", text)

    def _load_preview(self, path: Path) -> None:
        """Decode once in the background; resizing then uses the cached pixels."""
        if path == self._preview_path and self._preview_source is not None:
            self._schedule_preview_render()
            return
        self._preview_generation += 1
        generation = self._preview_generation
        self._preview_path = path
        self._preview_source = None
        self._preview_size = None
        self.preview_label.configure(image="", text="Loading preview…")

        def decode() -> None:
            try:
                with Image.open(path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    image.load()
                # The preview never needs full camera resolution. Build a
                # high-quality display master off-thread to bound subsequent
                # UI-thread copies and resizes even for 50–100 MP originals.
                image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                error = None
            except (OSError, ValueError) as exc:
                image, error = None, str(exc)

            def finish() -> None:
                if generation != self._preview_generation or path != self._preview_path:
                    return
                if image is None:
                    self.preview_label.configure(image="", text=f"Preview unavailable\n{error}")
                    return
                self._preview_source = image
                self._schedule_preview_render(immediate=True)

            self.root.after(0, finish)

        threading.Thread(target=decode, daemon=True).start()

    def _schedule_preview_render(self, _event=None, *, immediate: bool = False) -> None:
        """Render once after a burst of geometry changes, keeping sash drags fluid."""
        if self._preview_after is not None:
            self.root.after_cancel(self._preview_after)
        delay = 0 if immediate else 75
        self._preview_after = self.root.after(delay, self._render_preview)

    def _render_preview(self) -> None:
        self._preview_after = None
        if self._preview_source is None or not hasattr(self, "preview_label"):
            return
        target = (
            max(100, self.preview_label.winfo_width() - 16),
            max(100, self.preview_label.winfo_height() - 16),
        )
        if target == self._preview_size:
            return
        # Copying an already-decoded RGB image is substantially cheaper than
        # reopening and EXIF-normalizing the original on every resize.
        image = self._preview_source.copy()
        image.thumbnail(target, Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self._preview_size = target
        self.preview_label.configure(image=self.preview_photo, text="")

    def _save_caption(self) -> None:
        if not self.images: return
        path = self.images[self.image_index]
        sidecar = sidecar_path_for(path, Path(self.input_var.get()), self._effective_output_dir(), ".txt")
        try: write_caption(sidecar, self.caption_editor.get("1.0", "end").strip())
        except OSError as exc: messagebox.showerror("llamacap", f"Could not save caption:\n{exc}"); return
        self.status_var.set(f"Saved {sidecar.name}"); self._refresh_dataset()

    def _validate_form(self) -> bool:
        errors = []
        if not self.profile_var.get(): errors.append("Choose a profile.")
        folder = Path(self.input_var.get()) if self.input_var.get().strip() else None
        if folder is None or not folder.is_dir(): errors.append("Choose an existing image folder.")
        for value, label, integer, zero in ((self.size_var.get(), "Resize", False, True), (self.seed_var.get(), "Seed", True, True), (self.limit_var.get(), "Limit", True, False)):
            error = validate_number(value, label, integer=integer, allow_zero=zero)
            if error: errors.append(error)
        gguf, mmproj = self.gguf_var.get().strip(), self.mmproj_var.get().strip()
        if bool(gguf) != bool(mmproj): errors.append("Choose both GGUF and mmproj files, or neither.")
        if gguf and not Path(gguf).is_file(): errors.append("The selected GGUF file does not exist.")
        if mmproj and not Path(mmproj).is_file(): errors.append("The selected mmproj file does not exist.")
        self.validation_var.set("\n".join(f"• {error}" for error in errors))
        return not errors

    def _refresh_readiness(self) -> None:
        checks = []
        try:
            config = load_config(Path(self.config_var.get()) if self.config_var.get() else None) if self.config_var.get() else load_config()
            binary = resolve_llama_server(config); checks.append(f"✓ llama-server found: {binary.name}")
        except Exception as exc: checks.append(f"✗ llama-server: {str(exc).splitlines()[0]}")
        if self.gguf_var.get() and self.mmproj_var.get():
            checks.append("✓ exact model pair selected" if Path(self.gguf_var.get()).is_file() and Path(self.mmproj_var.get()).is_file() else "✗ selected model pair is incomplete")
        else:
            name = self.profile_var.get(); path = PROJECT_ROOT / "profiles" / f"{name}.toml"
            if not path.is_file():
                checks.append("✗ choose a profile")
            else:
                try:
                    with path.open("rb") as file:
                        model = tomllib.load(file).get("model", {})
                    active_config = config if "config" in locals() else load_config()
                    models_dir = PROJECT_ROOT / active_config.models.default_dir
                    gguf = Path(model.get("gguf_path")) if model.get("gguf_path") else models_dir / model.get("gguf_file", "")
                    mmproj = Path(model.get("mmproj_path")) if model.get("mmproj_path") else models_dir / model.get("mmproj_file", "")
                    if gguf.is_file() and mmproj.is_file(): checks.append("✓ profile model pair found")
                    else: checks.append("✗ profile model files are missing; select an override or install them")
                except (OSError, ValueError, TypeError): checks.append("✗ profile model configuration is invalid")
        checks.append(f"{'✓' if self.images else '✗'} {len(self.images)} supported images found")
        output = Path(self.output_var.get()) if self.output_var.get().strip() else Path(self.input_var.get() or PROJECT_ROOT)
        probe = output if output.exists() else output.parent
        checks.append("✓ output location writable" if probe.exists() and os.access(probe, os.W_OK) else "✗ output location is not writable")
        backend = "GPU/backend is reported by llama-server when the run starts"
        checks.append(f"ℹ {backend}")
        self.readiness_var.set("\n".join(checks)); self._validate_form()

    def _effective_output_dir(self) -> Path | None:
        if self.output_var.get().strip():
            return Path(self.output_var.get().strip())
        try:
            config = load_config(Path(self.config_var.get())) if self.config_var.get().strip() else load_config()
            if config.output.default_mode == "output_dir":
                return PROJECT_ROOT / config.output.default_dir
        except Exception:
            pass
        return None

    def _build_command(self, *, dry_run=False, single=False) -> list[str] | None:
        if not self._validate_form(): return None
        cmd = [sys.executable, "-m", "llamacap.cli", "--profile", self.profile_var.get(), "--input", self.input_var.get(), "--progress-json"]
        if self.output_var.get().strip(): cmd += ["--output-dir", self.output_var.get().strip()]
        if self.gguf_var.get().strip(): cmd += ["--model-gguf", self.gguf_var.get().strip(), "--model-mmproj", self.mmproj_var.get().strip()]
        if self.trigger_enabled.get(): cmd += ["--trigger", self.trigger_var.get()]
        for flag, value in (("--size", self.size_var.get()), ("--seed", self.seed_var.get()), ("--limit", self.limit_var.get())):
            if value.strip(): cmd += [flag, value.strip()]
        prompt = self.prompt_text.get("1.0", "end").strip()
        if prompt: cmd += ["--prompt", prompt]
        cmd.append("--recursive" if self.recursive_var.get() else "--no-recursive")
        cmd.append("--overwrite" if self.overwrite_var.get() or single else "--no-overwrite")
        if self.config_var.get().strip(): cmd += ["--config", self.config_var.get().strip()]
        if self.verbose_var.get(): cmd.append("--verbose")
        if dry_run: cmd.append("--dry-run")
        if single and self.images: cmd += ["--image", str(self.images[self.image_index])]
        return cmd

    def _run_single(self) -> None:
        if not self.images: return
        self._run(single=True)

    def _run(self, dry_run=False, single=False) -> None:
        if self.proc is not None: return
        if self.overwrite_var.get() and not dry_run and not single and self.warning_var.get():
            if not messagebox.askyesno("Replace captions?", self.warning_var.get() + "\n\nContinue?"): return
        cmd = self._build_command(dry_run=dry_run, single=single)
        if cmd is None: return
        self._clear_log(); self._append_line("$ " + subprocess.list2cmdline(cmd)); self._report_path = None
        self.report_btn.configure(state="disabled")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        try:
            self.proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0)
        except OSError as exc: messagebox.showerror("llamacap", f"Could not start:\n{exc}"); self.proc = None; return
        self._started_at = time.monotonic(); self._last_done = self._last_total = 0
        self.ok_var.set("0 complete"); self.skipped_var.set("0 skipped"); self.failed_var.set("0 failed")
        self.run_btn.configure(state="disabled"); self.stop_btn.configure(state="normal")
        self.progress.configure(mode="indeterminate"); self.progress.start(12)
        self.status_var.set("Loading model…"); self.time_var.set("Elapsed 0:00")
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start(); self._tick_clock()

    def _reader(self, proc) -> None:
        assert proc.stdout is not None
        for line in proc.stdout: self.out_queue.put(("line", line.rstrip("\r\n")))
        self.out_queue.put(("done", str(proc.wait())))

    def _tick_clock(self) -> None:
        if self.proc is None: return
        elapsed = time.monotonic() - self._started_at
        eta = elapsed / self._last_done * (self._last_total - self._last_done) if self._last_done else None
        self.time_var.set(f"Elapsed {format_duration(elapsed)}" + (f"  ·  ETA {format_duration(eta)}" if eta is not None else ""))
        self.root.after(500, self._tick_clock)

    def _stop(self) -> None:
        if self.proc is None: return
        self.status_var.set("Stopping…"); self.stop_btn.configure(state="disabled")
        pid = self.proc.pid
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/T", "/PID", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self._stop_escalation = self.root.after(3000, lambda: self._force_kill(pid))
            else: self.proc.terminate()
        except OSError as exc: self._append_line(f"Stop failed: {exc}")

    def _force_kill(self, pid: int) -> None:
        if self.proc is None or self.proc.pid != pid: return
        if sys.platform == "win32": subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else: self.proc.kill()

    def _drain_output(self) -> None:
        try:
            while True:
                kind, text = self.out_queue.get_nowait()
                if kind == "done": self._on_done(int(text)); continue
                if text.startswith(PROGRESS_SENTINEL): self._on_progress(text[len(PROGRESS_SENTINEL):]); continue
                match = FAILURE_REPORT_RE.search(text)
                if match: self._report_path = match.group(1).strip(); self.report_btn.configure(state="normal")
                self._append_line(text)
        except queue.Empty: pass
        self.root.after(60, self._drain_output)

    def _on_progress(self, payload: str) -> None:
        try: data = json.loads(payload)
        except ValueError: return
        event = data.get("event")
        if event == "phase":
            self.status_var.set("Loading model…" if data.get("phase") == "loading" else str(data.get("phase", "Working…")))
        elif event == "start":
            self.progress.stop(); self.progress.configure(mode="determinate", maximum=max(1, data.get("total", 1))); self.status_var.set("Captioning images…")
        elif event == "image":
            self._last_done, self._last_total = data.get("done", 0), data.get("total", 0); self.progress["value"] = self._last_done
            self.status_var.set(f"Captioning {data.get('current', '')}")
            self.ok_var.set(f"{data.get('ok', 0)} complete")
            self.skipped_var.set(f"{data.get('skip', 0)} skipped")
            self.failed_var.set(f"{data.get('fail', 0)} failed")
        elif event == "end": self.status_var.set("Finishing…")

    def _on_done(self, returncode: int) -> None:
        self.progress.stop(); self.proc = None; self.run_btn.configure(state="normal"); self.stop_btn.configure(state="disabled")
        self.status_var.set("Complete" if returncode == 0 else f"Finished with errors (code {returncode})")
        self._refresh_dataset(); self._show_image()

    def _append_line(self, text: str) -> None:
        if not hasattr(self, "log"): return
        self.log.configure(state="normal"); self.log.insert("end", text + "\n"); self.log.see("end"); self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal"); self.log.delete("1.0", "end"); self.log.configure(state="disabled")

    def _open_input(self) -> None:
        path = self.images[self.image_index].parent if self.images else Path(self.input_var.get())
        if path.is_dir(): os.startfile(path) if sys.platform == "win32" else subprocess.Popen(["xdg-open", str(path)])

    def _open_report(self) -> None:
        if self._report_path and Path(self._report_path).is_file(): os.startfile(self._report_path)

    def _apply_theme_choice(self) -> None:
        self.theme = self._resolved_theme(); self.palette = apply_theme(self.root, self.theme)
        for widget in (self.prompt_text, self.caption_editor, self.log): self._style_text_widget(widget)

    def close(self) -> None:
        if self.proc is not None:
            if not messagebox.askyesno("Quit llamacap?", "A captioning run is active. Stop it and quit?"): return
            self._stop()
        prefs = GuiPreferences(
            input_dir=self.input_var.get(), output_dir=self.output_var.get(), profile=self.profile_var.get(),
            recursive=self.recursive_var.get(), theme=self.theme_var.get(), geometry=self.root.geometry(),
            advanced_open=self.advanced.is_open, details_open=self.details.is_open,
        )
        try: prefs.save()
        except OSError: pass
        self.root.destroy()


def main() -> int:
    enable_dpi_awareness()
    root = TkinterDnD.Tk() if TkinterDnD is not None else tk.Tk()
    app = LlamacapGUI(root, apply_scaling(root))
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
