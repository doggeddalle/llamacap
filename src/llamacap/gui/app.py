"""Tkinter front-end for llamacap.

Launch with:  uv run scripts/gui.py   (or llamacap-gui.bat)

Drives the llamacap CLI as a subprocess so the GUI stays fully decoupled from
the captioning internals and mirrors the documented CLI exactly. Anything left
blank is simply not passed, so the profile / config defaults win.

The subprocess runs with --progress-json; sentinel lines feed the progress bar
and are hidden from the log pane.

DPI handling: the app declares itself DPI-aware and rescales Tk so it renders
crisply at 100% / 125% / 150% / 200% Windows display scaling.
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

try:
    import darkdetect
except ImportError:  # pragma: no cover
    darkdetect = None

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:  # optional: GUI still works without drag-and-drop
    DND_FILES = None
    TkinterDnD = None

from llamacap.config import PROJECT_ROOT
from llamacap.gui.profile_editor import ProfileEditor
from llamacap.gui.theme import apply_theme
from llamacap.gui.widgets import Tooltip, parse_dnd_paths
from llamacap.image_utils import list_images

PROGRESS_SENTINEL = "@@LLAMACAP@@ "
FAILURE_REPORT_RE = re.compile(r"Failure details written to (.+)")


# --------------------------------------------------------------------------- #
# DPI / scaling
# --------------------------------------------------------------------------- #
def enable_dpi_awareness() -> None:
    """Tell Windows we handle DPI ourselves so it won't bitmap-scale (blur) us."""
    if sys.platform != "win32":
        return
    try:
        from ctypes import windll

        # Per-Monitor-Aware v2 (Win10 1703+); fall back to older APIs.
        try:
            windll.user32.SetProcessDpiAwarenessContext(-4)  # PMv2
        except Exception:
            try:
                windll.shcore.SetProcessDpiAwareness(2)  # per-monitor
            except Exception:
                windll.user32.SetProcessDPIAware()  # system aware
    except Exception:
        pass


def apply_scaling(root: tk.Tk) -> float:
    """Sync Tk's point->pixel scaling to the real monitor DPI.

    Returns a UI scale factor relative to 100% DPI (1.0 at 96 dpi) that we use
    to size paddings/geometry in device pixels.
    """
    dpi = root.winfo_fpixels("1i")  # device pixels per inch
    root.tk.call("tk", "scaling", dpi / 72.0)
    return dpi / 96.0


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
class LlamacapGUI:
    def __init__(self, root: tk.Tk, scale: float) -> None:
        self.root = root
        self.scale = scale
        self.proc: subprocess.Popen | None = None
        self.out_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._stop_escalation: str | None = None
        self._count_generation = 0
        self._report_path: str | None = None

        root.title("llamacap")
        root.minsize(self.px(700), self.px(680))

        self.theme = self._detect_theme()
        self.palette = apply_theme(root, self.theme)

        # Form variables --------------------------------------------------- #
        self.profile_var = tk.StringVar()
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.gguf_var = tk.StringVar()
        self.mmproj_var = tk.StringVar()
        self.trigger_enabled = tk.BooleanVar(value=False)
        self.trigger_var = tk.StringVar()
        self.size_var = tk.StringVar()
        self.seed_var = tk.StringVar()
        self.limit_var = tk.StringVar()
        self.config_var = tk.StringVar()
        self.overwrite_var = tk.BooleanVar(value=False)
        self.recursive_var = tk.BooleanVar(value=False)
        self.dryrun_var = tk.BooleanVar(value=False)
        self.verbose_var = tk.BooleanVar(value=False)
        self.count_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready.")

        self._build_ui()
        self._setup_dnd()
        self._load_profiles()
        self.input_var.trace_add("write", lambda *_: self._refresh_image_count())
        self.recursive_var.trace_add("write", lambda *_: self._refresh_image_count())
        self.root.after(60, self._drain_output)

    # -- small helpers ---------------------------------------------------- #
    def px(self, n: int) -> int:
        """Device pixels for a value authored at 100% DPI."""
        return int(round(n * self.scale))

    def _pad(self) -> dict:
        p = self.px(6)
        return {"padx": p, "pady": p}

    @staticmethod
    def _detect_theme() -> str:
        if darkdetect is not None:
            detected = (darkdetect.theme() or "").lower()
            if detected in ("light", "dark"):
                return detected
        return "light"

    def _style_text_widget(self, widget: tk.Text) -> None:
        widget.configure(
            background=self.palette["text_bg"],
            foreground=self.palette["text_fg"],
            insertbackground=self.palette["text_fg"],
            relief="flat",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=self.palette["muted"],
        )

    # -- layout ----------------------------------------------------------- #
    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=self.px(10))
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)

        self._section_profile(outer)
        self._section_model(outer)
        self._section_overrides(outer)
        self._section_options(outer)
        self._section_actions(outer)
        self._section_log(outer)

        # Only the log row grows when the window is resized.
        outer.rowconfigure(5, weight=1)

    def _labeled_path(self, parent, row, label, var, browse):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", **self._pad())
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", **self._pad())
        ttk.Button(parent, text="Browse…", command=browse).grid(
            row=row, column=2, sticky="w", **self._pad()
        )
        parent.columnconfigure(1, weight=1)
        return entry

    def _section_profile(self, parent):
        f = ttk.LabelFrame(parent, text="Profile & paths", padding=self.px(6))
        f.grid(row=0, column=0, sticky="ew", pady=(0, self.px(8)))
        f.columnconfigure(1, weight=1)

        ttk.Label(f, text="Profile").grid(row=0, column=0, sticky="w", **self._pad())
        row = ttk.Frame(f)
        row.grid(row=0, column=1, columnspan=2, sticky="ew", **self._pad())
        row.columnconfigure(0, weight=1)
        self.profile_combo = ttk.Combobox(
            row, textvariable=self.profile_var, state="readonly"
        )
        self.profile_combo.grid(row=0, column=0, sticky="ew")
        ttk.Button(row, text="Edit…", command=self._edit_profile).grid(
            row=0, column=1, padx=(self.px(6), 0)
        )
        ttk.Button(row, text="New…", command=self._new_profile).grid(
            row=0, column=2, padx=(self.px(6), 0)
        )
        ttk.Button(row, text="Refresh", command=self._load_profiles).grid(
            row=0, column=3, padx=(self.px(6), 0)
        )

        self.input_entry = self._labeled_path(
            f, 1, "Input folder", self.input_var, self._pick_input
        )
        ttk.Button(f, text="Open", command=self._open_input, width=6).grid(
            row=1, column=3, sticky="w", padx=(0, self.px(6))
        )
        ttk.Label(f, textvariable=self.count_var, style="Muted.TLabel").grid(
            row=2, column=1, sticky="w", padx=self.px(6)
        )

        self.output_entry = self._labeled_path(
            f, 3, "Output folder", self.output_var, self._pick_output
        )
        ttk.Label(
            f,
            text="Output folder is optional — leave blank to write .txt sidecars in place.",
            style="Muted.TLabel",
        ).grid(row=4, column=1, columnspan=2, sticky="w", padx=self.px(6))

    def _section_model(self, parent):
        f = ttk.LabelFrame(
            parent, text="Model override (optional)", padding=self.px(6)
        )
        f.grid(row=1, column=0, sticky="ew", pady=(0, self.px(8)))
        f.columnconfigure(1, weight=1)

        self._labeled_path(f, 0, "GGUF model", self.gguf_var, self._pick_gguf)
        mmproj_entry = self._labeled_path(
            f, 1, "mmproj file", self.mmproj_var, self._pick_mmproj
        )
        Tooltip(
            mmproj_entry,
            "The vision projector (*mmproj*.gguf) that pairs with the main model. "
            "Both files must live in the same folder.",
        )
        ttk.Label(
            f,
            text="Overrides the profile's [model] for this run; leave blank to use the profile.",
            style="Muted.TLabel",
        ).grid(row=2, column=1, columnspan=2, sticky="w", padx=self.px(6))

    def _section_overrides(self, parent):
        f = ttk.LabelFrame(
            parent, text="Per-run overrides (optional)", padding=self.px(6)
        )
        f.grid(row=2, column=0, sticky="ew", pady=(0, self.px(8)))
        f.columnconfigure(1, weight=1)

        # Trigger word: checkbox gates it so an empty string can be sent on
        # purpose (--trigger "" disables a profile's trigger word).
        trigger_check = ttk.Checkbutton(
            f,
            text="Override trigger word",
            variable=self.trigger_enabled,
            command=self._sync_trigger_state,
        )
        trigger_check.grid(row=0, column=0, sticky="w", **self._pad())
        Tooltip(
            trigger_check,
            "Checked with an empty field: disables the profile's trigger word for "
            "this run. Checked with text: that text is prefixed to every caption.",
        )
        self.trigger_entry = ttk.Entry(f, textvariable=self.trigger_var)
        self.trigger_entry.grid(row=0, column=1, columnspan=2, sticky="ew", **self._pad())
        self._sync_trigger_state()

        ttk.Label(f, text="Resize (megapixels)").grid(
            row=1, column=0, sticky="w", **self._pad()
        )
        size_entry = ttk.Entry(f, textvariable=self.size_var, width=12)
        size_entry.grid(row=1, column=1, sticky="w", **self._pad())
        Tooltip(
            size_entry,
            "Resize images to ~this many megapixels (aspect ratio kept) before "
            "captioning. 0 disables resizing; blank uses the config default.",
        )

        ttk.Label(f, text="Seed").grid(row=2, column=0, sticky="w", **self._pad())
        ttk.Entry(f, textvariable=self.seed_var, width=12).grid(
            row=2, column=1, sticky="w", **self._pad()
        )

        ttk.Label(f, text="Limit (first N images)").grid(
            row=3, column=0, sticky="w", **self._pad()
        )
        ttk.Entry(f, textvariable=self.limit_var, width=12).grid(
            row=3, column=1, sticky="w", **self._pad()
        )

        ttk.Label(f, text="Prompt").grid(row=4, column=0, sticky="nw", **self._pad())
        self.prompt_text = tk.Text(f, height=4, wrap="word", font=("Segoe UI", 10))
        self._style_text_widget(self.prompt_text)
        self.prompt_text.grid(row=4, column=1, columnspan=2, sticky="ew", **self._pad())
        ttk.Label(
            f,
            text="Blank prompt = use the profile's prompt.",
            style="Muted.TLabel",
        ).grid(row=5, column=1, sticky="w", padx=self.px(6), pady=(0, self.px(4)))

        self._labeled_path(f, 6, "Config file", self.config_var, self._pick_config)

    def _section_options(self, parent):
        f = ttk.LabelFrame(parent, text="Options", padding=self.px(6))
        f.grid(row=3, column=0, sticky="ew", pady=(0, self.px(8)))
        ttk.Checkbutton(
            f, text="Overwrite existing sidecars", variable=self.overwrite_var
        ).grid(row=0, column=0, sticky="w", **self._pad())
        ttk.Checkbutton(
            f, text="Recurse into subfolders", variable=self.recursive_var
        ).grid(row=0, column=1, sticky="w", **self._pad())
        dryrun_check = ttk.Checkbutton(
            f, text="Dry run (report only)", variable=self.dryrun_var
        )
        dryrun_check.grid(row=1, column=0, sticky="w", **self._pad())
        Tooltip(
            dryrun_check,
            "Resolves the model, prompt, and images and reports what would be "
            "captioned/skipped — without starting the server or writing files.",
        )
        ttk.Checkbutton(
            f, text="Verbose logging", variable=self.verbose_var
        ).grid(row=1, column=1, sticky="w", **self._pad())

    def _section_actions(self, parent):
        f = ttk.Frame(parent)
        f.grid(row=4, column=0, sticky="ew", pady=(0, self.px(6)))
        f.columnconfigure(3, weight=1)

        self.run_btn = ttk.Button(
            f, text="Run captioning", style="Accent.TButton", command=self._run
        )
        self.run_btn.grid(row=0, column=0, padx=(0, self.px(6)))
        self.stop_btn = ttk.Button(f, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=(0, self.px(6)))
        ttk.Button(f, text="Clear log", command=self._clear_log).grid(
            row=0, column=2, padx=(0, self.px(6))
        )
        self.report_btn = ttk.Button(
            f, text="Open failure report", command=self._open_report, state="disabled"
        )
        self.report_btn.grid(row=0, column=4, sticky="e")

        self.progress = ttk.Progressbar(f, mode="determinate")
        self.progress.grid(
            row=1, column=0, columnspan=5, sticky="ew", pady=(self.px(8), 0)
        )
        ttk.Label(f, textvariable=self.status_var, style="Muted.TLabel").grid(
            row=2, column=0, columnspan=5, sticky="w", pady=(self.px(4), 0)
        )

    def _section_log(self, parent):
        f = ttk.LabelFrame(parent, text="Output", padding=self.px(6))
        f.grid(row=5, column=0, sticky="nsew")
        f.rowconfigure(0, weight=1)
        f.columnconfigure(0, weight=1)
        self.log = ScrolledText(
            f, height=10, wrap="word", font=("Consolas", 10), state="disabled"
        )
        self._style_text_widget(self.log)
        self.log.grid(row=0, column=0, sticky="nsew")

    # -- drag and drop ---------------------------------------------------- #
    def _setup_dnd(self) -> None:
        if DND_FILES is None or not hasattr(self.root, "drop_target_register"):
            return

        def bind_drop(widget, var):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", lambda e: self._on_drop(e, var))

        # Window-wide drops set the input folder; the two entries are precise.
        bind_drop(self.root, self.input_var)
        bind_drop(self.input_entry, self.input_var)
        bind_drop(self.output_entry, self.output_var)

    def _on_drop(self, event, var: tk.StringVar) -> None:
        paths = parse_dnd_paths(self.root, event.data)
        if not paths:
            return
        p = Path(paths[0])
        if p.is_file():
            p = p.parent
        var.set(str(p))

    # -- profile loading / editing ---------------------------------------- #
    def _load_profiles(self, select: str | None = None) -> None:
        try:
            from llamacap.config import load_config
            from llamacap.profiles import list_profile_names

            names = list_profile_names(load_config())
        except Exception as e:  # config missing / import error -> empty list
            names = []
            self._append_line(f"Could not load profiles: {e}")
        self.profile_combo["values"] = names
        if select and select in names:
            self.profile_var.set(select)
        elif names and self.profile_var.get() not in names:
            self.profile_var.set(names[0])

    def _edit_profile(self):
        name = self.profile_var.get().strip()
        if not name:
            messagebox.showerror("llamacap", "Choose a profile to edit.")
            return
        ProfileEditor(self.root, self, existing_name=name)

    def _new_profile(self):
        template = self.profile_var.get().strip() or None
        ProfileEditor(self.root, self, template_name=template)

    # -- file/dir pickers ------------------------------------------------- #
    def _pick_input(self):
        d = filedialog.askdirectory(title="Select image folder")
        if d:
            self.input_var.set(d)

    def _pick_output(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.output_var.set(d)

    def _pick_gguf(self):
        f = filedialog.askopenfilename(
            title="Select main GGUF model",
            filetypes=[("GGUF model", "*.gguf"), ("All files", "*.*")],
        )
        if f:
            self.gguf_var.set(f)
            # Best-effort: auto-find a sibling *mmproj*.gguf if not set yet.
            if not self.mmproj_var.get():
                for sib in Path(f).parent.glob("*.gguf"):
                    if "mmproj" in sib.name.lower():
                        self.mmproj_var.set(str(sib))
                        break

    def _pick_mmproj(self):
        f = filedialog.askopenfilename(
            title="Select mmproj file",
            filetypes=[("GGUF mmproj", "*.gguf"), ("All files", "*.*")],
        )
        if f:
            self.mmproj_var.set(f)

    def _pick_config(self):
        f = filedialog.askopenfilename(
            title="Select alternate config.toml",
            filetypes=[("TOML", "*.toml"), ("All files", "*.*")],
        )
        if f:
            self.config_var.set(f)

    def _sync_trigger_state(self):
        self.trigger_entry.configure(
            state="normal" if self.trigger_enabled.get() else "disabled"
        )

    # -- image count preview ---------------------------------------------- #
    def _refresh_image_count(self) -> None:
        self._count_generation += 1
        generation = self._count_generation
        folder = self.input_var.get().strip()
        recursive = self.recursive_var.get()

        if not folder or not Path(folder).is_dir():
            self.count_var.set("")
            return

        def count():
            try:
                n = len(list_images(Path(folder), recursive))
                text = f"≈{n} image{'s' if n != 1 else ''} found"
            except OSError:
                text = ""
            if generation == self._count_generation:
                self.root.after(0, lambda: self.count_var.set(text))

        threading.Thread(target=count, daemon=True).start()

    # -- open folder / report --------------------------------------------- #
    def _open_input(self):
        folder = self.input_var.get().strip()
        if folder and Path(folder).is_dir():
            os.startfile(folder)
        else:
            messagebox.showerror("llamacap", "Input folder does not exist.")

    def _open_report(self):
        if self._report_path and Path(self._report_path).is_file():
            os.startfile(self._report_path)

    # -- command assembly ------------------------------------------------- #
    def _resolve_model_dir(self) -> str | None:
        """Validate the gguf/mmproj pair and return the shared folder for --model.

        Returns None if no override was requested. Raises ValueError on a bad
        selection so the caller can surface a message box.
        """
        gguf = self.gguf_var.get().strip()
        mmproj = self.mmproj_var.get().strip()
        if not gguf and not mmproj:
            return None
        if not gguf or not mmproj:
            raise ValueError("Set BOTH the GGUF model and the mmproj file, or neither.")
        gp, mp = Path(gguf), Path(mmproj)
        if not gp.is_file():
            raise ValueError(f"GGUF file not found:\n{gp}")
        if not mp.is_file():
            raise ValueError(f"mmproj file not found:\n{mp}")
        if gp.parent != mp.parent:
            raise ValueError(
                "The GGUF and mmproj files must be in the same folder\n"
                "(--model takes a single directory)."
            )
        return str(gp.parent)

    def _build_command(self) -> list[str] | None:
        profile = self.profile_var.get().strip()
        input_dir = self.input_var.get().strip()
        if not profile:
            messagebox.showerror("llamacap", "Choose a profile.")
            return None
        if not input_dir:
            messagebox.showerror("llamacap", "Choose an input folder.")
            return None
        if not Path(input_dir).is_dir():
            messagebox.showerror("llamacap", f"Input folder does not exist:\n{input_dir}")
            return None

        try:
            model_dir = self._resolve_model_dir()
        except ValueError as e:
            messagebox.showerror("llamacap", str(e))
            return None

        cmd = [
            sys.executable, "-m", "llamacap.cli",
            "--profile", profile,
            "--input", input_dir,
            "--progress-json",
        ]

        out = self.output_var.get().strip()
        if out:
            cmd += ["--output-dir", out]
        if model_dir:
            cmd += ["--model", model_dir]
        if self.trigger_enabled.get():
            cmd += ["--trigger", self.trigger_var.get()]

        size = self.size_var.get().strip()
        if size:
            try:
                float(size)
            except ValueError:
                messagebox.showerror("llamacap", f"Resize megapixels must be a number: {size}")
                return None
            cmd += ["--size", size]

        seed = self.seed_var.get().strip()
        if seed:
            if not _is_int(seed):
                messagebox.showerror("llamacap", f"Seed must be an integer: {seed}")
                return None
            cmd += ["--seed", seed]

        limit = self.limit_var.get().strip()
        if limit:
            if not _is_int(limit):
                messagebox.showerror("llamacap", f"Limit must be an integer: {limit}")
                return None
            cmd += ["--limit", limit]

        prompt = self.prompt_text.get("1.0", "end").strip()
        if prompt:
            cmd += ["--prompt", prompt]

        config = self.config_var.get().strip()
        if config:
            cmd += ["--config", config]

        if self.overwrite_var.get():
            cmd.append("--overwrite")
        if self.recursive_var.get():
            cmd.append("--recursive")
        if self.dryrun_var.get():
            cmd.append("--dry-run")
        if self.verbose_var.get():
            cmd.append("--verbose")
        return cmd

    # -- run / stop ------------------------------------------------------- #
    def _run(self):
        if self.proc is not None:
            return
        cmd = self._build_command()
        if cmd is None:
            return

        self._clear_log()
        self._append_line("$ " + _display_cmd(cmd))
        self._append_line("")
        self._report_path = None
        self.report_btn.configure(state="disabled")

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        # Keep tqdm's progress bar single-line even though stdout isn't a TTY.
        env.setdefault("COLUMNS", "80")

        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )
        except Exception as e:
            messagebox.showerror("llamacap", f"Failed to launch:\n{e}")
            self.proc = None
            return

        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.status_var.set("Starting llama-server / loading model…")
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc: subprocess.Popen):
        """Read child output char-by-char so '\\r' progress updates stream live."""
        buf: list[str] = []
        assert proc.stdout is not None
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            if ch == "\n":
                self.out_queue.put(("line", "".join(buf)))
                buf.clear()
            elif ch == "\r":
                self.out_queue.put(("cr", "".join(buf)))
                buf.clear()
            else:
                buf.append(ch)
        if buf:
            self.out_queue.put(("line", "".join(buf)))
        rc = proc.wait()
        self.out_queue.put(("done", str(rc)))

    def _stop(self):
        if self.proc is None:
            return
        self._append_line("\n[stopping…]")
        self.stop_btn.configure(state="disabled")
        pid = self.proc.pid
        try:
            if sys.platform == "win32":
                # Polite first: no /F sends a close request to the tree.
                subprocess.run(
                    ["taskkill", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                # Escalate to force-kill if it's still alive in 3 seconds.
                self._stop_escalation = self.root.after(
                    3000, lambda: self._force_kill(pid)
                )
            else:
                self.proc.terminate()
        except Exception as e:
            self._append_line(f"[stop failed: {e}]")

    def _force_kill(self, pid: int):
        self._stop_escalation = None
        if self.proc is None or self.proc.pid != pid:
            return  # already exited
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception as e:
            self._append_line(f"[force stop failed: {e}]")

    # -- output pump ------------------------------------------------------ #
    def _drain_output(self):
        try:
            while True:
                kind, text = self.out_queue.get_nowait()
                if kind == "line":
                    if text.startswith(PROGRESS_SENTINEL):
                        self._on_progress(text[len(PROGRESS_SENTINEL):])
                        continue
                    match = FAILURE_REPORT_RE.search(text)
                    if match:
                        self._report_path = match.group(1).strip()
                        self.report_btn.configure(state="normal")
                    self._write_current(text, terminate=True)
                elif kind == "cr":
                    self._write_current(text, terminate=False)
                elif kind == "done":
                    self._on_done(int(text))
        except queue.Empty:
            pass
        self.root.after(60, self._drain_output)

    def _on_progress(self, payload: str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return
        event = data.get("event")
        if event == "start":
            self.progress.stop()
            self.progress.configure(mode="determinate", maximum=max(data.get("total", 1), 1))
            self.progress["value"] = 0
            self.status_var.set(f"0/{data.get('total', '?')}")
        elif event == "image":
            self.progress["value"] = data.get("done", 0)
            self.status_var.set(
                f"{data.get('done', 0)}/{data.get('total', '?')}  —  "
                f"{data.get('ok', 0)} ok · {data.get('skip', 0)} skipped · "
                f"{data.get('fail', 0)} failed   ({data.get('current', '')})"
            )
        elif event == "end":
            self.status_var.set(
                f"Finished: {data.get('ok', 0)} ok · {data.get('skip', 0)} skipped · "
                f"{data.get('fail', 0)} failed"
            )

    def _on_done(self, rc: int):
        self.proc = None
        if self._stop_escalation is not None:
            self.root.after_cancel(self._stop_escalation)
            self._stop_escalation = None
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self._append_line("")
        if rc == 0:
            self._append_line("[finished: success]")
            if self.status_var.get().startswith("Starting"):
                self.status_var.set("Done.")
        else:
            self._append_line(f"[finished: exit code {rc}]")
            self.status_var.set(f"Exited with code {rc} — see log.")

    # -- log widget helpers ---------------------------------------------- #
    def _write_current(self, text: str, terminate: bool):
        """Set the last (live) log line to `text`; terminate starts a new line.

        Non-terminated writes come from '\\r' progress updates and overwrite the
        current line in place.
        """
        self.log.configure(state="normal")
        self.log.delete("end-1c linestart", "end-1c")
        self.log.insert("end-1c", text + ("\n" if terminate else ""))
        self.log.see("end")
        self.log.configure(state="disabled")

    def _append_line(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end-1c", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def _display_cmd(cmd: list[str]) -> str:
    out = []
    for part in cmd:
        out.append(f'"{part}"' if " " in part else part)
    return " ".join(out)


def main() -> int:
    enable_dpi_awareness()
    if TkinterDnD is not None:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    scale = apply_scaling(root)
    app = LlamacapGUI(root, scale)

    def on_close():
        if app.proc is not None:
            app._stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
