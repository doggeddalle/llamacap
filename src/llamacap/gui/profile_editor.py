"""Profile editor dialog: create and modify profiles/*.toml from the GUI.

Reads profiles as raw TOML (not via load_profile) so a profile whose model
files are missing can still be opened and fixed. Writes with tomli-w.
"""
from __future__ import annotations

import re
import tkinter as tk
import tomllib
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import tomli_w

from llamacap.config import PROJECT_ROOT, load_config
from llamacap.profiles import PROFILES_DIR_NAME, GenerationParams

PLACEMENTS = ("prefix_comma", "prefix_period", "none")
NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")

GEN_FIELDS: list[tuple[str, str, type]] = [
    ("ctx_size", "Context size", int),
    ("n_predict", "Max caption tokens", int),
    ("temperature", "Temperature", float),
    ("top_p", "Top-p", float),
    ("top_k", "Top-k", int),
    ("repeat_penalty", "Repeat penalty", float),
    ("ngl", "GPU layers (ngl)", int),
    ("image_min_tokens", "Image min tokens", int),
    ("seed", "Seed", int),
]


def _profiles_dir() -> Path:
    return PROJECT_ROOT / PROFILES_DIR_NAME


def _models_dir() -> Path:
    try:
        return PROJECT_ROOT / load_config().models.default_dir
    except Exception:
        return PROJECT_ROOT / "models"


def _split_model_value(value: str) -> tuple[str, str]:
    """Map a user-visible model value to (`*_path`, `*_file`) TOML fields.

    Absolute paths directly inside the models dir are stored portably as bare
    filenames; other absolute paths stay absolute; bare names are filenames.
    """
    value = value.strip()
    if not value:
        return "", ""
    p = Path(value)
    if p.is_absolute():
        try:
            rel = p.relative_to(_models_dir())
            if len(rel.parts) == 1:
                return "", rel.name
        except ValueError:
            pass
        return str(p), ""
    return "", value


class ProfileEditor(tk.Toplevel):
    def __init__(
        self,
        master: tk.Misc,
        app,
        existing_name: str | None = None,
        template_name: str | None = None,
    ):
        super().__init__(master)
        self.app = app
        self.px = app.px
        self.existing_name = existing_name

        source = existing_name or template_name
        self.title(f"Edit profile: {existing_name}" if existing_name else "New profile")
        self.transient(master)
        self.grab_set()
        self.minsize(self.px(560), self.px(640))

        data = self._load_raw(source) if source else {}

        profile_meta = data.get("profile", {})
        model = data.get("model", {})
        prompt = data.get("prompt", {})
        trigger = data.get("trigger_word", {})
        generation = data.get("generation", {})
        output = data.get("output", {})
        gen_defaults = GenerationParams()

        # Vars ------------------------------------------------------------- #
        self.name_var = tk.StringVar(value=existing_name or "")
        self.desc_var = tk.StringVar(value=profile_meta.get("description", ""))
        self.gguf_var = tk.StringVar(
            value=model.get("gguf_path", "") or model.get("gguf_file", "")
        )
        self.mmproj_var = tk.StringVar(
            value=model.get("mmproj_path", "") or model.get("mmproj_file", "")
        )
        initial_mode = "text" if prompt.get("text", "") else "file"
        self.prompt_mode = tk.StringVar(value=initial_mode)
        self.prompt_file_var = tk.StringVar(value=prompt.get("file", ""))
        self.trigger_var = tk.StringVar(value=trigger.get("value", ""))
        self.placement_var = tk.StringVar(
            value=trigger.get("placement", "prefix_comma")
        )
        self.gen_vars: dict[str, tk.StringVar] = {
            key: tk.StringVar(value=str(generation.get(key, getattr(gen_defaults, key))))
            for key, _, _ in GEN_FIELDS
        }
        self.no_warmup_var = tk.BooleanVar(
            value=bool(generation.get("no_warmup", gen_defaults.no_warmup))
        )
        self.extra_args_var = tk.StringVar(
            value=" ".join(generation.get("extra_args", []))
        )
        self.suffix_var = tk.StringVar(value=output.get("suffix", ".txt"))

        self._build_ui()

        # Populate the prompt editor with inline text or the file's contents.
        if initial_mode == "text":
            self.prompt_text.insert("1.0", prompt.get("text", ""))
        else:
            self._load_prompt_file()

    # -- data ------------------------------------------------------------- #
    def _load_raw(self, name: str) -> dict:
        path = _profiles_dir() / f"{name}.toml"
        try:
            with path.open("rb") as f:
                return tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            messagebox.showerror("llamacap", f"Could not read {path.name}:\n{e}", parent=self)
            return {}

    # -- layout ----------------------------------------------------------- #
    def _build_ui(self) -> None:
        pad = {"padx": self.px(6), "pady": self.px(4)}
        outer = ttk.Frame(self, padding=self.px(10))
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)

        r = 0
        ttk.Label(outer, text="Name").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(outer, textvariable=self.name_var).grid(
            row=r, column=1, columnspan=2, sticky="ew", **pad
        )
        r += 1
        ttk.Label(outer, text="Description").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(outer, textvariable=self.desc_var).grid(
            row=r, column=1, columnspan=2, sticky="ew", **pad
        )

        # Model ------------------------------------------------------------ #
        r += 1
        mf = ttk.LabelFrame(outer, text="Model", padding=self.px(6))
        mf.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        mf.columnconfigure(1, weight=1)
        ttk.Label(mf, text="GGUF").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(mf, textvariable=self.gguf_var).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(mf, text="Browse…", command=lambda: self._pick_model(self.gguf_var)).grid(
            row=0, column=2, **pad
        )
        ttk.Label(mf, text="mmproj").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(mf, textvariable=self.mmproj_var).grid(row=1, column=1, sticky="ew", **pad)
        ttk.Button(mf, text="Browse…", command=lambda: self._pick_model(self.mmproj_var)).grid(
            row=1, column=2, **pad
        )
        ttk.Label(
            mf,
            text="Bare filenames resolve under the models/ folder; absolute paths are kept as is.",
            style="Muted.TLabel",
        ).grid(row=2, column=1, columnspan=2, sticky="w", **pad)

        # Prompt ----------------------------------------------------------- #
        r += 1
        pf = ttk.LabelFrame(outer, text="Prompt", padding=self.px(6))
        pf.grid(row=r, column=0, columnspan=3, sticky="nsew", **pad)
        pf.columnconfigure(1, weight=1)
        outer.rowconfigure(r, weight=1)

        ttk.Radiobutton(
            pf, text="From file", variable=self.prompt_mode, value="file",
            command=self._on_prompt_mode,
        ).grid(row=0, column=0, sticky="w", **pad)
        file_row = ttk.Frame(pf)
        file_row.grid(row=0, column=1, sticky="ew", **pad)
        file_row.columnconfigure(0, weight=1)
        self.prompt_file_entry = ttk.Entry(file_row, textvariable=self.prompt_file_var)
        self.prompt_file_entry.grid(row=0, column=0, sticky="ew")
        self.prompt_file_entry.bind("<FocusOut>", lambda _e: self._load_prompt_file())
        self.prompt_browse = ttk.Button(file_row, text="Browse…", command=self._pick_prompt_file)
        self.prompt_browse.grid(row=0, column=1, padx=(self.px(6), 0))

        ttk.Radiobutton(
            pf, text="Inline text", variable=self.prompt_mode, value="text",
            command=self._on_prompt_mode,
        ).grid(row=1, column=0, sticky="w", **pad)

        self.prompt_text = tk.Text(pf, height=7, wrap="word", font=("Segoe UI", 10))
        self.app._style_text_widget(self.prompt_text)
        self.prompt_text.grid(row=2, column=0, columnspan=2, sticky="nsew", **pad)
        pf.rowconfigure(2, weight=1)
        ttk.Label(
            pf,
            text="In file mode, edits here are saved back to the prompt file.",
            style="Muted.TLabel",
        ).grid(row=3, column=0, columnspan=2, sticky="w", **pad)
        self._on_prompt_mode()

        # Trigger word ------------------------------------------------------ #
        r += 1
        tf = ttk.LabelFrame(outer, text="Trigger word", padding=self.px(6))
        tf.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        tf.columnconfigure(1, weight=1)
        ttk.Label(tf, text="Word (empty = disabled)").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(tf, textvariable=self.trigger_var).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Label(tf, text="Placement").grid(row=0, column=2, sticky="w", **pad)
        ttk.Combobox(
            tf, textvariable=self.placement_var, values=PLACEMENTS,
            state="readonly", width=14,
        ).grid(row=0, column=3, **pad)

        # Generation -------------------------------------------------------- #
        r += 1
        gf = ttk.LabelFrame(outer, text="Generation", padding=self.px(6))
        gf.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        for col in (1, 3):
            gf.columnconfigure(col, weight=1)
        for i, (key, label, _type) in enumerate(GEN_FIELDS):
            row_i, col_i = divmod(i, 2)
            ttk.Label(gf, text=label).grid(row=row_i, column=col_i * 2, sticky="w", **pad)
            ttk.Entry(gf, textvariable=self.gen_vars[key], width=12).grid(
                row=row_i, column=col_i * 2 + 1, sticky="w", **pad
            )
        last_row = (len(GEN_FIELDS) + 1) // 2
        ttk.Checkbutton(gf, text="Skip warmup (--no-warmup)", variable=self.no_warmup_var).grid(
            row=last_row, column=0, columnspan=2, sticky="w", **pad
        )
        ttk.Label(gf, text="Extra llama-server args").grid(
            row=last_row + 1, column=0, sticky="w", **pad
        )
        ttk.Entry(gf, textvariable=self.extra_args_var).grid(
            row=last_row + 1, column=1, columnspan=3, sticky="ew", **pad
        )

        # Output / buttons --------------------------------------------------- #
        r += 1
        of = ttk.Frame(outer)
        of.grid(row=r, column=0, columnspan=3, sticky="ew", **pad)
        of.columnconfigure(2, weight=1)
        ttk.Label(of, text="Sidecar suffix").grid(row=0, column=0, sticky="w", padx=(0, self.px(6)))
        ttk.Entry(of, textvariable=self.suffix_var, width=8).grid(row=0, column=1, sticky="w")
        ttk.Button(of, text="Cancel", command=self.destroy).grid(row=0, column=3, padx=(0, self.px(6)))
        ttk.Button(of, text="Save", style="Accent.TButton", command=self._save).grid(row=0, column=4)

    # -- prompt handling --------------------------------------------------- #
    def _on_prompt_mode(self) -> None:
        file_mode = self.prompt_mode.get() == "file"
        state = "normal" if file_mode else "disabled"
        self.prompt_file_entry.configure(state=state)
        self.prompt_browse.configure(state=state)
        if file_mode:
            self._load_prompt_file()

    def _prompt_file_path(self) -> Path | None:
        raw = self.prompt_file_var.get().strip()
        if not raw:
            return None
        p = Path(raw)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def _load_prompt_file(self) -> None:
        if self.prompt_mode.get() != "file":
            return
        path = self._prompt_file_path()
        self.prompt_text.delete("1.0", "end")
        if path is not None and path.is_file():
            try:
                self.prompt_text.insert("1.0", path.read_text(encoding="utf-8"))
            except OSError as e:
                messagebox.showerror("llamacap", f"Could not read prompt file:\n{e}", parent=self)

    def _pick_prompt_file(self) -> None:
        f = filedialog.askopenfilename(
            parent=self,
            title="Select prompt file",
            initialdir=str(PROJECT_ROOT / "prompts"),
            filetypes=[("Text", "*.txt"), ("All files", "*.*")],
        )
        if f:
            p = Path(f)
            try:
                p = p.relative_to(PROJECT_ROOT)
            except ValueError:
                pass
            self.prompt_file_var.set(str(p).replace("\\", "/"))
            self._load_prompt_file()

    def _pick_model(self, var: tk.StringVar) -> None:
        f = filedialog.askopenfilename(
            parent=self,
            title="Select GGUF file",
            initialdir=str(_models_dir()),
            filetypes=[("GGUF", "*.gguf"), ("All files", "*.*")],
        )
        if f:
            var.set(f)

    # -- saving ------------------------------------------------------------ #
    def _save(self) -> None:
        name = self.name_var.get().strip()
        if not NAME_RE.match(name):
            messagebox.showerror(
                "llamacap",
                "Profile name must be letters, digits, '-' or '_' (used as the filename).",
                parent=self,
            )
            return

        generation: dict = {}
        for key, label, typ in GEN_FIELDS:
            raw = self.gen_vars[key].get().strip()
            try:
                generation[key] = typ(raw)
            except ValueError:
                messagebox.showerror(
                    "llamacap", f"{label} must be {'an integer' if typ is int else 'a number'}: {raw!r}",
                    parent=self,
                )
                return
        generation["no_warmup"] = self.no_warmup_var.get()
        generation["extra_args"] = self.extra_args_var.get().split()

        prompt_file = self.prompt_file_var.get().strip().replace("\\", "/")
        prompt_body = self.prompt_text.get("1.0", "end").strip()
        if self.prompt_mode.get() == "file":
            if not prompt_file:
                messagebox.showerror("llamacap", "Set a prompt file (or switch to inline text).", parent=self)
                return
            prompt_section = {"file": prompt_file, "text": ""}
        else:
            if not prompt_body:
                messagebox.showerror("llamacap", "Prompt text is empty.", parent=self)
                return
            prompt_section = {"file": prompt_file, "text": prompt_body}

        gguf_path, gguf_file = _split_model_value(self.gguf_var.get())
        mmproj_path, mmproj_file = _split_model_value(self.mmproj_var.get())

        doc = {
            "profile": {"name": name, "description": self.desc_var.get().strip()},
            "model": {
                "gguf_path": gguf_path,
                "mmproj_path": mmproj_path,
                "gguf_file": gguf_file,
                "mmproj_file": mmproj_file,
            },
            "prompt": prompt_section,
            "trigger_word": {
                "value": self.trigger_var.get().strip(),
                "placement": self.placement_var.get(),
            },
            "generation": generation,
            "output": {"suffix": self.suffix_var.get().strip() or ".txt"},
        }

        target = _profiles_dir() / f"{name}.toml"
        if target.exists() and name != self.existing_name:
            if not messagebox.askyesno(
                "llamacap", f"Profile '{name}' already exists. Overwrite it?", parent=self
            ):
                return

        try:
            if self.prompt_mode.get() == "file":
                path = self._prompt_file_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(prompt_body + "\n", encoding="utf-8")
            target.write_text(tomli_w.dumps(doc), encoding="utf-8")
        except OSError as e:
            messagebox.showerror("llamacap", f"Could not save profile:\n{e}", parent=self)
            return

        self.app._load_profiles(select=name)
        self.destroy()
