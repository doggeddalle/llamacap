"""Fast Windows-11-style flat theme for ttk, light and dark.

Built on the vector-drawn `clam` engine instead of an image-based theme
(sv-ttk): image themes re-scale dozens of 9-patch bitmaps on every window
re-layout, which makes interactive resizing lag badly (~8x slower re-layout
measured on this UI). Everything here is plain colors, so resizing stays smooth.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

PALETTES = {
    "light": {
        "bg": "#fafafa",
        "surface": "#ffffff",
        "border": "#d0d0d0",
        "fg": "#1b1b1b",
        "muted": "#5f5f5f",
        "accent": "#005fb8",
        "accent_hover": "#1a6fc4",
        "accent_fg": "#ffffff",
        "hover": "#f0f0f0",
        "pressed": "#e6e6e6",
        "trough": "#e8e8e8",
        "text_bg": "#ffffff",
        "text_fg": "#1b1b1b",
        "error": "#c42b1c",
        "warning": "#8a5700",
        "success": "#107c10",
    },
    "dark": {
        "bg": "#202020",
        "surface": "#2b2b2b",
        "border": "#454545",
        "fg": "#f0f0f0",
        "muted": "#9a9a9a",
        "accent": "#4cc2ff",
        "accent_hover": "#6ccdff",
        "accent_fg": "#1b1b1b",
        "hover": "#333333",
        "pressed": "#3a3a3a",
        "trough": "#3a3a3a",
        "text_bg": "#1c1c1c",
        "text_fg": "#f0f0f0",
        "error": "#ff99a4",
        "warning": "#f5c26b",
        "success": "#6ccb5f",
    },
}

BASE_FONT = ("Segoe UI", 10)


def _set_windows_titlebar(root: tk.Tk, dark: bool) -> None:
    """Ask DWM to match the native title bar to the app theme."""
    if root.tk.call("tk", "windowingsystem") != "win32":
        return
    try:
        from ctypes import byref, c_int, windll

        root.update_idletasks()
        value = c_int(1 if dark else 0)
        hwnd = windll.user32.GetParent(root.winfo_id())
        # Attribute 20 is supported by current Windows 10/11; 19 covers older builds.
        result = windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, byref(value), 4)
        if result != 0:
            windll.dwmapi.DwmSetWindowAttribute(hwnd, 19, byref(value), 4)
    except Exception:
        pass


def apply_theme(root: tk.Tk, theme: str) -> dict:
    """Apply the flat theme for `theme` ("light"/"dark"); returns its palette."""
    p = PALETTES["dark" if theme == "dark" else "light"]
    style = ttk.Style(root)
    style.theme_use("clam")

    root.configure(background=p["bg"])
    _set_windows_titlebar(root, theme == "dark")

    style.configure(
        ".",
        background=p["bg"],
        foreground=p["fg"],
        bordercolor=p["border"],
        darkcolor=p["bg"],
        lightcolor=p["bg"],
        troughcolor=p["trough"],
        focuscolor=p["accent"],
        selectbackground=p["accent"],
        selectforeground=p["accent_fg"],
        insertcolor=p["fg"],
        fieldbackground=p["surface"],
        font=BASE_FONT,
    )

    # Explicitly own every container surface. Clam otherwise leaks platform
    # defaults (notably white notebook tabs on Windows dark mode).
    style.configure("TFrame", background=p["bg"])
    style.configure("TLabel", background=p["bg"], foreground=p["fg"])
    style.configure("TPanedwindow", background=p["border"], sashwidth=5)
    style.configure("Sash", sashthickness=5, background=p["border"])

    style.configure(
        "TNotebook",
        background=p["bg"],
        bordercolor=p["border"],
        lightcolor=p["border"],
        darkcolor=p["border"],
        tabmargins=(8, 8, 8, 0),
    )
    style.configure(
        "TNotebook.Tab",
        background=p["bg"],
        foreground=p["muted"],
        bordercolor=p["bg"],
        lightcolor=p["bg"],
        darkcolor=p["bg"],
        padding=(16, 8),
        focuscolor=p["accent"],
    )
    style.map(
        "TNotebook.Tab",
        background=[
            ("selected", p["surface"]),
            ("active", p["hover"]),
            ("!selected", p["bg"]),
        ],
        foreground=[
            ("selected", p["fg"]),
            ("active", p["fg"]),
            ("disabled", p["muted"]),
            ("!selected", p["muted"]),
        ],
        bordercolor=[("selected", p["accent"]), ("!selected", p["bg"])],
        lightcolor=[("selected", p["accent"]), ("!selected", p["bg"])],
        darkcolor=[("selected", p["accent"]), ("!selected", p["bg"])],
    )

    style.configure("TLabelframe", background=p["bg"], bordercolor=p["border"])
    style.configure(
        "TLabelframe.Label",
        background=p["bg"],
        foreground=p["fg"],
        font=("Segoe UI", 10, "bold"),
    )
    style.configure("Muted.TLabel", foreground=p["muted"])
    style.configure("Error.TLabel", foreground=p["error"])
    style.configure("Warning.TLabel", foreground=p["warning"])
    style.configure("Success.TLabel", foreground=p["success"])
    style.configure("Status.TLabel", font=("Segoe UI", 11, "bold"))
    style.configure(
        "Tooltip.TLabel",
        background=p["surface"],
        foreground=p["fg"],
        relief="solid",
        borderwidth=1,
    )

    style.configure(
        "TButton",
        background=p["surface"],
        foreground=p["fg"],
        bordercolor=p["border"],
        padding=(10, 4),
        relief="flat",
    )
    style.configure(
        "Disclosure.TButton", background=p["bg"], borderwidth=0,
        padding=(4, 6), anchor="w",
    )
    style.map(
        "TButton",
        background=[
            ("disabled", p["bg"]),
            ("pressed", p["pressed"]),
            ("active", p["hover"]),
        ],
        foreground=[("disabled", p["muted"])],
        bordercolor=[("focus", p["accent"]), ("disabled", p["border"])],
    )
    style.configure(
        "Accent.TButton",
        background=p["accent"],
        foreground=p["accent_fg"],
        bordercolor=p["accent"],
    )
    style.map(
        "Accent.TButton",
        background=[("pressed", p["accent"]), ("active", p["accent_hover"])],
        foreground=[("disabled", p["muted"])],
    )

    for widget in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(
            widget,
            fieldbackground=p["surface"],
            foreground=p["fg"],
            bordercolor=p["border"],
            insertcolor=p["fg"],
            padding=4,
        )
        style.map(
            widget,
            bordercolor=[("focus", p["accent"])],
            lightcolor=[("focus", p["accent"])],
            fieldbackground=[("readonly", p["surface"]), ("disabled", p["bg"])],
        )
    style.map("TCombobox", selectbackground=[("readonly", p["surface"])])
    style.map("TCombobox", selectforeground=[("readonly", p["fg"])])

    for widget in ("TCheckbutton", "TRadiobutton"):
        style.configure(
            widget,
            background=p["bg"],
            foreground=p["fg"],
            indicatorbackground=p["surface"],
            indicatorforeground=p["accent"],
        )
        style.map(
            widget,
            background=[("active", p["bg"])],
            indicatorbackground=[("selected", p["surface"])],
        )

    style.configure(
        "Horizontal.TProgressbar",
        background=p["accent"],
        troughcolor=p["trough"],
        bordercolor=p["border"],
        lightcolor=p["accent"],
        darkcolor=p["accent"],
    )

    style.configure(
        "Vertical.TScrollbar",
        background=p["surface"],
        troughcolor=p["bg"],
        bordercolor=p["bg"],
        arrowcolor=p["muted"],
    )
    style.map("Vertical.TScrollbar", background=[("active", p["hover"])])
    style.configure(
        "Horizontal.TScrollbar",
        background=p["surface"], troughcolor=p["bg"], bordercolor=p["bg"],
        arrowcolor=p["muted"],
    )
    style.map("Horizontal.TScrollbar", background=[("active", p["hover"])])

    # The Combobox dropdown is a plain tk listbox; style it via the option db.
    root.option_add("*TCombobox*Listbox.background", p["surface"])
    root.option_add("*TCombobox*Listbox.foreground", p["fg"])
    root.option_add("*TCombobox*Listbox.selectBackground", p["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", p["accent_fg"])

    return p
