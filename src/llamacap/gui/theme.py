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
    },
}

BASE_FONT = ("Segoe UI", 10)


def apply_theme(root: tk.Tk, theme: str) -> dict:
    """Apply the flat theme for `theme` ("light"/"dark"); returns its palette."""
    p = PALETTES["dark" if theme == "dark" else "light"]
    style = ttk.Style(root)
    style.theme_use("clam")

    root.configure(background=p["bg"])

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

    style.configure("TLabelframe", background=p["bg"], bordercolor=p["border"])
    style.configure(
        "TLabelframe.Label",
        background=p["bg"],
        foreground=p["fg"],
        font=("Segoe UI", 10, "bold"),
    )
    style.configure("Muted.TLabel", foreground=p["muted"])
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
    style.map(
        "TButton",
        background=[("pressed", p["pressed"]), ("active", p["hover"])],
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

    # The Combobox dropdown is a plain tk listbox; style it via the option db.
    root.option_add("*TCombobox*Listbox.background", p["surface"])
    root.option_add("*TCombobox*Listbox.foreground", p["fg"])
    root.option_add("*TCombobox*Listbox.selectBackground", p["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", p["accent_fg"])

    return p
