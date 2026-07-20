"""Small reusable Tk helpers for the llamacap GUI."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class Tooltip:
    """Hover tooltip for any widget. Shows after a short delay, follows theme
    colors passed in by the app."""

    def __init__(self, widget: tk.Widget, text: str, *, delay_ms: int = 500):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id: str | None = None
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self):
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self):
        if self._tip is not None:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(
            self._tip,
            text=self.text,
            style="Tooltip.TLabel",
            padding=(8, 4),
            wraplength=360,
            justify="left",
        )
        label.pack()

    def _hide(self, _event=None):
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


def parse_dnd_paths(root: tk.Misc, data: str) -> list[str]:
    """Turn a tkinterdnd2 drop event's data string into a list of paths.

    Paths with spaces arrive wrapped in braces; Tcl's splitlist handles that.
    """
    try:
        return list(root.tk.splitlist(data))
    except tk.TclError:
        return [data.strip("{}")]
