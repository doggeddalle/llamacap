#!/usr/bin/env python
"""Launcher for the llamacap GUI.

Launch with:  uv run scripts/gui.py   (or double-click llamacap-gui.bat)

The actual application lives in the llamacap.gui package (src/llamacap/gui/).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make llamacap importable when run straight from a checkout without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llamacap.gui import main

if __name__ == "__main__":
    sys.exit(main())
