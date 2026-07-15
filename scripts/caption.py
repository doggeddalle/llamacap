#!/usr/bin/env python
"""Entrypoint: uv run scripts/caption.py --profile krea2 --input <dir>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llamacap.cli import main

if __name__ == "__main__":
    sys.exit(main())
