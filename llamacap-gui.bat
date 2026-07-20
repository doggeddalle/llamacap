@echo off
rem Launch the llamacap Tkinter GUI.
rem Uses pythonw so no console window lingers behind the app.
cd /d "%~dp0"
start "" uv run pythonw scripts\gui.py
