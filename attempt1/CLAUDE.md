# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See parent directory CLAUDE.md for full project context. This subdirectory is the first working prototype of the hand tracking subsystem.

## Entry Points

- `mobrecon_live.py` — primary script; live webcam + static image modes
- `mobrecon_demo.ipynb` — exploratory notebook; `cell5_demo_fixed.py` is Cell 5 extracted for standalone use

## Checkpoint Location

Place MobRecon checkpoint at:
```
HandMesh/mobrecon/checkpoints/<name>.pt
```
Script auto-discovers any `.pt`/`.pth` under that dir, preferring filenames containing "mobrecon".

## Python Version

Shebang pins `~/.pyenv/versions/3.8.10`. HandMesh conda env uses Python 3.9. Run `mobrecon_live.py` with the pyenv 3.8 interpreter or ensure deps installed for whichever Python is active.
