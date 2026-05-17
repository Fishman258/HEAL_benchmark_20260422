#!/usr/bin/env python3
"""Compatibility wrapper for the legacy AP noise-sweep plotting script."""

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "opencood" / "tools" / "plot_noise_sweep.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")

