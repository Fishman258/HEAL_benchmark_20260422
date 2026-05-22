#!/usr/bin/env python3
"""Compatibility wrapper for benchmark AP noise-sweep plotting."""

from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "benchmarks" / "plotting" / "plot_noise_sweep.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
