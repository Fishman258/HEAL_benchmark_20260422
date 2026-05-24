#!/usr/bin/env python3
"""Adapter around the current `opencood/tools/inference_w_noise.py` entry.

This is intentionally a thin compatibility layer.  It lets the new staged
pipeline call the legacy executor while we migrate logic into smaller modules.
"""

from __future__ import annotations

import sys
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from benchmarks.pipelines.artifacts import ROOT, resolve_path, run_command


INFERENCE_SCRIPT = ROOT / "opencood" / "tools" / "inference_w_noise.py"
OUTPUT_DIR_RE = re.compile(r"Benchmark outputs will be written to:\s*(.+)")

LEGACY_UNDERSCORE_FLAGS = {
    "also_laplace",
    "fusion_method",
    "model_dir",
    "save_vis_interval",
}


def _cli_key(key: str) -> str:
    normalized = str(key).strip()
    if normalized in LEGACY_UNDERSCORE_FLAGS:
        return "--" + normalized
    return "--" + normalized.replace("_", "-")


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    return str(value)


def append_cli_args(cmd: List[str], args: Mapping[str, Any]) -> None:
    for key, value in args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(_cli_key(key))
            continue
        cmd.extend([_cli_key(key), _format_value(value)])


def build_inference_command(
    args: Mapping[str, Any],
    *,
    python_bin: Optional[Union[str, Path]] = None,
    extra_args: Optional[Sequence[Any]] = None,
) -> List[str]:
    if not INFERENCE_SCRIPT.exists():
        raise FileNotFoundError(f"inference_w_noise.py not found: {INFERENCE_SCRIPT}")
    cmd = [str(python_bin or sys.executable), str(INFERENCE_SCRIPT)]
    append_cli_args(cmd, args)
    if extra_args:
        cmd.extend(str(x) for x in extra_args)
    return cmd


def run_inference_job(
    args: Mapping[str, Any],
    *,
    python_bin: Optional[Union[str, Path]] = None,
    extra_args: Optional[Sequence[Any]] = None,
    dry_run: bool = False,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    cmd = build_inference_command(args, python_bin=python_bin, extra_args=extra_args)
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    env = {
        "PYTHONPATH": str(ROOT) + (os.pathsep + current_pythonpath if current_pythonpath else ""),
    }
    record = run_command(cmd, cwd=ROOT, env=env, dry_run=dry_run, log_path=log_path)
    output_dirs = []
    stdout_path = record.get("stdout_path")
    if stdout_path and Path(str(stdout_path)).exists():
        text = Path(str(stdout_path)).read_text(encoding="utf-8", errors="replace")
        output_dirs.extend(match.group(1).strip() for match in OUTPUT_DIR_RE.finditer(text))
    else:
        for line in record.get("stdout_tail") or []:
            match = OUTPUT_DIR_RE.search(str(line))
            if match:
                output_dirs.append(match.group(1).strip())
    record["legacy_output_dirs"] = sorted(set(output_dirs))
    return record


def normalize_existing_path(value: Any) -> str:
    return str(resolve_path(value)) if value else ""
