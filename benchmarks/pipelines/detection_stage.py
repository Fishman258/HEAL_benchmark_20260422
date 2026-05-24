#!/usr/bin/env python3
"""Detection-stage orchestration.

This module records or creates detection outputs.  The current first-stage
implementation mostly uses existing stage1 caches; exporter execution will be
added here as detection code is moved under `opencood/detection`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping

from benchmarks.pipelines.artifacts import make_stage_dirs, resolve_path, write_json


def run_detection_stage(config: Mapping[str, Any], *, run_dir: Path, dry_run: bool = False) -> Dict[str, Any]:
    stage_dirs = make_stage_dirs(run_dir, "detection")
    started = time.time()
    mode = str(config.get("mode") or "existing_stage1").strip()
    output: Dict[str, Any] = {
        "schema_version": "benchmark_stage_record_v1",
        "stage": "detection",
        "mode": mode,
        "status": "dry_run" if dry_run else "completed",
        "output": {},
        "result": {},
        "inputs": dict(config),
    }

    if mode == "existing_stage1":
        stage1_path = resolve_path(config.get("stage1_result") or config.get("stage1_path"))
        output["output"]["stage1_result"] = str(stage1_path)
        output["output"]["exists"] = bool(stage1_path.exists())
        if not dry_run and not stage1_path.exists():
            raise FileNotFoundError(f"stage1 cache not found: {stage1_path}")
    elif mode == "skip":
        output["status"] = "skipped"
    else:
        raise ValueError(f"unsupported detection stage mode: {mode}")

    output["elapsed_sec"] = float(time.time() - started)
    write_json(stage_dirs["result"] / "summary.json", output)
    return output

