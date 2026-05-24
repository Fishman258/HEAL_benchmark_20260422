#!/usr/bin/env python3
"""Registration-stage orchestration."""

from __future__ import annotations

import time
from pathlib import Path
from glob import glob
from typing import Any, Dict, Mapping

from benchmarks.pipelines.artifacts import load_mapping, make_stage_dirs, resolve_path, write_json
from benchmarks.pipelines.legacy_inference import run_inference_job


def run_registration_stage(
    config: Mapping[str, Any],
    *,
    run_dir: Path,
    detection_record: Mapping[str, Any],
    dry_run: bool = False,
) -> Dict[str, Any]:
    stage_dirs = make_stage_dirs(run_dir, "registration")
    started = time.time()
    mode = str(config.get("mode") or "pose_solver_only").strip()
    record: Dict[str, Any] = {
        "schema_version": "benchmark_stage_record_v1",
        "stage": "registration",
        "mode": mode,
        "status": "dry_run" if dry_run else "completed",
        "output": {},
        "result": {},
        "inputs": dict(config),
    }
    if mode == "skip":
        record["status"] = "skipped"
        write_json(stage_dirs["result"] / "summary.json", record)
        return record
    if mode != "pose_solver_only":
        raise ValueError(f"unsupported registration stage mode: {mode}")

    args = dict(config.get("args") or {})
    stage1 = args.get("stage1-result") or args.get("stage1_result")
    if not stage1:
        stage1 = (detection_record.get("output") or {}).get("stage1_result")
    if not stage1:
        raise ValueError("registration stage requires stage1-result")
    override_path = args.get("pose-override-export-path") or args.get("pose_override_export_path")
    if not override_path:
        override_path = stage_dirs["output"] / "pose_override.json"
    args["stage1-result"] = str(resolve_path(stage1))
    args["pose-override-export-path"] = str(resolve_path(override_path))
    args["pose-solver-only"] = True

    cmd_record = run_inference_job(
        args,
        python_bin=config.get("python_bin") or config.get("python-bin"),
        dry_run=dry_run,
        log_path=stage_dirs["logs"] / "command.sh",
    )
    record["output"]["pose_override_path"] = args["pose-override-export-path"]
    record["output"]["stage1_result"] = args["stage1-result"]
    legacy_dirs = cmd_record.get("legacy_output_dirs") or []
    record["output"]["legacy_output_dirs"] = legacy_dirs
    artifacts = []
    for out_dir in legacy_dirs:
        artifacts.extend(glob(str(Path(out_dir) / "POSE_SOLVER_ONLY_*.yaml")))
        artifacts.extend(glob(str(Path(out_dir) / "POSE_SOLVER_ONLY_*.json")))
    record["output"]["artifacts"] = sorted(set(artifacts))
    record["result"]["command"] = cmd_record
    for artifact in record["output"]["artifacts"]:
        if not Path(artifact).name.startswith("POSE_SOLVER_ONLY_"):
            continue
        try:
            payload = load_mapping(Path(artifact))
        except Exception:
            continue
        stats = payload.get("pose_solver_only_stats") if isinstance(payload, dict) else None
        if isinstance(stats, list) and stats:
            first = stats[0]
            if isinstance(first, dict):
                record["result"]["metrics_path"] = artifact
                record["result"]["pose_solver_only_stats"] = stats
                record["result"]["samples"] = first.get("samples")
                record["result"]["applied"] = first.get("applied")
        break
    record["elapsed_sec"] = float(time.time() - started)
    write_json(stage_dirs["result"] / "summary.json", record)
    return record
