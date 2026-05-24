#!/usr/bin/env python3
"""Fusion-stage orchestration."""

from __future__ import annotations

import time
from pathlib import Path
from glob import glob
from typing import Any, Dict, Mapping

from benchmarks.pipelines.artifacts import load_mapping, make_stage_dirs, resolve_path, write_json
from benchmarks.pipelines.legacy_inference import run_inference_job


def run_fusion_stage(
    config: Mapping[str, Any],
    *,
    run_dir: Path,
    registration_record: Mapping[str, Any],
    dry_run: bool = False,
) -> Dict[str, Any]:
    stage_dirs = make_stage_dirs(run_dir, "fusion")
    started = time.time()
    mode = str(config.get("mode") or "legacy_inference").strip()
    record: Dict[str, Any] = {
        "schema_version": "benchmark_stage_record_v1",
        "stage": "fusion",
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
    if mode != "legacy_inference":
        raise ValueError(f"unsupported fusion stage mode: {mode}")

    args = dict(config.get("args") or {})
    override_path = args.get("pose-override-path") or args.get("pose_override_path")
    if not override_path:
        override_path = (registration_record.get("output") or {}).get("pose_override_path")
    if override_path:
        args["pose-override-path"] = str(resolve_path(override_path))
        args.setdefault("pose-correction", "none")

    cmd_record = run_inference_job(
        args,
        python_bin=config.get("python_bin") or config.get("python-bin"),
        dry_run=dry_run,
        log_path=stage_dirs["logs"] / "command.sh",
    )
    record["output"]["pose_override_path"] = args.get("pose-override-path", "")
    legacy_dirs = cmd_record.get("legacy_output_dirs") or []
    record["output"]["legacy_output_dirs"] = legacy_dirs
    artifacts = []
    for out_dir in legacy_dirs:
        artifacts.extend(glob(str(Path(out_dir) / "AP030507_*.yaml")))
        artifacts.extend(glob(str(Path(out_dir) / "AP030507_*.json")))
        artifacts.extend(glob(str(Path(out_dir) / "eval_*.yaml")))
        artifacts.extend(glob(str(Path(out_dir) / "eval_*.json")))
    artifacts = sorted(set(artifacts))
    record["output"]["artifacts"] = artifacts
    for artifact in artifacts:
        name = Path(artifact).name
        if name.startswith("AP030507_"):
            record["output"].setdefault("metric_path", artifact)
            try:
                payload = load_mapping(Path(artifact))
            except Exception:
                payload = None
            if isinstance(payload, dict):
                record["result"]["metric_path"] = artifact
                record["result"]["ap30"] = payload.get("ap30")
                record["result"]["ap50"] = payload.get("ap50")
                record["result"]["ap70"] = payload.get("ap70")
                record["result"]["pos_std_list"] = payload.get("pos_std_list")
                record["result"]["rot_std_list"] = payload.get("rot_std_list")
            break
    record["result"]["command"] = cmd_record
    record["elapsed_sec"] = float(time.time() - started)
    write_json(stage_dirs["result"] / "summary.json", record)
    return record
