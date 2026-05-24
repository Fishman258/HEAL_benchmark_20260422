#!/usr/bin/env python3
"""Run a staged benchmark pipeline from YAML/JSON.

This is the new orchestration entry point.  Version 1 is intentionally thin:
it records stage boundaries and delegates actual inference to the existing
legacy executor until the core logic is migrated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT_HINT = Path(__file__).resolve().parents[2]
if str(ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(ROOT_HINT))

from benchmarks.pipelines.artifacts import (
    DEFAULT_OUTPUT_ROOT,
    ROOT,
    load_mapping,
    make_run_dir,
    resolve_path,
    write_json,
    write_yaml_or_json,
)
from benchmarks.pipelines.benchmark_stage import run_benchmark_stage
from benchmarks.pipelines.detection_stage import run_detection_stage
from benchmarks.pipelines.evaluation_stage import run_evaluation_stage
from benchmarks.pipelines.fusion_stage import run_fusion_stage
from benchmarks.pipelines.registration_stage import run_registration_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a staged benchmark pipeline.")
    parser.add_argument("config", type=Path, help="Pipeline YAML/JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="Create records and print commands without running jobs.")
    parser.add_argument("--print-run-dir", action="store_true", help="Print the run directory at the end.")
    return parser.parse_args()


def _stage_enabled(stage_cfg: Mapping[str, Any]) -> bool:
    return bool(stage_cfg.get("enabled", True))


def main() -> None:
    opt = parse_args()
    config = load_mapping(opt.config)
    name = str(config.get("name") or opt.config.stem)
    output_root = resolve_path(config.get("output_root") or DEFAULT_OUTPUT_ROOT)
    run_dir = make_run_dir(name, output_root=output_root, note=str(config.get("note") or name))
    write_yaml_or_json(run_dir / "config_resolved.yaml", config)

    manifest: Dict[str, Any] = {
        "schema_version": "benchmark_pipeline_manifest_v1",
        "workspace_root": str(ROOT),
        "config": str(opt.config.resolve()),
        "name": name,
        "run_dir": str(run_dir),
        "dry_run": bool(opt.dry_run),
        "stages": {},
    }
    write_json(run_dir / "manifest.json", manifest)

    stages = config.get("stages") or {}
    if not isinstance(stages, dict):
        raise ValueError("pipeline config requires mapping key: stages")

    detection_cfg = stages.get("detection") or {"enabled": False, "mode": "skip"}
    benchmark_cfg = stages.get("benchmark") or {"enabled": False, "mode": "skip"}
    registration_cfg = stages.get("registration") or {"enabled": False, "mode": "skip"}
    fusion_cfg = stages.get("fusion") or {"enabled": False, "mode": "skip"}
    evaluation_cfg = stages.get("evaluation") or {"enabled": False, "mode": "skip"}

    detection_record: Dict[str, Any] = {"stage": "detection", "status": "skipped", "output": {}}
    benchmark_record: Dict[str, Any] = {"stage": "benchmark", "status": "skipped", "output": {}}
    registration_record: Dict[str, Any] = {"stage": "registration", "status": "skipped", "output": {}}

    if _stage_enabled(detection_cfg):
        detection_record = run_detection_stage(detection_cfg, run_dir=run_dir, dry_run=opt.dry_run)
    manifest["stages"]["detection"] = detection_record
    write_json(run_dir / "manifest.json", manifest)

    if _stage_enabled(benchmark_cfg):
        benchmark_record = run_benchmark_stage(
            benchmark_cfg,
            run_dir=run_dir,
            detection_record=detection_record,
            dry_run=opt.dry_run,
        )
    manifest["stages"]["benchmark"] = benchmark_record
    write_json(run_dir / "manifest.json", manifest)

    if _stage_enabled(registration_cfg):
        registration_record = run_registration_stage(
            registration_cfg,
            run_dir=run_dir,
            detection_record=detection_record,
            dry_run=opt.dry_run,
        )
    manifest["stages"]["registration"] = registration_record
    write_json(run_dir / "manifest.json", manifest)

    if _stage_enabled(fusion_cfg):
        fusion_record = run_fusion_stage(
            fusion_cfg,
            run_dir=run_dir,
            registration_record=registration_record,
            dry_run=opt.dry_run,
        )
    else:
        fusion_record = {"stage": "fusion", "status": "skipped", "output": {}}
    manifest["stages"]["fusion"] = fusion_record
    write_json(run_dir / "manifest.json", manifest)

    if _stage_enabled(evaluation_cfg):
        evaluation_record = run_evaluation_stage(
            evaluation_cfg,
            run_dir=run_dir,
            dry_run=opt.dry_run,
            stage_records=manifest["stages"],
        )
    else:
        evaluation_record = {"stage": "evaluation", "status": "skipped", "output": {}}
    manifest["stages"]["evaluation"] = evaluation_record
    write_json(run_dir / "manifest.json", manifest)

    if opt.print_run_dir:
        print(run_dir)
    else:
        print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
