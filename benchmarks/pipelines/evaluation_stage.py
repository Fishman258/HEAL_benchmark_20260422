#!/usr/bin/env python3
"""Evaluation-stage summary collation."""

from __future__ import annotations

import glob
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from benchmarks.pipelines.artifacts import make_stage_dirs, resolve_path, write_json


def run_evaluation_stage(
    config: Mapping[str, Any],
    *,
    run_dir: Path,
    dry_run: bool = False,
    stage_records: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    stage_dirs = make_stage_dirs(run_dir, "evaluation")
    started = time.time()
    search_roots = list(config.get("search_roots") or [])
    patterns = config.get("patterns") or ["AP030507_*.yaml", "AP030507_*.yml", "AP030507_*.json"]
    artifacts = []
    explicit_artifacts = []
    if bool(config.get("include_stage_output_dirs", True)) and isinstance(stage_records, Mapping):
        for stage_record in stage_records.values():
            output = stage_record.get("output") if isinstance(stage_record, Mapping) else None
            if not isinstance(output, Mapping):
                continue
            search_roots.extend(output.get("legacy_output_dirs") or [])
            explicit_artifacts.extend(output.get("artifacts") or [])
    if not dry_run:
        for root in search_roots:
            root_path = resolve_path(root)
            for pattern in patterns:
                artifacts.extend(glob.glob(str(root_path / pattern)))
        artifacts.extend(str(resolve_path(path)) for path in explicit_artifacts)
    record: Dict[str, Any] = {
        "schema_version": "benchmark_stage_record_v1",
        "stage": "evaluation",
        "mode": str(config.get("mode") or "collect_existing_artifacts"),
        "status": "dry_run" if dry_run else "completed",
        "output": {"artifacts": sorted(set(artifacts))},
        "result": {"artifact_count": len(set(artifacts)), "search_roots": sorted(set(str(x) for x in search_roots))},
        "inputs": dict(config),
        "elapsed_sec": float(time.time() - started),
    }
    write_json(stage_dirs["result"] / "summary.json", record)
    return record
