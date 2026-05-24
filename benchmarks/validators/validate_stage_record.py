#!/usr/bin/env python3
"""Validate a benchmark stage summary record."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

ROOT_HINT = Path(__file__).resolve().parents[2]
if str(ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(ROOT_HINT))

from benchmarks.validators.common import load_json, print_result, require


VALID_STATUSES = {"completed", "dry_run", "skipped", "failed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a benchmark stage summary JSON.")
    parser.add_argument("stage_record", type=Path, help="Path to */result/summary.json.")
    parser.add_argument("--stage", default="", help="Expected stage name.")
    return parser.parse_args()


def main() -> None:
    opt = parse_args()
    errors: List[str] = []
    payload = load_json(opt.stage_record)

    require(isinstance(payload, dict), "stage record must be a top-level object", errors)
    if not isinstance(payload, dict):
        print_result(opt.stage_record, {}, errors)
        return

    require(payload.get("schema_version") == "benchmark_stage_record_v1", "invalid schema_version", errors)
    require(isinstance(payload.get("stage"), str) and bool(payload.get("stage")), "stage must be a non-empty string", errors)
    require(isinstance(payload.get("mode"), str) and bool(payload.get("mode")), "mode must be a non-empty string", errors)
    require(payload.get("status") in VALID_STATUSES, "invalid status: {}".format(payload.get("status")), errors)
    require(isinstance(payload.get("output"), dict), "output must be an object", errors)
    require(isinstance(payload.get("result"), dict), "result must be an object", errors)
    require(isinstance(payload.get("inputs"), dict), "inputs must be an object", errors)
    if opt.stage:
        require(payload.get("stage") == opt.stage, "expected stage {}, got {}".format(opt.stage, payload.get("stage")), errors)

    summary = {
        "stage": payload.get("stage"),
        "mode": payload.get("mode"),
        "status": payload.get("status"),
    }
    print_result(opt.stage_record, summary, errors)


if __name__ == "__main__":
    main()
