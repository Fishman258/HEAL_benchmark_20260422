#!/usr/bin/env python3
"""Validate a stage1 detection cache used by registration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_HINT = Path(__file__).resolve().parents[2]
if str(ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(ROOT_HINT))

from benchmarks.validators.common import (
    limited_items,
    load_json,
    print_result,
    require,
    validate_count_and_keys,
)


LIST_FIELDS = (
    "pred_corner3d_np_list",
    "pred_box3d_np_list",
    "pred_score_np_list",
    "uncertainty_np_list",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a stage1 detection cache JSON.")
    parser.add_argument("stage1", type=Path, help="Path to stage1_boxes*.json.")
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--require-contiguous-keys", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=0, help="Validate only the first N records; 0 validates all.")
    return parser.parse_args()


def validate_record(sample_key: str, record: Any, errors: List[str]) -> Dict[str, int]:
    prefix = "sample {}".format(sample_key)
    require(isinstance(record, dict), "{} must be an object".format(prefix), errors)
    if not isinstance(record, dict):
        return {"cav_count": 0}

    cav_ids = record.get("cav_id_list")
    require(isinstance(cav_ids, list), "{} cav_id_list must be a list".format(prefix), errors)
    cav_count = len(cav_ids) if isinstance(cav_ids, list) else 0
    require(cav_count > 0, "{} cav_id_list must be non-empty".format(prefix), errors)

    has_box_field = any(key in record for key in ("pred_corner3d_np_list", "pred_box3d_np_list"))
    require(has_box_field, "{} requires pred_corner3d_np_list or pred_box3d_np_list".format(prefix), errors)

    for key in LIST_FIELDS:
        if key in record:
            value = record.get(key)
            require(isinstance(value, list), "{} {} must be a list".format(prefix, key), errors)
            if isinstance(value, list):
                require(
                    len(value) == cav_count,
                    "{} {} length {} != cav count {}".format(prefix, key, len(value), cav_count),
                    errors,
                )

    for key in ("lidar_pose_np", "lidar_pose_clean_np"):
        if key in record:
            value = record.get(key)
            require(isinstance(value, list), "{} {} must be a list".format(prefix, key), errors)
            if isinstance(value, list):
                require(
                    len(value) == cav_count,
                    "{} {} length {} != cav count {}".format(prefix, key, len(value), cav_count),
                    errors,
                )
    return {"cav_count": cav_count}


def main() -> None:
    opt = parse_args()
    errors: List[str] = []
    payload = load_json(opt.stage1)
    require(isinstance(payload, dict), "stage1 cache must be a top-level object", errors)
    if not isinstance(payload, dict):
        print_result(opt.stage1, {}, errors)
        return

    validate_count_and_keys(
        payload,
        expected_samples=opt.expected_samples,
        require_contiguous_keys=bool(opt.require_contiguous_keys),
        errors=errors,
    )

    max_cav = 0
    checked = 0
    for sample_key, record in limited_items(payload, int(opt.sample_limit or 0)):
        stats = validate_record(str(sample_key), record, errors)
        max_cav = max(max_cav, int(stats.get("cav_count", 0)))
        checked += 1

    summary = {
        "samples": len(payload),
        "checked_records": checked,
        "max_cav_count_checked": max_cav,
    }
    print_result(opt.stage1, summary, errors)


if __name__ == "__main__":
    main()
