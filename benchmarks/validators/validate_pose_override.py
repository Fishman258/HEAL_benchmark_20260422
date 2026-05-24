#!/usr/bin/env python3
"""Validate a pose override JSON consumed by fusion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_HINT = Path(__file__).resolve().parents[2]
if str(ROOT_HINT) not in sys.path:
    sys.path.insert(0, str(ROOT_HINT))

from benchmarks.validators.common import (
    is_pose6,
    limited_items,
    load_json,
    print_result,
    require,
    validate_count_and_keys,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a pose override JSON.")
    parser.add_argument("pose_override", type=Path, help="Path to pose_override*.json.")
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--pose-field", default="lidar_pose_pred_np")
    parser.add_argument("--confidence-field", default="pose_confidence_np")
    parser.add_argument("--require-contiguous-keys", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=0, help="Validate only the first N records; 0 validates all.")
    return parser.parse_args()


def _validate_pose_map(prefix: str, cav_ids: List[Any], poses: Any, pose_field: str, errors: List[str]) -> None:
    if isinstance(poses, dict):
        for cav_id in cav_ids:
            require(str(cav_id) in poses, "{} missing pose for cav {}".format(prefix, cav_id), errors)
            if str(cav_id) in poses:
                require(is_pose6(poses.get(str(cav_id))), "{} pose for cav {} must be pose6".format(prefix, cav_id), errors)
        return
    require(isinstance(poses, list), "{} {} must be a list or object".format(prefix, pose_field), errors)
    if not isinstance(poses, list):
        return
    require(len(poses) == len(cav_ids), "{} pose count {} != cav count {}".format(prefix, len(poses), len(cav_ids)), errors)
    for idx, pose in enumerate(poses[: len(cav_ids)]):
        require(is_pose6(pose), "{} pose index {} must be pose6".format(prefix, idx), errors)


def validate_record(sample_key: str, record: Any, pose_field: str, confidence_field: str, errors: List[str]) -> Dict[str, int]:
    prefix = "sample {}".format(sample_key)
    require(isinstance(record, dict), "{} must be an object".format(prefix), errors)
    if not isinstance(record, dict):
        return {"cav_count": 0}
    if bool(record.get("force_ego_only") or record.get("ego_only") or record.get("drop_non_ego")):
        return {"cav_count": 1}

    cav_ids = record.get("cav_id_list") or record.get("cav_ids") or record.get("agent_ids")
    require(isinstance(cav_ids, list), "{} cav_id_list/cav_ids/agent_ids must be a list".format(prefix), errors)
    if not isinstance(cav_ids, list):
        return {"cav_count": 0}
    require(len(cav_ids) > 0, "{} cav id list must be non-empty".format(prefix), errors)

    _validate_pose_map(prefix, cav_ids, record.get(pose_field), pose_field, errors)
    confs = record.get(confidence_field)
    if confs is not None:
        if isinstance(confs, list):
            require(len(confs) == len(cav_ids), "{} confidence count mismatch".format(prefix), errors)
        else:
            require(isinstance(confs, dict), "{} confidence field must be a list or object".format(prefix), errors)
    return {"cav_count": len(cav_ids)}


def main() -> None:
    opt = parse_args()
    errors: List[str] = []
    payload = load_json(opt.pose_override)
    require(isinstance(payload, dict), "pose override must be a top-level object", errors)
    if not isinstance(payload, dict):
        print_result(opt.pose_override, {}, errors)
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
        stats = validate_record(str(sample_key), record, opt.pose_field, opt.confidence_field, errors)
        max_cav = max(max_cav, int(stats.get("cav_count", 0)))
        checked += 1

    summary = {
        "samples": len(payload),
        "checked_records": checked,
        "max_cav_count_checked": max_cav,
    }
    print_result(opt.pose_override, summary, errors)


if __name__ == "__main__":
    main()
