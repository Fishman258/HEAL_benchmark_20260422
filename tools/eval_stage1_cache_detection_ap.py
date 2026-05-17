#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import box_utils, eval_utils


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate per-CAV stage1 cache detection AP.")
    parser.add_argument("--hypes-yaml", type=str, required=True)
    parser.add_argument("--stage1-cache", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--test-dir-override", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--label-mode", choices=["camera", "lidar"], default="camera")
    parser.add_argument("--eval-scope", choices=["per_cav", "ego_only"], default="per_cav")
    parser.add_argument("--log-interval", type=int, default=200)
    return parser.parse_args()


def _resolve_hypes(hypes, split, test_dir_override):
    hypes = copy.deepcopy(hypes)
    if test_dir_override:
        hypes["test_dir"] = str(test_dir_override)
        hypes["validate_dir"] = str(test_dir_override)
    split = str(split or "test").lower()
    if split == "test":
        hypes["validate_dir"] = hypes["test_dir"]
        train = False
    elif split in {"val", "validate", "validation"}:
        train = False
    else:
        train = True
    hypes["label_type"] = str(hypes.get("label_type") or "camera")
    return hypes, train


def _sample_gt(dataset, base_data_dict, cav_id, label_mode):
    cav = copy.deepcopy(base_data_dict[cav_id])
    params = cav.setdefault("params", {})
    if "lidar_pose_clean" not in params:
        params["lidar_pose_clean"] = params.get("lidar_pose")
    reference_pose = params["lidar_pose_clean"]
    generator = dataset.generate_object_center_camera if label_mode == "camera" else dataset.generate_object_center_lidar
    centers, mask, _ = generator([cav], reference_pose)
    valid = centers[mask == 1]
    if valid.size == 0:
        return np.zeros((0, 8, 3), dtype=np.float32)
    return box_utils.boxes_to_corners_3d(valid, order=dataset.params["postprocess"]["order"]).astype(np.float32)


def main():
    opt = _parse_args()
    out_dir = Path(opt.output_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, SimpleNamespace(model_dir=""))
    hypes, train_flag = _resolve_hypes(hypes, opt.split, opt.test_dir_override)
    dataset = build_dataset(hypes, visualize=False, train=train_flag)
    with open(opt.stage1_cache, "r", encoding="utf-8") as f:
        cache = json.load(f, object_pairs_hook=OrderedDict)

    total = min(len(dataset), len(cache))
    if int(opt.max_samples) > 0:
        total = min(total, int(opt.max_samples))

    result_stat = {
        0.30: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.50: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.70: {"tp": [], "fp": [], "gt": 0, "score": []},
    }
    frame_count = 0
    cav_eval_count = 0
    pred_box_count = 0
    gt_box_count = 0
    missing_keys = []

    for idx in range(total):
        key = str(idx)
        rec = cache.get(key)
        if rec is None:
            missing_keys.append(key)
            continue
        base_data_dict = dataset.retrieve_base_data(idx)
        cav_ids = [str(x) for x in rec.get("cav_id_list") or []]
        pred_lists = rec.get("pred_corner3d_np_list") or []
        score_lists = rec.get("pred_score_np_list") or []
        if opt.eval_scope == "ego_only":
            cav_ids = cav_ids[:1]
            pred_lists = pred_lists[:1]
            score_lists = score_lists[:1]

        for agent_idx, cav_id_str in enumerate(cav_ids):
            if agent_idx >= len(pred_lists):
                continue
            cav_id = None
            for raw_id in base_data_dict.keys():
                if str(raw_id) == cav_id_str:
                    cav_id = raw_id
                    break
            if cav_id is None:
                continue
            gt_corners = _sample_gt(dataset, base_data_dict, cav_id, opt.label_mode)
            pred_corners = np.asarray(pred_lists[agent_idx], dtype=np.float32)
            scores = np.asarray(
                score_lists[agent_idx] if agent_idx < len(score_lists) else [],
                dtype=np.float32,
            )
            if pred_corners.size == 0:
                pred_tensor = None
                score_tensor = torch.zeros((0,), dtype=torch.float32)
            else:
                pred_corners = pred_corners.reshape(-1, 8, 3)
                if scores.shape[0] != pred_corners.shape[0]:
                    scores = np.ones((pred_corners.shape[0],), dtype=np.float32)
                pred_tensor = torch.from_numpy(pred_corners)
                score_tensor = torch.from_numpy(scores)

            gt_tensor = torch.from_numpy(gt_corners.reshape(-1, 8, 3))
            for iou in (0.30, 0.50, 0.70):
                eval_utils.caluclate_tp_fp(pred_tensor, score_tensor, gt_tensor, result_stat, iou)
            cav_eval_count += 1
            pred_box_count += int(pred_corners.reshape(-1, 8, 3).shape[0]) if pred_corners.size else 0
            gt_box_count += int(gt_corners.shape[0])
        frame_count += 1
        if opt.log_interval > 0 and frame_count % int(opt.log_interval) == 0:
            print("[stage1-ap] processed {} / {}".format(frame_count, total), flush=True)

    ap30, _, _ = eval_utils.calculate_ap(result_stat, 0.30)
    ap50, _, _ = eval_utils.calculate_ap(result_stat, 0.50)
    ap70, _, _ = eval_utils.calculate_ap(result_stat, 0.70)
    payload = OrderedDict()
    payload["stage1_cache"] = str(Path(opt.stage1_cache).resolve())
    payload["hypes_yaml"] = str(Path(opt.hypes_yaml).resolve())
    payload["dataset_root"] = str(dataset.root_dir)
    payload["label_mode"] = opt.label_mode
    payload["eval_scope"] = opt.eval_scope
    payload["frames_evaluated"] = int(frame_count)
    payload["cav_evaluations"] = int(cav_eval_count)
    payload["pred_box_count"] = int(pred_box_count)
    payload["gt_box_count"] = int(gt_box_count)
    payload["missing_keys"] = missing_keys
    payload["ap30"] = float(ap30)
    payload["ap50"] = float(ap50)
    payload["ap70"] = float(ap70)
    payload["result_stat_counts"] = {
        str(iou): {
            "tp_fp_len": len(result_stat[iou]["tp"]),
            "score_len": len(result_stat[iou]["score"]),
            "gt": int(result_stat[iou]["gt"]),
        }
        for iou in (0.30, 0.50, 0.70)
    }
    with (out_dir / "stage1_detection_ap.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
