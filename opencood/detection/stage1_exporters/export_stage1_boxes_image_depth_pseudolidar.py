#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Export OPV2V image+depth pseudo-LiDAR local boxes as a stage1 cache.

This is a smoke/MVP bridge for box-based pose-correction methods: RGB/depth is
back-projected into each CAV's local LiDAR frame, then an existing PointPillar
detector is reused to emit the same stage1 schema as LiDAR exporters.
"""

import argparse
import copy
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.detection.stage1_exporters.export_stage1_boxes_per_cav import _post_process_stage1, _to_jsonable
from opencood.utils.camera_utils import load_camera_data
from opencood.utils.pcd_utils import mask_ego_points, mask_points_by_range
from opencood.utils.transformation_utils import get_pairwise_transformation


IMAGE_DEPTH_CACHE_BASENAME = "stage1_boxes_image_depth_pseudolidar.json"
IMAGE_DEPTH_MANIFEST_BASENAME = "manifest_image_depth_pseudolidar.json"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Export OPV2V image/depth pseudo-LiDAR per-CAV stage1 boxes."
    )
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--test_dir_override", type=str, default="")
    parser.add_argument("--depth_root", type=str, default="/data2/opv2v_depth/test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--camera_ids", type=str, default="0,1,2,3")
    parser.add_argument("--pixel_stride", type=int, default=4)
    parser.add_argument("--max_points_per_camera", type=int, default=25000)
    parser.add_argument("--max_points_per_cav", type=int, default=100000)
    parser.add_argument("--min_depth", type=float, default=0.5)
    parser.add_argument("--max_depth", type=float, default=120.0)
    parser.add_argument(
        "--intensity_mode",
        choices=["constant", "rgb_luma", "inverse_depth"],
        default="rgb_luma",
    )
    parser.add_argument("--nms_pre_topk", type=int, default=0)
    parser.add_argument("--score_threshold", type=float, default=None)
    parser.add_argument(
        "--allow_existing_output",
        action="store_true",
        help="Allow reusing an existing output_dir. The image-depth cache file is still not overwritten.",
    )
    parser.add_argument("--note", type=str, default="")
    return parser.parse_args()


def _resolve_split_hypes(hypes, split, test_dir_override=""):
    hypes = copy.deepcopy(hypes)
    split = str(split or "test").lower()
    if test_dir_override:
        hypes["test_dir"] = str(test_dir_override)
        hypes["validate_dir"] = str(test_dir_override)
    if split == "test":
        hypes["validate_dir"] = hypes["test_dir"]
        train = False
    elif split in {"val", "validate", "validation"}:
        train = False
        split = "val"
    elif split == "train":
        train = True
    else:
        raise ValueError("Unsupported split: {}".format(split))
    hypes.setdefault("noise_setting", OrderedDict())
    hypes["noise_setting"]["add_noise"] = False
    return hypes, train, split


def _parse_camera_ids(text):
    out = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        cam_id = int(part)
        if cam_id < 0 or cam_id > 3:
            raise ValueError("OPV2V camera id must be in [0, 3], got {}".format(cam_id))
        out.append(cam_id)
    if not out:
        raise ValueError("--camera_ids resolved to an empty list")
    return out


def _load_checkpoint_or_path(checkpoint_path, model):
    checkpoint_path = str(checkpoint_path)
    if os.path.isfile(checkpoint_path):
        loaded_state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(loaded_state_dict, strict=False)
        return model
    if os.path.isdir(checkpoint_path):
        _, model = train_utils.load_saved_model(checkpoint_path, model)
        return model
    raise FileNotFoundError(checkpoint_path)


def _sample_record(dataset, idx):
    scenario_index = 0
    for i, ele in enumerate(dataset.len_record):
        if idx < ele:
            scenario_index = i
            break
    scenario_database = dataset.scenario_database[scenario_index]
    scenario_folder = dataset.scenario_folders[scenario_index]
    timestamp_index = idx if scenario_index == 0 else idx - dataset.len_record[scenario_index - 1]
    timestamp = dataset.return_timestamp_key(scenario_database, timestamp_index)
    return scenario_folder, scenario_database, timestamp


def _load_params(yaml_path):
    json_path = str(yaml_path).replace(".yaml", ".json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            params = json.load(f)
    else:
        params = yaml_utils.load_yaml(str(yaml_path))
    if "lidar_pose_clean" not in params:
        params["lidar_pose_clean"] = params.get("lidar_pose")
    return params


def _depth_path(depth_root, dataset_root, cav_path, timestamp, cam_id):
    rel = os.path.relpath(str(cav_path), str(dataset_root))
    return os.path.join(str(depth_root), rel, "{}_camera{}_depth.npy".format(timestamp, cam_id))


def _load_depth(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32, copy=False)
    depth = load_camera_data([path])[0]
    return np.asarray(depth, dtype=np.float32)


def _load_rgb_luma(path):
    if not os.path.exists(path):
        return None
    img = load_camera_data([path])[0].convert("RGB")
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _choice_without_replacement(rng, n, k):
    if k <= 0 or n <= k:
        return None
    return rng.choice(n, size=k, replace=False)


def _backproject_depth_to_lidar(
    *,
    dataset,
    params,
    camera_path,
    depth_path,
    cam_id,
    pixel_stride,
    max_points_per_camera,
    min_depth,
    max_depth,
    intensity_mode,
    rng,
):
    depth = _load_depth(depth_path)
    if depth.ndim != 2:
        depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError("Expected 2D depth for {}, got shape {}".format(depth_path, depth.shape))

    stride = max(1, int(pixel_stride))
    v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    valid = np.isfinite(z) & (z >= float(min_depth)) & (z <= float(max_depth))
    if not np.any(valid):
        return np.zeros((0, 4), dtype=np.float32)

    u = u[valid].astype(np.float32)
    v = v[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    if max_points_per_camera and z.shape[0] > int(max_points_per_camera):
        keep = _choice_without_replacement(rng, z.shape[0], int(max_points_per_camera))
        u = u[keep]
        v = v[keep]
        z = z[keep]

    camera_to_lidar, intrinsic = dataset.get_ext_int(params, int(cam_id))
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    if abs(fx) < 1e-6 or abs(fy) < 1e-6:
        raise ValueError("Invalid camera intrinsic for camera{}: {}".format(cam_id, intrinsic))

    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    ones = np.ones_like(z)
    cam_points = np.stack([x, y, z, ones], axis=0)
    lidar_xyz = (camera_to_lidar @ cam_points)[:3, :].T.astype(np.float32)

    if intensity_mode == "constant":
        intensity = np.ones((lidar_xyz.shape[0], 1), dtype=np.float32)
    elif intensity_mode == "inverse_depth":
        intensity = (1.0 - np.clip(z / max(float(max_depth), 1e-6), 0.0, 1.0)).reshape(-1, 1)
        intensity = intensity.astype(np.float32)
    else:
        luma = _load_rgb_luma(camera_path)
        if luma is None or luma.shape != depth.shape:
            intensity = np.ones((lidar_xyz.shape[0], 1), dtype=np.float32)
        else:
            sampled = luma[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride][valid]
            if max_points_per_camera and sampled.shape[0] > int(max_points_per_camera):
                sampled = sampled[keep]
            intensity = sampled.astype(np.float32).reshape(-1, 1)

    return np.concatenate([lidar_xyz, intensity], axis=1).astype(np.float32)


def _build_pseudolidar_for_cav(
    *,
    dataset,
    params,
    camera_files,
    depth_files,
    camera_ids,
    pixel_stride,
    max_points_per_camera,
    max_points_per_cav,
    min_depth,
    max_depth,
    intensity_mode,
    rng,
):
    pcs = []
    per_camera_counts = {}
    for cam_id in camera_ids:
        pc = _backproject_depth_to_lidar(
            dataset=dataset,
            params=params,
            camera_path=camera_files[cam_id],
            depth_path=depth_files[cam_id],
            cam_id=cam_id,
            pixel_stride=pixel_stride,
            max_points_per_camera=max_points_per_camera,
            min_depth=min_depth,
            max_depth=max_depth,
            intensity_mode=intensity_mode,
            rng=rng,
        )
        per_camera_counts[str(cam_id)] = int(pc.shape[0])
        if pc.shape[0]:
            pcs.append(pc)

    if pcs:
        pcd = np.concatenate(pcs, axis=0).astype(np.float32)
    else:
        pcd = np.zeros((0, 4), dtype=np.float32)

    pcd = mask_points_by_range(pcd, dataset.params["preprocess"]["cav_lidar_range"])
    pcd = mask_ego_points(pcd)
    if max_points_per_cav and pcd.shape[0] > int(max_points_per_cav):
        keep = _choice_without_replacement(rng, pcd.shape[0], int(max_points_per_cav))
        pcd = pcd[keep]

    stats = {
        "points_before_range_by_camera": per_camera_counts,
        "points_after_range_and_ego_mask": int(pcd.shape[0]),
    }
    return pcd.astype(np.float32), stats


def _process_single_cav(dataset, selected_cav_base):
    lidar_np = selected_cav_base["lidar_np"]
    lidar_np = mask_points_by_range(lidar_np, dataset.params["preprocess"]["cav_lidar_range"])
    lidar_np = mask_ego_points(lidar_np)
    processed_lidar = dataset.pre_processor.preprocess(lidar_np)
    object_bbx_center, object_bbx_mask, object_ids = dataset.generate_object_center(
        [selected_cav_base], selected_cav_base["params"]["lidar_pose_clean"]
    )
    label_dict = dataset.post_processor.generate_label(
        gt_box_center=object_bbx_center,
        anchors=dataset.anchor_box,
        mask=object_bbx_mask,
    )
    return {
        "processed_lidar": processed_lidar,
        "object_bbx_center": object_bbx_center,
        "object_bbx_mask": object_bbx_mask,
        "object_ids": object_ids,
        "label_dict": label_dict,
    }


def _single_cav_batch(dataset, processed, params):
    pose = np.asarray(params["lidar_pose"], dtype=np.float32).reshape(1, 6)
    pose_clean = np.asarray(params.get("lidar_pose_clean", params["lidar_pose"]), dtype=np.float32).reshape(1, 6)
    pairwise = np.tile(np.eye(4, dtype=np.float32), (1, dataset.max_cav, dataset.max_cav, 1, 1))
    batch = {
        "ego": {
            "processed_lidar": dataset.pre_processor.collate_batch([processed["processed_lidar"]]),
            "object_bbx_center": torch.from_numpy(np.asarray([processed["object_bbx_center"]])),
            "object_bbx_mask": torch.from_numpy(np.asarray([processed["object_bbx_mask"]])),
            "record_len": torch.from_numpy(np.asarray([1], dtype=np.int64)),
            "label_dict": dataset.post_processor.collate_batch([processed["label_dict"]]),
            "object_ids": processed["object_ids"],
            "pairwise_t_matrix": torch.from_numpy(pairwise),
            "lidar_pose_clean": torch.from_numpy(pose_clean),
            "lidar_pose": torch.from_numpy(pose),
            "anchor_box": dataset.anchor_box_torch,
        }
    }
    batch["ego"]["label_dict"]["pairwise_t_matrix"] = batch["ego"]["pairwise_t_matrix"]
    batch["ego"]["label_dict"]["record_len"] = batch["ego"]["record_len"]
    return batch


def _model_input(batch):
    ego = batch["ego"]
    if "record_len" in ego:
        return ego
    return {
        "processed_lidar": ego["processed_lidar"],
    }


def _validate_output_paths(output_dir, split_name, allow_existing):
    out_dir = Path(output_dir)
    split_dir = out_dir / split_name
    cache_path = split_dir / IMAGE_DEPTH_CACHE_BASENAME
    manifest_path = out_dir / IMAGE_DEPTH_MANIFEST_BASENAME
    if cache_path.exists():
        raise FileExistsError("Refusing to overwrite image-depth cache: {}".format(cache_path))
    if manifest_path.exists():
        raise FileExistsError("Refusing to overwrite image-depth manifest: {}".format(manifest_path))
    return out_dir, split_dir, cache_path, manifest_path


def _validate_cache_record(rec):
    cav_ids = rec.get("cav_id_list") or []
    n = len(cav_ids)
    for key in ("pred_corner3d_np_list", "pred_box3d_np_list", "pred_score_np_list", "uncertainty_np_list"):
        if len(rec.get(key) or []) != n:
            raise ValueError("{} length mismatch: {} vs {}".format(key, len(rec.get(key) or []), n))
    for key in ("lidar_pose_np", "lidar_pose_clean_np"):
        if len(rec.get(key) or []) != n:
            raise ValueError("{} length mismatch: {} vs {}".format(key, len(rec.get(key) or []), n))


def main():
    opt = _parse_args()
    camera_ids = _parse_camera_ids(opt.camera_ids)

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, SimpleNamespace(model_dir=""))
    hypes, train_flag, split_name = _resolve_split_hypes(hypes, opt.split, opt.test_dir_override)
    hypes["input_source"] = ["lidar"]

    if opt.score_threshold is not None:
        hypes["postprocess"]["target_args"]["score_threshold"] = float(opt.score_threshold)

    out_dir, split_dir, cache_path, manifest_path = _validate_output_paths(
        opt.output_dir, split_name, opt.allow_existing_output
    )

    dataset = build_dataset(hypes, visualize=False, train=train_flag)
    if not hasattr(dataset, "anchor_box"):
        dataset.anchor_box = dataset.post_processor.generate_anchor_box()
        dataset.anchor_box_torch = torch.from_numpy(dataset.anchor_box)

    model = train_utils.create_model(hypes)
    model = _load_checkpoint_or_path(opt.checkpoint_path, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    anchor_box = torch.from_numpy(dataset.post_processor.generate_anchor_box()).to(device)

    total_len = len(dataset)
    max_samples = opt.max_samples if opt.max_samples is not None else total_len
    max_samples = min(int(max_samples), int(total_len))
    rng = np.random.default_rng(20260513)
    stage1 = OrderedDict()
    sample_stats = OrderedDict()
    started = time.time()

    with torch.no_grad():
        for idx in range(max_samples):
            scenario_folder, scenario_database, timestamp = _sample_record(dataset, idx)
            scenario_name = os.path.basename(os.path.normpath(scenario_folder))

            base_for_pairwise = OrderedDict()
            cav_id_list = []
            pred_corner3d_np_list = []
            pred_box3d_np_list = []
            pred_score_np_list = []
            uncertainty_np_list = []
            lidar_pose_np = []
            lidar_pose_clean_np = []
            per_cav_stats = OrderedDict()

            for cav_id, cav_content in scenario_database.items():
                if cav_id == "ego":
                    continue
                cav_timestamp = cav_content.get(timestamp)
                if cav_timestamp is None:
                    continue
                cav_path = os.path.join(scenario_folder, str(cav_id))
                params = _load_params(cav_timestamp["yaml"])
                camera_files = list(cav_timestamp.get("cameras") or [])
                if not camera_files:
                    camera_files = dataset.find_camera_files(cav_path, timestamp)
                depth_files = [
                    _depth_path(opt.depth_root, dataset.root_dir, cav_path, timestamp, cam_id)
                    for cam_id in range(4)
                ]

                pseudo_lidar, stats = _build_pseudolidar_for_cav(
                    dataset=dataset,
                    params=params,
                    camera_files=camera_files,
                    depth_files=depth_files,
                    camera_ids=camera_ids,
                    pixel_stride=opt.pixel_stride,
                    max_points_per_camera=opt.max_points_per_camera,
                    max_points_per_cav=opt.max_points_per_cav,
                    min_depth=opt.min_depth,
                    max_depth=opt.max_depth,
                    intensity_mode=opt.intensity_mode,
                    rng=rng,
                )
                per_cav_stats[str(cav_id)] = stats

                single_base = OrderedDict()
                single_base["ego"] = bool(cav_content.get("ego", False))
                single_base["params"] = params
                single_base["lidar_np"] = pseudo_lidar
                processed = _process_single_cav(dataset, single_base)
                batch = _single_cav_batch(dataset, processed, params)
                batch = train_utils.to_device(batch, device)
                output = model(_model_input(batch))
                corners, boxes, scores, uncertainty = _post_process_stage1(
                    output, dataset.post_processor, anchor_box, opt.nms_pre_topk
                )

                cav_id_list.append(str(cav_id))
                pred_corner3d_np_list.append(corners[0] if corners else [])
                pred_box3d_np_list.append(boxes[0] if boxes else [])
                pred_score_np_list.append(scores[0] if scores else [])
                uncertainty_np_list.append(uncertainty[0] if uncertainty else [])
                lidar_pose_np.append(_to_jsonable(np.asarray(params.get("lidar_pose"), dtype=np.float32)))
                lidar_pose_clean_np.append(
                    _to_jsonable(np.asarray(params.get("lidar_pose_clean", params.get("lidar_pose")), dtype=np.float32))
                )
                base_for_pairwise[str(cav_id)] = {"params": params, "ego": bool(cav_content.get("ego", False))}

            rec = OrderedDict()
            rec["pred_corner3d_np_list"] = pred_corner3d_np_list
            rec["pred_box3d_np_list"] = pred_box3d_np_list
            rec["pred_score_np_list"] = pred_score_np_list
            rec["uncertainty_np_list"] = uncertainty_np_list
            rec["lidar_pose_np"] = lidar_pose_np
            rec["lidar_pose_clean_np"] = lidar_pose_clean_np
            rec["cav_id_list"] = cav_id_list
            rec["veh_frame_id"] = timestamp
            rec["infra_frame_id"] = None
            rec["scenario_name"] = scenario_name
            rec["timestamp"] = timestamp
            rec["stage1_source"] = "image_depth_pseudolidar"
            rec["bev_range"] = _to_jsonable(np.asarray(dataset.params["preprocess"]["cav_lidar_range"], dtype=np.float32))
            if base_for_pairwise:
                rec["pairwise_t_matrix_clean_np"] = _to_jsonable(
                    get_pairwise_transformation(base_for_pairwise, dataset.max_cav, False).astype(np.float32)
                )
            _validate_cache_record(rec)
            stage1[str(idx)] = rec
            sample_stats[str(idx)] = per_cav_stats

            if opt.log_interval and (len(stage1) % int(opt.log_interval) == 0 or idx == max_samples - 1):
                total_boxes = sum(len(x) for x in pred_corner3d_np_list)
                print(
                    "[image-depth-stage1] processed {}/{} samples (idx={}, cavs={}, boxes={})".format(
                        len(stage1), max_samples, idx, len(cav_id_list), total_boxes
                    ),
                    flush=True,
                )

    split_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(stage1, f, sort_keys=True)

    manifest = OrderedDict()
    manifest["schema_version"] = "image_depth_pseudolidar_stage1_manifest_v1"
    manifest["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["elapsed_sec"] = float(time.time() - started)
    manifest["stage1_cache"] = str(cache_path)
    manifest["stage1_cache_basename"] = IMAGE_DEPTH_CACHE_BASENAME
    manifest["stage1_source"] = "image_depth_pseudolidar"
    manifest["dataset_root"] = str(dataset.root_dir)
    manifest["depth_root"] = str(Path(opt.depth_root).resolve())
    manifest["hypes_yaml"] = str(Path(opt.hypes_yaml).resolve())
    manifest["checkpoint_path"] = str(Path(opt.checkpoint_path).resolve())
    manifest["split"] = split_name
    manifest["samples_written"] = int(len(stage1))
    manifest["dataset_len"] = int(total_len)
    manifest["camera_ids"] = camera_ids
    manifest["pixel_stride"] = int(opt.pixel_stride)
    manifest["max_points_per_camera"] = int(opt.max_points_per_camera)
    manifest["max_points_per_cav"] = int(opt.max_points_per_cav)
    manifest["depth_range_m"] = [float(opt.min_depth), float(opt.max_depth)]
    manifest["intensity_mode"] = str(opt.intensity_mode)
    manifest["score_threshold"] = float(hypes["postprocess"]["target_args"]["score_threshold"])
    manifest["nms_pre_topk"] = int(opt.nms_pre_topk)
    manifest["note"] = str(opt.note or "")
    manifest["sample_stats"] = sample_stats
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print("Wrote {} samples to {}".format(len(stage1), cache_path), flush=True)
    print("Wrote manifest to {}".format(manifest_path), flush=True)


if __name__ == "__main__":
    main()
