#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Export OPV2V image+depth camera-model local boxes as a stage1 cache.

This exporter runs an existing camera/depth detector per CAV, using each CAV as
its own local frame, and emits the same box-cache schema consumed by Reg++ style
pose-correction methods. It is intentionally separate from LiDAR and
pseudo-LiDAR exporters so benchmark artifacts remain distinguishable.
"""

import argparse
import copy
import importlib
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
from opencood.detection.stage1_exporters.export_stage1_boxes_per_cav import _to_jsonable
from opencood.utils.common_utils import merge_features_to_dict
from opencood.utils.pose_utils import attach_pose_confidence
from opencood.utils.transformation_utils import get_pairwise_transformation


IMAGE_DEPTH_CAMERA_CACHE_BASENAME = "stage1_boxes_image_depth_camera_model.json"
IMAGE_DEPTH_CAMERA_MANIFEST_BASENAME = "manifest_image_depth_camera_model.json"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Export OPV2V image/depth camera-model per-CAV stage1 boxes."
    )
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Checkpoint file or model directory. Directories are loaded with train_utils.load_saved_model().",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--test_dir_override", type=str, default="")
    parser.add_argument(
        "--depth_root_override",
        type=str,
        default="/data2/opv2v_depth/test",
        help="Parallel OPV2V depth root containing {scenario}/{cav}/{timestamp}_camera{i}_depth.npy.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--nms_pre_topk", type=int, default=0)
    parser.add_argument("--score_threshold", type=float, default=None)
    parser.add_argument("--num_shards", "--num-shards", type=int, default=1)
    parser.add_argument("--shard_index", "--shard-index", type=int, default=0)
    parser.add_argument(
        "--allow_existing_output",
        action="store_true",
        help="Allow an existing output_dir. The cache and manifest files are still not overwritten.",
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


def _parallel_depth_path(depth_root, dataset_root, cav_path, timestamp, cam_id):
    rel = os.path.relpath(str(cav_path), str(dataset_root))
    return os.path.join(str(depth_root), rel, "{}_camera{}_depth.npy".format(timestamp, cam_id))


def _apply_depth_root_override(dataset, depth_root):
    depth_root = str(depth_root or "").strip()
    if not depth_root:
        return {"enabled": False, "missing": 0, "total": 0}

    total = 0
    missing = 0
    for scenario_folder, scenario_database in zip(dataset.scenario_folders, dataset.scenario_database.values()):
        for cav_id, cav_content in scenario_database.items():
            if cav_id == "ego":
                continue
            cav_path = os.path.join(scenario_folder, str(cav_id))
            for timestamp, item in list(cav_content.items()):
                if timestamp == "ego" or not isinstance(item, dict):
                    continue
                depth_files = [
                    _parallel_depth_path(depth_root, dataset.root_dir, cav_path, timestamp, cam_id)
                    for cam_id in range(4)
                ]
                total += len(depth_files)
                missing += sum(0 if os.path.exists(p) else 1 for p in depth_files)
                item["depths"] = depth_files
    return {"enabled": True, "missing": int(missing), "total": int(total)}


def _validate_output_paths(output_dir, split_name, shard_index=0, num_shards=1):
    out_dir = Path(output_dir)
    split_dir = out_dir / split_name
    if int(num_shards) > 1:
        cache_path = split_dir / "stage1_boxes_image_depth_camera_model_shard{:02d}of{:02d}.json".format(
            int(shard_index), int(num_shards)
        )
        manifest_path = out_dir / "manifest_image_depth_camera_model_shard{:02d}of{:02d}.json".format(
            int(shard_index), int(num_shards)
        )
    else:
        cache_path = split_dir / IMAGE_DEPTH_CAMERA_CACHE_BASENAME
        manifest_path = out_dir / IMAGE_DEPTH_CAMERA_MANIFEST_BASENAME
    if cache_path.exists():
        raise FileExistsError("Refusing to overwrite image-depth camera cache: {}".format(cache_path))
    if manifest_path.exists():
        raise FileExistsError("Refusing to overwrite image-depth camera manifest: {}".format(manifest_path))
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


def _build_stage1_postprocessor(hypes):
    postprocessor_name = hypes["postprocess"].get("core_method", "VoxelPostprocessor")
    if str(postprocessor_name).lower() == "voxelpostprocessor":
        postprocessor_name = "uncertainty_voxel_postprocessor"
    postprocessor_lib = importlib.import_module("opencood.detection.postprocessing")
    target_postprocessor_name = str(postprocessor_name).replace("_", "").lower()
    stage1_postprocessor_class = None
    for name, cls in postprocessor_lib.__dict__.items():
        if name.lower() == target_postprocessor_name:
            stage1_postprocessor_class = cls
            break
    if stage1_postprocessor_class is None:
        raise ValueError("Cannot find postprocessor class: {}".format(postprocessor_name))
    return stage1_postprocessor_class(copy.deepcopy(hypes["postprocess"]), train=False)


def _build_single_cav_sample(dataset, base_data_dict, cav_id, sample_idx):
    selected_cav_base = copy.deepcopy(base_data_dict[cav_id])
    selected_cav_base["ego"] = True
    params = selected_cav_base.setdefault("params", {})
    if "lidar_pose_clean" not in params:
        params["lidar_pose_clean"] = params.get("lidar_pose")
    if "modality_name" not in selected_cav_base:
        raise KeyError("Missing modality_name for CAV {} in sample {}".format(cav_id, sample_idx))

    modality_name = selected_cav_base["modality_name"]
    sensor_type = dataset.sensor_type_dict[modality_name]
    if sensor_type != "camera":
        raise ValueError("Expected camera modality, got {} -> {}".format(modality_name, sensor_type))

    base_single = OrderedDict([(cav_id, selected_cav_base)])
    attach_pose_confidence(base_single)
    ego_cav_base = selected_cav_base
    selected_cav_processed = dataset.get_item_single_car(selected_cav_base, ego_cav_base)

    object_stack = [selected_cav_processed["object_bbx_center"]]
    object_id_stack = list(selected_cav_processed["object_ids"])
    unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
    object_stack = np.vstack(object_stack)
    object_stack = object_stack[unique_indices]

    max_num = int(dataset.params["postprocess"]["max_num"])
    object_bbx_center = np.zeros((max_num, 7))
    mask = np.zeros(max_num)
    object_bbx_center[: object_stack.shape[0], :] = object_stack
    mask[: object_stack.shape[0]] = 1
    label_dict = dataset.post_processor.generate_label(
        gt_box_center=object_bbx_center, anchors=dataset.anchor_box, mask=mask
    )

    processed_data_dict = OrderedDict()
    processed_data_dict["ego"] = {}
    for name in getattr(dataset, "modality_name_list", []):
        processed_data_dict["ego"][f"input_{name}"] = None

    merged_image_inputs_dict = merge_features_to_dict(
        [selected_cav_processed[f"image_inputs_{modality_name}"]],
        merge="stack",
    )
    processed_data_dict["ego"][f"input_{modality_name}"] = merged_image_inputs_dict
    processed_data_dict["ego"]["agent_modality_list"] = [modality_name]

    pairwise_t_matrix = get_pairwise_transformation(base_single, dataset.max_cav, dataset.proj_first)
    pose_confidence = np.asarray(
        [base_single[cav_id]["params"].get("pose_confidence", 1.0)],
        dtype=np.float32,
    )

    processed_data_dict["ego"].update(
        {
            "object_bbx_center": object_bbx_center,
            "object_bbx_mask": mask,
            "object_ids": [object_id_stack[i] for i in unique_indices],
            "anchor_box": dataset.anchor_box,
            "label_dict": label_dict,
            "cav_num": 1,
            "pairwise_t_matrix": pairwise_t_matrix,
            "lidar_poses_clean": np.asarray([params["lidar_pose_clean"]], dtype=np.float32).reshape(-1, 6),
            "lidar_poses": np.asarray([params["lidar_pose"]], dtype=np.float32).reshape(-1, 6),
            "pose_confidence": pose_confidence.reshape(-1),
            "sample_idx": sample_idx,
            "cav_id_list": [cav_id],
        }
    )
    if getattr(dataset, "supervise_single", False) or getattr(dataset, "heterogeneous", False):
        single_label_dicts = dataset.post_processor.collate_batch(
            [selected_cav_processed["single_label_dict"]]
        )
        processed_data_dict["ego"].update(
            {
                "single_label_dict_torch": single_label_dicts,
                "single_object_bbx_center_torch": torch.from_numpy(
                    np.asarray([selected_cav_processed["single_object_bbx_center"]])
                ),
                "single_object_bbx_mask_torch": torch.from_numpy(
                    np.asarray([selected_cav_processed["single_object_bbx_mask"]])
                ),
            }
        )
    return processed_data_dict


def _post_process_stage1_native(output, post_processor, anchor_box):
    if isinstance(output, dict) and "unc_preds" not in output:
        cls_preds_tensor = output.get("cls_preds")
        if cls_preds_tensor is not None:
            bsz, anchors, height, width = cls_preds_tensor.shape
            output["unc_preds"] = cls_preds_tensor.new_zeros(
                (bsz, anchors * 2, height, width),
                dtype=cls_preds_tensor.dtype,
                device=cls_preds_tensor.device,
            )

    stage1_outputs = post_processor.post_process_stage1(output, anchor_box)
    if not stage1_outputs:
        return [], [], [], []
    if isinstance(stage1_outputs, (list, tuple)) and len(stage1_outputs) == 3:
        pred_corner3d_list, pred_box3d_list, uncertainty_list = stage1_outputs
        score_list = None
    elif isinstance(stage1_outputs, (list, tuple)) and len(stage1_outputs) >= 4:
        pred_corner3d_list, pred_box3d_list, uncertainty_list, score_list = stage1_outputs[:4]
    else:
        raise ValueError("Unexpected post_process_stage1 output: {}".format(type(stage1_outputs)))

    if pred_corner3d_list is None:
        return [], [], [], []
    corners = pred_corner3d_list[0].detach().cpu().numpy().tolist()
    boxes = pred_box3d_list[0].detach().cpu().numpy().tolist()
    uncertainty = uncertainty_list[0].detach().cpu().numpy().tolist()
    scores = score_list[0].detach().cpu().numpy().tolist() if score_list is not None else []
    return corners, boxes, scores, uncertainty


def main():
    opt = _parse_args()
    if int(opt.num_shards) <= 0:
        raise ValueError("--num_shards must be >= 1")
    if int(opt.shard_index) < 0 or int(opt.shard_index) >= int(opt.num_shards):
        raise ValueError("--shard_index must be in [0, num_shards)")

    hypes = yaml_utils.load_yaml(opt.hypes_yaml, SimpleNamespace(model_dir=""))
    hypes, train_flag, split_name = _resolve_split_hypes(hypes, opt.split, opt.test_dir_override)

    if "camera" not in (hypes.get("input_source") or []):
        raise ValueError("Camera-model exporter requires input_source to include camera")
    if "depth" not in (hypes.get("input_source") or []):
        raise ValueError("Camera-model exporter requires input_source to include depth")

    if opt.score_threshold is not None:
        hypes["postprocess"]["target_args"]["score_threshold"] = float(opt.score_threshold)

    out_dir, split_dir, cache_path, manifest_path = _validate_output_paths(
        opt.output_dir,
        split_name,
        shard_index=opt.shard_index,
        num_shards=opt.num_shards,
    )
    dataset = build_dataset(hypes, visualize=False, train=train_flag)
    if not hasattr(dataset, "anchor_box"):
        dataset.anchor_box = dataset.post_processor.generate_anchor_box()
        dataset.anchor_box_torch = torch.from_numpy(dataset.anchor_box)

    depth_override_stats = _apply_depth_root_override(dataset, opt.depth_root_override)
    if depth_override_stats.get("missing", 0) > 0:
        raise FileNotFoundError(
            "Depth override has missing files: {}/{}".format(
                depth_override_stats.get("missing"), depth_override_stats.get("total")
            )
        )

    model = train_utils.create_model(hypes)
    model = _load_checkpoint_or_path(opt.checkpoint_path, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    stage1_postprocessor = _build_stage1_postprocessor(hypes)
    anchor_box = torch.from_numpy(stage1_postprocessor.generate_anchor_box()).to(device)

    total_len = len(dataset)
    max_samples = opt.max_samples if opt.max_samples is not None else total_len
    max_samples = min(int(max_samples), int(total_len))
    stage1 = OrderedDict()
    sample_stats = OrderedDict()
    started = time.time()

    total_to_process = sum(
        1
        for idx in range(max_samples)
        if int(opt.num_shards) == 1 or (idx % int(opt.num_shards)) == int(opt.shard_index)
    )

    with torch.no_grad():
        for idx in range(max_samples):
            if int(opt.num_shards) > 1 and (idx % int(opt.num_shards)) != int(opt.shard_index):
                continue
            scenario_folder, scenario_database, timestamp = _sample_record(dataset, idx)
            scenario_name = os.path.basename(os.path.normpath(scenario_folder))
            base_data_dict = dataset.retrieve_base_data(idx)

            base_for_pairwise = OrderedDict()
            cav_id_list = []
            pred_corner3d_np_list = []
            pred_box3d_np_list = []
            pred_score_np_list = []
            uncertainty_np_list = []
            lidar_pose_np = []
            lidar_pose_clean_np = []
            per_cav_stats = OrderedDict()

            for cav_id, cav_content in base_data_dict.items():
                sample = _build_single_cav_sample(dataset, base_data_dict, cav_id, idx)
                batch = dataset.collate_batch_test([sample])
                if batch is None:
                    corners, boxes, scores, uncertainty = [], [], [], []
                else:
                    batch = train_utils.to_device(batch, device)
                    output = model(batch["ego"])
                    corners, boxes, scores, uncertainty = _post_process_stage1_native(
                        output, stage1_postprocessor, anchor_box
                    )

                params = base_data_dict[cav_id].setdefault("params", {})
                cav_id_list.append(str(cav_id))
                pred_corner3d_np_list.append(corners)
                pred_box3d_np_list.append(boxes)
                pred_score_np_list.append(scores)
                uncertainty_np_list.append(uncertainty)
                lidar_pose_np.append(_to_jsonable(np.asarray(params.get("lidar_pose"), dtype=np.float32)))
                lidar_pose_clean_np.append(
                    _to_jsonable(np.asarray(params.get("lidar_pose_clean", params.get("lidar_pose")), dtype=np.float32))
                )
                per_cav_stats[str(cav_id)] = {
                    "boxes": int(len(pred_corner3d_np_list[-1])),
                    "modality_name": str(base_data_dict[cav_id]["modality_name"]),
                }
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
            rec["stage1_source"] = "image_depth_camera_model"
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
                    "[image-depth-camera-stage1] processed {}/{} samples (idx={}, cavs={}, boxes={})".format(
                        len(stage1), total_to_process, idx, len(cav_id_list), total_boxes
                    ),
                    flush=True,
                )

    split_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(stage1, f, sort_keys=True)

    manifest = OrderedDict()
    manifest["schema_version"] = "image_depth_camera_model_stage1_manifest_v1"
    manifest["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["elapsed_sec"] = float(time.time() - started)
    manifest["stage1_cache"] = str(cache_path)
    manifest["stage1_cache_basename"] = cache_path.name
    manifest["stage1_source"] = "image_depth_camera_model"
    manifest["dataset_root"] = str(dataset.root_dir)
    manifest["depth_root_override"] = str(Path(opt.depth_root_override).resolve()) if opt.depth_root_override else ""
    manifest["depth_override_stats"] = depth_override_stats
    manifest["hypes_yaml"] = str(Path(opt.hypes_yaml).resolve())
    manifest["checkpoint_path"] = str(Path(opt.checkpoint_path).resolve())
    manifest["split"] = split_name
    manifest["samples_written"] = int(len(stage1))
    manifest["dataset_len"] = int(total_len)
    manifest["max_samples"] = int(max_samples)
    manifest["num_shards"] = int(opt.num_shards)
    manifest["shard_index"] = int(opt.shard_index)
    manifest["indices_written"] = [int(k) for k in stage1.keys()]
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
