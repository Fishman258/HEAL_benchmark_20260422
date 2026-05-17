# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
from collections import OrderedDict
import hashlib
import json
import random
import os
import time
import warnings
import gzip

import torch
import open3d as o3d
from torch.utils.data import DataLoader
import numpy as np

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.extrinsics.path_utils import resolve_repo_path
from opencood.extrinsics.pose_correction.selection_policy import normalize_pose_selection_policy
from opencood.extrinsics.pose_correction import build_pose_corrector, run_pose_solver
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.utils.common_utils import read_json
from opencood.utils.transformation_utils import pose_to_tfm, x1_to_x2
from opencood.visualization import vis_utils, my_vis, simple_vis

torch.multiprocessing.set_sharing_strategy('file_system')
# This script can emit extremely noisy warnings (e.g., torch deprecated sigmoid)
# at every iteration, which slows evaluation drastically and floods stdout.
warnings.filterwarnings("ignore")

def _parse_float_list(raw: str):
    values = []
    for token in (raw or "").split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values


def _parse_hw(raw: str):
    raw = str(raw or "").strip()
    if not raw:
        return None
    if "," in raw:
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) == 2:
            try:
                return int(parts[0]), int(parts[1])
            except Exception:
                return None
    try:
        side = int(raw)
    except Exception:
        return None
    return side, side


def _parse_int_set(raw: str):
    values = set()
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.add(int(token))
        except Exception:
            continue
    return values


def _sha256_bytes(payload):
    h = hashlib.sha256()
    h.update(payload)
    return h.hexdigest()


def _jsonify_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonify_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_value(v) for v in value]
    if torch.is_tensor(value):
        return _jsonify_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonify_value(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if hasattr(value, "tolist"):
        try:
            return _jsonify_value(value.tolist())
        except Exception:
            pass
    return str(value)


def _normalize_pose_override_map_for_json(override_map):
    if not isinstance(override_map, dict):
        return {}
    return {str(sample_idx): _jsonify_value(entry) for sample_idx, entry in override_map.items()}


def _filename_number_tag(value):
    text = "{:g}".format(float(value))
    return text.replace("-", "m").replace(".", "p")


def _resolve_pose_override_export_path(export_path, *, pos_std, rot_std, use_laplace=False, multiple_exports=False):
    resolved = os.path.abspath(os.path.expanduser(str(export_path)))
    if not multiple_exports:
        return resolved
    root, ext = os.path.splitext(resolved)
    if not ext:
        ext = ".json"
    suffix = f"_pos{_filename_number_tag(pos_std)}_rot{_filename_number_tag(rot_std)}"
    if use_laplace:
        suffix += "_laplace"
    return f"{root}{suffix}{ext}"


def _repo_root_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def _benchmark_output_dir():
    root_dir = os.path.join(_repo_root_dir(), "outputs")
    os.makedirs(root_dir, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root_dir, f"run_{run_stamp}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _override_opv2v_depth_root(dataset, depth_root):
    depth_root = str(depth_root or "").strip()
    if not depth_root:
        return {"enabled": False, "missing": 0, "total": 0}
    depth_root = os.path.abspath(os.path.expanduser(depth_root))
    total = 0
    missing = 0
    root_dir = str(getattr(dataset, "root_dir", ""))
    scenario_folders = list(getattr(dataset, "scenario_folders", []) or [])
    scenario_databases = list((getattr(dataset, "scenario_database", {}) or {}).values())
    for scenario_folder, scenario_database in zip(scenario_folders, scenario_databases):
        if not isinstance(scenario_database, dict):
            continue
        for cav_id, cav_content in scenario_database.items():
            if cav_id == "ego" or not isinstance(cav_content, dict):
                continue
            cav_path = os.path.join(str(scenario_folder), str(cav_id))
            try:
                rel = os.path.relpath(cav_path, root_dir)
            except Exception:
                rel = os.path.join(os.path.basename(str(scenario_folder)), str(cav_id))
            for timestamp, item in list(cav_content.items()):
                if timestamp == "ego" or not isinstance(item, dict):
                    continue
                depth_files = [
                    os.path.join(depth_root, rel, "{}_camera{}_depth.npy".format(timestamp, cam_id))
                    for cam_id in range(4)
                ]
                total += len(depth_files)
                missing += sum(0 if os.path.exists(p) else 1 for p in depth_files)
                item["depths"] = depth_files
    return {"enabled": True, "root": depth_root, "missing": int(missing), "total": int(total)}


def _sync_late_fusion_transformations_from_pose(batch_data):
    """
    Late fusion projects each CAV's boxes with per-CAV transformation_matrix.
    Runtime pose correction updates ego['lidar_pose']; mirror that update back
    into late-fusion matrices so correction affects the actual fused boxes.
    """
    if not isinstance(batch_data, dict):
        return batch_data
    ego = batch_data.get("ego")
    if not isinstance(ego, dict):
        return batch_data
    lidar_pose = ego.get("lidar_pose")
    if not torch.is_tensor(lidar_pose) or lidar_pose.ndim < 2 or int(lidar_pose.shape[0]) <= 0:
        return batch_data
    cav_id_list = ego.get("cav_id_list")
    if not isinstance(cav_id_list, (list, tuple)) or not cav_id_list:
        return batch_data

    poses_np = lidar_pose.detach().cpu().numpy().reshape(-1, lidar_pose.shape[-1])
    clean_pose = ego.get("lidar_pose_clean")
    clean_np = None
    if torch.is_tensor(clean_pose) and clean_pose.ndim >= 2 and clean_pose.shape[0] >= lidar_pose.shape[0]:
        clean_np = clean_pose.detach().cpu().numpy().reshape(-1, clean_pose.shape[-1])

    for cav_idx, cav_id in enumerate(list(cav_id_list)[: poses_np.shape[0]]):
        key = "ego" if cav_idx == 0 else cav_id
        cav = batch_data.get(key)
        if not isinstance(cav, dict):
            continue
        mat = x1_to_x2(poses_np[cav_idx], poses_np[0])
        cav["transformation_matrix"] = torch.as_tensor(
            mat, device=lidar_pose.device, dtype=lidar_pose.dtype
        )
        if clean_np is not None and cav_idx < clean_np.shape[0]:
            clean_mat = x1_to_x2(clean_np[cav_idx], clean_np[0])
            cav["transformation_matrix_clean"] = torch.as_tensor(
                clean_mat, device=lidar_pose.device, dtype=lidar_pose.dtype
            )
    return batch_data


def _write_pose_override_map_json(path, override_map):
    payload = _normalize_pose_override_map_for_json(override_map)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2, sort_keys=True)
        fp.write("\n")
    return payload


def _append_jsonl(path, records):
    if not path or not records:
        return 0
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    count = 0
    with opener(path, "at", encoding="utf-8") as fp:
        for rec in records:
            if not isinstance(rec, dict):
                continue
            fp.write(json.dumps(_jsonify_value(rec), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _pose_no_fallback_all_non_ego_applied(timing_payload) -> bool:
    """
    Strict no-fallback gate for online pose correction.

    Sample-level `pose_provider_applied=True` is too weak for zero-interference:
    if one non-ego CAV is corrected but another still flows through with noisy
    extrinsics, the remaining noisy branch can still bend the curve. When
    per-pair stats are available, require all non-ego pairs to be corrected
    before allowing cooperative fusion to continue.
    """
    if not isinstance(timing_payload, dict):
        return False

    applied_any = bool(timing_payload.get("pose_provider_applied", False))
    if not applied_any:
        return False

    pair_total = timing_payload.get("pose_corr_pair_total_count")
    pair_applied = timing_payload.get("pose_corr_applied_pair_count")
    try:
        pair_total_i = int(round(float(pair_total)))
        pair_applied_i = int(round(float(pair_applied)))
    except Exception:
        # Older payloads only expose the coarse sample-level bool.
        return applied_any

    if pair_total_i <= 0:
        return True
    return pair_applied_i >= pair_total_i


def _numeric_stats(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "abs_sum": 0.0,
        }
    flat = arr.astype(np.float64, copy=False).reshape(-1)
    return {
        "count": int(flat.size),
        "min": _safe_float(np.min(flat)),
        "max": _safe_float(np.max(flat)),
        "mean": _safe_float(np.mean(flat)),
        "std": _safe_float(np.std(flat)),
        "abs_sum": _safe_float(np.sum(np.abs(flat))),
    }


def _quantized_sha256(arr, decimals):
    arr = np.asarray(arr)
    if arr.size == 0:
        return _sha256_bytes(b"")
    q = np.round(arr.astype(np.float32, copy=False), decimals=int(decimals))
    q = np.ascontiguousarray(q)
    return _sha256_bytes(q.tobytes())


def _tensor_digest(tensor):
    if not torch.is_tensor(tensor):
        return None
    t = tensor.detach().cpu().contiguous()
    arr = t.numpy()
    out = {
        "kind": "tensor",
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "sha256": _sha256_bytes(arr.tobytes()),
    }
    if np.issubdtype(arr.dtype, np.number):
        out["stats"] = _numeric_stats(arr)
        if np.issubdtype(arr.dtype, np.floating):
            out["sha256_round_4"] = _quantized_sha256(arr, 4)
            out["sha256_round_3"] = _quantized_sha256(arr, 3)
    return out


def _ndarray_digest(arr):
    arr = np.ascontiguousarray(arr)
    out = {
        "kind": "ndarray",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": _sha256_bytes(arr.tobytes()),
    }
    if np.issubdtype(arr.dtype, np.number):
        out["stats"] = _numeric_stats(arr)
        if np.issubdtype(arr.dtype, np.floating):
            out["sha256_round_4"] = _quantized_sha256(arr, 4)
            out["sha256_round_3"] = _quantized_sha256(arr, 3)
    return out


def _voxel_dict_digest(value):
    coords = value.get("voxel_coords")
    feats = value.get("voxel_features")
    nums = value.get("voxel_num_points")
    if not (torch.is_tensor(coords) and torch.is_tensor(feats) and torch.is_tensor(nums)):
        return None
    coords_np = coords.detach().cpu().numpy()
    feats_np = feats.detach().cpu().numpy()
    nums_np = nums.detach().cpu().numpy()
    if coords_np.ndim != 2 or feats_np.ndim < 1 or nums_np.ndim < 1:
        return None
    if not (coords_np.shape[0] == feats_np.shape[0] == nums_np.shape[0]):
        return None
    if coords_np.shape[0] == 0:
        order = np.empty((0,), dtype=np.int64)
    else:
        sort_keys = tuple(coords_np[:, col] for col in reversed(range(coords_np.shape[1])))
        order = np.lexsort(sort_keys)
    coords_sorted = np.ascontiguousarray(coords_np[order])
    feats_sorted = np.ascontiguousarray(feats_np[order])
    nums_sorted = np.ascontiguousarray(nums_np[order])
    feats_canonical = feats_sorted.copy()
    if feats_canonical.ndim == 3 and feats_canonical.shape[1] > 1:
        for idx in range(feats_canonical.shape[0]):
            pts = feats_canonical[idx]
            sort_keys = tuple(pts[:, col] for col in reversed(range(pts.shape[1])))
            pt_order = np.lexsort(sort_keys)
            feats_canonical[idx] = pts[pt_order]
    return {
        "kind": "voxel_dict",
        "rows": int(coords_np.shape[0]),
        "coords": _ndarray_digest(coords_sorted),
        "features": _ndarray_digest(feats_canonical),
        "num_points": _ndarray_digest(nums_sorted),
    }


def _scalar_preview(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(type(value).__name__)


def _summarize_audit_value(value, depth=0):
    if torch.is_tensor(value):
        return _tensor_digest(value)
    if isinstance(value, np.ndarray):
        return _ndarray_digest(value)
    if isinstance(value, dict):
        voxel_digest = _voxel_dict_digest(value)
        if voxel_digest is not None:
            return voxel_digest
        keys = sorted([str(k) for k in value.keys()])
        summary = {"kind": "dict", "keys": keys}
        if depth < 2:
            summary["items"] = {
                str(k): _summarize_audit_value(v, depth + 1)
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            }
        return summary
    if isinstance(value, (list, tuple)):
        summary = {"kind": type(value).__name__, "len": len(value)}
        if depth < 2:
            if all(not torch.is_tensor(v) and not isinstance(v, (dict, list, tuple, np.ndarray)) for v in value[:8]):
                summary["head"] = [_scalar_preview(v) for v in value[:8]]
            else:
                summary["items"] = [_summarize_audit_value(v, depth + 1) for v in value[:4]]
        return summary
    return {"kind": type(value).__name__, "value": _scalar_preview(value)}


def _record_len_to_list(record_len):
    if torch.is_tensor(record_len):
        return [int(x) for x in record_len.detach().cpu().reshape(-1).tolist()]
    if isinstance(record_len, (list, tuple)):
        out = []
        for item in record_len:
            try:
                out.append(int(item))
            except Exception:
                pass
        return out
    return []


def _norm_text(value, default=""):
    return str(value if value is not None else default).strip().lower()


def _resolve_pose_selection_policy(opt) -> str:
    raw = str(getattr(opt, "pose_selection_policy", "") or "").strip()
    if not raw:
        raw = "compare_current_proxy" if bool(getattr(opt, "pose_compare_current", False)) else "solver_only"
    return normalize_pose_selection_policy(raw)


def _build_zero_interference_attestation(opt, hypes, pose_correction):
    enabled = bool(getattr(opt, "pose_assert_zero_interference", False))
    pose_provider_cfg = dict(hypes.get("pose_provider") or {}) if isinstance(hypes, dict) else {}
    pose_override_cfg = dict(hypes.get("pose_override") or {}) if isinstance(hypes, dict) else {}
    online_args = dict(pose_provider_cfg.get("online_args") or {})
    pose_selection_policy = _resolve_pose_selection_policy(opt)

    observed = {
        "pose_correction": str(pose_correction or ""),
        "runtime_mode": _norm_text(pose_provider_cfg.get("runtime_mode")),
        "solver_backend": _norm_text(pose_provider_cfg.get("solver_backend"), "offline_map"),
        "pose_source": _norm_text(
            pose_provider_cfg.get("pose_source"),
            getattr(opt, "pose_source", "noisy_input"),
        ),
        "online_method": _norm_text(pose_provider_cfg.get("online_method")),
        "online_mode": _norm_text(online_args.get("mode")),
        "recompute_pairwise": bool(pose_provider_cfg.get("recompute_pairwise", False)),
        "pose_override_enabled": bool(pose_override_cfg.get("enabled", False)),
        "comm_range_gating_requested": _norm_text(getattr(opt, "comm_range_gating", "auto"), "auto"),
        "comm_range_use_clean_pose_effective": bool(hypes.get("comm_range_use_clean_pose", False)),
        "noise_target": _norm_text(getattr(opt, "noise_target", "all"), "all"),
        "sweep_mode": _norm_text(getattr(opt, "sweep_mode", "paired"), "paired"),
        "pose_timing_enabled": bool(getattr(opt, "pose_timing", False)),
        "pose_compare_current": bool(getattr(opt, "pose_compare_current", False)),
        "pose_selection_policy": pose_selection_policy,
        "vips_use_prior": bool(getattr(opt, "vips_use_prior", False)),
        "cbm_use_prior": bool(getattr(opt, "cbm_use_prior", False)),
        "pose_no_fallback": bool(getattr(opt, "pose_no_fallback", False)),
        "require_no_fallback": bool(getattr(opt, "pose_assert_require_no_fallback", False)),
        "stage1_result_present": bool(str(getattr(opt, "stage1_result", "") or "").strip()),
        "force_pose_confidence": getattr(opt, "force_pose_confidence", None),
    }

    if not enabled:
        return {
            "schema_version": "zero_interference_noinit_contract_v1",
            "enabled": False,
            "verdict": "SKIP",
            "attested_strategy": None,
            "errors": [],
            "observed": observed,
        }

    errors = []
    pose_corr_txt = _norm_text(pose_correction)
    if not pose_corr_txt or pose_corr_txt == "none":
        errors.append("pose_correction must be a strict no-init method, got none")
    if pose_corr_txt.endswith("stable"):
        errors.append("stable pose_correction is forbidden for zero-interference")
    if observed["runtime_mode"] != "register_and_fuse":
        errors.append("runtime_mode must be register_and_fuse")
    if observed["solver_backend"] != "online_box":
        errors.append("solver_backend must be online_box")
    if observed["pose_source"] not in {"noisy_input", "identity"}:
        errors.append("pose_source must be noisy_input or identity")
    if observed["online_mode"] not in {"initfree", "gt"}:
        errors.append("online pose mode must be initfree (or gt for oracle)")
    if not observed["recompute_pairwise"]:
        errors.append("recompute_pairwise must be enabled")
    if observed["pose_override_enabled"]:
        errors.append("dataset-side pose_override must be disabled when runtime attestation is enabled")
    if observed["comm_range_gating_requested"] != "clean":
        errors.append("comm_range_gating must be explicitly set to clean")
    if not observed["comm_range_use_clean_pose_effective"]:
        errors.append("comm_range_use_clean_pose must be effective at runtime")
    if observed["noise_target"] != "non-ego":
        errors.append("noise_target must be non-ego")
    if observed["sweep_mode"] != "paired":
        errors.append("sweep_mode must be paired")
    if not observed["pose_timing_enabled"]:
        errors.append("pose_timing must be enabled so runtime evidence is written to YAML")
    if observed["pose_compare_current"]:
        errors.append("compare-current is forbidden")
    if observed["pose_selection_policy"] != "solver_only":
        errors.append("pose_selection_policy must be solver_only")
    if observed["vips_use_prior"]:
        errors.append("vips prior is forbidden")
    if observed["cbm_use_prior"]:
        errors.append("cbm prior is forbidden")
    if pose_corr_txt not in {"oracle_gt"} and not observed["stage1_result_present"]:
        errors.append("stage1_result is required for strict no-init methods")
    if observed["require_no_fallback"] and not observed["pose_no_fallback"]:
        errors.append("pose_no_fallback must be enabled for this attested run")

    if observed["pose_source"] == "identity":
        attested_strategy = "noinit_identity"
    else:
        attested_strategy = "noinit_nofallback" if observed["pose_no_fallback"] else "noinit"
    return {
        "schema_version": "zero_interference_noinit_contract_v1",
        "enabled": True,
        "verdict": "PASS" if not errors else "FAIL",
        "attested_strategy": attested_strategy,
        "errors": errors,
        "observed": observed,
    }


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--also_laplace', action='store_true',
                        help="whether to use laplace to simulate noise. Otherwise Gaussian")
    parser.add_argument('--fusion_method', type=str,
                        default='intermediate',
                        help='no, no_w_uncertainty, late, early or intermediate')
    parser.add_argument('--save_vis_interval', type=int, default=40,
                        help='save how many numbers of visualization result?')
    parser.add_argument('--note', default="", type=str, help="any other thing?")
    parser.add_argument(
        "--test-dir-override",
        type=str,
        default="",
        help="Override hypes['test_dir'] from checkpoint config, useful for immutable converted benchmark views.",
    )
    parser.add_argument(
        "--validate-dir-override",
        type=str,
        default="",
        help="Override hypes['validate_dir']; defaults to --test-dir-override when omitted.",
    )
    parser.add_argument(
        "--depth-root-override",
        type=str,
        default="",
        help=(
            "Override OPV2V depth file root for camera+depth configs. "
            "Expected layout mirrors the OPV2V split: {scenario}/{cav}/{timestamp}_camera{i}_depth.npy."
        ),
    )
    parser.add_argument(
        "--max-cav-override",
        type=int,
        default=0,
        help="Override hypes['train_params']['max_cav']; use 2 for paired OPV2V views.",
    )
    parser.add_argument(
        "--pos-std-list",
        type=str,
        default="0,0.2,0.4,0.6",
        help="Comma-separated translation noise std list (meters).",
    )
    parser.add_argument(
        "--rot-std-list",
        type=str,
        default="0,0.2,0.4,0.6",
        help="Comma-separated yaw noise std list (degrees).",
    )
    parser.add_argument(
        "--sweep-mode",
        choices=["paired", "grid"],
        default="paired",
        help="paired: zip(pos,rot); grid: Cartesian product.",
    )
    parser.add_argument(
        "--noise-target",
        choices=["all", "ego", "non-ego"],
        default="non-ego",
        help="Which agent(s) receive synthetic pose noise. "
             "'all' matches legacy behavior; 'non-ego' makes relative error scale match the requested std.",
    )
    parser.add_argument(
        "--pose-dropout-prob",
        type=float,
        default=0.0,
        help="Probability of simulating localization dropout per sample. "
             "When triggered, poses reuse the last noisy state (if available).",
    )
    parser.add_argument(
        "--pose-correction",
        choices=[
            "none",
            "v2xregpp_initfree",
            "v2xregpp_stable",
            "freealign_paper",
            "freealign_paper_stable",
            "freealign_repo",
            "freealign_repo_stable",
            "vips_initfree",
            "vips_stable",
            "cbm_initfree",
            "cbm_stable",
            "image_match_initfree",
            "image_match_stable",
            "lidar_reg_initfree",
            "lidar_reg_stable",
            # V2VLoc-style pose override/correction from cached pose JSON.
            "v2vloc_pgc_initfree",
            "v2vloc_pgc_stable",
            # Oracle pose override from clean poses (GT).
            "v2vloc_oracle_initfree",
            "v2vloc_oracle_stable",
            # GT pose override from dataset clean poses.
            "oracle_gt",
        ],
        default="none",
        help="Optional extrinsic correction inside the dataset, before building pairwise transforms.",
    )
    parser.add_argument(
        "--pose-device",
        type=str,
        default="auto",
        help="Device for pose correction (v2xregpp/freealign/vips/cbm). "
             "Use 'auto' to pick cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--runtime-mode",
        type=str,
        default="",
        choices=["", "single_only", "fusion_only", "register_only", "register_and_fuse"],
        help="Optional unified runtime mode override for pose_provider."
             " Empty keeps legacy behavior.",
    )
    parser.add_argument(
        "--solver-backend",
        type=str,
        default="offline_map",
        choices=["offline_map", "online_box", "online_box_feat_refine"],
        help="Pose solver backend. offline_map keeps legacy pre-pass; online_* runs solver inside pose_provider.",
    )
    parser.add_argument(
        "--online-gpu-stage1-solver",
        action="store_true",
        help="Enable experimental GPU stage1 solver path when using online backends.",
    )
    parser.add_argument(
        "--online-skip-pairwise-rebuild",
        action="store_true",
        help=(
            "When using online backends, reuse dataset pairwise instead of runtime rebuild. "
            "Useful for strict oracle parity checks."
        ),
    )
    parser.add_argument(
        "--online-refine-steps",
        type=int,
        default=4,
        help="online_box_feat_refine: local SE(2) refinement iterations.",
    )
    parser.add_argument(
        "--online-refine-init-step-xy",
        type=float,
        default=0.4,
        help="online_box_feat_refine: initial translation search step (meters).",
    )
    parser.add_argument(
        "--online-refine-init-step-yaw",
        type=float,
        default=1.5,
        help="online_box_feat_refine: initial yaw search step (degrees).",
    )
    parser.add_argument(
        "--online-refine-decay",
        type=float,
        default=0.6,
        help="online_box_feat_refine: per-iteration step decay factor.",
    )
    parser.add_argument(
        "--online-refine-min-step-xy",
        type=float,
        default=0.03,
        help="online_box_feat_refine: minimum translation step (meters).",
    )
    parser.add_argument(
        "--online-refine-min-step-yaw",
        type=float,
        default=0.15,
        help="online_box_feat_refine: minimum yaw step (degrees).",
    )
    parser.add_argument(
        "--online-refine-min-improvement",
        type=float,
        default=1e-4,
        help="online_box_feat_refine: minimum residual improvement per update (meters).",
    )
    parser.add_argument(
        "--deterministic-strict",
        action="store_true",
        help=(
            "Enable stricter determinism controls for parity gates. "
            "This may reduce throughput and can raise errors if nondeterministic ops are used."
        ),
    )
    parser.add_argument(
        "--pose-source",
        type=str,
        default="noisy_input",
        choices=["noisy_input", "gt", "identity"],
        help="Pose source for fusion_only runtime mode.",
    )
    parser.add_argument(
        "--stage1-result",
        type=str,
        default="",
        help="Cache JSON used by the pose corrector (required when --pose-correction != none). "
             "For v2xregpp_* / freealign_* this is a stage1_boxes.json. For v2vloc_pgc_* it is a PGC pose json; "
             "for v2vloc_oracle_* it can reuse stage1_boxes.json (lidar_pose_clean_np). "
             "oracle_gt ignores this path.",
    )
    parser.add_argument(
        "--pose-override-path",
        type=str,
        default="",
        help=(
            "Optional dataset-side pose override JSON (load_pose_override_map format). "
            "Use this to run downstream cooperative perception from a precomputed registration cache."
        ),
    )
    parser.add_argument(
        "--freealign-max-boxes",
        type=int,
        default=60,
        help="FreeAlign: max number of boxes per agent used for matching.",
    )
    parser.add_argument(
        "--freealign-min-nodes",
        type=int,
        default=5,
        help="FreeAlign(paper): minimum number of nodes/boxes required to attempt matching.",
    )
    parser.add_argument(
        "--freealign-sim-threshold",
        type=float,
        default=0.6,
        help="FreeAlign(paper): similarity threshold used for anchor selection/matching.",
    )
    parser.add_argument(
        "--freealign-affine-method",
        type=str,
        default="lmeds",
        choices=["lmeds", "ransac"],
        help="FreeAlign(paper): OpenCV estimateAffinePartial2D method.",
    )
    parser.add_argument(
        "--freealign-ransac-reproj-threshold",
        type=float,
        default=1.0,
        help="FreeAlign(paper): RANSAC reprojection threshold (meters) in estimateAffinePartial2D.",
    )
    parser.add_argument(
        "--freealign-min-anchors",
        type=int,
        default=3,
        help="FreeAlign(repo): minimum anchors for initial subgraph match.",
    )
    parser.add_argument(
        "--freealign-anchor-error",
        type=float,
        default=0.3,
        help="FreeAlign(repo): anchor error threshold.",
    )
    parser.add_argument(
        "--freealign-box-error",
        type=float,
        default=0.5,
        help="FreeAlign(repo): box error threshold.",
    )
    parser.add_argument(
        "--v2xregpp-config",
        type=str,
        default="configs/dair/midfusion/pipeline_midfusion_detection_occ.yaml",
        help="V2X-Reg++ pipeline config used by the pose corrector.",
    )
    parser.add_argument(
        "--v2xregpp-stage1-field",
        type=str,
        default="pred_corner3d_np_list",
        help="Stage1 result field name that contains per-agent 3D boxes (e.g., pred_corner3d_np_list, feature_corner3d_np_list).",
    )
    parser.add_argument(
        "--v2xregpp-bbox-type",
        type=str,
        default="detected",
        help="BBox type tag used inside V2X-Reg++ matching thresholds (e.g., detected, feature).",
    )
    parser.add_argument(
        "--v2xregpp-use-occ-hint",
        action="store_true",
        help="Enable occ-hint seed when stage1_result contains occ_map_level0/paths.",
    )
    parser.add_argument(
        "--v2xregpp-use-occ-pose",
        action="store_true",
        help="Treat occ-hint as a first-class pose candidate (and allow occ-only pose updates when boxes are sparse).",
    )
    parser.add_argument(
        "--v2xregpp-force-occ-pose",
        action="store_true",
        help="Always use occ-based pose (optionally refined), making the output independent of injected pose noise.",
    )
    parser.add_argument(
        "--v2xregpp-occ-from-lidar",
        action="store_true",
        help="Build BEV occupancy maps from raw lidar points (ignores stage1 occ_map fields).",
    )
    parser.add_argument(
        "--v2xregpp-occ-grid",
        type=str,
        default="256",
        help="Occupancy grid size for --v2xregpp-occ-from-lidar (int or 'H,W').",
    )
    parser.add_argument(
        "--v2xregpp-occ-max-delta-xy",
        type=float,
        default=20.0,
        help="Reject occ-hint if it differs from current pose by more than this translation (meters).",
    )
    parser.add_argument(
        "--v2xregpp-occ-max-delta-yaw",
        type=float,
        default=45.0,
        help="Reject occ-hint if it differs from current pose by more than this yaw (degrees).",
    )
    parser.add_argument(
        "--v2xregpp-icp-refine",
        action="store_true",
        help="Run Open3D ICP refinement on raw lidar points using the estimated transform as init.",
    )
    parser.add_argument("--v2xregpp-icp-voxel", type=float, default=1.0, help="ICP voxel downsample size (meters).")
    parser.add_argument("--v2xregpp-icp-max-corr", type=float, default=2.0, help="ICP max correspondence distance (meters).")
    parser.add_argument("--v2xregpp-icp-max-iter", type=int, default=30, help="ICP max iterations.")
    parser.add_argument("--v2xregpp-min-matches", type=int, default=3)
    parser.add_argument("--v2xregpp-min-stability", type=float, default=0.0)
    parser.add_argument(
        "--v2xregpp-min-precision",
        type=float,
        default=0.0,
        help="Absolute precision threshold (CorrespondingDetector) required to apply an estimated pose. 0 disables.",
    )
    parser.add_argument("--v2xregpp-ema-alpha", type=float, default=0.5)
    parser.add_argument("--v2xregpp-max-step-xy", type=float, default=3.0)
    parser.add_argument("--v2xregpp-max-step-yaw", type=float, default=10.0)
    parser.add_argument("--pose-compare-current", action="store_true", help="Compare estimated pose with current pose and keep the better one.")
    parser.add_argument(
        "--pose-selection-policy",
        type=str,
        default="",
        choices=["", "solver_only", "compare_current_proxy", "choose_better_pose_error"],
        help="Explicit policy for selecting between current/init pose and estimated pose.",
    )
    parser.add_argument("--pose-compare-distance-threshold", type=float, default=3.0, help="Distance threshold for pose comparison (meters).")
    parser.add_argument(
        "--pose-current-precision-threshold",
        type=float,
        default=None,
        help=(
            "Skip update if current precision exceeds this (set <0 to disable). "
            "Note: CorrespondingDetector precision is in [-3,0] (0 best, -3=no matches); "
            "for this threshold we use a shifted quality score in [0,3] (quality = precision + 3)."
        ),
    )
    parser.add_argument("--pose-min-precision-improvement", type=float, default=None, help="Minimum precision improvement over current pose.")
    parser.add_argument("--pose-min-matched-improvement", type=int, default=None, help="Minimum matched-count improvement over current pose.")
    parser.add_argument("--pose-min-precision", type=float, default=None, help="Absolute minimum precision required to apply estimated pose.")
    parser.add_argument(
        "--pose-solver-only",
        action="store_true",
        help=(
            "Run only the offline pose-solver pre-pass (stage1 matching + extrinsic estimation) and exit. "
            "Useful for exporting raw matches / raw estimated extrinsics without running AP inference."
        ),
    )
    parser.add_argument(
        "--pose-override-export-path",
        type=str,
        default="",
        help=(
            "Optional JSON path to dump offline_map solver_result.overrides in load_pose_override_map format "
            "(top-level sample_idx -> entry)."
        ),
    )
    parser.add_argument(
        "--force-pose-confidence",
        type=float,
        default=None,
        help=(
            "If set, overwrite per-agent pose_confidence with a constant after noise/override application. "
            "Use this to disable clean-pose-derived confidence leakage during evaluation."
        ),
    )
    parser.add_argument(
        "--pose-telemetry-dir",
        type=str,
        default="",
        help=(
            "Optional directory to export per-sample pose telemetry as JSONL(.gz). "
            "When set, stage1 pose correctors will record raw estimated rel_T and match pairs."
        ),
    )
    parser.add_argument(
        "--pose-telemetry-distance-threshold",
        type=float,
        default=3.0,
        help="Distance threshold (meters) used to compute telemetry match pairs.",
    )
    parser.add_argument(
        "--pose-telemetry-max-pairs",
        type=int,
        default=50,
        help="Top-k match pairs to store per sample (0 disables storing the pairs list).",
    )
    parser.add_argument(
        "--effective-input-audit-dir",
        type=str,
        default="",
        help=(
            "Optional directory to export per-sample effective-input audit JSONL. "
            "This records post-provider batch inputs plus model-side pre/post-fusion feature digests "
            "for selected eval sample indices."
        ),
    )
    parser.add_argument(
        "--effective-input-audit-samples",
        type=str,
        default="",
        help="Comma-separated eval sample indices to audit (loop index i). Empty disables filtering.",
    )
    parser.add_argument(
        "--effective-input-audit-max-samples",
        type=int,
        default=0,
        help="Optional cap on the number of audited samples for this invocation. 0 disables the cap.",
    )
    parser.add_argument(
        "--effective-input-audit-stop-after-write",
        action="store_true",
        help="Exit early once the requested effective-input audit samples have been written.",
    )
    parser.add_argument("--vips-use-prior", action="store_true", help="VIPS: use current pose as initialization.")
    parser.add_argument("--vips-match-threshold", type=float, default=0.5, help="VIPS: matching threshold.")
    parser.add_argument("--vips-match-distance", type=float, default=8.0, help="VIPS: match distance threshold (meters).")
    parser.add_argument("--cbm-use-prior", action="store_true", help="CBM: use current pose as initialization.")
    parser.add_argument("--cbm-sigma1-deg", type=float, default=10.0, help="CBM: sigma1 (deg).")
    parser.add_argument("--cbm-sigma2-m", type=float, default=3.0, help="CBM: sigma2 (meters).")
    parser.add_argument("--cbm-sigma3-m", type=float, default=1.0, help="CBM: sigma3 (meters).")
    parser.add_argument("--cbm-absolute-dis-lim", type=float, default=20.0, help="CBM: absolute distance limit (meters).")
    parser.add_argument(
        "--image-match-matcher",
        type=str,
        default="orb",
        choices=["orb", "sift", "loftr", "disk", "lightglue"],
    )
    parser.add_argument("--image-match-max-features", type=int, default=4000)
    parser.add_argument("--image-match-ratio-test", type=float, default=0.75)
    parser.add_argument("--image-match-cross-check", action="store_true")
    parser.add_argument("--image-match-ransac-thresh", type=float, default=1.0)
    parser.add_argument("--image-match-ransac-confidence", type=float, default=0.999)
    parser.add_argument("--image-match-ransac-max-iters", type=int, default=2000)
    parser.add_argument("--image-match-min-matches", type=int, default=20)
    parser.add_argument("--image-match-min-inliers", type=int, default=15)
    parser.add_argument("--image-match-resize-max-dim", type=int, default=1024)
    parser.add_argument("--image-match-allow-no-intrinsics", action="store_true")
    parser.add_argument("--image-match-t-scale", type=float, default=None)
    parser.add_argument("--image-match-device", type=str, default="cpu")
    parser.add_argument("--image-match-camera-index", type=int, default=0)
    parser.add_argument("--image-match-camera-indices", type=str, default="")
    parser.add_argument("--image-match-try-all-cameras", action="store_true")
    parser.add_argument("--image-match-init-source", type=str, default="current", choices=["current", "clean", "none"])
    parser.add_argument("--image-match-min-stability", type=float, default=0.0)
    parser.add_argument("--lidar-reg-voxel-size", type=float, default=1.0)
    parser.add_argument("--lidar-reg-max-corr", type=float, default=2.0)
    parser.add_argument("--lidar-reg-ransac-n", type=int, default=4)
    parser.add_argument("--lidar-reg-ransac-max-iter", type=int, default=50000)
    parser.add_argument("--lidar-reg-ransac-confidence", type=float, default=0.999)
    parser.add_argument("--lidar-reg-use-fgr", action="store_true")
    parser.add_argument(
        "--lidar-reg-global-method",
        type=str,
        default="auto",
        choices=["auto", "ransac", "fgr", "teaser_gnctls", "teaser_fgr", "teaser_quatro"],
        help="Global registration backend before ICP.",
    )
    parser.add_argument(
        "--lidar-reg-icp-method",
        type=str,
        default="point_to_plane",
        choices=["point_to_plane", "point_to_point", "gicp"],
    )
    parser.add_argument("--lidar-reg-icp-max-iter", type=int, default=50)
    parser.add_argument("--lidar-reg-min-points", type=int, default=200)
    parser.add_argument("--lidar-reg-max-points", type=int, default=60000)
    parser.add_argument("--lidar-reg-min-fitness", type=float, default=0.0)
    parser.add_argument("--lidar-reg-max-inlier-rmse", type=float, default=0.0)
    parser.add_argument("--lidar-reg-teaser-noise-bound", type=float, default=2.0)
    parser.add_argument("--lidar-reg-teaser-max-correspondences", type=int, default=8000)
    parser.add_argument(
        "--lidar-reg-cache",
        type=str,
        default=None,
        help="Optional JSON cache for lidar_reg/hkust methods (keyed by sample|ego|cav) to avoid re-running global registration.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers. For v2xregpp_stable, keep this at 0 to preserve temporal state.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional cap on evaluated samples per noise setting (debug only).",
    )
    parser.add_argument(
        "--eval-sample-start",
        type=int,
        default=0,
        help=(
            "Skip the first N samples of the DataLoader before counting --max-eval-samples. "
            "Useful for smoke runs when the dataset head is degenerate under comm-range pruning "
            "(e.g., first K samples have no in-range partners under clean gating)."
        ),
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print progress every N samples (set 1 to print every sample).",
    )
    parser.add_argument(
        "--comm-range-override",
        type=float,
        default=None,
        help="Optional override for hypes['comm_range'] (useful for pose-correction stress tests).",
    )
    parser.add_argument(
        "--comm-range-gating",
        type=str,
        default="auto",
        choices=["auto", "clean", "noisy"],
        help=(
            "Which pose to use for comm-range pruning: clean (use lidar_pose_clean) or noisy (use lidar_pose). "
            "auto may change implicitly with solver_backend/runtime_mode/pose_correction; for benchmarks, pass an explicit value."
        ),
    )
    parser.add_argument(
        "--force-ego-input-only",
        action="store_true",
        help=(
            "For intermediate/early fusion datasets, drop non-ego CAV inputs for the forward pass "
            "(record_len=1 and filter voxel coords), while keeping merged GT labels. "
            "This makes the 'single' baseline comparable to cooperative runs under the same comm_range."
        ),
    )
    parser.add_argument(
        "--pose-no-fallback",
        action="store_true",
        help=(
            "No-fallback mode for pose correction: if the online pose provider does not fully take over all "
            "non-ego pairs on a sample, drop non-ego CAV inputs for that sample (ego-only forward pass) "
            "instead of letting any noisy extrinsics remain in fusion. Intended for strict no-init "
            "invariance probes."
        ),
    )
    parser.add_argument(
        "--pose-assert-zero-interference",
        action="store_true",
        help=(
            "Enable runtime contract assertion for the strict zero-interference no-init benchmark: "
            "register_and_fuse + online_box + initfree + clean comm-range gating + non-ego noise + "
            "no compare-current + no prior. The attestation is written back into the AP YAML."
        ),
    )
    parser.add_argument(
        "--pose-assert-require-no-fallback",
        action="store_true",
        help="When runtime attestation is enabled, additionally require --pose-no-fallback.",
    )
    parser.add_argument(
        "--pose-timing",
        action="store_true",
        help="Collect pose correction timing per sample.",
    )
    parser.add_argument(
        "--debug-pose-k",
        type=int,
        default=0,
        help=(
            "Debug only: print pose/pairwise sanity checks for the first K evaluated samples per noise point. "
            "This is meant to diagnose why pose correction may not reflect in AP curves."
        ),
    )
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'no', 'no_w_uncertainty', 'single']
    benchmark_output_dir = _benchmark_output_dir()
    print(f"Benchmark outputs will be written to: {benchmark_output_dir}")

    hypes = yaml_utils.load_yaml(None, opt)

    test_dir_override = str(getattr(opt, "test_dir_override", "") or "").strip()
    if test_dir_override:
        hypes["test_dir"] = os.path.abspath(os.path.expanduser(test_dir_override))
    validate_dir_override = str(getattr(opt, "validate_dir_override", "") or "").strip()
    if validate_dir_override:
        hypes["validate_dir"] = os.path.abspath(os.path.expanduser(validate_dir_override))
    elif test_dir_override:
        hypes["validate_dir"] = hypes["test_dir"]
    max_cav_override = int(getattr(opt, "max_cav_override", 0) or 0)
    if max_cav_override > 0:
        hypes.setdefault("train_params", {})["max_cav"] = int(max_cav_override)

    if opt.comm_range_override is not None:
        hypes["comm_range"] = float(opt.comm_range_override)
    if getattr(opt, "force_pose_confidence", None) is not None:
        hypes["force_pose_confidence"] = float(opt.force_pose_confidence)
    
    hypes['validate_dir'] = hypes['test_dir']
    test_dir_lower = str(hypes['test_dir']).lower()
    if "opv2v" in test_dir_lower or "v2xsim" in test_dir_lower:
        assert "test" in hypes['validate_dir']
    left_hand = True if ("opv2v" in test_dir_lower or 'v2xset' in test_dir_lower) else False
    print(f"Left hand visualizing: {left_hand}")

    model = None
    raw_model = None
    device = torch.device('cpu')
    if not bool(getattr(opt, 'pose_solver_only', False)):
        print('Creating Model')
        model = train_utils.create_model(hypes)
        # we assume gpu is necessary
        if torch.cuda.is_available():
            model.cuda()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('Loading Model from checkpoint')
        saved_path = opt.model_dir
        _, model = train_utils.load_saved_model(saved_path, model)
        model.eval()
        raw_model = model.module if hasattr(model, "module") else model

    pos_std_list = _parse_float_list(opt.pos_std_list)
    rot_std_list = _parse_float_list(opt.rot_std_list)
    if not pos_std_list:
        pos_std_list = [0.0]
    if not rot_std_list:
        rot_std_list = [0.0]
    sweep_mode = opt.sweep_mode
    if sweep_mode == "paired":
        if len(pos_std_list) == 1 and len(rot_std_list) > 1:
            pos_std_list = pos_std_list * len(rot_std_list)
        if len(rot_std_list) == 1 and len(pos_std_list) > 1:
            rot_std_list = rot_std_list * len(pos_std_list)
        if len(pos_std_list) != len(rot_std_list):
            raise ValueError("paired sweep requires pos-std-list and rot-std-list to have equal length (or one of them length=1).")
        noise_pairs = list(zip(pos_std_list, rot_std_list))
    else:
        noise_pairs = [(p, r) for p in pos_std_list for r in rot_std_list]

    # Optional pose correction config injection.
    pose_correction = opt.pose_correction
    pose_solver_only = bool(getattr(opt, "pose_solver_only", False))
    pose_override_export_path = str(getattr(opt, "pose_override_export_path", "") or "").strip()
    pose_override_path = str(getattr(opt, "pose_override_path", "") or "").strip()
    pose_telemetry_dir = str(getattr(opt, "pose_telemetry_dir", "") or "").strip()
    pose_telemetry_enable = bool(pose_telemetry_dir)
    pose_telemetry_distance_threshold = float(getattr(opt, "pose_telemetry_distance_threshold", 0.0) or 0.0)
    pose_telemetry_max_pairs = int(getattr(opt, "pose_telemetry_max_pairs", 0) or 0)
    effective_input_audit_dir = str(getattr(opt, "effective_input_audit_dir", "") or "").strip()
    effective_input_audit_enable = bool(effective_input_audit_dir)
    effective_input_audit_samples = _parse_int_set(getattr(opt, "effective_input_audit_samples", ""))
    effective_input_audit_max_samples = int(getattr(opt, "effective_input_audit_max_samples", 0) or 0)
    effective_input_audit_path = ""
    effective_input_audit_fp = None
    effective_input_audit_written = 0
    if effective_input_audit_enable:
        os.makedirs(effective_input_audit_dir, exist_ok=True)
        effective_input_audit_path = os.path.join(effective_input_audit_dir, "effective_input_audit.jsonl")
        effective_input_audit_fp = open(effective_input_audit_path, "a", encoding="utf-8")
        if raw_model is not None and hasattr(raw_model, "record_runtime_features"):
            raw_model.record_runtime_features = True
        print("Effective-input audit JSONL:", effective_input_audit_path)
    pose_selection_policy = _resolve_pose_selection_policy(opt)
    compare_current = bool(opt.pose_compare_current)
    if pose_selection_policy == "compare_current_proxy":
        compare_current = True
    elif pose_selection_policy in {"solver_only", "choose_better_pose_error"}:
        compare_current = False
    compare_distance_threshold = float(opt.pose_compare_distance_threshold)
    apply_if_current_precision_below = opt.pose_current_precision_threshold
    min_precision_improvement = opt.pose_min_precision_improvement
    min_matched_improvement = opt.pose_min_matched_improvement
    min_precision = opt.pose_min_precision
    if compare_current:
        if min_precision_improvement is None:
            min_precision_improvement = 0.0
        if min_matched_improvement is None:
            min_matched_improvement = 0
        if apply_if_current_precision_below is None:
            apply_if_current_precision_below = -1.0
    pose_compare_args = {}
    if compare_current:
        pose_compare_args["compare_with_current"] = True
    if compare_distance_threshold is not None:
        pose_compare_args["compare_distance_threshold_m"] = float(compare_distance_threshold)
    if apply_if_current_precision_below is not None:
        pose_compare_args["apply_if_current_precision_below"] = float(apply_if_current_precision_below)
    if min_precision_improvement is not None:
        pose_compare_args["min_precision_improvement"] = float(min_precision_improvement)
    if min_matched_improvement is not None:
        pose_compare_args["min_matched_improvement"] = int(min_matched_improvement)
    if min_precision is not None:
        pose_compare_args["min_precision"] = float(min_precision)
    pose_solver_spec = None
    pose_solver_stage1 = None
    pose_solver_pose = None
    simple_override_cfg = None
    runtime_mode_opt = str(opt.runtime_mode or "").lower().strip()
    solver_backend_opt = str(opt.solver_backend or "offline_map").lower().strip()
    pose_source_opt = str(opt.pose_source or "noisy_input").lower().strip()
    pose_device = str(opt.pose_device or "auto").lower()
    if pose_device == "auto":
        pose_device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 303
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        # These flags reduce nondeterminism in cuDNN. Keep them on by default because
        # this script is used for evidence-grade benchmark runs.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    if bool(getattr(opt, "deterministic_strict", False)):
        # Parity gates can require tighter numeric stability than typical benchmark runs.
        # Prefer correctness over speed when explicitly requested.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
        try:
            # `warn_only=False` enforces deterministic kernels; some ops may throw.
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Fall back to best-effort determinism rather than crashing.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass
    if pose_correction != "none":
        if pose_solver_only and solver_backend_opt != "offline_map":
            raise ValueError("--pose-solver-only currently supports --solver-backend offline_map only")
        if pose_override_export_path and solver_backend_opt != "offline_map":
            raise ValueError("--pose-override-export-path requires --solver-backend offline_map")
        if (
            not opt.stage1_result
            and not pose_correction.startswith(("image_match", "lidar_reg"))
            and pose_correction != "oracle_gt"
        ):
            raise ValueError("--stage1-result is required when --pose-correction != none")
        if pose_override_path:
            raise ValueError("--pose-override-path requires --pose-correction none")
        # Ensure we don't accidentally combine multiple alignment blocks.
        hypes.pop("box_align", None)
        hypes.pop("v2xregpp_align", None)
        hypes.pop("freealign_align", None)
        hypes.pop("vips_align", None)
        hypes.pop("cbm_align", None)
        hypes.pop("pgc_pose", None)
        hypes.pop("image_match_align", None)
        raw_pose_override_cfg = hypes.get("pose_override")
        if isinstance(raw_pose_override_cfg, dict) and raw_pose_override_cfg.get("mode"):
            simple_override_cfg = dict(raw_pose_override_cfg)
        elif solver_backend_opt == "offline_map" and pose_source_opt == "identity":
            # For strict zero-init cache export, copy the ego pose onto each non-ego
            # agent so the initial relative transform is identity. Zeroing non-ego
            # world poses would still leak the ego world frame into the solver input.
            simple_override_cfg = {
                "mode": "ego",
                "apply_to": "non-ego",
            }
        if pose_correction.startswith("v2xregpp"):
            mode = "stable" if pose_correction.endswith("stable") else "initfree"
            method = "v2xregpp"
            args = {
                "config_path": opt.v2xregpp_config,
                "device": pose_device,
                "mode": mode,
                "use_occ_hint": bool(opt.v2xregpp_use_occ_hint),
                "use_occ_pose": bool(opt.v2xregpp_use_occ_pose),
                "force_occ_pose": bool(opt.v2xregpp_force_occ_pose),
                "occ_from_lidar": bool(opt.v2xregpp_occ_from_lidar),
                "occ_grid_hw": _parse_hw(opt.v2xregpp_occ_grid) or (256, 256),
                "occ_max_delta_xy_m": float(opt.v2xregpp_occ_max_delta_xy),
                "occ_max_delta_yaw_deg": float(opt.v2xregpp_occ_max_delta_yaw),
                "icp_refine": bool(opt.v2xregpp_icp_refine),
                "icp_voxel_size_m": float(opt.v2xregpp_icp_voxel),
                "icp_max_corr_dist_m": float(opt.v2xregpp_icp_max_corr),
                "icp_max_iterations": int(opt.v2xregpp_icp_max_iter),
                "min_matches": int(opt.v2xregpp_min_matches),
                "min_stability": float(opt.v2xregpp_min_stability),
                "min_precision": float(opt.v2xregpp_min_precision),
                "ema_alpha": float(opt.v2xregpp_ema_alpha),
                "max_step_xy_m": float(opt.v2xregpp_max_step_xy),
                "max_step_yaw_deg": float(opt.v2xregpp_max_step_yaw),
                "stage1_field": str(opt.v2xregpp_stage1_field or "pred_corner3d_np_list"),
                "bbox_type": str(opt.v2xregpp_bbox_type or "detected"),
                "selection_policy": pose_selection_policy,
            }
            v2xregpp_compare_args = {}
            if compare_current:
                v2xregpp_compare_args["compare_with_current"] = True
            if compare_distance_threshold is not None:
                v2xregpp_compare_args["compare_distance_threshold_m"] = float(compare_distance_threshold)
            if apply_if_current_precision_below is not None:
                v2xregpp_compare_args["apply_if_current_precision_below"] = float(apply_if_current_precision_below)
            if min_precision_improvement is not None:
                v2xregpp_compare_args["min_precision_improvement"] = float(min_precision_improvement)
            if min_matched_improvement is not None:
                v2xregpp_compare_args["min_matched_improvement"] = int(min_matched_improvement)
            if v2xregpp_compare_args:
                args.update(v2xregpp_compare_args)
            if pose_telemetry_enable:
                args["telemetry_enable"] = True
                args["telemetry_distance_threshold_m"] = float(
                    pose_telemetry_distance_threshold or compare_distance_threshold or 3.0
                )
                args["telemetry_max_pairs"] = int(pose_telemetry_max_pairs or 0)
        elif pose_correction.startswith("freealign"):
            backend = "repo" if "repo" in pose_correction else "paper"
            mode = "stable" if pose_correction.endswith("stable") else "initfree"
            method = "freealign"
            args = {
                "backend": backend,
                "mode": mode,
                "device": str(pose_device),
                # Reuse V2X-Reg++ stable hyperparams so "stable" means the same across methods.
                "ema_alpha": float(opt.v2xregpp_ema_alpha),
                "max_step_xy_m": float(opt.v2xregpp_max_step_xy),
                "max_step_yaw_deg": float(opt.v2xregpp_max_step_yaw),
                "max_boxes": int(opt.freealign_max_boxes),
                # Paper configs (repo backend will ignore unknown fields).
                "min_nodes": int(opt.freealign_min_nodes),
                "sim_threshold": float(opt.freealign_sim_threshold),
                "affine_method": str(opt.freealign_affine_method),
                "ransac_reproj_threshold": float(opt.freealign_ransac_reproj_threshold),
                # Repo configs (paper backend will ignore unknown fields).
                "min_anchors": int(opt.freealign_min_anchors),
                "anchor_error": float(opt.freealign_anchor_error),
                "box_error": float(opt.freealign_box_error),
                "selection_policy": pose_selection_policy,
            }
            if pose_compare_args:
                args.update(pose_compare_args)
            if pose_telemetry_enable:
                args["telemetry_enable"] = True
                args["telemetry_distance_threshold_m"] = float(
                    pose_telemetry_distance_threshold or compare_distance_threshold or 3.0
                )
                args["telemetry_max_pairs"] = int(pose_telemetry_max_pairs or 0)
        elif pose_correction.startswith("vips"):
            mode = "stable" if pose_correction.endswith("stable") else "initfree"
            method = "vips"
            args = {
                "stage1_field": str(opt.v2xregpp_stage1_field or "pred_corner3d_np_list"),
                "bbox_type": str(opt.v2xregpp_bbox_type or "detected"),
                "mode": mode,
                "use_prior": bool(opt.vips_use_prior),
                "device": str(pose_device),
                "ema_alpha": float(opt.v2xregpp_ema_alpha),
                "max_step_xy_m": float(opt.v2xregpp_max_step_xy),
                "max_step_yaw_deg": float(opt.v2xregpp_max_step_yaw),
                "match_threshold": float(opt.vips_match_threshold),
                "match_distance_thr_m": float(opt.vips_match_distance),
                "selection_policy": pose_selection_policy,
            }
            if pose_compare_args:
                args.update(pose_compare_args)
            if pose_telemetry_enable:
                args["telemetry_enable"] = True
                args["telemetry_distance_threshold_m"] = float(
                    pose_telemetry_distance_threshold or compare_distance_threshold or 3.0
                )
                args["telemetry_max_pairs"] = int(pose_telemetry_max_pairs or 0)
        elif pose_correction.startswith("cbm"):
            mode = "stable" if pose_correction.endswith("stable") else "initfree"
            method = "cbm"
            args = {
                "stage1_field": str(opt.v2xregpp_stage1_field or "pred_corner3d_np_list"),
                "bbox_type": str(opt.v2xregpp_bbox_type or "detected"),
                "mode": mode,
                "use_prior": bool(opt.cbm_use_prior),
                "device": str(pose_device),
                "ema_alpha": float(opt.v2xregpp_ema_alpha),
                "max_step_xy_m": float(opt.v2xregpp_max_step_xy),
                "max_step_yaw_deg": float(opt.v2xregpp_max_step_yaw),
                "sigma1_deg": float(opt.cbm_sigma1_deg),
                "sigma2_m": float(opt.cbm_sigma2_m),
                "sigma3_m": float(opt.cbm_sigma3_m),
                "absolute_dis_lim_m": float(opt.cbm_absolute_dis_lim),
                "selection_policy": pose_selection_policy,
            }
            if pose_compare_args:
                args.update(pose_compare_args)
            if pose_telemetry_enable:
                args["telemetry_enable"] = True
                args["telemetry_distance_threshold_m"] = float(
                    pose_telemetry_distance_threshold or compare_distance_threshold or 3.0
                )
                args["telemetry_max_pairs"] = int(pose_telemetry_max_pairs or 0)
        elif pose_correction.startswith("image_match"):
            mode = "stable" if pose_correction.endswith("stable") else "initfree"
            method = "image_match"
            args = {
                "mode": mode,
                "camera_index": int(opt.image_match_camera_index),
                "camera_indices": str(opt.image_match_camera_indices or ""),
                "try_all_cameras": bool(opt.image_match_try_all_cameras),
                "init_source": str(opt.image_match_init_source),
                "min_stability": float(opt.image_match_min_stability),
                "ema_alpha": float(opt.v2xregpp_ema_alpha),
                "max_step_xy_m": float(opt.v2xregpp_max_step_xy),
                "max_step_yaw_deg": float(opt.v2xregpp_max_step_yaw),
                "matcher": str(opt.image_match_matcher),
                "max_features": int(opt.image_match_max_features),
                "ratio_test": float(opt.image_match_ratio_test),
                "cross_check": bool(opt.image_match_cross_check),
                "ransac_thresh_px": float(opt.image_match_ransac_thresh),
                "ransac_confidence": float(opt.image_match_ransac_confidence),
                "ransac_max_iters": int(opt.image_match_ransac_max_iters),
                "min_matches": int(opt.image_match_min_matches),
                "min_inliers": int(opt.image_match_min_inliers),
                "resize_max_dim": int(opt.image_match_resize_max_dim),
                "allow_no_intrinsics": bool(opt.image_match_allow_no_intrinsics),
                "t_scale": None if opt.image_match_t_scale is None else float(opt.image_match_t_scale),
                "device": str(opt.image_match_device),
            }
            if pose_compare_args:
                image_compare_args = {}
                for key in ("compare_with_current", "compare_distance_threshold_m"):
                    if key in pose_compare_args:
                        image_compare_args[key] = pose_compare_args[key]
                if image_compare_args:
                    args.update(image_compare_args)
        elif pose_correction.startswith("lidar_reg"):
            mode = "stable" if pose_correction.endswith("stable") else "initfree"
            method = "lidar_reg"
            global_method = str(opt.lidar_reg_global_method or "auto").strip().lower()
            if global_method == "auto":
                global_method = "fgr" if bool(opt.lidar_reg_use_fgr) else "ransac"
            args = {
                "mode": mode,
                "ema_alpha": float(opt.v2xregpp_ema_alpha),
                "max_step_xy_m": float(opt.v2xregpp_max_step_xy),
                "max_step_yaw_deg": float(opt.v2xregpp_max_step_yaw),
                "min_fitness": float(opt.lidar_reg_min_fitness),
                "max_inlier_rmse": float(opt.lidar_reg_max_inlier_rmse),
                "voxel_size_m": float(opt.lidar_reg_voxel_size),
                "max_corr_dist_m": float(opt.lidar_reg_max_corr),
                "ransac_n": int(opt.lidar_reg_ransac_n),
                "ransac_max_iter": int(opt.lidar_reg_ransac_max_iter),
                "ransac_confidence": float(opt.lidar_reg_ransac_confidence),
                "use_fgr": bool(opt.lidar_reg_use_fgr),
                "global_method": str(global_method),
                "icp_method": str(opt.lidar_reg_icp_method),
                "icp_max_iter": int(opt.lidar_reg_icp_max_iter),
                "min_points": int(opt.lidar_reg_min_points),
                "max_points": int(opt.lidar_reg_max_points),
                "teaser_noise_bound_m": float(opt.lidar_reg_teaser_noise_bound),
                "teaser_max_correspondences": int(opt.lidar_reg_teaser_max_correspondences),
            }
            if opt.lidar_reg_cache:
                args["cache_path"] = str(resolve_repo_path(opt.lidar_reg_cache))
            if pose_compare_args:
                lidar_compare_args = {}
                for key in ("compare_with_current", "compare_distance_threshold_m"):
                    if key in pose_compare_args:
                        lidar_compare_args[key] = pose_compare_args[key]
                if lidar_compare_args:
                    args.update(lidar_compare_args)
        elif pose_correction.startswith("v2vloc_"):
            mode = "stable" if pose_correction.endswith("stable") else "initfree"
            # We intentionally *don't* write pose_confidence here, so that pose confidence
            # is computed consistently via `attach_pose_confidence` from (pose - pose_clean)
            # for all methods, isolating the effect of pose alignment.
            if "oracle" in pose_correction:
                pose_field = "lidar_pose_clean_np"
            else:
                pose_field = "lidar_pose_pred_np"
            method = "pgc"
            args = {
                "pose_field": str(pose_field),
                "confidence_field": "",
                "min_confidence": 0.0,
                "mode": str(mode),
                "ema_alpha": float(opt.v2xregpp_ema_alpha),
                "max_step_xy_m": float(opt.v2xregpp_max_step_xy),
                "max_step_yaw_deg": float(opt.v2xregpp_max_step_yaw),
                "freeze_ego": True,
            }
        elif pose_correction == "oracle_gt":
            method = "gt"
            args = {
                "freeze_ego": True,
            }
        else:
            raise ValueError(f"Unsupported --pose-correction: {pose_correction}")

        stage1_path = None
        if opt.stage1_result and method not in {"image_match", "lidar_reg"}:
            stage1_path = str(resolve_repo_path(opt.stage1_result))
            if method == "pgc":
                pose_solver_pose = read_json(stage1_path)
            else:
                pose_solver_stage1 = read_json(stage1_path)
        pose_solver_spec = {"method": method, "args": args}

        pose_provider_cfg = dict(hypes.get("pose_provider") or {})
        if solver_backend_opt in {"online_box", "online_box_feat_refine"}:
            pose_provider_cfg["enabled"] = True
            pose_provider_cfg["runtime_mode"] = runtime_mode_opt or "register_and_fuse"
            pose_provider_cfg["solver_backend"] = solver_backend_opt
            pose_provider_cfg["pose_source"] = pose_source_opt
            pose_provider_cfg["online_method"] = method
            online_args = dict(args)
            if bool(getattr(opt, "online_gpu_stage1_solver", False)):
                online_args["gpu_stage1_solver"] = True
            if bool(getattr(opt, "online_skip_pairwise_rebuild", False)):
                online_args["skip_pairwise_rebuild"] = True
            if solver_backend_opt == "online_box_feat_refine":
                online_args["refine_steps"] = int(opt.online_refine_steps)
                online_args["refine_init_step_xy_m"] = float(opt.online_refine_init_step_xy)
                online_args["refine_init_step_yaw_deg"] = float(opt.online_refine_init_step_yaw)
                online_args["refine_decay"] = float(opt.online_refine_decay)
                online_args["refine_min_step_xy_m"] = float(opt.online_refine_min_step_xy)
                online_args["refine_min_step_yaw_deg"] = float(opt.online_refine_min_step_yaw)
                online_args["refine_min_improvement_m"] = float(opt.online_refine_min_improvement)
            pose_provider_cfg["online_args"] = online_args
            pose_provider_cfg["recompute_pairwise"] = True
            if stage1_path:
                if method == "pgc":
                    pose_provider_cfg["pose_result"] = stage1_path
                else:
                    pose_provider_cfg["stage1_result"] = stage1_path
            hypes["pose_provider"] = pose_provider_cfg
            # Keep comm-range gating consistent with offline-map correction path:
            # use clean pose for dataset-side pruning when runtime correction is active.
            hypes["comm_range_use_clean_pose"] = True
            # Online backend should not rely on dataset-side override maps.
            hypes["pose_override"] = {"enabled": False}
        else:
            pose_override_cfg = dict(hypes.get("pose_override") or {})
            for key in ("path", "pose_path", "pose_result", "pose_map"):
                pose_override_cfg.pop(key, None)
            pose_override_cfg["enabled"] = True
            pose_override_cfg.setdefault("pose_field", "lidar_pose_pred_np")
            pose_override_cfg.setdefault("confidence_field", "pose_confidence_np")
            hypes["pose_override"] = pose_override_cfg

    if pose_correction == "none" and runtime_mode_opt:
        pose_provider_cfg = dict(hypes.get("pose_provider") or {})
        pose_provider_cfg["enabled"] = True
        pose_provider_cfg["runtime_mode"] = runtime_mode_opt
        pose_provider_cfg["solver_backend"] = solver_backend_opt
        pose_provider_cfg["pose_source"] = pose_source_opt
        pose_provider_cfg["recompute_pairwise"] = True
        hypes["pose_provider"] = pose_provider_cfg

    if pose_correction == "none" and pose_override_path:
        pose_override_cfg = dict(hypes.get("pose_override") or {})
        pose_override_cfg["enabled"] = True
        pose_override_cfg["path"] = str(pose_override_path)
        pose_override_cfg.setdefault("pose_field", "lidar_pose_pred_np")
        pose_override_cfg.setdefault("confidence_field", "pose_confidence_np")
        hypes["pose_override"] = pose_override_cfg

    if pose_override_export_path and pose_correction == "none":
        raise ValueError("--pose-override-export-path requires --pose-correction != none")

    # Allow explicit override of comm-range gating semantics (debug + contract freezing).
    # This must run *after* pose-correction/runtime-mode config since those paths may set
    # comm_range_use_clean_pose automatically for consistency with offline-map overrides.
    comm_range_gating = str(getattr(opt, "comm_range_gating", "auto") or "auto").strip().lower()
    if comm_range_gating == "clean":
        hypes["comm_range_use_clean_pose"] = True
    elif comm_range_gating == "noisy":
        hypes["comm_range_use_clean_pose"] = False

    if opt.pose_timing:
        hypes["pose_timing"] = True

    runtime_contract_attestation = _build_zero_interference_attestation(opt, hypes, pose_correction)
    if runtime_contract_attestation.get("enabled") and runtime_contract_attestation.get("verdict") != "PASS":
        raise RuntimeError(
            "zero-interference runtime contract failed: {}".format(
                "; ".join(runtime_contract_attestation.get("errors") or ["unknown error"])
            )
        )

    def _force_ego_input_only(batch_data):
        """
        Keep only ego CAV inputs for the forward pass (intermediate/early fusion style),
        without changing merged GT labels stored in object_bbx_center/object_ids.

        This is needed on datasets like V2V4Real where comm_range pruning changes the
        GT set; comm_range=0 single-agent would otherwise become a different task.
        """
        try:
            ego = batch_data.get("ego")
            if not isinstance(ego, dict):
                return batch_data
            record_len = ego.get("record_len")
            if record_len is None:
                return batch_data

            # Late fusion stores one top-level dict per CAV. Keep non-ego
            # entries for merged GT labels, but mark the batch so
            # inference_late_fusion() only forwards ego predictions.
            is_late_fusion_batch = any(
                k != "ego" and isinstance(v, dict) and "transformation_matrix" in v
                for k, v in batch_data.items()
            )
            if is_late_fusion_batch:
                ego["_force_ego_input_only"] = True

            def _prune_voxel_dict(proc_dict: dict) -> None:
                """
                Keep only voxels that belong to ego cav (cav_idx==0).

                NOTE: In OpenCOOD-style intermediate fusion, per-cav features are
                concatenated and `voxel_coords[:, 0]` stores the cav index when
                batch_size=1 (testing).
                """
                coords = proc_dict.get("voxel_coords")
                if not torch.is_tensor(coords) or coords.ndim != 2 or int(coords.shape[0]) <= 0:
                    return
                cav_idx = coords[:, 0]
                mask = cav_idx == 0
                n = int(coords.shape[0])
                for k, vv in list(proc_dict.items()):
                    if torch.is_tensor(vv) and vv.ndim >= 1 and int(vv.shape[0]) == n:
                        proc_dict[k] = vv[mask]
                coords2 = proc_dict.get("voxel_coords")
                if torch.is_tensor(coords2) and coords2.numel() > 0:
                    proc_dict["voxel_coords"][:, 0] = 0

            def _prune_camera_dict(cam_dict: dict) -> None:
                """
                Slice camera tensors to ego only (first cav entry).

                For camera pipelines, the leading dim is typically `sum(record_len)`
                (== num_cavs when batch_size=1). Simply setting record_len=1 is NOT
                sufficient because fusion masks are derived from the actual tensor
                shapes; we must physically drop non-ego entries.
                """
                if not cam_dict:
                    return
                n_cav = None
                imgs = cam_dict.get("imgs")
                if torch.is_tensor(imgs) and imgs.ndim >= 1 and int(imgs.shape[0]) > 0:
                    n_cav = int(imgs.shape[0])
                if n_cav is None:
                    for vv in cam_dict.values():
                        if torch.is_tensor(vv) and vv.ndim >= 1 and int(vv.shape[0]) > 0:
                            n_cav = int(vv.shape[0])
                            break
                if n_cav is None:
                    return
                for k, vv in list(cam_dict.items()):
                    if torch.is_tensor(vv) and vv.ndim >= 1 and int(vv.shape[0]) == n_cav:
                        cam_dict[k] = vv[:1]

            # Force single-agent forward.
            if torch.is_tensor(record_len):
                ego["record_len"] = record_len.clone()
                ego["record_len"].fill_(1)
            else:
                ego["record_len"] = torch.tensor([1], dtype=torch.int64)

            # Per-CAV pose/confidence (concat along cav dim) -> keep ego only.
            for key in ("lidar_pose", "lidar_pose_clean"):
                v = ego.get(key)
                if torch.is_tensor(v) and v.ndim >= 2 and v.shape[0] >= 1:
                    ego[key] = v[:1]
            v = ego.get("pose_confidence")
            if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] >= 1:
                ego["pose_confidence"] = v[:1]

            # Optional: keep cav_id_list aligned with the forward pass.
            cav_id_list = ego.get("cav_id_list")
            if isinstance(cav_id_list, list) and cav_id_list:
                ego["cav_id_list"] = [cav_id_list[0]]

            # Optional: keep agent_modality_list aligned for heter models.
            agent_modality_list = ego.get("agent_modality_list")
            if isinstance(agent_modality_list, list) and agent_modality_list:
                ego["agent_modality_list"] = [agent_modality_list[0]]

            # Optional: keep raw per-cav lidar list aligned (online lidar_reg path).
            lidar_np_by_cav = ego.get("lidar_np_by_cav")
            if isinstance(lidar_np_by_cav, list) and lidar_np_by_cav:
                ego["lidar_np_by_cav"] = [lidar_np_by_cav[0]]

            # Voxel features are concatenated across cavs; voxel_coords[:,0] stores cav index.
            proc = ego.get("processed_lidar")
            if isinstance(proc, dict):
                _prune_voxel_dict(proc)

            # Camera-only datasets use image_inputs; heter datasets use inputs_{tag}.
            image_inputs = ego.get("image_inputs")
            if isinstance(image_inputs, dict):
                _prune_camera_dict(image_inputs)

            # Heterogeneous pipelines store modality inputs as inputs_* dicts.
            for key, val in list(ego.items()):
                if not isinstance(key, str) or not key.startswith("inputs_"):
                    continue
                if not isinstance(val, dict):
                    continue
                # Lidar-like inputs (voxelized).
                if "voxel_coords" in val:
                    _prune_voxel_dict(val)
                    continue
                # Camera-like inputs.
                _prune_camera_dict(val)

            # Keep pairwise transforms conservative: in ego-only mode, only ego feature is valid.
            pairwise = ego.get("pairwise_t_matrix")
            if torch.is_tensor(pairwise) and pairwise.ndim >= 5 and pairwise.shape[-2:] == (4, 4):
                try:
                    eye = torch.eye(4, device=pairwise.device, dtype=pairwise.dtype)
                    ego["pairwise_t_matrix"] = eye.view(1, 1, 1, 4, 4).repeat(
                        int(pairwise.shape[0]), int(pairwise.shape[1]), int(pairwise.shape[2]), 1, 1
                    )
                    label_dict = ego.get("label_dict")
                    if isinstance(label_dict, dict):
                        label_dict["pairwise_t_matrix"] = ego["pairwise_t_matrix"]
                except Exception:
                    pass
            return batch_data
        except Exception:
            # Best-effort only; never crash the benchmark because of pruning.
            return batch_data

    if opt.also_laplace:
        use_laplace_options = [False, True]
    else:
        use_laplace_options = [False]
    pose_override_export_multiple = bool(pose_override_export_path) and (
        (len(use_laplace_options) * len(noise_pairs)) > 1
    )

    for use_laplace in use_laplace_options:
        AP30 = []
        AP50 = []
        AP70 = []
        timing_stats_all = []
        rel_error_stats_all = []
        pose_solver_only_stats = []
        mean_pos = 0.0
        mean_rot = 0.0
        # Build the dataset once and only mutate the noise_setting per sweep entry.
        np.random.seed(303)
        noise_setting = OrderedDict()
        noise_setting['add_noise'] = True
        noise_setting['args'] = {
            'pos_std': 0.0,
            'rot_std': 0.0,
            'pos_mean': mean_pos,
            'rot_mean': mean_rot,
            'target': opt.noise_target,
        }
        if float(opt.pose_dropout_prob or 0.0) > 0.0:
            noise_setting['args']['dropout_prob'] = float(opt.pose_dropout_prob)
        if use_laplace:
            noise_setting['args']['laplace'] = True
        hypes.update({"noise_setting": noise_setting})
        print('Dataset Building')
        visualize_dataset = bool(int(getattr(opt, "save_vis_interval", 0) or 0) > 0)
        opencood_dataset = build_dataset(hypes, visualize=visualize_dataset, train=False)
        depth_override_stats = _override_opv2v_depth_root(
            opencood_dataset,
            getattr(opt, "depth_root_override", ""),
        )
        if depth_override_stats.get("enabled"):
            print("Depth root override:", depth_override_stats)
            if int(depth_override_stats.get("missing", 0) or 0) > 0:
                raise FileNotFoundError(
                    "Depth override has missing files: {}/{}".format(
                        depth_override_stats.get("missing"), depth_override_stats.get("total")
                    )
                )
        num_workers = opt.num_workers
        if num_workers is None:
            # When pose correction loads a large stage1 cache JSON, multi-worker
            # DataLoader will fork/copy the dict into each worker and can easily
            # OOM. Default to 0 workers for correction modes unless explicitly
            # overridden.
            num_workers = 0 if pose_correction != "none" else 4
            if pose_correction.endswith("stable"):
                num_workers = 0

        for pos_std, rot_std in noise_pairs:
            # setting noise
            noise_setting = OrderedDict()
            noise_args = {
                'pos_std': pos_std,
                'rot_std': rot_std,
                'pos_mean': mean_pos,
                'rot_mean': mean_rot,
                'target': opt.noise_target,
            }
            if float(opt.pose_dropout_prob or 0.0) > 0.0:
                noise_args['dropout_prob'] = float(opt.pose_dropout_prob)

            noise_setting['add_noise'] = True
            noise_setting['args'] = noise_args

            suffix = ""
            if use_laplace:
                noise_setting['args']['laplace'] = True
                suffix = "_laplace"

            pose_solver_metrics = None
            pose_solver_cuda_memory_stats = None
            if pose_solver_spec is not None and solver_backend_opt == "offline_map":
                corrector = build_pose_corrector(pose_solver_spec["method"], args=pose_solver_spec["args"])
                telemetry_path = None
                telemetry_context = None
                if pose_telemetry_enable:
                    os.makedirs(pose_telemetry_dir, exist_ok=True)
                    corr_safe = str(pose_correction or "none").replace("/", "_").replace(" ", "")
                    fname = f"{corr_safe}_pos{float(pos_std):.1f}_rot{float(rot_std):.1f}.jsonl.gz"
                    telemetry_path = os.path.join(pose_telemetry_dir, fname)
                    telemetry_context = {
                        "pose_correction": str(pose_correction or "none"),
                        "note": str(opt.note or ""),
                        "model_dir": str(opt.model_dir),
                        "stage1_result": str(opt.stage1_result or ""),
                        "solver_backend": str(solver_backend_opt),
                        "runtime_mode": str(runtime_mode_opt),
                        "fusion_method": str(opt.fusion_method),
                    }
                solver_cuda_peak_allocated_before = None
                solver_cuda_peak_reserved_before = None
                if torch.cuda.is_available():
                    try:
                        torch.cuda.reset_peak_memory_stats()
                        solver_cuda_peak_allocated_before = int(torch.cuda.memory_allocated())
                        solver_cuda_peak_reserved_before = int(torch.cuda.memory_reserved())
                    except Exception:
                        solver_cuda_peak_allocated_before = None
                        solver_cuda_peak_reserved_before = None
                solver_result = run_pose_solver(
                    opencood_dataset,
                    corrector=corrector,
                    stage1_result=pose_solver_stage1,
                    pose_result=pose_solver_pose,
                    noise_setting=noise_setting,
                    max_samples=opt.max_eval_samples,
                    seed=303,
                    simple_override_cfg=simple_override_cfg,
                    telemetry_path=telemetry_path,
                    telemetry_context=telemetry_context,
                    force_ego_only_when_incomplete=bool(getattr(opt, "pose_no_fallback", False)),
                )
                if torch.cuda.is_available():
                    try:
                        pose_solver_cuda_memory_stats = {
                            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                            "allocated_before_bytes": solver_cuda_peak_allocated_before,
                            "reserved_before_bytes": solver_cuda_peak_reserved_before,
                            "allocated_after_bytes": int(torch.cuda.memory_allocated()),
                            "reserved_after_bytes": int(torch.cuda.memory_reserved()),
                        }
                    except Exception:
                        pose_solver_cuda_memory_stats = None
                pose_solver_metrics = solver_result.metrics
                if hasattr(opencood_dataset, "set_pose_override_map"):
                    opencood_dataset.set_pose_override_map(solver_result.overrides)
                else:
                    opencood_dataset.pose_override_map = solver_result.overrides
                    opencood_dataset.pose_override_enabled = True
                if pose_override_export_path:
                    export_path = _resolve_pose_override_export_path(
                        pose_override_export_path,
                        pos_std=pos_std,
                        rot_std=rot_std,
                        use_laplace=use_laplace,
                        multiple_exports=pose_override_export_multiple,
                    )
                    _write_pose_override_map_json(export_path, solver_result.overrides)
                    print("Pose override JSON:", export_path)
                no_noise_setting = OrderedDict()
                no_noise_setting["add_noise"] = False
                no_noise_setting["args"] = {
                    "pos_std": 0.0,
                    "rot_std": 0.0,
                    "pos_mean": mean_pos,
                    "rot_mean": mean_rot,
                    "target": opt.noise_target,
                }
                opencood_dataset.params['noise_setting'] = no_noise_setting
            elif (
                pose_solver_spec is not None
                and solver_backend_opt in {"online_box", "online_box_feat_refine"}
                and str(pose_solver_spec.get("method") or "").lower().strip() == "gt"
            ):
                # For oracle GT runtime, keep eval geometry identical to offline-map path.
                no_noise_setting = OrderedDict()
                no_noise_setting["add_noise"] = False
                no_noise_setting["args"] = {
                    "pos_std": 0.0,
                    "rot_std": 0.0,
                    "pos_mean": mean_pos,
                    "rot_mean": mean_rot,
                    "target": opt.noise_target,
                }
                opencood_dataset.params['noise_setting'] = no_noise_setting
            else:
                opencood_dataset.params['noise_setting'] = noise_setting

            if pose_solver_only:
                if isinstance(pose_solver_metrics, dict):
                    pose_solver_only_stats.append(
                        {
                            "pos_std": float(pos_std),
                            "rot_std": float(rot_std),
                            **pose_solver_metrics,
                            "cuda_memory": pose_solver_cuda_memory_stats,
                        }
                    )
                print(
                    f"[POSE_SOLVER_ONLY] pos_std={float(pos_std):.1f} rot_std={float(rot_std):.1f} "
                    f"metrics={(pose_solver_metrics or {})}"
                )
                continue
            # If pre-processing already produces CUDA tensors (e.g., GPU voxelization),
            # DataLoader pin_memory will fail and worker forking will crash. Disable
            # pin_memory and force num_workers=0 in that case.
            pin_memory = torch.cuda.is_available()
            force_workers_zero = False
            if pin_memory:
                try:
                    pre_processors = []
                    if hasattr(opencood_dataset, "pre_processor"):
                        pre_processors.append(getattr(opencood_dataset, "pre_processor"))
                    for attr in dir(opencood_dataset):
                        if attr.startswith("pre_processor_"):
                            pre_processors.append(getattr(opencood_dataset, attr))
                    if any(getattr(pp, "use_gpu", False) for pp in pre_processors if pp is not None):
                        pin_memory = False
                        force_workers_zero = True
                except Exception:
                    # Fall back to default pin_memory behavior.
                    pass
            if force_workers_zero and int(num_workers) > 0:
                print("[WARN] GPU preprocessor detected; forcing num_workers=0 to avoid CUDA fork issues.")
                num_workers = 0
            data_loader = DataLoader(
                opencood_dataset,
                batch_size=1,
                num_workers=int(num_workers),
                collate_fn=opencood_dataset.collate_batch_test,
                shuffle=False,
                pin_memory=pin_memory,
                drop_last=False,
            )
            print(f"Noise Added: {pos_std}/{rot_std}/{mean_pos}/{mean_rot}.")
            
            # Create the dictionary for evaluation
            result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                           0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                           0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
            
            noise_level = f"{pos_std}_{rot_std}_{mean_pos}_{mean_rot}_" + opt.fusion_method + suffix + opt.note
            if pose_correction != "none":
                noise_level = f"{noise_level}_{pose_correction}"

            rel_trans_errors = []
            rel_yaw_errors = []
            pose_timing = {}
            online_telemetry_path = ""
            if (
                pose_telemetry_enable
                and solver_backend_opt in {"online_box", "online_box_feat_refine"}
                and pose_correction != "none"
            ):
                os.makedirs(pose_telemetry_dir, exist_ok=True)
                corr_safe = str(pose_correction or "none").replace("/", "_").replace(" ", "")
                fname = f"{corr_safe}_pos{float(pos_std):.1f}_rot{float(rot_std):.1f}_online.jsonl.gz"
                online_telemetry_path = os.path.join(pose_telemetry_dir, fname)
                if os.path.exists(online_telemetry_path):
                    os.remove(online_telemetry_path)
            cuda_peak_allocated_before = None
            cuda_peak_reserved_before = None
            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                    cuda_peak_allocated_before = int(torch.cuda.memory_allocated())
                    cuda_peak_reserved_before = int(torch.cuda.memory_reserved())
                except Exception:
                    cuda_peak_allocated_before = None
                    cuda_peak_reserved_before = None
            infer_start = time.perf_counter()
            sample_count = 0


            eval_sample_start = int(getattr(opt, "eval_sample_start", 0) or 0)
            if eval_sample_start < 0:
                eval_sample_start = 0

            for i, batch_data in enumerate(data_loader):
                if i < eval_sample_start:
                    continue
                if int(opt.log_interval or 0) > 0 and ((i - eval_sample_start) % int(opt.log_interval) == 0):
                    print(f"{noise_level}_{i}")
                if batch_data is None:
                    continue
                sample_count += 1
                if bool(opt.force_ego_input_only):
                    batch_data = _force_ego_input_only(batch_data)
                with torch.no_grad():
                    batch_data = train_utils.to_device(batch_data, device)
                    debug_k = int(getattr(opt, "debug_pose_k", 0) or 0)
                    if debug_k > 0 and sample_count <= debug_k:
                        try:
                            rec = batch_data.get("ego", {}).get("record_len")
                            pw = batch_data.get("ego", {}).get("pairwise_t_matrix")
                            lp = batch_data.get("ego", {}).get("lidar_pose")
                            lpc = batch_data.get("ego", {}).get("lidar_pose_clean")
                            print(
                                f"[DEBUG_POSE][pre] i={i} record_len={rec} "
                                f"pairwise_shape={(tuple(pw.shape) if hasattr(pw, 'shape') else None)} "
                                f"lidar_pose_shape={(tuple(lp.shape) if hasattr(lp, 'shape') else None)}"
                            )
                            if lp is not None and lpc is not None:
                                poses = lp.detach().cpu().numpy().reshape(-1, 6)
                                poses_clean = lpc.detach().cpu().numpy().reshape(-1, 6)
                                T_world = pose_to_tfm(poses)
                                T_world_clean = pose_to_tfm(poses_clean)
                                ego_T_world = T_world[0]
                                ego_T_world_clean = T_world_clean[0]
                                rel_trans = []
                                rel_yaw = []
                                for cav_idx in range(1, min(T_world.shape[0], T_world_clean.shape[0])):
                                    rel = np.linalg.inv(ego_T_world) @ T_world[cav_idx]
                                    rel_clean = np.linalg.inv(ego_T_world_clean) @ T_world_clean[cav_idx]
                                    err = np.linalg.inv(rel_clean) @ rel
                                    rel_trans.append(float(np.linalg.norm(err[:2, 3])))
                                    yaw = float(np.degrees(np.arctan2(err[1, 0], err[0, 0])))
                                    rel_yaw.append(abs(yaw))
                                if rel_trans:
                                    print(
                                        f"[DEBUG_POSE][pre] rel_trans_mean={float(np.mean(rel_trans)):.3f} "
                                        f"rel_yaw_mean={float(np.mean(rel_yaw)):.3f}"
                                    )
                            if isinstance(pw, torch.Tensor) and pw.ndim == 5 and pw.shape[1] >= 2:
                                t01 = pw[0, 0, 1, :2, 3].detach().cpu().numpy().tolist()
                                t10 = pw[0, 1, 0, :2, 3].detach().cpu().numpy().tolist()
                                print(f"[DEBUG_POSE][pre] pairwise t(0->1) xy={t01}  t(1->0) xy={t10}")
                        except Exception as e:
                            print(f"[DEBUG_POSE][pre] failed: {type(e).__name__}: {e}")
                    batch_data = train_utils.maybe_apply_pose_provider(batch_data, hypes)
                    timing_payload = batch_data.get('ego', {}).get('pose_timing')
                    if online_telemetry_path:
                        try:
                            cfg = hypes.get("_pose_provider_runtime_cfg") if isinstance(hypes, dict) else None
                            corrector = getattr(cfg, "_online_corrector", None)
                            rows = getattr(corrector, "last_telemetry", None)
                            if isinstance(rows, list) and rows:
                                enriched_rows = []
                                for row in rows:
                                    if not isinstance(row, dict):
                                        continue
                                    rec = dict(row)
                                    rec.update(
                                        {
                                            "eval_index": int(i),
                                            "sample_count": int(sample_count),
                                            "noise_pos_std": float(pos_std),
                                            "noise_rot_std": float(rot_std),
                                            "pose_correction": str(pose_correction or "none"),
                                            "solver_backend": str(solver_backend_opt),
                                            "runtime_mode": str(runtime_mode_opt),
                                            "fusion_method": str(opt.fusion_method),
                                            "note": str(opt.note or ""),
                                            "model_dir": str(opt.model_dir),
                                            "stage1_result": str(opt.stage1_result or ""),
                                        }
                                    )
                                    enriched_rows.append(rec)
                                _append_jsonl(online_telemetry_path, enriched_rows)
                        except Exception as e:
                            if int(getattr(opt, "debug_pose_k", 0) or 0) > 0:
                                print(f"[POSE_TELEMETRY] failed: {type(e).__name__}: {e}")
                    if timing_payload:
                        def _ingest(payload):
                            if isinstance(payload, list):
                                for item in payload:
                                    _ingest(item)
                                return
                            if not isinstance(payload, dict):
                                return
                            for key, val in payload.items():
                                if isinstance(val, bool) or not isinstance(val, (int, float)):
                                    continue
                                key_str = str(key)
                                if not (key_str.endswith("_sec") or key_str.endswith("_count")):
                                    continue
                                pose_timing.setdefault(key_str, []).append(float(val))
                        _ingest(timing_payload)
                    applied = None
                    no_fallback_ready = None
                    if isinstance(timing_payload, dict):
                        applied = bool(timing_payload.get("pose_provider_applied", False))
                        no_fallback_ready = _pose_no_fallback_all_non_ego_applied(timing_payload)

                    # Strict no-fallback: if any non-ego pair remains uncorrected,
                    # do not fuse under noisy extrinsics for this sample.
                    if bool(getattr(opt, "pose_no_fallback", False)):
                        if not bool(no_fallback_ready):
                            batch_data = _force_ego_input_only(batch_data)

                    if opt.fusion_method == 'late':
                        batch_data = _sync_late_fusion_transformations_from_pose(batch_data)

                    if debug_k > 0 and sample_count <= debug_k:
                        try:
                            rec = batch_data.get("ego", {}).get("record_len")
                            pw = batch_data.get("ego", {}).get("pairwise_t_matrix")
                            lp = batch_data.get("ego", {}).get("lidar_pose")
                            lpc = batch_data.get("ego", {}).get("lidar_pose_clean")
                            print(
                                f"[DEBUG_POSE][post] i={i} applied={applied} "
                                f"no_fallback_ready={no_fallback_ready} record_len={rec} "
                                f"pairwise_shape={(tuple(pw.shape) if hasattr(pw, 'shape') else None)}"
                            )
                            if lp is not None and lpc is not None:
                                poses = lp.detach().cpu().numpy().reshape(-1, 6)
                                poses_clean = lpc.detach().cpu().numpy().reshape(-1, 6)
                                T_world = pose_to_tfm(poses)
                                T_world_clean = pose_to_tfm(poses_clean)
                                ego_T_world = T_world[0]
                                ego_T_world_clean = T_world_clean[0]
                                rel_trans = []
                                rel_yaw = []
                                for cav_idx in range(1, min(T_world.shape[0], T_world_clean.shape[0])):
                                    rel = np.linalg.inv(ego_T_world) @ T_world[cav_idx]
                                    rel_clean = np.linalg.inv(ego_T_world_clean) @ T_world_clean[cav_idx]
                                    err = np.linalg.inv(rel_clean) @ rel
                                    rel_trans.append(float(np.linalg.norm(err[:2, 3])))
                                    yaw = float(np.degrees(np.arctan2(err[1, 0], err[0, 0])))
                                    rel_yaw.append(abs(yaw))
                                if rel_trans:
                                    print(
                                        f"[DEBUG_POSE][post] rel_trans_mean={float(np.mean(rel_trans)):.3f} "
                                        f"rel_yaw_mean={float(np.mean(rel_yaw)):.3f}"
                                    )
                            if isinstance(pw, torch.Tensor) and pw.ndim == 5 and pw.shape[1] >= 2:
                                t01 = pw[0, 0, 1, :2, 3].detach().cpu().numpy().tolist()
                                t10 = pw[0, 1, 0, :2, 3].detach().cpu().numpy().tolist()
                                print(f"[DEBUG_POSE][post] pairwise t(0->1) xy={t01}  t(1->0) xy={t10}")
                        except Exception as e:
                            print(f"[DEBUG_POSE][post] failed: {type(e).__name__}: {e}")

                    should_audit_effective_input = False
                    if effective_input_audit_fp is not None:
                        sample_selected = (not effective_input_audit_samples) or (int(i) in effective_input_audit_samples)
                        under_cap = (effective_input_audit_max_samples <= 0) or (
                            effective_input_audit_written < effective_input_audit_max_samples
                        )
                        should_audit_effective_input = sample_selected and under_cap
                    
                    if opt.fusion_method == 'late':
                        infer_result = inference_utils.inference_late_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
                    elif opt.fusion_method == 'early':
                        infer_result = inference_utils.inference_early_fusion(batch_data,
                                                                model,
                                                                opencood_dataset)
                    elif opt.fusion_method == 'intermediate':
                        infer_result = inference_utils.inference_intermediate_fusion(batch_data,
                                                                        model,
                                                                        opencood_dataset)
                    elif opt.fusion_method == 'no':
                        infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                        model,
                                                                        opencood_dataset)
                    elif opt.fusion_method == 'no_w_uncertainty':
                        infer_result = inference_utils.inference_no_fusion_w_uncertainty(batch_data,
                                                                        model,
                                                                        opencood_dataset)
                    elif opt.fusion_method == 'single':
                        infer_result = inference_utils.inference_no_fusion(batch_data,
                                                                        model,
                                                                        opencood_dataset,
                                                                        single_gt=True)
                    else:
                        raise NotImplementedError('Only single, no, no_w_uncertainty, early, late and intermediate'
                                                'fusion is supported.')

                    if should_audit_effective_input:
                        try:
                            ego = batch_data.get("ego", {}) if isinstance(batch_data, dict) else {}
                            runtime_store = getattr(raw_model, "_runtime_feature_store", {}) if raw_model is not None else {}
                            modality_inputs = {}
                            if isinstance(ego, dict):
                                for key, value in ego.items():
                                    if isinstance(key, str) and key.startswith("inputs_"):
                                        modality_inputs[key] = _summarize_audit_value(value)
                            audit_record = {
                                "schema_version": "effective_input_audit_v1",
                                "eval_index": int(i),
                                "sample_count": int(sample_count),
                                "noise_pos_std": float(pos_std),
                                "noise_rot_std": float(rot_std),
                                "noise_level_tag": str(noise_level),
                                "pose_correction": str(pose_correction),
                                "pose_no_fallback": bool(getattr(opt, "pose_no_fallback", False)),
                                "pose_provider_applied": applied,
                                "pose_no_fallback_ready": no_fallback_ready,
                                "model_dir": str(opt.model_dir),
                                "note": str(opt.note or ""),
                                "post_provider_batch": {
                                    "record_len": _record_len_to_list(ego.get("record_len")) if isinstance(ego, dict) else [],
                                    "cav_id_list": list(ego.get("cav_id_list") or []) if isinstance(ego, dict) else [],
                                    "agent_modality_list": list(ego.get("agent_modality_list") or []) if isinstance(ego, dict) else [],
                                    "lidar_pose": _summarize_audit_value(ego.get("lidar_pose")) if isinstance(ego, dict) else None,
                                    "lidar_pose_clean": _summarize_audit_value(ego.get("lidar_pose_clean")) if isinstance(ego, dict) else None,
                                    "pairwise_t_matrix": _summarize_audit_value(ego.get("pairwise_t_matrix")) if isinstance(ego, dict) else None,
                                    "pose_confidence": _summarize_audit_value(ego.get("pose_confidence")) if isinstance(ego, dict) else None,
                                    "processed_lidar": _summarize_audit_value(ego.get("processed_lidar")) if isinstance(ego, dict) else None,
                                    "image_inputs": _summarize_audit_value(ego.get("image_inputs")) if isinstance(ego, dict) else None,
                                    "modality_inputs": modality_inputs,
                                },
                                "model_runtime_store": {
                                    "keys": sorted([str(k) for k in runtime_store.keys()]) if isinstance(runtime_store, dict) else [],
                                    "record_len": _summarize_audit_value(runtime_store.get("record_len")) if isinstance(runtime_store, dict) else None,
                                    "agent_modality_list": _summarize_audit_value(runtime_store.get("agent_modality_list")) if isinstance(runtime_store, dict) else None,
                                    "agent_bev": _summarize_audit_value(runtime_store.get("agent_bev")) if isinstance(runtime_store, dict) else None,
                                    "pairwise_t_matrix": _summarize_audit_value(runtime_store.get("pairwise_t_matrix")) if isinstance(runtime_store, dict) else None,
                                    "affine_matrix": _summarize_audit_value(runtime_store.get("affine_matrix")) if isinstance(runtime_store, dict) else None,
                                    "fused_feature": _summarize_audit_value(runtime_store.get("fused_feature")) if isinstance(runtime_store, dict) else None,
                                },
                            }
                            effective_input_audit_fp.write(json.dumps(audit_record, ensure_ascii=False) + "\n")
                            effective_input_audit_fp.flush()
                            effective_input_audit_written += 1
                            if bool(getattr(opt, "effective_input_audit_stop_after_write", False)):
                                cap_hit = (effective_input_audit_max_samples > 0) and (
                                    effective_input_audit_written >= effective_input_audit_max_samples
                                )
                                if cap_hit:
                                    print("[EFFECTIVE_INPUT_AUDIT] requested samples captured; exiting early.")
                                    if effective_input_audit_fp is not None:
                                        effective_input_audit_fp.close()
                                        effective_input_audit_fp = None
                                    return
                        except Exception as e:
                            print(f"[EFFECTIVE_INPUT_AUDIT] failed at i={i}: {type(e).__name__}: {e}")

                    pred_box_tensor = infer_result['pred_box_tensor']
                    gt_box_tensor = infer_result['gt_box_tensor']
                    pred_score = infer_result['pred_score']

                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.3)
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.5)
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.7)

                    try:
                        poses = batch_data['ego']['lidar_pose'].detach().cpu().numpy().reshape(-1, 6)
                        poses_clean = batch_data['ego']['lidar_pose_clean'].detach().cpu().numpy().reshape(-1, 6)
                        T_world = pose_to_tfm(poses)
                        T_world_clean = pose_to_tfm(poses_clean)
                        ego_T_world = T_world[0]
                        ego_T_world_clean = T_world_clean[0]
                        for cav_idx in range(1, min(T_world.shape[0], T_world_clean.shape[0])):
                            rel = np.linalg.inv(ego_T_world) @ T_world[cav_idx]
                            rel_clean = np.linalg.inv(ego_T_world_clean) @ T_world_clean[cav_idx]
                            err = np.linalg.inv(rel_clean) @ rel
                            rel_trans_errors.append(float(np.linalg.norm(err[:2, 3])))
                            yaw = float(np.degrees(np.arctan2(err[1, 0], err[0, 0])))
                            rel_yaw_errors.append(abs(yaw))
                    except Exception:
                        pass
                    vis_interval = int(getattr(opt, "save_vis_interval", 0) or 0)
                    if (
                        vis_interval > 0
                        and (i % vis_interval == 0)
                        and (pred_box_tensor is not None or gt_box_tensor is not None)
                        and (use_laplace is False)
                    ):
                        vis_save_path_root = os.path.join(benchmark_output_dir, f'vis_{noise_level}')
                        if not os.path.exists(vis_save_path_root):
                            os.makedirs(vis_save_path_root)

                        """ If you want to 3d vis, uncomment lines below """
                        # vis_save_path = os.path.join(vis_save_path_root, '3d_%05d.png' % i)
                        # simple_vis.visualize(infer_result,
                        #                     batch_data['ego'][
                        #                         'origin_lidar'][0],
                        #                     hypes['postprocess']['gt_range'],
                        #                     vis_save_path,
                        #                     method='3d',
                        #                     left_hand=left_hand)

                        vis_save_path = os.path.join(vis_save_path_root, 'bev_%05d.png' % i)
                        simple_vis.visualize(infer_result,
                                            batch_data['ego'][
                                                'origin_lidar'][0],
                                            hypes['postprocess']['gt_range'],
                                            vis_save_path,
                                            method='bev',
                                            left_hand=left_hand)

                torch.cuda.empty_cache()
                if opt.max_eval_samples is not None and sample_count >= int(opt.max_eval_samples):
                    break

            infer_elapsed = float(time.perf_counter() - infer_start)
            cuda_memory_stats = None
            if torch.cuda.is_available():
                try:
                    cuda_memory_stats = {
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                        "allocated_before_bytes": cuda_peak_allocated_before,
                        "reserved_before_bytes": cuda_peak_reserved_before,
                        "allocated_after_bytes": int(torch.cuda.memory_allocated()),
                        "reserved_after_bytes": int(torch.cuda.memory_reserved()),
                    }
                except Exception:
                    cuda_memory_stats = None
            timing_summary = {}
            if pose_timing:
                timing_summary = {
                    str(k): float(np.mean(v)) for k, v in pose_timing.items() if v
                }
            pose_total_sec = 0.0
            if timing_summary:
                pose_total_sec = float(sum(val for key, val in timing_summary.items() if str(key).endswith("_sec")))
            timing_stats = {
                "samples": int(sample_count),
                "infer_sec": float(infer_elapsed),
                "infer_fps": float(sample_count / infer_elapsed) if infer_elapsed > 0.0 and sample_count > 0 else None,
                "pose_sec": float(pose_total_sec) if pose_total_sec > 0.0 else None,
                "pose_fps": float(1.0 / pose_total_sec) if pose_total_sec > 0.0 else None,
                "pose_timing": timing_summary if timing_summary else None,
                "cuda_memory": cuda_memory_stats,
            }
            if pose_solver_metrics is not None:
                timing_stats["pose_solver"] = pose_solver_metrics

            ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat,
                                        benchmark_output_dir, noise_level)
            AP30.append(ap30)
            AP50.append(ap50)
            AP70.append(ap70)
            timing_stats_all.append(timing_stats)

            rel_stats = {}
            if rel_trans_errors:
                arr = np.asarray(rel_trans_errors, dtype=np.float64)
                rel_stats['rel_trans_m'] = {
                    'mean': float(np.mean(arr)),
                    'median': float(np.median(arr)),
                    'p90': float(np.percentile(arr, 90)),
                }
                thresholds_m = [1.0, 2.0, 3.0, 5.0, 10.0]
                rel_stats["rel_success_at_m"] = {
                    (str(int(thr)) if float(thr).is_integer() else str(thr)): float(np.mean(arr < float(thr)))
                    for thr in thresholds_m
                }
            if rel_yaw_errors:
                arr = np.asarray(rel_yaw_errors, dtype=np.float64)
                rel_stats['rel_yaw_deg'] = {
                    'mean': float(np.mean(arr)),
                    'median': float(np.median(arr)),
                    'p90': float(np.percentile(arr, 90)),
                }
            rel_error_stats_all.append(
                {
                    'pos_std': float(pos_std),
                    'rot_std': float(rot_std),
                    **rel_stats,
                }
            )

            dump_dict = {
                'pos_std_list': [float(p) for p, _ in noise_pairs],
                'rot_std_list': [float(r) for _, r in noise_pairs],
                'noise_target': str(opt.noise_target or "all"),
                'ap30': AP30,
                'ap50': AP50,
                'ap70': AP70,
                'pose_correction': pose_correction,
                'runtime_contract_attestation': runtime_contract_attestation,
                'timing_stats': timing_stats_all,
                'rel_error_stats': rel_error_stats_all,
            }
            tag_safe = str(opt.note or "").replace("/", "_").replace(" ", "")
            corr_safe = str(pose_correction or "none")
            yaml_utils.save_yaml(
                dump_dict,
                os.path.join(benchmark_output_dir, f'AP030507_{corr_safe}{suffix}{tag_safe}.yaml'),
            )

        if pose_solver_only:
            corr_safe = str(pose_correction or "none").replace("/", "_").replace(" ", "")
            tag_safe = str(opt.note or "").replace("/", "_").replace(" ", "")
            out_dir = pose_telemetry_dir or benchmark_output_dir
            yaml_utils.save_yaml(
                {
                    "pos_std_list": [float(p) for p, _ in noise_pairs],
                    "rot_std_list": [float(r) for _, r in noise_pairs],
                    "noise_target": str(opt.noise_target or "all"),
                    "pose_correction": str(pose_correction),
                    "runtime_contract_attestation": runtime_contract_attestation,
                    "pose_solver_only_stats": pose_solver_only_stats,
                },
                os.path.join(out_dir, f"POSE_SOLVER_ONLY_{corr_safe}{suffix}{tag_safe}.yaml"),
            )
            if effective_input_audit_fp is not None:
                effective_input_audit_fp.close()
            return

    if effective_input_audit_fp is not None:
        effective_input_audit_fp.close()


if __name__ == '__main__':
    main()
