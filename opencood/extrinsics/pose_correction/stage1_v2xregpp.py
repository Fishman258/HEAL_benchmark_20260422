from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from opencood.extrinsics.bbox_utils import corners_to_bbox3d_list
from opencood.extrinsics.path_utils import ensure_v2xreg_root_on_path, resolve_repo_path
from opencood.extrinsics.pose_correction.selection_policy import select_estimate_by_policy
from opencood.utils.transformation_utils import pose_to_tfm, tfm_to_pose


def _as_hw(raw: Any, *, default_hw: Tuple[int, int] = (256, 256)) -> Tuple[int, int]:
    if raw is None:
        return default_hw
    if isinstance(raw, int):
        side = max(1, int(raw))
        return side, side
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        try:
            h = max(1, int(raw[0]))
            w = max(1, int(raw[1]))
        except Exception:
            return default_hw
        return h, w
    return default_hw


def _wrap_angle_deg(angle: float) -> float:
    return float(((angle + 180.0) % 360.0) - 180.0)


def _delta_angle_deg(a: float, b: float) -> float:
    return float(_wrap_angle_deg(a - b))


def _topk_by_confidence(boxes: Sequence[object], k: int) -> List[object]:
    if k is None or int(k) <= 0 or len(boxes) <= int(k):
        return list(boxes)
    scored = []
    for idx, box in enumerate(boxes):
        conf = None
        if hasattr(box, "get_confidence"):
            try:
                conf = float(box.get_confidence())
            except Exception:
                conf = None
        if conf is None:
            try:
                conf = float(getattr(box, "confidence", 0.0) or 0.0)
            except Exception:
                conf = 0.0
        scored.append((conf, idx))
    scored.sort(key=lambda x: x[0], reverse=True)
    keep_idx = [idx for _, idx in scored[: int(k)]]
    return [boxes[i] for i in keep_idx]


def _extract_agent_indices_by_str(all_agent_ids: Sequence[Any], ego_id: Any, cav_id: Any) -> Optional[Tuple[int, int]]:
    ego_str = str(ego_id)
    cav_str = str(cav_id)
    ego_idx = None
    cav_idx = None
    for idx, agent_id in enumerate(all_agent_ids):
        if ego_idx is None and str(agent_id) == ego_str:
            ego_idx = idx
        if cav_idx is None and str(agent_id) == cav_str:
            cav_idx = idx
    if ego_idx is None or cav_idx is None:
        return None
    return int(ego_idx), int(cav_idx)


def _normalize_agent_role(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    name = str(raw).lower()
    if "veh" in name or "vehicle" in name:
        return "vehicle"
    if "infra" in name or "rsu" in name or "infrastructure" in name:
        return "infrastructure"
    return None


def _infer_dair_role_from_base(base_data_dict: Mapping[Any, Any], cav_id: Any) -> Optional[str]:
    entry = base_data_dict.get(cav_id)
    if not isinstance(entry, Mapping):
        return None
    params = entry.get("params") or {}
    if not isinstance(params, Mapping):
        return None
    vehicles_all = params.get("vehicles_all")
    if isinstance(vehicles_all, list):
        return "vehicle" if len(vehicles_all) > 0 else "infrastructure"
    vehicles_front = params.get("vehicles_front")
    if isinstance(vehicles_front, list):
        return "vehicle" if len(vehicles_front) > 0 else "infrastructure"
    return None


def _extract_agent_indices_by_role(
    all_agent_ids: Sequence[Any],
    *,
    ego_id: Any,
    cav_id: Any,
    base_data_dict: Mapping[Any, Any],
) -> Optional[Tuple[int, int]]:
    role_to_idx: Dict[str, int] = {}
    for idx, agent_id in enumerate(all_agent_ids):
        role = _normalize_agent_role(agent_id)
        if role and role not in role_to_idx:
            role_to_idx[role] = int(idx)
    if not role_to_idx:
        return None

    ego_role = _infer_dair_role_from_base(base_data_dict, ego_id)
    cav_role = _infer_dair_role_from_base(base_data_dict, cav_id)
    if ego_role is None or cav_role is None:
        return None

    ego_idx = role_to_idx.get(ego_role)
    cav_idx = role_to_idx.get(cav_role)
    if ego_idx is None or cav_idx is None:
        return None
    return int(ego_idx), int(cav_idx)


def _extract_agent_indices(all_agent_ids: Sequence[Any], ego_id: Any, cav_id: Any) -> Optional[Tuple[int, int]]:
    try:
        ego_idx = all_agent_ids.index(ego_id)
    except ValueError:
        return None
    try:
        cav_idx = all_agent_ids.index(cav_id)
    except ValueError:
        return None
    return ego_idx, cav_idx


def _extract_boxes(
    stage1_content: Mapping[str, Any],
    *,
    agent_idx: int,
    field: str,
    bbox_type: str,
) -> List[object]:
    preds = stage1_content.get(field) or []
    if not isinstance(preds, list) or agent_idx < 0 or agent_idx >= len(preds):
        return []
    boxes_raw = preds[agent_idx]
    if not isinstance(boxes_raw, list) or not boxes_raw:
        return []

    score_all = None
    for key in ("pred_score_np_list", "score_np_list", "scores_np_list"):
        if key in stage1_content:
            score_all = stage1_content.get(key)
            break
    scores = None
    if isinstance(score_all, list) and agent_idx < len(score_all) and isinstance(score_all[agent_idx], list):
        scores = score_all[agent_idx]

    corners_list = []
    score_list = []
    descriptor_list = []
    for box_idx, box in enumerate(boxes_raw):
        if isinstance(box, dict):
            corners = box.get("corners") or box.get("points") or box.get("bbox")
            if corners is None:
                continue
            corners_list.append(corners)
            score_list.append(float(box.get("score", box.get("confidence", 1.0))))
            descriptor_list.append(box.get("descriptor"))
        else:
            corners_list.append(box)
            if scores is not None and box_idx < len(scores):
                score_list.append(float(scores[box_idx]))
            else:
                score_list.append(1.0)
            descriptor_list.append(None)

    if not corners_list:
        return []
    corners_np = np.asarray(corners_list, dtype=np.float32)
    descriptors_np = []
    for d in descriptor_list:
        if d is None:
            descriptors_np.append(None)
        else:
            try:
                descriptors_np.append(np.asarray(d, dtype=np.float32).reshape(-1))
            except Exception:
                descriptors_np.append(None)
    return corners_to_bbox3d_list(
        corners_np,
        bbox_type=bbox_type,
        scores=score_list,
        descriptors=descriptors_np,
    )


def _new_pose_corr_stats(pair_total: int) -> Dict[str, int]:
    total = max(0, int(pair_total))
    return {
        "pose_corr_pair_total_count": total,
        "pose_corr_applied_pair_count": 0,
        "pose_corr_skip_empty_boxes_count": 0,
        "pose_corr_skip_no_matches_count": 0,
        "pose_corr_skip_svd_failed_count": 0,
        "pose_corr_skip_compare_gate_count": 0,
        "pose_corr_skip_exception_count": 0,
        "pose_corr_skip_other_count": 0,
    }


def _bump_pose_corr_skip(stats: Dict[str, int], *, reason: str) -> None:
    r = str(reason or "").lower().strip()
    if r in {"empty_boxes", "insufficient_boxes", "empty_centers", "empty_after_filter"}:
        stats["pose_corr_skip_empty_boxes_count"] += 1
        return
    if r in {"no_matches", "no_common_subgraph", "no_matches_after_distance_filter", "min_matches", "min_stability"}:
        stats["pose_corr_skip_no_matches_count"] += 1
        return
    if r in {"svd_failed"}:
        stats["pose_corr_skip_svd_failed_count"] += 1
        return
    if r in {"compare_gate", "compare_gate_reject", "current_pose_good_enough"}:
        stats["pose_corr_skip_compare_gate_count"] += 1
        return
    if "exception" in r:
        stats["pose_corr_skip_exception_count"] += 1
        return
    stats["pose_corr_skip_other_count"] += 1


def _extract_occ_map(
    stage1_content: Mapping[str, Any],
    *,
    agent_idx: int,
) -> Optional[np.ndarray]:
    if "occ_map_level0" in stage1_content:
        occ_raw = stage1_content.get("occ_map_level0")
        if isinstance(occ_raw, list) and agent_idx < len(occ_raw):
            occ_raw = occ_raw[agent_idx]
        if occ_raw is None:
            return None
        try:
            occ = np.asarray(occ_raw, dtype=np.float32)
        except Exception:
            return None
        if occ.ndim >= 3 and occ.shape[0] <= 16 and agent_idx < int(occ.shape[0]):
            occ = occ[int(agent_idx)]
        return occ
    if "occ_map_level0_path" in stage1_content:
        paths = stage1_content.get("occ_map_level0_path")
        if isinstance(paths, list) and agent_idx < len(paths):
            path = paths[agent_idx]
        else:
            path = paths
        if not path:
            return None
        resolved = resolve_repo_path(str(path))
        try:
            payload = np.load(str(resolved), allow_pickle=False)
        except Exception:
            return None
        if isinstance(payload, np.lib.npyio.NpzFile):
            for key in ("occ", "occ_map_level0", "arr_0"):
                if key in payload:
                    occ = np.asarray(payload[key], dtype=np.float32)
                    payload.close()
                    if occ.ndim >= 3 and occ.shape[0] <= 16 and agent_idx < int(occ.shape[0]):
                        occ = occ[int(agent_idx)]
                    return occ
            payload.close()
            return None
        occ = np.asarray(payload, dtype=np.float32)
        if occ.ndim >= 3 and occ.shape[0] <= 16 and agent_idx < int(occ.shape[0]):
            occ = occ[int(agent_idx)]
        return occ
    return None


def _build_lidar_occ_map(
    lidar_np: Any,
    *,
    bev_range: Sequence[float],
    grid_hw: Tuple[int, int],
) -> Optional[np.ndarray]:
    if lidar_np is None:
        return None
    try:
        pts = np.asarray(lidar_np, dtype=np.float32)
    except Exception:
        return None
    if pts.ndim != 2 or pts.shape[1] < 2:
        return None

    min_x, min_y, min_z, max_x, max_y, max_z = [float(v) for v in (bev_range or [])[:6]]
    extent_x = max(max_x - min_x, 1e-6)
    extent_y = max(max_y - min_y, 1e-6)
    H, W = _as_hw(grid_hw)

    x = pts[:, 0]
    y = pts[:, 1]
    mask = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
    if pts.shape[1] >= 3 and np.isfinite(min_z) and np.isfinite(max_z) and max_z > min_z:
        z = pts[:, 2]
        mask &= (z >= min_z) & (z <= max_z)

    if not np.any(mask):
        return np.zeros((H, W), dtype=np.float32)

    x = x[mask]
    y = y[mask]

    col = np.floor((x - min_x) / extent_x * float(W)).astype(np.int64, copy=False)
    row = np.floor((max_y - y) / extent_y * float(H)).astype(np.int64, copy=False)
    col = np.clip(col, 0, W - 1)
    row = np.clip(row, 0, H - 1)

    occ = np.zeros((H, W), dtype=np.float32)
    occ[row, col] = 1.0
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore

        occ = gaussian_filter(occ, sigma=1.0, mode="nearest")
    except Exception:
        pass
    return occ.astype(np.float32, copy=False)


def _estimate_occ_hint(
    occ_src: np.ndarray,
    occ_dst: np.ndarray,
    *,
    bev_range: Sequence[float],
    rotation_max_deg: float,
    rotation_step_deg: float,
    min_peak: float,
    min_peak_ratio: float,
) -> Optional[Dict[str, Any]]:
    ensure_v2xreg_root_on_path()
    from opencood.extrinsics.pose_estimation.v2x_calib.utils import convert_6DOF_to_T  # type: ignore

    def _squeeze(raw):
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim >= 3:
            arr = arr.squeeze()
        return arr

    occ_src = _squeeze(occ_src)
    occ_dst = _squeeze(occ_dst)
    if occ_src.ndim != 2 or occ_dst.ndim != 2:
        return None
    if occ_src.shape != occ_dst.shape:
        return None

    H, W = int(occ_src.shape[-2]), int(occ_src.shape[-1])
    if H <= 0 or W <= 0:
        return None

    # Keep higher spatial resolution for more accurate translation/yaw hints.
    # The default 256x256 BEV grid is still cheap to process with FFT.
    max_dim = 256
    stride = int(np.ceil(max(H, W) / float(max_dim))) if max(H, W) > max_dim else 1
    if stride > 1:
        occ_src = occ_src[::stride, ::stride]
        occ_dst = occ_dst[::stride, ::stride]
        H, W = int(occ_src.shape[-2]), int(occ_src.shape[-1])

    occ_dst_zm = occ_dst - float(np.mean(occ_dst))
    Fb_conj = np.conj(np.fft.fft2(occ_dst_zm))

    def _phase_corr(a, Fb_conj_local):
        a = a - float(np.mean(a))
        Fa = np.fft.fft2(a)
        R = Fa * Fb_conj_local
        R /= (np.abs(R) + 1e-6)
        corr = np.fft.ifft2(R)
        corr_abs = np.abs(corr)
        idx = np.unravel_index(int(np.argmax(corr_abs)), corr_abs.shape)
        peak = float(corr_abs[idx])

        def _parabola_offset(v_m1: float, v_0: float, v_p1: float) -> float:
            denom = (v_m1 - 2.0 * v_0 + v_p1)
            if abs(denom) < 1e-9:
                return 0.0
            delta = 0.5 * (v_m1 - v_p1) / denom
            if not np.isfinite(delta):
                return 0.0
            return float(np.clip(delta, -0.5, 0.5))

        row, col = int(idx[0]), int(idx[1])
        row_m1 = (row - 1) % H if H else row
        row_p1 = (row + 1) % H if H else row
        col_m1 = (col - 1) % W if W else col
        col_p1 = (col + 1) % W if W else col

        delta_row = _parabola_offset(
            float(corr_abs[row_m1, col]),
            float(corr_abs[row, col]),
            float(corr_abs[row_p1, col]),
        )
        delta_col = _parabola_offset(
            float(corr_abs[row, col_m1]),
            float(corr_abs[row, col]),
            float(corr_abs[row, col_p1]),
        )
        shift_row = float(row) + float(delta_row)
        shift_col = float(col) + float(delta_col)
        if shift_row > H / 2.0:
            shift_row -= float(H)
        if shift_col > W / 2.0:
            shift_col -= float(W)
        return peak, float(shift_row), float(shift_col)

    best_peak = -1.0
    best_shift_row = 0.0
    best_shift_col = 0.0
    best_yaw = 0.0
    second_peak = 0.0

    if rotation_max_deg > 0.0 and rotation_step_deg > 0.0:
        try:
            from scipy.ndimage import rotate as _rotate
        except Exception:
            _rotate = None
        if _rotate is None:
            return None
        angles = np.arange(-rotation_max_deg, rotation_max_deg + 1e-3, rotation_step_deg, dtype=np.float32)
        coarse = []
        for angle in angles.tolist():
            rotated = _rotate(
                occ_src,
                angle=float(angle),
                reshape=False,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            peak, shift_row, shift_col = _phase_corr(rotated, Fb_conj)
            coarse.append((float(peak), float(angle), float(shift_row), float(shift_col)))
        coarse.sort(key=lambda x: x[0], reverse=True)
        if len(coarse) > 1:
            second_peak = float(coarse[1][0])
        top_n = min(5, len(coarse))
        refine_step = max(0.5, float(rotation_step_deg) / 10.0)
        refine_radius = float(rotation_step_deg)
        for peak0, angle0, _, _ in coarse[:top_n]:
            angle_min = max(-rotation_max_deg, angle0 - refine_radius)
            angle_max = min(rotation_max_deg, angle0 + refine_radius)
            refine_angles = np.arange(angle_min, angle_max + 1e-3, refine_step, dtype=np.float32)
            for angle in refine_angles.tolist():
                rotated = _rotate(
                    occ_src,
                    angle=float(angle),
                    reshape=False,
                    order=1,
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                )
                peak, shift_row, shift_col = _phase_corr(rotated, Fb_conj)
                if peak > best_peak:
                    best_peak = peak
                    best_shift_row = shift_row
                    best_shift_col = shift_col
                    best_yaw = float(angle)

        # Final local refinement around the best yaw for sub-degree accuracy.
        fine_step = max(0.25, float(rotation_step_deg) / 30.0)
        fine_radius = max(1.0, min(2.0, float(rotation_step_deg)))
        angle_min = max(-rotation_max_deg, best_yaw - fine_radius)
        angle_max = min(rotation_max_deg, best_yaw + fine_radius)
        fine_angles = np.arange(angle_min, angle_max + 1e-3, fine_step, dtype=np.float32)
        for angle in fine_angles.tolist():
            rotated = _rotate(
                occ_src,
                angle=float(angle),
                reshape=False,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            peak, shift_row, shift_col = _phase_corr(rotated, Fb_conj)
            if peak > best_peak:
                best_peak = peak
                best_shift_row = shift_row
                best_shift_col = shift_col
                best_yaw = float(angle)
    else:
        best_peak, best_shift_row, best_shift_col = _phase_corr(occ_src, Fb_conj)

    if min_peak > 0.0 and best_peak < min_peak:
        return None
    if min_peak_ratio > 0.0:
        ratio = float(best_peak) / float(second_peak + 1e-6)
        if ratio < min_peak_ratio:
            return None
    else:
        ratio = float(best_peak) / float(second_peak + 1e-6) if second_peak > 0.0 else float("inf")

    extent_x = float(bev_range[3]) - float(bev_range[0])
    extent_y = float(bev_range[4]) - float(bev_range[1])
    resolution_x = extent_x / float(W) if W else 1.0
    resolution_y = extent_y / float(H) if H else 1.0
    offset = np.array(
        [
            -best_shift_col * resolution_x,
            -best_shift_row * resolution_y,
            0.0,
            0.0,
            0.0,
            -best_yaw,
        ],
        dtype=np.float32,
    )
    return {
        "T": convert_6DOF_to_T(offset),
        "peak": float(best_peak),
        "second_peak": float(second_peak),
        "peak_ratio": float(ratio),
        "shift_row": float(best_shift_row),
        "shift_col": float(best_shift_col),
        "yaw_deg": float(best_yaw),
        "stride": int(stride),
        "hw": (int(H), int(W)),
    }


def _estimate_occ_hint_T(
    occ_src: np.ndarray,
    occ_dst: np.ndarray,
    *,
    bev_range: Sequence[float],
    rotation_max_deg: float,
    rotation_step_deg: float,
    min_peak: float,
    min_peak_ratio: float,
) -> Optional[np.ndarray]:
    hint = _estimate_occ_hint(
        occ_src,
        occ_dst,
        bev_range=bev_range,
        rotation_max_deg=rotation_max_deg,
        rotation_step_deg=rotation_step_deg,
        min_peak=min_peak,
        min_peak_ratio=min_peak_ratio,
    )
    if hint is None:
        return None
    T = hint.get("T")
    if T is None:
        return None
    return np.asarray(T, dtype=np.float64)


def _icp_refine_T(
    src_lidar_np: Any,
    dst_lidar_np: Any,
    *,
    T_init: np.ndarray,
    voxel_size_m: float,
    max_corr_dist_m: float,
    max_iterations: int,
) -> Optional[np.ndarray]:
    if src_lidar_np is None or dst_lidar_np is None:
        return None
    try:
        src_pts = np.asarray(src_lidar_np, dtype=np.float64)
        dst_pts = np.asarray(dst_lidar_np, dtype=np.float64)
    except Exception:
        return None
    if src_pts.ndim != 2 or dst_pts.ndim != 2 or src_pts.shape[1] < 3 or dst_pts.shape[1] < 3:
        return None
    if src_pts.shape[0] < 50 or dst_pts.shape[0] < 50:
        return None

    try:
        import open3d as o3d  # type: ignore
    except Exception:
        return None

    def _to_pcd(pts: np.ndarray) -> "o3d.geometry.PointCloud":
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64, copy=False))
        return pcd

    source = _to_pcd(src_pts)
    target = _to_pcd(dst_pts)
    vs = float(voxel_size_m or 0.0)
    if vs > 0.0:
        source = source.voxel_down_sample(vs)
        target = target.voxel_down_sample(vs)

    if len(source.points) < 50 or len(target.points) < 50:
        return None

    max_corr = float(max_corr_dist_m or 0.0)
    if max_corr <= 0.0:
        return None
    max_it = max(1, int(max_iterations or 1))

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_it)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    try:
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_corr,
            np.asarray(T_init, dtype=np.float64),
            estimation,
            criteria,
        )
    except Exception:
        return None
    T = np.asarray(result.transformation, dtype=np.float64)
    if T.shape != (4, 4):
        return None
    return T


@dataclass
class Stage1V2XRegPPPoseCorrector:
    """
    Correct agent poses using V2X-Reg++ (box matching + robust SVD) with an optional mid-fusion hint.

    This corrector is designed to run inside HEAL dataset code, before `pairwise_t_matrix`
    is built, so that cooperative fusion consumes corrected transforms.
    """

    config_path: str
    stage1_field: str = "pred_corner3d_np_list"
    bbox_type: str = "detected"
    max_boxes: int = 0
    mode: str = "initfree"  # initfree | stable
    use_occ_hint: bool = False
    use_occ_pose: bool = False
    force_occ_pose: bool = False
    occ_from_lidar: bool = False
    occ_grid_hw: Tuple[int, int] = (256, 256)
    # Keep square pixels in physical space (important for yaw estimation via pixel-space rotation).
    # When occ_grid_hw is square, we will automatically adjust W/H to match bev_range aspect ratio.
    occ_preserve_aspect: bool = True
    occ_max_delta_xy_m: float = 20.0
    occ_max_delta_yaw_deg: float = 45.0
    icp_refine: bool = False
    icp_voxel_size_m: float = 1.0
    icp_max_corr_dist_m: float = 2.0
    icp_max_iterations: int = 30
    min_matches: int = 3
    min_stability: float = 0.0
    # Absolute quality gate on CorrespondingDetector precision. Keep at 0 to disable.
    # This is intentionally independent of the current (potentially noisy) pose.
    min_precision: float = 0.0
    # Optional: compare against current pose and keep the better alignment.
    compare_with_current: bool = False
    compare_distance_threshold_m: float = 3.0
    apply_if_current_precision_below: float = -1.0
    min_precision_improvement: float = 0.1
    min_matched_improvement: int = 1
    selection_policy: str = "solver_only"
    ema_alpha: float = 0.5
    max_step_xy_m: float = 3.0
    max_step_yaw_deg: float = 10.0
    freeze_ego: bool = True
    device: Optional[str] = None
    _state: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    # Cache the estimated relative transform per (sample_idx, cav_id) so noise sweeps
    # don't re-run expensive matching/occ correlation for each noise level.
    #
    # This is safe as long as the estimator inputs (stage1 boxes / raw lidar) do not
    # change across sweeps, which is the case in `inference_w_noise.py` where only
    # `lidar_pose` is perturbed.
    _rel_T_est_cache: Dict[Tuple[int, str], Optional[np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )

    telemetry_enable: bool = False
    telemetry_distance_threshold_m: float = 3.0
    telemetry_max_pairs: int = 0
    last_telemetry: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    last_stats: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        ensure_v2xreg_root_on_path()
        from opencood.extrinsics.pose_estimation.calib.config import load_config  # type: ignore
        from opencood.extrinsics.pose_estimation.calib.filters.pipeline import FilterPipeline  # type: ignore
        from opencood.extrinsics.pose_estimation.calib.matching.engine import MatchingEngine  # type: ignore

        cfg_path = resolve_repo_path(self.config_path)
        self._cfg = load_config(str(cfg_path))
        self._filters = FilterPipeline(self._cfg.filters)
        self._device = self.device
        self._matcher = MatchingEngine(self._cfg.matching, device=self._device)

        mode = str(self.mode or "initfree").lower().strip()
        if mode not in {"initfree", "stable"}:
            mode = "initfree"
        self.mode = mode

        self.min_matches = max(1, int(self.min_matches or 1))
        self.min_stability = float(self.min_stability or 0.0)
        self.min_precision = max(0.0, float(self.min_precision or 0.0))
        self.compare_with_current = bool(self.compare_with_current)
        self.compare_distance_threshold_m = float(self.compare_distance_threshold_m or 0.0)
        self.apply_if_current_precision_below = float(self.apply_if_current_precision_below)
        self.min_precision_improvement = float(self.min_precision_improvement or 0.0)
        self.min_matched_improvement = max(0, int(self.min_matched_improvement or 0))
        self.ema_alpha = float(self.ema_alpha or 0.0)
        self.ema_alpha = float(np.clip(self.ema_alpha, 0.0, 1.0))
        self.max_boxes = max(0, int(self.max_boxes or 0))
        self.max_step_xy_m = float(self.max_step_xy_m or 0.0)
        self.max_step_yaw_deg = float(self.max_step_yaw_deg or 0.0)
        self.use_occ_hint = bool(self.use_occ_hint)
        self.use_occ_pose = bool(self.use_occ_pose)
        self.force_occ_pose = bool(self.force_occ_pose)
        self.occ_from_lidar = bool(self.occ_from_lidar)
        self.occ_grid_hw = _as_hw(self.occ_grid_hw)
        self.occ_preserve_aspect = bool(self.occ_preserve_aspect)
        self.occ_max_delta_xy_m = float(self.occ_max_delta_xy_m or 0.0)
        self.occ_max_delta_yaw_deg = float(self.occ_max_delta_yaw_deg or 0.0)
        self.icp_refine = bool(self.icp_refine)
        self.icp_voxel_size_m = float(self.icp_voxel_size_m or 0.0)
        self.icp_max_corr_dist_m = float(self.icp_max_corr_dist_m or 0.0)
        self.icp_max_iterations = max(1, int(self.icp_max_iterations or 1))

        self.telemetry_enable = bool(self.telemetry_enable)
        self.telemetry_distance_threshold_m = float(self.telemetry_distance_threshold_m or 0.0)
        if self.telemetry_distance_threshold_m <= 0.0:
            self.telemetry_distance_threshold_m = float(self.compare_distance_threshold_m or 3.0)
        self.telemetry_max_pairs = max(0, int(self.telemetry_max_pairs or 0))

    def _reset_if_new_epoch(self, sample_idx: int) -> None:
        last = self._state.get("last_sample_idx")
        if last is None or int(sample_idx) < int(last):
            self._state.clear()
        self._state["last_sample_idx"] = int(sample_idx)

    def _get_prev_delta(self, key: str) -> Optional[Tuple[float, float, float]]:
        prev = self._state.get("delta_se2", {}).get(key)
        if prev is None:
            return None
        try:
            x, y, yaw = float(prev[0]), float(prev[1]), float(prev[2])
        except Exception:
            return None
        return x, y, yaw

    def _set_prev_delta(self, key: str, delta: Tuple[float, float, float]) -> None:
        store = self._state.setdefault("delta_se2", {})
        store[key] = (float(delta[0]), float(delta[1]), float(delta[2]))

    def _clear_prev_delta(self, key: str) -> None:
        store = self._state.get("delta_se2")
        if not isinstance(store, dict):
            return
        store.pop(key, None)

    def _smooth_se2(self, prev: Tuple[float, float, float], cur: Tuple[float, float, float]) -> Tuple[float, float, float]:
        a = self.ema_alpha
        if a <= 0.0:
            return prev
        if a >= 1.0:
            return cur
        x = (1.0 - a) * prev[0] + a * cur[0]
        y = (1.0 - a) * prev[1] + a * cur[1]
        dyaw = _delta_angle_deg(cur[2], prev[2])
        yaw = _wrap_angle_deg(prev[2] + a * dyaw)
        return float(x), float(y), float(yaw)

    def _limit_step_se2(self, prev: Tuple[float, float, float], cur: Tuple[float, float, float]) -> bool:
        if self.max_step_xy_m > 0.0:
            dx = float(cur[0] - prev[0])
            dy = float(cur[1] - prev[1])
            if float(np.hypot(dx, dy)) > self.max_step_xy_m + 1e-6:
                return False
        if self.max_step_yaw_deg > 0.0:
            if abs(_delta_angle_deg(cur[2], prev[2])) > self.max_step_yaw_deg + 1e-6:
                return False
        return True

    def _estimate_rel_T(
        self,
        src_boxes,
        dst_boxes,
        occ_src,
        occ_dst,
        bev_range,
        *,
        T_current: Optional[np.ndarray] = None,
        allow_gate_reject_return: bool = False,
    ) -> Optional[Dict[str, Any]]:
        ensure_v2xreg_root_on_path()
        from opencood.extrinsics.pose_estimation.v2x_calib.search import Matches2Extrinsics  # type: ignore
        from opencood.extrinsics.pose_estimation.v2x_calib.utils import convert_6DOF_to_T, implement_T_3dbox_object_list  # type: ignore
        from opencood.extrinsics.pose_estimation.v2x_calib.corresponding import CorrespondingDetector  # type: ignore

        filtered_src, filtered_dst = self._filters.apply(src_boxes or [], dst_boxes or [])
        has_boxes = bool(filtered_src) and bool(filtered_dst)

        occ_hint = None
        T_hint = None
        if (self.use_occ_hint or self.use_occ_pose) and occ_src is not None and occ_dst is not None:
            bev_range = list(bev_range or [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5])
            occ_hint = _estimate_occ_hint(
                occ_src,
                occ_dst,
                bev_range=bev_range,
                rotation_max_deg=float(getattr(self._cfg.matching, "occ_hint_rotation_max_deg", 0.0) or 0.0),
                rotation_step_deg=float(getattr(self._cfg.matching, "occ_hint_rotation_step_deg", 0.0) or 0.0),
                min_peak=float(getattr(self._cfg.matching, "occ_hint_min_peak", 0.0) or 0.0),
                min_peak_ratio=float(getattr(self._cfg.matching, "occ_hint_min_peak_ratio", 0.0) or 0.0),
            )
            if occ_hint is not None:
                try:
                    T_hint = np.asarray(occ_hint.get("T"), dtype=np.float64)
                except Exception:
                    T_hint = None

        if T_hint is not None and T_current is not None:
            max_xy = float(self.occ_max_delta_xy_m)
            max_yaw = float(self.occ_max_delta_yaw_deg)
            if max_xy > 0.0 or max_yaw > 0.0:
                try:
                    err = np.linalg.inv(np.asarray(T_current, dtype=np.float64)) @ np.asarray(T_hint, dtype=np.float64)
                    delta_xy = float(np.linalg.norm(err[:2, 3]))
                    delta_yaw = abs(float(np.degrees(np.arctan2(err[1, 0], err[0, 0]))))
                except Exception:
                    delta_xy = None
                    delta_yaw = None
                if delta_xy is not None and max_xy > 0.0 and delta_xy > max_xy + 1e-6:
                    T_hint = None
                    occ_hint = None
                elif delta_yaw is not None and max_yaw > 0.0 and delta_yaw > max_yaw + 1e-6:
                    T_hint = None
                    occ_hint = None

        if not has_boxes:
            if self.use_occ_pose and T_hint is not None:
                payload: Dict[str, Any] = {
                    "source": "occ",
                    "T": np.asarray(T_hint, dtype=np.float64),
                    "matches": [],
                    "matched": 0,
                    "stability": 0.0,
                    "precision": 0.0,
                }
                if occ_hint is not None:
                    payload.update({f"occ_{k}": v for k, v in occ_hint.items() if k != "T"})
                return payload
            return None

        def _quality_from_T_init(T_init, source: str):
            if T_init is None:
                return None
            try:
                converted = implement_T_3dbox_object_list(np.asarray(T_init, dtype=np.float64), filtered_src)
            except Exception:
                return None
            detector = CorrespondingDetector(
                converted,
                filtered_dst,
                distance_threshold=self._cfg.matching.distance_thresholds,
                parallel=getattr(self._cfg.matching, "corresponding_parallel", False),
                resolve_180_ambiguity=getattr(self._cfg.matching, "resolve_180_ambiguity", False),
            )
            try:
                matches_with_score = detector.get_matches_with_score()
            except Exception:
                matches_with_score = {}
            if not isinstance(matches_with_score, dict) or len(matches_with_score) < 2:
                return None
            refined_matches = []
            for pair, score in matches_with_score.items():
                try:
                    score_f = float(score)
                except Exception:
                    continue
                refined_matches.append((pair, float(np.exp(score_f))))
            refined_matches.sort(key=lambda x: x[1], reverse=True)
            if not refined_matches:
                return None
            solver_ref = Matches2Extrinsics(
                filtered_src,
                filtered_dst,
                matches_score_list=refined_matches,
                svd_strategy=self._cfg.matching.svd_strategy,
                resolve_180_ambiguity=getattr(self._cfg.matching, "resolve_180_ambiguity", False),
                max_iterations=getattr(self._cfg.solver, "max_iterations", 1),
                inlier_threshold_m=getattr(self._cfg.solver, "inlier_threshold_m", 0.0),
                mad_scale=getattr(self._cfg.solver, "mad_scale", 2.5),
                min_inliers=getattr(self._cfg.solver, "min_inliers", 1),
                device=self._device,
            )
            T6_ref = solver_ref.get_combined_extrinsic(matches2extrinsic_strategies="weightedSVD")
            T_ref = convert_6DOF_to_T(T6_ref)
            try:
                converted_ref = implement_T_3dbox_object_list(T_ref, filtered_src)
                detector_ref = CorrespondingDetector(
                    converted_ref,
                    filtered_dst,
                    distance_threshold=self._cfg.matching.distance_thresholds,
                    parallel=getattr(self._cfg.matching, "corresponding_parallel", False),
                    resolve_180_ambiguity=getattr(self._cfg.matching, "resolve_180_ambiguity", False),
                )
                precision_ref = float(detector_ref.get_distance_corresponding_precision())
                matched_ref = int(detector_ref.get_matched_num())
            except Exception:
                return None
            if matched_ref <= 0:
                return None
            return {
                "source": source,
                "T": np.asarray(T_ref, dtype=np.float64),
                "matches": refined_matches,
                "stability": float(refined_matches[0][1]) if refined_matches else 0.0,
                "precision": precision_ref,
                "matched": matched_ref,
            }

        if self.force_occ_pose and self.use_occ_pose and T_hint is not None:
            forced = _quality_from_T_init(T_hint, "occ_refined")
            if forced is None:
                forced = {
                    "source": "occ",
                    "T": np.asarray(T_hint, dtype=np.float64),
                    "matches": [],
                    "matched": 0,
                    "stability": 0.0,
                    "precision": 0.0,
                }
            if occ_hint is not None:
                forced.update({f"occ_{k}": v for k, v in occ_hint.items() if k != "T"})
            return forced

        matches_base, stability_base = self._matcher.compute(
            filtered_src,
            filtered_dst,
            T_hint=None,
            T_eval=None,
            sensor_combo="lidar-lidar",
        )

        matches_hint = []
        stability_hint = 0.0
        if self.use_occ_hint and T_hint is not None:
            matches_hint, stability_hint = self._matcher.compute(
                filtered_src,
                filtered_dst,
                T_hint=T_hint,
                T_eval=None,
                sensor_combo="lidar-lidar",
            )

        matches_aligned = []
        stability_aligned = 0.0
        if self.use_occ_hint and T_hint is not None:
            try:
                aligned_src = implement_T_3dbox_object_list(T_hint, filtered_src)
            except Exception:
                aligned_src = None
            if aligned_src is not None:
                matches_aligned, stability_aligned = self._matcher.compute(
                    aligned_src,
                    filtered_dst,
                    T_hint=None,
                    T_eval=None,
                    sensor_combo="lidar-lidar",
                )

        def _quality(matches, stability, source: str):
            if not matches:
                return None
            solver = Matches2Extrinsics(
                filtered_src,
                filtered_dst,
                matches_score_list=matches,
                svd_strategy=self._cfg.matching.svd_strategy,
                resolve_180_ambiguity=getattr(self._cfg.matching, "resolve_180_ambiguity", False),
                max_iterations=getattr(self._cfg.solver, "max_iterations", 1),
                inlier_threshold_m=getattr(self._cfg.solver, "inlier_threshold_m", 0.0),
                mad_scale=getattr(self._cfg.solver, "mad_scale", 2.5),
                min_inliers=getattr(self._cfg.solver, "min_inliers", 1),
                device=self._device,
            )
            T6 = solver.get_combined_extrinsic(matches2extrinsic_strategies=self._cfg.matching.matches2extrinsic)
            T_est = convert_6DOF_to_T(T6)
            try:
                converted = implement_T_3dbox_object_list(T_est, filtered_src)
            except Exception:
                return None
            detector = CorrespondingDetector(
                converted,
                filtered_dst,
                distance_threshold=self._cfg.matching.distance_thresholds,
                parallel=getattr(self._cfg.matching, "corresponding_parallel", False),
                resolve_180_ambiguity=getattr(self._cfg.matching, "resolve_180_ambiguity", False),
            )
            precision = float(detector.get_distance_corresponding_precision())
            matched = int(detector.get_matched_num())
            if matched <= 0:
                return None

            # Second-pass refinement: expand correspondences under the estimated transform and
            # re-solve using distance-weighted SVD for better stability/coverage.
            try:
                matches_with_score = detector.get_matches_with_score()
            except Exception:
                matches_with_score = {}
            if isinstance(matches_with_score, dict) and len(matches_with_score) >= 2:
                refined_matches = []
                for pair, score in matches_with_score.items():
                    try:
                        score_f = float(score)
                    except Exception:
                        continue
                    refined_matches.append((pair, float(np.exp(score_f))))
                refined_matches.sort(key=lambda x: x[1], reverse=True)
                if refined_matches:
                    solver_ref = Matches2Extrinsics(
                        filtered_src,
                        filtered_dst,
                        matches_score_list=refined_matches,
                        svd_strategy=self._cfg.matching.svd_strategy,
                        resolve_180_ambiguity=getattr(self._cfg.matching, "resolve_180_ambiguity", False),
                        max_iterations=getattr(self._cfg.solver, "max_iterations", 1),
                        inlier_threshold_m=getattr(self._cfg.solver, "inlier_threshold_m", 0.0),
                        mad_scale=getattr(self._cfg.solver, "mad_scale", 2.5),
                        min_inliers=getattr(self._cfg.solver, "min_inliers", 1),
                        device=self._device,
                    )
                    T6_ref = solver_ref.get_combined_extrinsic(matches2extrinsic_strategies="weightedSVD")
                    T_ref = convert_6DOF_to_T(T6_ref)
                    try:
                        converted_ref = implement_T_3dbox_object_list(T_ref, filtered_src)
                        detector_ref = CorrespondingDetector(
                            converted_ref,
                            filtered_dst,
                            distance_threshold=self._cfg.matching.distance_thresholds,
                            parallel=getattr(self._cfg.matching, "corresponding_parallel", False),
                            resolve_180_ambiguity=getattr(self._cfg.matching, "resolve_180_ambiguity", False),
                        )
                        precision_ref = float(detector_ref.get_distance_corresponding_precision())
                        matched_ref = int(detector_ref.get_matched_num())
                        if matched_ref > matched or (
                            matched_ref == matched and precision_ref > precision + 1e-9
                        ):
                            T_est = T_ref
                            precision = precision_ref
                            matched = matched_ref
                            matches = refined_matches
                    except Exception:
                        pass
            return {
                "source": source,
                "T": np.asarray(T_est, dtype=np.float64),
                "matches": matches,
                "stability": float(stability or 0.0),
                "precision": precision,
                "matched": matched,
            }

        def _quality_fixed_T(T_fixed, source: str):
            if T_fixed is None:
                return None
            try:
                converted = implement_T_3dbox_object_list(np.asarray(T_fixed, dtype=np.float64), filtered_src)
            except Exception:
                return None
            detector = CorrespondingDetector(
                converted,
                filtered_dst,
                distance_threshold=self._cfg.matching.distance_thresholds,
                parallel=getattr(self._cfg.matching, "corresponding_parallel", False),
                resolve_180_ambiguity=getattr(self._cfg.matching, "resolve_180_ambiguity", False),
            )
            precision = float(detector.get_distance_corresponding_precision())
            matched = int(detector.get_matched_num())
            if matched <= 0:
                return None
            return {
                "source": source,
                "T": np.asarray(T_fixed, dtype=np.float64),
                "matches": [],
                "stability": 0.0,
                "precision": precision,
                "matched": matched,
            }

        candidates = []
        cand_base = _quality(matches_base, stability_base, "late")
        if cand_base is not None:
            candidates.append(cand_base)
        cand_hint = _quality(matches_hint, stability_hint, "hint")
        if cand_hint is not None:
            candidates.append(cand_hint)
        cand_aligned = _quality(matches_aligned, stability_aligned, "aligned")
        if cand_aligned is not None:
            candidates.append(cand_aligned)

        # When requested, also treat the occ-hint as an explicit pose candidate and refine
        # correspondences under that hypothesis. This helps reduce reliance on the current pose
        # and can improve robustness when pure box matching is ambiguous.
        if self.use_occ_pose and T_hint is not None:
            cand_occ_refined = _quality_from_T_init(T_hint, "occ_refined")
            if cand_occ_refined is not None:
                if occ_hint is not None:
                    cand_occ_refined.update({f"occ_{k}": v for k, v in occ_hint.items() if k != "T"})
                candidates.append(cand_occ_refined)

        if self.use_occ_pose and T_hint is not None:
            cand_occ = _quality_fixed_T(T_hint, "occ")
            if cand_occ is not None:
                if occ_hint is not None:
                    cand_occ.update({f"occ_{k}": v for k, v in occ_hint.items() if k != "T"})
                candidates.append(cand_occ)
        if not candidates:
            if self.use_occ_pose and T_hint is not None:
                payload = {
                    "source": "occ",
                    "T": np.asarray(T_hint, dtype=np.float64),
                    "matches": [],
                    "matched": 0,
                    "stability": 0.0,
                    "precision": 0.0,
                }
                if occ_hint is not None:
                    payload.update({f"occ_{k}": v for k, v in occ_hint.items() if k != "T"})
                return payload
            return None

        best = candidates[0]
        for cand in candidates[1:]:
            if cand["precision"] > best["precision"] + 1e-9:
                best = cand
            elif abs(cand["precision"] - best["precision"]) <= 1e-9:
                if cand["matched"] > best["matched"]:
                    best = cand
                elif cand["matched"] == best["matched"] and cand["stability"] > best["stability"]:
                    best = cand

        if T_current is not None:
            use_compare_threshold = bool(self.compare_with_current) and float(self.compare_distance_threshold_m) > 0.0
            compare_threshold = None
            if use_compare_threshold:
                compare_threshold = {
                    str(self.bbox_type or "detected"): float(self.compare_distance_threshold_m)
                }

            def _match_stats_for_T(T_eval, *, threshold_override):
                if T_eval is None:
                    return None, None
                try:
                    aligned = implement_T_3dbox_object_list(np.asarray(T_eval, dtype=np.float64), filtered_src)
                except Exception:
                    return None, None
                try:
                    detector = CorrespondingDetector(
                        aligned,
                        filtered_dst,
                        distance_threshold=threshold_override,
                        parallel=getattr(self._cfg.matching, "corresponding_parallel", False),
                        resolve_180_ambiguity=getattr(self._cfg.matching, "resolve_180_ambiguity", False),
                    )
                    return (
                        float(detector.get_distance_corresponding_precision()),
                        int(detector.get_matched_num()),
                    )
                except Exception:
                    return None, None

            # When --pose-compare-current is enabled we compute precision/matched under a
            # single threshold profile (compare_distance_threshold_m) so the gate is
            # consistent with other stage1 correctors (freealign/vips/cbm).
            est_precision_gate = float(best.get("precision") or 0.0)
            est_matched_gate = int(best.get("matched") or 0)
            if compare_threshold is not None:
                est_precision2, est_matched2 = _match_stats_for_T(best.get("T"), threshold_override=compare_threshold)
                cur_precision2, cur_matched2 = _match_stats_for_T(T_current, threshold_override=compare_threshold)
                if est_precision2 is not None and est_matched2 is not None and cur_precision2 is not None and cur_matched2 is not None:
                    best["compare_precision"] = float(est_precision2)
                    best["compare_matched"] = int(est_matched2)
                    best["current_precision"] = float(cur_precision2)
                    best["current_matched"] = int(cur_matched2)
                    est_precision_gate = float(est_precision2)
                    est_matched_gate = int(est_matched2)

            try:
                if "current_precision" in best and "current_matched" in best:
                    cur_precision = float(best["current_precision"])
                    cur_matched = int(best["current_matched"])
                else:
                    cur_boxes = implement_T_3dbox_object_list(np.asarray(T_current, dtype=np.float64), filtered_src)
                    detector_cur = CorrespondingDetector(
                        cur_boxes,
                        filtered_dst,
                        distance_threshold=self._cfg.matching.distance_thresholds,
                        parallel=getattr(self._cfg.matching, "corresponding_parallel", False),
                        resolve_180_ambiguity=getattr(self._cfg.matching, "resolve_180_ambiguity", False),
                    )
                    cur_precision = float(detector_cur.get_distance_corresponding_precision())
                    cur_matched = int(detector_cur.get_matched_num())
            except Exception:
                cur_precision = None
                cur_matched = None

            if cur_precision is not None and cur_matched is not None:
                best.setdefault("current_precision", float(cur_precision))
                best.setdefault("current_matched", int(cur_matched))

                # Optional guard: if the current pose already aligns boxes well enough, avoid overriding it
                # with a potentially biased estimate from noisy detections.
                if float(self.apply_if_current_precision_below) >= 0.0:
                    # NOTE: CorrespondingDetector "precision" is an average negative distance in [-3, 0],
                    # where 0 is best and -3 means "no matches". Shift to [0, 3] so the CLI threshold
                    # (which is defined as a non-negative number) becomes meaningful.
                    cur_quality = float(cur_precision) + 3.0
                    if int(cur_matched) > 0 and float(cur_quality) > float(self.apply_if_current_precision_below):
                        if allow_gate_reject_return:
                            best["gate_reject"] = True
                            best["gate_reject_reason"] = "current_pose_good_enough"
                            return best
                        return None

                if int(cur_matched) > 0:
                    precision_ok = float(est_precision_gate) >= float(cur_precision) + float(self.min_precision_improvement)
                    matched_ok = int(est_matched_gate) >= int(cur_matched) + int(self.min_matched_improvement)
                    if not (precision_ok or matched_ok):
                        if allow_gate_reject_return:
                            best["gate_reject"] = True
                            best["gate_reject_reason"] = "compare_gate_reject"
                            best["gate_precision_ok"] = bool(precision_ok)
                            best["gate_matched_ok"] = bool(matched_ok)
                            return best
                        return None

        return best

    def apply(
        self,
        *,
        sample_idx: int,
        cav_id_list: Sequence[Any],
        base_data_dict: MutableMapping[Any, Dict[str, Any]],
        stage1_result: Mapping[str, Any],
    ) -> bool:
        """
        Update `base_data_dict[cav_id]['params']['lidar_pose']` in-place.

        Returns:
            bool: True if any pose updated, False otherwise.
        """
        stats = _new_pose_corr_stats(max(0, len(cav_id_list) - 1))
        self.last_stats = stats
        self._reset_if_new_epoch(int(sample_idx))
        self.last_telemetry = []

        key = str(sample_idx)
        stage1_content = stage1_result.get(key)
        if stage1_content is None:
            stats["pose_corr_skip_other_count"] = int(stats.get("pose_corr_pair_total_count") or 0)
            return False
        if not isinstance(stage1_content, Mapping):
            stats["pose_corr_skip_other_count"] = int(stats.get("pose_corr_pair_total_count") or 0)
            return False

        all_agent_ids = stage1_content.get("cav_id_list") or []
        all_agent_boxes = stage1_content.get(self.stage1_field) or []
        if not isinstance(all_agent_ids, list) or not isinstance(all_agent_boxes, list):
            stats["pose_corr_skip_other_count"] = int(stats.get("pose_corr_pair_total_count") or 0)
            return False
        if not all_agent_ids or not all_agent_boxes:
            stats["pose_corr_skip_other_count"] = int(stats.get("pose_corr_pair_total_count") or 0)
            return False

        if not cav_id_list:
            return False
        ego_id = cav_id_list[0]

        if ego_id not in base_data_dict:
            stats["pose_corr_skip_other_count"] = int(stats.get("pose_corr_pair_total_count") or 0)
            return False

        bev_range = stage1_content.get("bev_range") or [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5]
        # For occ-hint, yaw is estimated by rotating the pixel grid. This only matches physical
        # yaw when pixels represent square meters. If the user configured a square grid (H==W),
        # we adjust W/H to match bev_range aspect ratio so resolution_x == resolution_y.
        occ_grid_hw = self.occ_grid_hw
        if self.occ_preserve_aspect:
            try:
                H, W = _as_hw(occ_grid_hw)
                if int(H) > 0 and int(W) == int(H):
                    extent_x = float(bev_range[3]) - float(bev_range[0])
                    extent_y = float(bev_range[4]) - float(bev_range[1])
                    if extent_y > 1e-6 and extent_x > 1e-6:
                        W_new = int(round(float(H) * extent_x / extent_y))
                        if W_new > 0:
                            occ_grid_hw = (int(H), int(W_new))
            except Exception:
                occ_grid_hw = self.occ_grid_hw
        lidar_occ_cache: Dict[Any, Optional[np.ndarray]] = {}

        updated_any = False
        ego_pose = base_data_dict[ego_id]["params"]["lidar_pose"]
        ego_T_world = pose_to_tfm(np.asarray([ego_pose], dtype=np.float64))[0]

        for cav_id in cav_id_list:
            if cav_id == ego_id and self.freeze_ego:
                continue
            if cav_id not in base_data_dict:
                _bump_pose_corr_skip(stats, reason="other")
                continue
            idx_pair = _extract_agent_indices(all_agent_ids, ego_id, cav_id)
            if idx_pair is None:
                idx_pair = _extract_agent_indices_by_str(all_agent_ids, ego_id, cav_id)
            if idx_pair is None:
                idx_pair = _extract_agent_indices_by_role(
                    all_agent_ids,
                    ego_id=ego_id,
                    cav_id=cav_id,
                    base_data_dict=base_data_dict,
                )
            if idx_pair is None:
                _bump_pose_corr_skip(stats, reason="other")
                continue
            ego_idx, cav_idx = idx_pair

            rel_key = str(cav_id)
            prev_delta = self._get_prev_delta(rel_key) if self.mode == "stable" else None

            cav_pose_current = base_data_dict[cav_id]["params"]["lidar_pose"]
            cav_T_world_current = pose_to_tfm(np.asarray([cav_pose_current], dtype=np.float64))[0]
            rel_current_T = np.linalg.inv(ego_T_world) @ cav_T_world_current

            rel_T_corrected: Optional[np.ndarray] = None
            rel_T_est: Optional[np.ndarray] = None
            rel_T_est_raw: Optional[np.ndarray] = None
            gate_reason: Optional[str] = None
            src_boxes = []
            dst_boxes = []

            use_cache = not bool(self.telemetry_enable)

            cache_key = (int(sample_idx), str(cav_id))
            missing = object()
            cached = self._rel_T_est_cache.get(cache_key, missing) if use_cache else missing
            if cached is not missing:
                rel_T_est = None if cached is None else np.asarray(cached, dtype=np.float64)
            else:
                dst_boxes = _extract_boxes(
                    stage1_content, agent_idx=ego_idx, field=self.stage1_field, bbox_type=self.bbox_type
                )
                src_boxes = _extract_boxes(
                    stage1_content, agent_idx=cav_idx, field=self.stage1_field, bbox_type=self.bbox_type
                )
                if int(self.max_boxes or 0) > 0:
                    dst_boxes = _topk_by_confidence(dst_boxes, int(self.max_boxes))
                    src_boxes = _topk_by_confidence(src_boxes, int(self.max_boxes))

                occ_dst = None
                occ_src = None
                if self.use_occ_hint or self.use_occ_pose:
                    occ_dst = _extract_occ_map(stage1_content, agent_idx=ego_idx)
                    occ_src = _extract_occ_map(stage1_content, agent_idx=cav_idx)
                if self.occ_from_lidar:
                    if occ_dst is None:
                        if ego_id not in lidar_occ_cache:
                            lidar_occ_cache[ego_id] = _build_lidar_occ_map(
                                base_data_dict.get(ego_id, {}).get("lidar_np"),
                                bev_range=bev_range,
                                grid_hw=occ_grid_hw,
                            )
                        occ_dst = lidar_occ_cache.get(ego_id)
                    if occ_src is None:
                        if cav_id not in lidar_occ_cache:
                            lidar_occ_cache[cav_id] = _build_lidar_occ_map(
                                base_data_dict.get(cav_id, {}).get("lidar_np"),
                                bev_range=bev_range,
                                grid_hw=occ_grid_hw,
                            )
                        occ_src = lidar_occ_cache.get(cav_id)

                # "initfree" should be independent of the (potentially noisy) current pose, otherwise
                # the update decision can vary with injected noise even if the estimator itself doesn't.
                #
                # We keep T_current for stable mode (used for delta correction). For initfree, only
                # pass T_current when explicitly requested via --pose-compare-current (compare_with_current).
                T_current_for_est = rel_current_T if (self.mode == "stable" or bool(self.compare_with_current)) else None
                est = self._estimate_rel_T(
                    src_boxes,
                    dst_boxes,
                    occ_src,
                    occ_dst,
                    bev_range,
                    T_current=T_current_for_est,
                    allow_gate_reject_return=bool(self.telemetry_enable),
                )

                if est is not None:
                    if self.icp_refine:
                        T_init = np.asarray(est.get("T"), dtype=np.float64)
                        T_icp = _icp_refine_T(
                            base_data_dict.get(cav_id, {}).get("lidar_np"),
                            base_data_dict.get(ego_id, {}).get("lidar_np"),
                            T_init=T_init,
                            voxel_size_m=self.icp_voxel_size_m,
                            max_corr_dist_m=self.icp_max_corr_dist_m,
                            max_iterations=self.icp_max_iterations,
                        )
                        if T_icp is not None:
                            est["T"] = T_icp
                    rel_T_est = np.asarray(est.get("T"), dtype=np.float64) if est.get("T") is not None else None
                    rel_T_est_raw = None if rel_T_est is None else np.asarray(rel_T_est, dtype=np.float64)
                    if bool(est.get("gate_reject")):
                        gate_reason = str(est.get("gate_reject_reason") or "compare_gate_reject")
                        rel_T_est = None

                    if str(est.get("source")) != "occ":
                        matches_count = int(est.get("matched") or len(est.get("matches") or []))
                        stability = float(est.get("stability") or 0.0)
                        if matches_count < self.min_matches or stability < self.min_stability:
                            if gate_reason is None:
                                gate_reason = "min_matches" if matches_count < self.min_matches else "min_stability"
                            rel_T_est = None

                    if rel_T_est is not None and self.min_precision > 0.0:
                        try:
                            precision = float(est.get("precision") or 0.0)
                        except Exception:
                            precision = 0.0
                        if precision < float(self.min_precision) - 1e-9:
                            if gate_reason is None:
                                gate_reason = "min_precision"
                            rel_T_est = None
                else:
                    if gate_reason is None:
                        gate_reason = 'no_estimate'


                if use_cache:
                    self._rel_T_est_cache[cache_key] = None if rel_T_est is None else np.asarray(rel_T_est, dtype=np.float64)

            telemetry_row = None
            if self.telemetry_enable:
                from opencood.extrinsics.pose_correction.telemetry_utils import compute_match_pairs, se2_from_tfm, serialize_tfm

                telemetry_row = {
                    'sample_idx': int(sample_idx),
                    'ego_id': str(ego_id),
                    'cav_id': str(cav_id),
                    'mode': str(self.mode),
                    'bbox_type': str(self.bbox_type),
                    'compare_with_current': bool(self.compare_with_current),
                    'rel_T_current': serialize_tfm(rel_current_T),
                    'rel_se2_current': se2_from_tfm(rel_current_T),
                    'rel_T_est_raw': serialize_tfm(rel_T_est_raw),
                    'rel_se2_est_raw': se2_from_tfm(rel_T_est_raw),
                    'rel_T_est_filtered': serialize_tfm(rel_T_est),
                    'rel_se2_est_filtered': se2_from_tfm(rel_T_est),
                    'gate_reason': gate_reason,
                    'src_box_count': int(len(src_boxes)),
                    'dst_box_count': int(len(dst_boxes)),
                }
                telemetry_row['matches_current'] = compute_match_pairs(
                    src_boxes,
                    dst_boxes,
                    rel_current_T,
                    bbox_type=self.bbox_type,
                    distance_threshold_m=self.telemetry_distance_threshold_m,
                    topk_pairs=self.telemetry_max_pairs,
                )
                telemetry_row['matches_est_raw'] = compute_match_pairs(
                    src_boxes,
                    dst_boxes,
                    rel_T_est_raw,
                    bbox_type=self.bbox_type,
                    distance_threshold_m=self.telemetry_distance_threshold_m,
                    topk_pairs=self.telemetry_max_pairs,
                )

            if self.mode == "stable":
                if rel_T_est is not None:
                    try:
                        delta_T = np.asarray(rel_T_est, dtype=np.float64) @ np.linalg.inv(np.asarray(rel_current_T, dtype=np.float64))
                        delta_pose6 = tfm_to_pose(delta_T)
                        delta_xyyaw = (
                            float(delta_pose6[0]),
                            float(delta_pose6[1]),
                            _wrap_angle_deg(float(delta_pose6[4])),
                        )
                    except Exception:
                        delta_xyyaw = None

                    if delta_xyyaw is not None:
                        if prev_delta is not None:
                            if not self._limit_step_se2(prev_delta, delta_xyyaw):
                                delta_xyyaw = prev_delta
                            else:
                                delta_xyyaw = self._smooth_se2(prev_delta, delta_xyyaw)
                        self._set_prev_delta(rel_key, delta_xyyaw)
                        prev_delta = delta_xyyaw

                if prev_delta is not None:
                    delta_T_se2 = pose_to_tfm(np.asarray([[prev_delta[0], prev_delta[1], prev_delta[2]]], dtype=np.float64))[0]
                    rel_T_corrected = np.asarray(delta_T_se2, dtype=np.float64) @ np.asarray(rel_current_T, dtype=np.float64)
                else:
                    if telemetry_row is not None:
                        telemetry_row["applied"] = False
                        telemetry_row["rel_T_applied"] = None
                        telemetry_row["rel_se2_applied"] = None
                        self.last_telemetry.append(telemetry_row)
                    continue
            else:
                if rel_T_est is None:
                    _bump_pose_corr_skip(stats, reason=gate_reason or "other")
                    if telemetry_row is not None:
                        telemetry_row["applied"] = False
                        telemetry_row["rel_T_applied"] = None
                        telemetry_row["rel_se2_applied"] = None
                        self.last_telemetry.append(telemetry_row)
                    continue
                rel_T_corrected = rel_T_est

            if not select_estimate_by_policy(
                selection_policy=self.selection_policy,
                base_data_dict=base_data_dict,
                ego_id=ego_id,
                cav_id=cav_id,
                T_current=rel_current_T,
                T_est=rel_T_corrected,
                telemetry_row=telemetry_row,
            ):
                _bump_pose_corr_skip(stats, reason=gate_reason or "other")
                if telemetry_row is not None:
                    telemetry_row["applied"] = False
                    telemetry_row["rel_T_applied"] = None
                    telemetry_row["rel_se2_applied"] = None
                    self.last_telemetry.append(telemetry_row)
                continue

            cav_T_world_new = ego_T_world @ np.asarray(rel_T_corrected, dtype=np.float64)
            cav_pose_new = tfm_to_pose(cav_T_world_new)
            cav_pose = list(base_data_dict[cav_id]["params"]["lidar_pose"])
            cav_pose[0] = float(cav_pose_new[0])
            cav_pose[1] = float(cav_pose_new[1])
            cav_pose[4] = float(cav_pose_new[4])
            base_data_dict[cav_id]["params"]["lidar_pose"] = cav_pose
            updated_any = True
            stats["pose_corr_applied_pair_count"] += 1
            if telemetry_row is not None:
                from opencood.extrinsics.pose_correction.telemetry_utils import se2_from_tfm, serialize_tfm

                telemetry_row["applied"] = True
                telemetry_row["rel_T_applied"] = serialize_tfm(rel_T_corrected)
                telemetry_row["rel_se2_applied"] = se2_from_tfm(rel_T_corrected)
                self.last_telemetry.append(telemetry_row)

        return updated_any


__all__ = ["Stage1V2XRegPPPoseCorrector"]
