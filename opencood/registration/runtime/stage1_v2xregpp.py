from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from opencood.registration.utils.bbox import corners_to_bbox3d_list
from opencood.utils.path_utils import resolve_repo_path
from opencood.registration.runtime.selection_policy import select_estimate_by_policy
from opencood.registration.estimators.v2xregpp import (
    V2XRegPPRuntimeEstimator,
    build_lidar_occ_map,
)
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
    _estimator: V2XRegPPRuntimeEstimator = field(default=None, init=False, repr=False)
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
        self._estimator = V2XRegPPRuntimeEstimator(
            config_path=self.config_path,
            bbox_type=self.bbox_type,
            use_occ_hint=self.use_occ_hint,
            use_occ_pose=self.use_occ_pose,
            force_occ_pose=self.force_occ_pose,
            occ_max_delta_xy_m=self.occ_max_delta_xy_m,
            occ_max_delta_yaw_deg=self.occ_max_delta_yaw_deg,
            compare_with_current=self.compare_with_current,
            compare_distance_threshold_m=self.compare_distance_threshold_m,
            apply_if_current_precision_below=self.apply_if_current_precision_below,
            min_precision_improvement=self.min_precision_improvement,
            min_matched_improvement=self.min_matched_improvement,
            icp_refine=self.icp_refine,
            icp_voxel_size_m=self.icp_voxel_size_m,
            icp_max_corr_dist_m=self.icp_max_corr_dist_m,
            icp_max_iterations=self.icp_max_iterations,
            device=self.device,
        )

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
                            lidar_occ_cache[ego_id] = build_lidar_occ_map(
                                base_data_dict.get(ego_id, {}).get("lidar_np"),
                                bev_range=bev_range,
                                grid_hw=occ_grid_hw,
                            )
                        occ_dst = lidar_occ_cache.get(ego_id)
                    if occ_src is None:
                        if cav_id not in lidar_occ_cache:
                            lidar_occ_cache[cav_id] = build_lidar_occ_map(
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
                est = self._estimator.estimate_rel_T(
                    src_boxes,
                    dst_boxes,
                    occ_src,
                    occ_dst,
                    bev_range,
                    T_current=T_current_for_est,
                    allow_gate_reject_return=bool(self.telemetry_enable),
                    src_lidar_np=base_data_dict.get(cav_id, {}).get("lidar_np"),
                    dst_lidar_np=base_data_dict.get(ego_id, {}).get("lidar_np"),
                )

                if est is not None:
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
                from opencood.registration.runtime.telemetry_utils import compute_match_pairs, se2_from_tfm, serialize_tfm

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
                from opencood.registration.runtime.telemetry_utils import se2_from_tfm, serialize_tfm

                telemetry_row["applied"] = True
                telemetry_row["rel_T_applied"] = serialize_tfm(rel_T_corrected)
                telemetry_row["rel_se2_applied"] = se2_from_tfm(rel_T_corrected)
                self.last_telemetry.append(telemetry_row)

        return updated_any


__all__ = ["Stage1V2XRegPPPoseCorrector"]
