from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from opencood.registration.utils.bbox import corners_to_bbox3d_list
from opencood.registration.runtime.selection_policy import select_estimate_by_policy
from opencood.registration.estimators.freealign_paper import FreeAlignPaperConfig, FreeAlignPaperEstimator
from opencood.registration.estimators.freealign_repo import FreeAlignRepoConfig, FreeAlignRepoEstimator
from opencood.utils.box_utils import project_box3d
from opencood.utils.transformation_utils import pose_to_tfm, tfm_to_pose, x1_to_x2, x_to_world


def _wrap_angle_deg(angle: float) -> float:
    return float(((angle + 180.0) % 360.0) - 180.0)


def _delta_angle_deg(a: float, b: float) -> float:
    return float(_wrap_angle_deg(a - b))


def _smooth_se2(
    prev: Tuple[float, float, float],
    cur: Tuple[float, float, float],
    *,
    alpha: float,
) -> Tuple[float, float, float]:
    a = float(alpha)
    if a <= 0.0:
        return prev
    if a >= 1.0:
        return cur
    x = (1.0 - a) * float(prev[0]) + a * float(cur[0])
    y = (1.0 - a) * float(prev[1]) + a * float(cur[1])
    dyaw = _delta_angle_deg(float(cur[2]), float(prev[2]))
    yaw = _wrap_angle_deg(float(prev[2]) + a * dyaw)
    return float(x), float(y), float(yaw)


def _limit_step_se2(
    prev: Tuple[float, float, float],
    cur: Tuple[float, float, float],
    *,
    max_step_xy_m: float,
    max_step_yaw_deg: float,
) -> bool:
    max_xy = float(max_step_xy_m or 0.0)
    if max_xy > 0.0:
        dx = float(cur[0] - prev[0])
        dy = float(cur[1] - prev[1])
        if float(np.hypot(dx, dy)) > max_xy + 1e-6:
            return False
    max_yaw = float(max_step_yaw_deg or 0.0)
    if max_yaw > 0.0:
        if abs(_delta_angle_deg(float(cur[2]), float(prev[2]))) > max_yaw + 1e-6:
            return False
    return True


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
    if r in {"insufficient_boxes", "empty_centers", "empty_boxes", "empty_after_filter"}:
        stats["pose_corr_skip_empty_boxes_count"] += 1
        return
    if r in {"no_common_subgraph", "no_matches", "no_matches_after_distance_filter", "no_temporal_match"}:
        stats["pose_corr_skip_no_matches_count"] += 1
        return
    if r in {"compare_gate", "compare_gate_reject"}:
        stats["pose_corr_skip_compare_gate_count"] += 1
        return
    if "exception" in r:
        stats["pose_corr_skip_exception_count"] += 1
        return
    stats["pose_corr_skip_other_count"] += 1


def _compute_match_stats(src_boxes, dst_boxes, T_rel: np.ndarray, *, bbox_type: str, distance_threshold_m: float):
    if not src_boxes or not dst_boxes:
        return None, None
    try:
        from opencood.utils.path_utils import ensure_v2xreg_root_on_path

        ensure_v2xreg_root_on_path()
        from opencood.registration.utils.geometry import implement_T_3dbox_object_list  # type: ignore
        from opencood.registration.estimators.box_matching import CorrespondingDetector  # type: ignore
    except Exception:
        return None, None

    try:
        aligned_src = implement_T_3dbox_object_list(np.asarray(T_rel, dtype=np.float64), src_boxes)
        bbox_key = str(bbox_type or "detected")
        threshold = {bbox_key: float(distance_threshold_m)}
        detector = CorrespondingDetector(aligned_src, dst_boxes, distance_threshold=threshold)
        precision = float(detector.get_distance_corresponding_precision())
        matched = int(detector.get_matched_num())
        return precision, matched
    except Exception:
        return None, None


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


def _extract_agent_indices(
    all_agent_ids: Sequence[Any],
    *,
    ego_id: Any,
    cav_id: Any,
    base_data_dict: Mapping[Any, Any],
) -> Optional[Tuple[int, int]]:
    try:
        ego_idx = all_agent_ids.index(ego_id)
        cav_idx = all_agent_ids.index(cav_id)
        return int(ego_idx), int(cav_idx)
    except ValueError:
        pass

    ego_str = str(ego_id)
    cav_str = str(cav_id)
    ego_idx = None
    cav_idx = None
    for idx, agent_id in enumerate(all_agent_ids):
        if ego_idx is None and str(agent_id) == ego_str:
            ego_idx = int(idx)
        if cav_idx is None and str(agent_id) == cav_str:
            cav_idx = int(idx)
    if ego_idx is not None and cav_idx is not None:
        return int(ego_idx), int(cav_idx)

    role_to_idx: Dict[str, int] = {}
    for idx, agent_id in enumerate(all_agent_ids):
        role = _normalize_agent_role(agent_id)
        if role and role not in role_to_idx:
            role_to_idx[role] = int(idx)
    if role_to_idx:
        ego_role = _infer_dair_role_from_base(base_data_dict, ego_id)
        cav_role = _infer_dair_role_from_base(base_data_dict, cav_id)
        if ego_role and cav_role and ego_role in role_to_idx and cav_role in role_to_idx:
            return int(role_to_idx[ego_role]), int(role_to_idx[cav_role])
    return None


def _extract_boxes(
    stage1_content: Mapping[str, Any],
    *,
    agent_idx: int,
    bbox_type: str,
) -> List[object]:
    corners_all = stage1_content.get("pred_corner3d_np_list")
    if not isinstance(corners_all, list) or agent_idx < 0 or agent_idx >= len(corners_all):
        return []
    corners_list = corners_all[agent_idx]
    if not isinstance(corners_list, list) or not corners_list:
        return []

    scores_all = stage1_content.get("pred_score_np_list")
    scores = None
    if isinstance(scores_all, list) and agent_idx < len(scores_all) and isinstance(scores_all[agent_idx], list):
        scores = [float(x) for x in scores_all[agent_idx]]

    corners_np = np.asarray(corners_list, dtype=np.float32)
    if corners_np.ndim != 3 or corners_np.shape[1:] != (8, 3):
        return []
    if scores is not None and len(scores) != int(corners_np.shape[0]):
        scores = None
    return corners_to_bbox3d_list(corners_np, bbox_type=bbox_type, scores=scores)


def _frame_id_to_int(frame_id: Any) -> Optional[int]:
    if frame_id is None:
        return None
    try:
        return int(str(frame_id))
    except Exception:
        return None


def _extract_pose_from_stage1(
    stage1_content: Mapping[str, Any],
    *,
    agent_idx: int,
    use_clean: bool,
) -> Optional[Sequence[float]]:
    key = "lidar_pose_clean_np" if use_clean else "lidar_pose_np"
    poses = stage1_content.get(key)
    if not isinstance(poses, list) or agent_idx < 0 or agent_idx >= len(poses):
        return None
    pose = poses[agent_idx]
    if not isinstance(pose, (list, tuple)) or len(pose) < 6:
        return None
    return pose


def _transform_boxes_to_current(
    boxes: Sequence[object],
    *,
    T_cur_from_past: np.ndarray,
) -> List[object]:
    if not boxes:
        return []
    corners = []
    scores = []
    bbox_type = None
    for bbox in boxes:
        try:
            corners.append(np.asarray(bbox.get_bbox3d_8_3(), dtype=np.float32))
            scores.append(float(getattr(bbox, "confidence", 1.0)))
            if bbox_type is None:
                bbox_type = str(bbox.get_bbox_type())
        except Exception:
            continue
    if not corners:
        return []
    corners_np = np.stack(corners, axis=0)
    projected = project_box3d(corners_np, T_cur_from_past)
    return corners_to_bbox3d_list(projected, bbox_type=bbox_type or "detected", scores=scores)


@dataclass
class Stage1FreeAlignPoseCorrector:
    """
    Apply FreeAlign pose correction using cached stage-1 boxes.

    This keeps FreeAlign logic out of dataset code.
    """

    cfg: FreeAlignPaperConfig = field(default_factory=FreeAlignPaperConfig)
    bbox_type: str = "detected"

    _estimator: FreeAlignPaperEstimator = field(init=False, repr=False)
    _state: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    # Cache per (sample_idx, cav_id) to avoid repeating matching in noise sweeps.
    _rel_T_est_cache: Dict[Tuple[int, str], Optional[np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )
    last_stats: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    last_stats: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    telemetry_enable: bool = False
    telemetry_distance_threshold_m: float = 0.0
    telemetry_max_pairs: int = 0
    selection_policy: str = "solver_only"
    last_telemetry: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._estimator = FreeAlignPaperEstimator(self.cfg)

    def _reset_if_new_epoch(self, sample_idx: int) -> None:
        last = self._state.get("last_sample_idx")
        if last is None or int(sample_idx) < int(last):
            self._state.clear()
        self._state["last_sample_idx"] = int(sample_idx)

    def _get_prev_delta(self, key: str) -> Optional[Tuple[float, float, float]]:
        store = self._state.get("delta_se2", {})
        if not isinstance(store, dict):
            return None
        prev = store.get(str(key))
        if prev is None:
            return None
        try:
            x, y, yaw = float(prev[0]), float(prev[1]), float(prev[2])
        except Exception:
            return None
        return x, y, yaw

    def _set_prev_delta(self, key: str, delta: Tuple[float, float, float]) -> None:
        store = self._state.setdefault("delta_se2", {})
        if not isinstance(store, dict):
            return
        store[str(key)] = (float(delta[0]), float(delta[1]), float(delta[2]))

    def _ensure_time_index(self, stage1_result: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        cache = self._state.get("time_index")
        cache_id = self._state.get("time_index_id")
        if cache is not None and cache_id == id(stage1_result):
            return cache

        time_index: Dict[str, Dict[str, List[Tuple[int, str]]]] = {}
        for key, entry in stage1_result.items():
            if not isinstance(entry, Mapping):
                continue
            seq_id = entry.get("sequence_id")
            seq_key = str(seq_id) if seq_id is not None else "default"
            veh_frame = _frame_id_to_int(entry.get("veh_frame_id"))
            infra_frame = _frame_id_to_int(entry.get("infra_frame_id"))
            if veh_frame is not None:
                time_index.setdefault("vehicle", {}).setdefault(seq_key, []).append((int(veh_frame), str(key)))
            if infra_frame is not None:
                time_index.setdefault("infrastructure", {}).setdefault(seq_key, []).append((int(infra_frame), str(key)))
            frame_generic = _frame_id_to_int(entry.get("frame_id"))
            if frame_generic is not None:
                time_index.setdefault("generic", {}).setdefault(seq_key, []).append((int(frame_generic), str(key)))

        resolved: Dict[str, Dict[str, Any]] = {}
        for role, seq_map in time_index.items():
            role_payload: Dict[str, Any] = {"sequences": {}}
            for seq_key, pairs in seq_map.items():
                if not pairs:
                    continue
                pairs.sort(key=lambda x: x[0])
                frames = [p[0] for p in pairs]
                keys = [p[1] for p in pairs]
                pos_map = {int(frame): int(idx) for idx, frame in enumerate(frames)}
                role_payload["sequences"][str(seq_key)] = {"frames": frames, "keys": keys, "pos_map": pos_map}
            if role_payload["sequences"]:
                resolved[str(role)] = role_payload

        self._state["time_index"] = resolved
        self._state["time_index_id"] = id(stage1_result)
        return resolved

    def _estimate_temporal_rel_T(
        self,
        *,
        sample_idx: int,
        ego_id: Any,
        cav_id: Any,
        ego_idx: int,
        cav_idx: int,
        all_agent_ids: Sequence[Any],
        base_data_dict: Mapping[Any, Any],
        stage1_content: Mapping[str, Any],
        stage1_result: Mapping[str, Any],
    ) -> Tuple[Optional[np.ndarray], float, int, Dict[str, Any]]:
        time_buffer = int(getattr(self.cfg, "time_buffer", 0) or 0)
        if time_buffer <= 0:
            return None, 0.0, 0, {"reason": "time_buffer_disabled"}
        time_stride = max(1, int(getattr(self.cfg, "time_stride", 1) or 1))
        use_clean = bool(getattr(self.cfg, "time_use_clean_pose", True))

        time_index = self._ensure_time_index(stage1_result)
        role = _normalize_agent_role(ego_id) or _infer_dair_role_from_base(base_data_dict, ego_id)
        if role is None and "generic" in time_index:
            role = "generic"
        if role is None or role not in time_index:
            return None, 0.0, 0, {"reason": "missing_role"}
        if role == "vehicle":
            frame_field = "veh_frame_id"
        elif role == "infrastructure":
            frame_field = "infra_frame_id"
        else:
            frame_field = "frame_id"
        cur_frame = _frame_id_to_int(stage1_content.get(frame_field))
        if cur_frame is None:
            return None, 0.0, 0, {"reason": "missing_frame"}

        seq_id = stage1_content.get("sequence_id")
        seq_key = str(seq_id) if seq_id is not None else "default"
        seq_map = time_index[role].get("sequences") or {}
        seq_info = seq_map.get(seq_key) or seq_map.get("default")
        if not seq_info:
            return None, 0.0, 0, {"reason": "missing_sequence"}
        pos_map = seq_info["pos_map"]
        keys = seq_info["keys"]
        pos = pos_map.get(int(cur_frame))
        if pos is None:
            return None, 0.0, 0, {"reason": "frame_not_indexed"}

        cav_boxes = _extract_boxes(stage1_content, agent_idx=cav_idx, bbox_type=self.bbox_type)
        if not cav_boxes:
            return None, 0.0, 0, {"reason": "empty_cav_boxes"}

        cur_pose = _extract_pose_from_stage1(stage1_content, agent_idx=ego_idx, use_clean=use_clean)

        best_T = None
        best_eps = float("inf")
        best_matches = -1
        best_meta: Dict[str, Any] = {}
        best_step = None

        for step in range(0, time_buffer + 1):
            pos_k = int(pos - step * time_stride)
            if pos_k < 0:
                break
            key_k = keys[pos_k]
            entry_k = stage1_result.get(str(key_k))
            if not isinstance(entry_k, Mapping):
                continue
            indices_k = _extract_agent_indices(
                entry_k.get("cav_id_list") or [],
                ego_id=ego_id,
                cav_id=cav_id,
                base_data_dict=base_data_dict,
            )
            if indices_k is None:
                continue
            ego_idx_k, _ = indices_k
            ego_boxes = _extract_boxes(entry_k, agent_idx=ego_idx_k, bbox_type=self.bbox_type)
            if not ego_boxes:
                continue
            if cur_pose is not None:
                past_pose = _extract_pose_from_stage1(entry_k, agent_idx=ego_idx_k, use_clean=use_clean)
                if past_pose is not None:
                    T_cur_from_past = x1_to_x2(past_pose, cur_pose)
                    ego_boxes = _transform_boxes_to_current(ego_boxes, T_cur_from_past=T_cur_from_past)
                    if not ego_boxes:
                        continue

            rel_T_est, stability, matches, meta = self._estimator.estimate(
                cav_boxes=cav_boxes, ego_boxes=ego_boxes, T_init=None
            )
            if rel_T_est is None:
                continue
            eps = float(meta.get("eps", float("inf")))
            if eps < best_eps - 1e-9 or (abs(eps - best_eps) <= 1e-9 and matches > best_matches):
                best_T = np.asarray(rel_T_est, dtype=np.float64)
                best_eps = float(eps)
                best_matches = int(matches)
                best_meta = dict(meta or {})
                best_step = int(step)

        if best_T is None:
            return None, 0.0, 0, {"reason": "no_temporal_match"}
        best_meta.update(
            {
                "time_offset_steps": int(best_step if best_step is not None else 0),
                "time_buffer": int(time_buffer),
                "time_stride": int(time_stride),
                "time_role": str(role),
            }
        )
        return best_T, float(best_matches), int(best_matches), best_meta

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
            bool: True if any CAV pose was updated, False otherwise.
        """
        stats = _new_pose_corr_stats(max(0, len(cav_id_list) - 1))
        self.last_stats = stats
        self.last_telemetry = []
        key = str(sample_idx)
        if key not in stage1_result:
            stats["pose_corr_skip_other_count"] = int(stats.get("pose_corr_pair_total_count") or 0)
            return False
        stage1_content = stage1_result.get(key)
        if stage1_content is None:
            stats["pose_corr_skip_other_count"] = int(stats.get("pose_corr_pair_total_count") or 0)
            return False

        mode = str(getattr(self.cfg, "mode", "initfree") or "initfree").lower().strip()
        stable = mode == "stable"
        if stable:
            self._reset_if_new_epoch(int(sample_idx))

        all_agent_ids = stage1_content.get("cav_id_list") or []
        if not isinstance(all_agent_ids, list) or len(all_agent_ids) < 2:
            stats["pose_corr_skip_other_count"] = int(stats.get("pose_corr_pair_total_count") or 0)
            return False
        if not cav_id_list:
            return False

        ego_id = cav_id_list[0]
        if ego_id not in base_data_dict:
            stats["pose_corr_skip_other_count"] = int(stats.get("pose_corr_pair_total_count") or 0)
            return False

        ego_pose = base_data_dict[ego_id]["params"]["lidar_pose"]
        Tw_ego = x_to_world(ego_pose)
        ego_T_world = np.asarray(Tw_ego, dtype=np.float64)

        updated = False
        for cav_id in cav_id_list[1:]:
            if cav_id not in base_data_dict:
                _bump_pose_corr_skip(stats, reason="other")
                continue
            indices = _extract_agent_indices(all_agent_ids, ego_id=ego_id, cav_id=cav_id, base_data_dict=base_data_dict)
            if indices is None:
                _bump_pose_corr_skip(stats, reason="other")
                continue
            ego_idx, cav_idx = indices

            use_cache = not bool(self.telemetry_enable)

            cache_key = (int(sample_idx), str(cav_id))
            missing = object()
            cached = self._rel_T_est_cache.get(cache_key, missing) if use_cache else missing
            if cached is not missing:
                rel_T_est = None if cached is None else np.asarray(cached, dtype=np.float64)
                stability = 0.0
                matches = 0
                meta = {}
            else:
                time_buffer = int(getattr(self.cfg, "time_buffer", 0) or 0)
                if time_buffer > 0:
                    rel_T_est, stability, matches, meta = self._estimate_temporal_rel_T(
                        sample_idx=sample_idx,
                        ego_id=ego_id,
                        cav_id=cav_id,
                        ego_idx=ego_idx,
                        cav_idx=cav_idx,
                        all_agent_ids=all_agent_ids,
                        base_data_dict=base_data_dict,
                        stage1_content=stage1_content,
                        stage1_result=stage1_result,
                    )
                else:
                    ego_boxes = _extract_boxes(stage1_content, agent_idx=ego_idx, bbox_type=self.bbox_type)
                    cav_boxes = _extract_boxes(stage1_content, agent_idx=cav_idx, bbox_type=self.bbox_type)
                    T_ego_cav, stability, matches, meta = self._estimator.estimate(
                        cav_boxes=cav_boxes, ego_boxes=ego_boxes, T_init=None
                    )
                    rel_T_est = np.asarray(T_ego_cav, dtype=np.float64) if T_ego_cav is not None else None
                if use_cache:
                    self._rel_T_est_cache[cache_key] = None if rel_T_est is None else np.asarray(rel_T_est, dtype=np.float64)

            cav_pose_current = base_data_dict[cav_id]["params"]["lidar_pose"]
            cav_T_world_current = x_to_world(cav_pose_current)
            rel_current_T = np.linalg.inv(ego_T_world) @ np.asarray(cav_T_world_current, dtype=np.float64)

            rel_T_est_raw = None if rel_T_est is None else np.asarray(rel_T_est, dtype=np.float64)

            fail_reason = None
            if rel_T_est is None:
                if isinstance(meta, Mapping):
                    fail_reason = str(meta.get("reason") or "other")
                else:
                    fail_reason = "other"

            compare_with_current = bool(getattr(self.cfg, "compare_with_current", False))
            min_precision = float(getattr(self.cfg, "min_precision", 0.0) or 0.0)
            apply_if_current_precision_below = float(
                getattr(self.cfg, "apply_if_current_precision_below", -1.0) or -1.0
            )
            min_precision_improvement = float(getattr(self.cfg, "min_precision_improvement", 0.0) or 0.0)
            min_matched_improvement = int(getattr(self.cfg, "min_matched_improvement", 0) or 0)
            compare_threshold = float(getattr(self.cfg, "compare_distance_threshold_m", 3.0) or 0.0)

            if rel_T_est is not None:
                needs_compare = compare_with_current or min_precision > 0.0 or apply_if_current_precision_below >= 0.0 \
                    or min_precision_improvement > 0.0 or min_matched_improvement > 0
                if needs_compare:
                    ego_boxes_eval = _extract_boxes(stage1_content, agent_idx=ego_idx, bbox_type=self.bbox_type)
                    cav_boxes_eval = _extract_boxes(stage1_content, agent_idx=cav_idx, bbox_type=self.bbox_type)
                    est_precision, est_matched = _compute_match_stats(
                        cav_boxes_eval, ego_boxes_eval, rel_T_est,
                        bbox_type=self.bbox_type, distance_threshold_m=compare_threshold
                    )
                    if est_precision is None or est_matched is None:
                        rel_T_est = None
                        fail_reason = "compare_gate"
                    else:
                        if min_precision > 0.0 and float(est_precision) < float(min_precision):
                            rel_T_est = None
                            fail_reason = "compare_gate"
                        else:
                            cur_precision, cur_matched = _compute_match_stats(
                                cav_boxes_eval, ego_boxes_eval, rel_current_T,
                                bbox_type=self.bbox_type, distance_threshold_m=compare_threshold
                            )
                            if cur_precision is not None and cur_matched is not None:
                                if apply_if_current_precision_below >= 0.0:
                                    # CorrespondingDetector precision is in [-3, 0] (0 is best). Shift to [0, 3]
                                    # so a non-negative CLI threshold is meaningful.
                                    cur_quality = float(cur_precision) + 3.0
                                    if int(cur_matched) > 0 and float(cur_quality) > float(apply_if_current_precision_below):
                                        rel_T_est = None
                                        fail_reason = "compare_gate"
                                if rel_T_est is not None and int(cur_matched) > 0:
                                    precision_ok = float(est_precision) >= float(cur_precision) + float(min_precision_improvement)
                                    matched_ok = int(est_matched) >= int(cur_matched) + int(min_matched_improvement)
                                    if not (precision_ok or matched_ok):
                                        rel_T_est = None
                                        fail_reason = "compare_gate"

            telemetry_row = None
            if self.telemetry_enable:
                from opencood.registration.runtime.telemetry_utils import compute_match_pairs, se2_from_tfm, serialize_tfm

                thr = float(self.telemetry_distance_threshold_m or 0.0)
                if thr <= 0.0:
                    thr = float(compare_threshold or 3.0)
                ego_boxes_eval = _extract_boxes(stage1_content, agent_idx=ego_idx, bbox_type=self.bbox_type)
                cav_boxes_eval = _extract_boxes(stage1_content, agent_idx=cav_idx, bbox_type=self.bbox_type)
                telemetry_row = {
                    "sample_idx": int(sample_idx),
                    "ego_id": str(ego_id),
                    "cav_id": str(cav_id),
                    "mode": str(mode),
                    "bbox_type": str(self.bbox_type),
                    "fail_reason": fail_reason,
                    "rel_T_current": serialize_tfm(rel_current_T),
                    "rel_se2_current": se2_from_tfm(rel_current_T),
                    "rel_T_est_raw": serialize_tfm(rel_T_est_raw),
                    "rel_se2_est_raw": se2_from_tfm(rel_T_est_raw),
                    "rel_T_est_filtered": serialize_tfm(rel_T_est),
                    "rel_se2_est_filtered": se2_from_tfm(rel_T_est),
                    "src_box_count": int(len(cav_boxes_eval)),
                    "dst_box_count": int(len(ego_boxes_eval)),
                }
                telemetry_row["matches_current"] = compute_match_pairs(
                    cav_boxes_eval,
                    ego_boxes_eval,
                    rel_current_T,
                    bbox_type=self.bbox_type,
                    distance_threshold_m=thr,
                    topk_pairs=int(self.telemetry_max_pairs or 0),
                )
                telemetry_row["matches_est_raw"] = compute_match_pairs(
                    cav_boxes_eval,
                    ego_boxes_eval,
                    rel_T_est_raw,
                    bbox_type=self.bbox_type,
                    distance_threshold_m=thr,
                    topk_pairs=int(self.telemetry_max_pairs or 0),
                )

            rel_T_corrected: Optional[np.ndarray]
            if not stable:
                if rel_T_est is None:
                    _bump_pose_corr_skip(stats, reason=str(fail_reason or "other"))
                    if telemetry_row is not None:
                        telemetry_row["applied"] = False
                        telemetry_row["rel_T_applied"] = None
                        telemetry_row["rel_se2_applied"] = None
                        self.last_telemetry.append(telemetry_row)
                    continue
                rel_T_corrected = rel_T_est
            else:
                rel_key = str(cav_id)
                prev_delta = self._get_prev_delta(rel_key)
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
                            if not _limit_step_se2(
                                prev_delta,
                                delta_xyyaw,
                                max_step_xy_m=float(getattr(self.cfg, "max_step_xy_m", 0.0) or 0.0),
                                max_step_yaw_deg=float(getattr(self.cfg, "max_step_yaw_deg", 0.0) or 0.0),
                            ):
                                delta_xyyaw = prev_delta
                            else:
                                delta_xyyaw = _smooth_se2(prev_delta, delta_xyyaw, alpha=float(getattr(self.cfg, "ema_alpha", 0.5) or 0.0))
                        self._set_prev_delta(rel_key, delta_xyyaw)
                        prev_delta = delta_xyyaw

                if prev_delta is None:
                    _bump_pose_corr_skip(stats, reason=str(fail_reason or "other"))
                    if telemetry_row is not None:
                        telemetry_row["applied"] = False
                        telemetry_row["rel_T_applied"] = None
                        telemetry_row["rel_se2_applied"] = None
                        self.last_telemetry.append(telemetry_row)
                    continue
                delta_T_se2 = pose_to_tfm(np.asarray([[prev_delta[0], prev_delta[1], prev_delta[2]]], dtype=np.float64))[0]
                rel_T_corrected = np.asarray(delta_T_se2, dtype=np.float64) @ np.asarray(rel_current_T, dtype=np.float64)

            if not select_estimate_by_policy(
                selection_policy=self.selection_policy,
                base_data_dict=base_data_dict,
                ego_id=ego_id,
                cav_id=cav_id,
                T_current=rel_current_T,
                T_est=rel_T_corrected,
                telemetry_row=telemetry_row,
            ):
                if telemetry_row is not None:
                    telemetry_row["applied"] = False
                    telemetry_row["rel_T_applied"] = None
                    telemetry_row["rel_se2_applied"] = None
                    self.last_telemetry.append(telemetry_row)
                continue
            cav_T_world_new = ego_T_world @ np.asarray(rel_T_corrected, dtype=np.float64)
            pose_new = tfm_to_pose(cav_T_world_new)
            pose_old = list(base_data_dict[cav_id]["params"]["lidar_pose"])
            if len(pose_old) < 6:
                continue
            pose_old[0] = float(pose_new[0])
            pose_old[1] = float(pose_new[1])
            pose_old[4] = float(pose_new[4])
            base_data_dict[cav_id]["params"]["lidar_pose"] = pose_old

            updated = True
            stats["pose_corr_applied_pair_count"] += 1
            if telemetry_row is not None:
                from opencood.registration.runtime.telemetry_utils import se2_from_tfm, serialize_tfm

                telemetry_row["applied"] = True
                telemetry_row["rel_T_applied"] = serialize_tfm(rel_T_corrected)
                telemetry_row["rel_se2_applied"] = se2_from_tfm(rel_T_corrected)
                self.last_telemetry.append(telemetry_row)
            _ = (stability, matches, meta)  # keep for optional future logging
        return updated


@dataclass
class Stage1FreeAlignRepoPoseCorrector:
    """
    Apply FreeAlign (released repo matching: match_v7_with_detection) pose correction using cached stage-1 boxes.

    This keeps FreeAlign logic out of dataset code.
    """

    cfg: FreeAlignRepoConfig = field(default_factory=FreeAlignRepoConfig)
    bbox_type: str = "detected"

    _estimator: FreeAlignRepoEstimator = field(init=False, repr=False)
    _state: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    # Cache per (sample_idx, cav_id) to avoid repeating matching in noise sweeps.
    _rel_T_est_cache: Dict[Tuple[int, str], Optional[np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )

    telemetry_enable: bool = False
    telemetry_distance_threshold_m: float = 0.0
    telemetry_max_pairs: int = 0
    selection_policy: str = "solver_only"
    last_telemetry: list = field(default_factory=list, init=False, repr=False)
    last_stats: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._estimator = FreeAlignRepoEstimator(self.cfg)

    def _reset_if_new_epoch(self, sample_idx: int) -> None:
        last = self._state.get("last_sample_idx")
        if last is None or int(sample_idx) < int(last):
            self._state.clear()
        self._state["last_sample_idx"] = int(sample_idx)

    def _get_prev_delta(self, key: str) -> Optional[Tuple[float, float, float]]:
        store = self._state.get("delta_se2", {})
        if not isinstance(store, dict):
            return None
        prev = store.get(str(key))
        if prev is None:
            return None
        try:
            x, y, yaw = float(prev[0]), float(prev[1]), float(prev[2])
        except Exception:
            return None
        return x, y, yaw

    def _set_prev_delta(self, key: str, delta: Tuple[float, float, float]) -> None:
        store = self._state.setdefault("delta_se2", {})
        if not isinstance(store, dict):
            return
        store[str(key)] = (float(delta[0]), float(delta[1]), float(delta[2]))

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
            bool: True if any CAV pose was updated, False otherwise.
        """
        stats = _new_pose_corr_stats(max(0, len(cav_id_list) - 1))
        self.last_stats = stats
        self.last_telemetry = []
        key = str(sample_idx)
        if key not in stage1_result:
            stats["pose_corr_skip_other_count"] = int(stats.get("pose_corr_pair_total_count") or 0)
            return False
        stage1_content = stage1_result.get(key)
        if stage1_content is None:
            stats["pose_corr_skip_other_count"] = int(stats.get("pose_corr_pair_total_count") or 0)
            return False

        mode = str(getattr(self.cfg, "mode", "initfree") or "initfree").lower().strip()
        stable = mode == "stable"
        if stable:
            self._reset_if_new_epoch(int(sample_idx))

        all_agent_ids = stage1_content.get("cav_id_list") or []
        if not isinstance(all_agent_ids, list) or len(all_agent_ids) < 2:
            stats["pose_corr_skip_other_count"] = int(stats.get("pose_corr_pair_total_count") or 0)
            return False
        if not cav_id_list:
            return False

        ego_id = cav_id_list[0]
        if ego_id not in base_data_dict:
            stats["pose_corr_skip_other_count"] = int(stats.get("pose_corr_pair_total_count") or 0)
            return False

        ego_pose = base_data_dict[ego_id]["params"]["lidar_pose"]
        Tw_ego = x_to_world(ego_pose)
        ego_T_world = np.asarray(Tw_ego, dtype=np.float64)

        updated = False
        for cav_id in cav_id_list[1:]:
            if cav_id not in base_data_dict:
                _bump_pose_corr_skip(stats, reason="other")
                continue
            indices = _extract_agent_indices(all_agent_ids, ego_id=ego_id, cav_id=cav_id, base_data_dict=base_data_dict)
            if indices is None:
                _bump_pose_corr_skip(stats, reason="other")
                continue
            ego_idx, cav_idx = indices

            use_cache = not bool(self.telemetry_enable)

            cache_key = (int(sample_idx), str(cav_id))
            missing = object()
            cached = self._rel_T_est_cache.get(cache_key, missing) if use_cache else missing
            if cached is not missing:
                rel_T_est = None if cached is None else np.asarray(cached, dtype=np.float64)
                stability = 0.0
                matches = 0
                meta = {}
            else:
                ego_boxes = _extract_boxes(stage1_content, agent_idx=ego_idx, bbox_type=self.bbox_type)
                cav_boxes = _extract_boxes(stage1_content, agent_idx=cav_idx, bbox_type=self.bbox_type)
                T_ego_cav, stability, matches, meta = self._estimator.estimate(cav_boxes=cav_boxes, ego_boxes=ego_boxes)
                rel_T_est = np.asarray(T_ego_cav, dtype=np.float64) if T_ego_cav is not None else None
                if use_cache:
                    self._rel_T_est_cache[cache_key] = None if rel_T_est is None else np.asarray(rel_T_est, dtype=np.float64)

            cav_pose_current = base_data_dict[cav_id]["params"]["lidar_pose"]
            cav_T_world_current = x_to_world(cav_pose_current)
            rel_current_T = np.linalg.inv(ego_T_world) @ np.asarray(cav_T_world_current, dtype=np.float64)

            rel_T_est_raw = None if rel_T_est is None else np.asarray(rel_T_est, dtype=np.float64)

            fail_reason = None
            if rel_T_est is None:
                if isinstance(meta, Mapping):
                    fail_reason = str(meta.get("reason") or "other")
                else:
                    fail_reason = "other"

            compare_with_current = bool(getattr(self.cfg, "compare_with_current", False))
            min_precision = float(getattr(self.cfg, "min_precision", 0.0) or 0.0)
            apply_if_current_precision_below = float(
                getattr(self.cfg, "apply_if_current_precision_below", -1.0) or -1.0
            )
            min_precision_improvement = float(getattr(self.cfg, "min_precision_improvement", 0.0) or 0.0)
            min_matched_improvement = int(getattr(self.cfg, "min_matched_improvement", 0) or 0)
            compare_threshold = float(getattr(self.cfg, "compare_distance_threshold_m", 3.0) or 0.0)

            if rel_T_est is not None:
                needs_compare = compare_with_current or min_precision > 0.0 or apply_if_current_precision_below >= 0.0 \
                    or min_precision_improvement > 0.0 or min_matched_improvement > 0
                if needs_compare:
                    ego_boxes_eval = _extract_boxes(stage1_content, agent_idx=ego_idx, bbox_type=self.bbox_type)
                    cav_boxes_eval = _extract_boxes(stage1_content, agent_idx=cav_idx, bbox_type=self.bbox_type)
                    est_precision, est_matched = _compute_match_stats(
                        cav_boxes_eval, ego_boxes_eval, rel_T_est,
                        bbox_type=self.bbox_type, distance_threshold_m=compare_threshold
                    )
                    if est_precision is None or est_matched is None:
                        rel_T_est = None
                        fail_reason = "compare_gate"
                    else:
                        if min_precision > 0.0 and float(est_precision) < float(min_precision):
                            rel_T_est = None
                            fail_reason = "compare_gate"
                        else:
                            cur_precision, cur_matched = _compute_match_stats(
                                cav_boxes_eval, ego_boxes_eval, rel_current_T,
                                bbox_type=self.bbox_type, distance_threshold_m=compare_threshold
                            )
                            if cur_precision is not None and cur_matched is not None:
                                if apply_if_current_precision_below >= 0.0:
                                    # CorrespondingDetector precision is in [-3, 0] (0 is best). Shift to [0, 3]
                                    # so a non-negative CLI threshold is meaningful.
                                    cur_quality = float(cur_precision) + 3.0
                                    if int(cur_matched) > 0 and float(cur_quality) > float(apply_if_current_precision_below):
                                        rel_T_est = None
                                        fail_reason = "compare_gate"
                                if rel_T_est is not None and int(cur_matched) > 0:
                                    precision_ok = float(est_precision) >= float(cur_precision) + float(min_precision_improvement)
                                    matched_ok = int(est_matched) >= int(cur_matched) + int(min_matched_improvement)
                                    if not (precision_ok or matched_ok):
                                        rel_T_est = None
                                        fail_reason = "compare_gate"

            telemetry_row = None
            if self.telemetry_enable:
                from opencood.registration.runtime.telemetry_utils import compute_match_pairs, se2_from_tfm, serialize_tfm

                thr = float(self.telemetry_distance_threshold_m or 0.0)
                if thr <= 0.0:
                    thr = float(compare_threshold or 3.0)
                ego_boxes_eval = _extract_boxes(stage1_content, agent_idx=ego_idx, bbox_type=self.bbox_type)
                cav_boxes_eval = _extract_boxes(stage1_content, agent_idx=cav_idx, bbox_type=self.bbox_type)
                telemetry_row = {
                    "sample_idx": int(sample_idx),
                    "ego_id": str(ego_id),
                    "cav_id": str(cav_id),
                    "mode": str(mode),
                    "bbox_type": str(self.bbox_type),
                    "fail_reason": fail_reason,
                    "rel_T_current": serialize_tfm(rel_current_T),
                    "rel_se2_current": se2_from_tfm(rel_current_T),
                    "rel_T_est_raw": serialize_tfm(rel_T_est_raw),
                    "rel_se2_est_raw": se2_from_tfm(rel_T_est_raw),
                    "rel_T_est_filtered": serialize_tfm(rel_T_est),
                    "rel_se2_est_filtered": se2_from_tfm(rel_T_est),
                    "src_box_count": int(len(cav_boxes_eval)),
                    "dst_box_count": int(len(ego_boxes_eval)),
                }
                telemetry_row["matches_current"] = compute_match_pairs(
                    cav_boxes_eval,
                    ego_boxes_eval,
                    rel_current_T,
                    bbox_type=self.bbox_type,
                    distance_threshold_m=thr,
                    topk_pairs=int(self.telemetry_max_pairs or 0),
                )
                telemetry_row["matches_est_raw"] = compute_match_pairs(
                    cav_boxes_eval,
                    ego_boxes_eval,
                    rel_T_est_raw,
                    bbox_type=self.bbox_type,
                    distance_threshold_m=thr,
                    topk_pairs=int(self.telemetry_max_pairs or 0),
                )

            rel_T_corrected: Optional[np.ndarray]
            if not stable:
                if rel_T_est is None:
                    _bump_pose_corr_skip(stats, reason=str(fail_reason or "other"))
                    if telemetry_row is not None:
                        telemetry_row["applied"] = False
                        telemetry_row["rel_T_applied"] = None
                        telemetry_row["rel_se2_applied"] = None
                        self.last_telemetry.append(telemetry_row)
                    continue
                rel_T_corrected = rel_T_est
            else:
                rel_key = str(cav_id)
                prev_delta = self._get_prev_delta(rel_key)
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
                            if not _limit_step_se2(
                                prev_delta,
                                delta_xyyaw,
                                max_step_xy_m=float(getattr(self.cfg, "max_step_xy_m", 0.0) or 0.0),
                                max_step_yaw_deg=float(getattr(self.cfg, "max_step_yaw_deg", 0.0) or 0.0),
                            ):
                                delta_xyyaw = prev_delta
                            else:
                                delta_xyyaw = _smooth_se2(prev_delta, delta_xyyaw, alpha=float(getattr(self.cfg, "ema_alpha", 0.5) or 0.0))
                        self._set_prev_delta(rel_key, delta_xyyaw)
                        prev_delta = delta_xyyaw

                if prev_delta is None:
                    _bump_pose_corr_skip(stats, reason=str(fail_reason or "other"))
                    if telemetry_row is not None:
                        telemetry_row["applied"] = False
                        telemetry_row["rel_T_applied"] = None
                        telemetry_row["rel_se2_applied"] = None
                        self.last_telemetry.append(telemetry_row)
                    continue
                delta_T_se2 = pose_to_tfm(np.asarray([[prev_delta[0], prev_delta[1], prev_delta[2]]], dtype=np.float64))[0]
                rel_T_corrected = np.asarray(delta_T_se2, dtype=np.float64) @ np.asarray(rel_current_T, dtype=np.float64)

            if not select_estimate_by_policy(
                selection_policy=self.selection_policy,
                base_data_dict=base_data_dict,
                ego_id=ego_id,
                cav_id=cav_id,
                T_current=rel_current_T,
                T_est=rel_T_corrected,
                telemetry_row=telemetry_row,
            ):
                if telemetry_row is not None:
                    telemetry_row["applied"] = False
                    telemetry_row["rel_T_applied"] = None
                    telemetry_row["rel_se2_applied"] = None
                    self.last_telemetry.append(telemetry_row)
                continue
            cav_T_world_new = ego_T_world @ np.asarray(rel_T_corrected, dtype=np.float64)
            pose_new = tfm_to_pose(cav_T_world_new)
            pose_old = list(base_data_dict[cav_id]["params"]["lidar_pose"])
            if len(pose_old) < 6:
                continue
            pose_old[0] = float(pose_new[0])
            pose_old[1] = float(pose_new[1])
            pose_old[4] = float(pose_new[4])
            base_data_dict[cav_id]["params"]["lidar_pose"] = pose_old

            updated = True
            _ = (stability, matches, meta)  # keep for optional future logging
        return updated


__all__ = ["Stage1FreeAlignPoseCorrector", "Stage1FreeAlignRepoPoseCorrector"]
