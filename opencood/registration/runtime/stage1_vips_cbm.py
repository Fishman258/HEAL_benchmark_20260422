from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from opencood.registration.estimators.cbm import CBMEstimator
from opencood.registration.estimators.vips import VIPSEstimator
from opencood.registration.runtime.selection_policy import select_estimate_by_policy
from opencood.registration.utils.types import ExtrinsicInit
from opencood.utils.transformation_utils import pose_to_tfm, tfm_to_pose


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
        name = str(agent_id).lower()
        role = None
        if "veh" in name or "vehicle" in name:
            role = "vehicle"
        elif "infra" in name or "rsu" in name or "infrastructure" in name:
            role = "infrastructure"
        if role and role not in role_to_idx:
            role_to_idx[role] = int(idx)
    if role_to_idx:
        params = base_data_dict.get(ego_id, {}).get("params") or {}
        ego_role = "vehicle" if params.get("vehicles_all") else "infrastructure"
        params = base_data_dict.get(cav_id, {}).get("params") or {}
        cav_role = "vehicle" if params.get("vehicles_all") else "infrastructure"
        if ego_role in role_to_idx and cav_role in role_to_idx:
            return int(role_to_idx[ego_role]), int(role_to_idx[cav_role])
    return None


def _extract_boxes(
    stage1_content: Mapping[str, Any],
    *,
    agent_idx: int,
    field: str,
    bbox_type: str,
) -> list:
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
    for box_idx, box in enumerate(boxes_raw):
        if isinstance(box, dict):
            corners = box.get("corners") or box.get("points") or box.get("bbox")
            if corners is None:
                continue
            corners_list.append(corners)
            score_list.append(float(box.get("score", box.get("confidence", 1.0))))
        else:
            corners_list.append(box)
            if scores is not None and box_idx < len(scores):
                score_list.append(float(scores[box_idx]))
            else:
                score_list.append(1.0)

    if not corners_list:
        return []
    corners_np = np.asarray(corners_list, dtype=np.float32)
    from opencood.registration.utils.bbox import corners_to_bbox3d_list

    return corners_to_bbox3d_list(corners_np, bbox_type=bbox_type, scores=score_list)


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
    if r in {"no_matches", "no_common_subgraph", "no_matches_after_distance_filter"}:
        stats["pose_corr_skip_no_matches_count"] += 1
        return
    if r in {"svd_failed"}:
        stats["pose_corr_skip_svd_failed_count"] += 1
        return
    if r in {"compare_gate", "compare_gate_reject"}:
        stats["pose_corr_skip_compare_gate_count"] += 1
        return
    if r in {"vips_exception"} or "exception" in r:
        stats["pose_corr_skip_exception_count"] += 1
        return
    stats["pose_corr_skip_other_count"] += 1


@dataclass
class _Stage1BasePoseCorrector:
    stage1_field: str = "pred_corner3d_np_list"
    bbox_type: str = "detected"
    mode: str = "initfree"  # initfree | stable
    use_prior: bool = False
    compare_with_current: bool = False
    compare_distance_threshold_m: float = 3.0
    ema_alpha: float = 0.5
    max_step_xy_m: float = 3.0
    max_step_yaw_deg: float = 10.0
    min_precision: float = 0.0
    apply_if_current_precision_below: float = -1.0
    min_precision_improvement: float = 0.0
    min_matched_improvement: int = 0
    selection_policy: str = "solver_only"
    freeze_ego: bool = True
    telemetry_enable: bool = False
    telemetry_distance_threshold_m: float = 0.0
    telemetry_max_pairs: int = 0
    last_telemetry: list = field(default_factory=list, init=False, repr=False)
    _state: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _rel_T_est_cache: Dict[Tuple[int, str], Optional[np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )
    last_stats: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

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

    def _estimate_rel_T(
        self,
        src_boxes,
        dst_boxes,
        *,
        init: Optional[ExtrinsicInit],
    ):
        raise NotImplementedError

    def _compute_match_stats(self, src_boxes, dst_boxes, T_rel: np.ndarray):
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
            bbox_key = str(self.bbox_type or "detected")
            threshold = {bbox_key: float(self.compare_distance_threshold_m)}
            detector = CorrespondingDetector(aligned_src, dst_boxes, distance_threshold=threshold)
            precision = float(detector.get_distance_corresponding_precision())
            matched = int(detector.get_matched_num())
            return precision, matched
        except Exception:
            return None, None

    def apply(
        self,
        *,
        sample_idx: int,
        cav_id_list: Sequence[Any],
        base_data_dict: MutableMapping[Any, Dict[str, Any]],
        stage1_result: Mapping[str, Any],
    ) -> bool:
        stats = _new_pose_corr_stats(max(0, len(cav_id_list) - 1))
        self.last_stats = stats
        self.last_telemetry = []
        self._reset_if_new_epoch(int(sample_idx))

        key = str(sample_idx)
        stage1_content = stage1_result.get(key)
        if stage1_content is None or not isinstance(stage1_content, Mapping):
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

        mode = str(self.mode or "initfree").lower().strip()
        stable = mode == "stable"

        updated_any = False
        ego_pose = base_data_dict[ego_id]["params"]["lidar_pose"]
        ego_T_world = pose_to_tfm(np.asarray([ego_pose], dtype=np.float64))[0]

        for cav_id in cav_id_list:
            if cav_id == ego_id and self.freeze_ego:
                continue
            if cav_id not in base_data_dict:
                _bump_pose_corr_skip(stats, reason="other")
                continue

            idx_pair = _extract_agent_indices(all_agent_ids, ego_id=ego_id, cav_id=cav_id, base_data_dict=base_data_dict)
            if idx_pair is None:
                _bump_pose_corr_skip(stats, reason="other")
                continue
            ego_idx, cav_idx = idx_pair

            # Convention: all transforms in this file are (cav -> ego). Keep the
            # estimator and match-stat calls consistent with that direction.
            ego_boxes = _extract_boxes(
                stage1_content, agent_idx=ego_idx, field=self.stage1_field, bbox_type=self.bbox_type
            )
            cav_boxes = _extract_boxes(
                stage1_content, agent_idx=cav_idx, field=self.stage1_field, bbox_type=self.bbox_type
            )
            src_boxes = cav_boxes
            dst_boxes = ego_boxes

            rel_key = str(cav_id)
            prev_delta = self._get_prev_delta(rel_key) if stable else None

            cav_pose_current = base_data_dict[cav_id]["params"]["lidar_pose"]
            cav_T_world_current = pose_to_tfm(np.asarray([cav_pose_current], dtype=np.float64))[0]
            rel_current_T = np.linalg.inv(ego_T_world) @ cav_T_world_current

            rel_T_est: Optional[np.ndarray] = None
            rel_T_est_raw: Optional[np.ndarray] = None
            fail_reason: Optional[str] = None
            use_cache = not bool(self.telemetry_enable)

            cache_key = (int(sample_idx), str(cav_id))
            missing = object()
            cached = self._rel_T_est_cache.get(cache_key, missing) if use_cache else missing
            if cached is not missing:
                rel_T_est = None if cached is None else np.asarray(cached, dtype=np.float64)
            else:
                init = None
                if self.use_prior:
                    init = ExtrinsicInit(T_init=np.asarray(rel_current_T, dtype=np.float64), source="current_pose")
                estimate = None
                try:
                    estimate = self._estimate_rel_T(src_boxes, dst_boxes, init=init)
                except Exception:
                    estimate = None
                    fail_reason = "exception"
                if estimate is not None and estimate.success and estimate.T is not None:
                    rel_T_est = np.asarray(estimate.T, dtype=np.float64)
                else:
                    extra = getattr(estimate, "extra", None) if estimate is not None else None
                    if isinstance(extra, Mapping):
                        fail_reason = str(extra.get("reason") or fail_reason or "other")
                    elif fail_reason is None:
                        fail_reason = "other"
                if use_cache:
                    self._rel_T_est_cache[cache_key] = None if rel_T_est is None else np.asarray(rel_T_est, dtype=np.float64)

            rel_T_est_raw = None if rel_T_est is None else np.asarray(rel_T_est, dtype=np.float64)

            if rel_T_est is not None:
                needs_compare = bool(self.compare_with_current) or float(self.min_precision) > 0.0 \
                    or float(self.apply_if_current_precision_below) >= 0.0 \
                    or float(self.min_precision_improvement) > 0.0 \
                    or int(self.min_matched_improvement) > 0
                if needs_compare:
                    est_precision, est_matched = self._compute_match_stats(src_boxes, dst_boxes, rel_T_est)
                    if est_precision is None or est_matched is None:
                        rel_T_est = None
                        fail_reason = "compare_gate"
                    else:
                        if float(self.min_precision) > 0.0 and float(est_precision) < float(self.min_precision):
                            rel_T_est = None
                            fail_reason = "compare_gate"
                        else:
                            cur_precision, cur_matched = self._compute_match_stats(src_boxes, dst_boxes, rel_current_T)
                            if cur_precision is not None and cur_matched is not None:
                                if float(self.apply_if_current_precision_below) >= 0.0:
                                    # CorrespondingDetector precision is in [-3, 0] (0 is best). Shift to [0, 3]
                                    # so a non-negative CLI threshold is meaningful.
                                    cur_quality = float(cur_precision) + 3.0
                                    if int(cur_matched) > 0 and float(cur_quality) > float(self.apply_if_current_precision_below):
                                        rel_T_est = None
                                        fail_reason = "compare_gate"
                                if rel_T_est is not None and int(cur_matched) > 0:
                                    precision_ok = float(est_precision) >= float(cur_precision) + float(self.min_precision_improvement)
                                    matched_ok = int(est_matched) >= int(cur_matched) + int(self.min_matched_improvement)
                                    if not (precision_ok or matched_ok):
                                        rel_T_est = None
                                        fail_reason = "compare_gate"

            telemetry_row = None
            if self.telemetry_enable:
                from opencood.registration.runtime.telemetry_utils import compute_match_pairs, se2_from_tfm, serialize_tfm

                thr = float(self.telemetry_distance_threshold_m or 0.0)
                if thr <= 0.0:
                    thr = float(self.compare_distance_threshold_m or 3.0)
                telemetry_row = {
                    "sample_idx": int(sample_idx),
                    "ego_id": str(ego_id),
                    "cav_id": str(cav_id),
                    "mode": str(self.mode),
                    "bbox_type": str(self.bbox_type),
                    "compare_with_current": bool(self.compare_with_current),
                    "use_prior": bool(self.use_prior),
                    "fail_reason": fail_reason,
                    "src_box_count": int(len(src_boxes)),
                    "dst_box_count": int(len(dst_boxes)),
                    "rel_T_current": serialize_tfm(rel_current_T),
                    "rel_se2_current": se2_from_tfm(rel_current_T),
                    "rel_T_est_raw": serialize_tfm(rel_T_est_raw),
                    "rel_se2_est_raw": se2_from_tfm(rel_T_est_raw),
                    "rel_T_est_filtered": serialize_tfm(rel_T_est),
                    "rel_se2_est_filtered": se2_from_tfm(rel_T_est),
                }
                telemetry_row["matches_current"] = compute_match_pairs(
                    src_boxes,
                    dst_boxes,
                    rel_current_T,
                    bbox_type=self.bbox_type,
                    distance_threshold_m=thr,
                    topk_pairs=int(self.telemetry_max_pairs or 0),
                )
                telemetry_row["matches_est_raw"] = compute_match_pairs(
                    src_boxes,
                    dst_boxes,
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
                            if not _limit_step_se2(prev_delta, delta_xyyaw, max_step_xy_m=self.max_step_xy_m, max_step_yaw_deg=self.max_step_yaw_deg):
                                delta_xyyaw = prev_delta
                            else:
                                delta_xyyaw = _smooth_se2(prev_delta, delta_xyyaw, alpha=self.ema_alpha)
                        self._set_prev_delta(rel_key, delta_xyyaw)
                        prev_delta = delta_xyyaw

                if prev_delta is not None:
                    delta_T_se2 = pose_to_tfm(np.asarray([[prev_delta[0], prev_delta[1], prev_delta[2]]], dtype=np.float64))[0]
                    rel_T_corrected = np.asarray(delta_T_se2, dtype=np.float64) @ np.asarray(rel_current_T, dtype=np.float64)
                else:
                    _bump_pose_corr_skip(stats, reason=str(fail_reason or "other"))
                    if telemetry_row is not None:
                        telemetry_row["applied"] = False
                        telemetry_row["rel_T_applied"] = None
                        telemetry_row["rel_se2_applied"] = None
                        self.last_telemetry.append(telemetry_row)
                    continue

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


@dataclass
class Stage1CBMPoseCorrector(_Stage1BasePoseCorrector):
    sigma1_deg: float = 10.0
    sigma2_m: float = 3.0
    sigma3_m: float = 1.0
    absolute_dis_lim_m: float = 20.0
    device: Optional[str] = None
    _estimator: CBMEstimator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._estimator = CBMEstimator(
            sigma1_deg=self.sigma1_deg,
            sigma2_m=self.sigma2_m,
            sigma3_m=self.sigma3_m,
            absolute_dis_lim_m=self.absolute_dis_lim_m,
            device=self.device,
        )

    def _estimate_rel_T(self, src_boxes, dst_boxes, *, init: Optional[ExtrinsicInit]):
        return self._estimator.estimate(src_boxes, dst_boxes, init=init)


@dataclass
class Stage1VIPSPoseCorrector(_Stage1BasePoseCorrector):
    match_threshold: float = 0.5
    match_distance_thr_m: float = 8.0
    device: Optional[str] = None
    _estimator: VIPSEstimator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._estimator = VIPSEstimator(
            match_threshold=float(self.match_threshold),
            match_distance_thr_m=float(self.match_distance_thr_m),
            device=self.device,
        )

    def _estimate_rel_T(self, src_boxes, dst_boxes, *, init: Optional[ExtrinsicInit]):
        return self._estimator.estimate(src_boxes, dst_boxes, init=init)


__all__ = ["Stage1CBMPoseCorrector", "Stage1VIPSPoseCorrector"]
