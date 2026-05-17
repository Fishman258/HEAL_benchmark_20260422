from __future__ import annotations

import math
from typing import Any, Mapping, MutableMapping, Optional

import numpy as np

from opencood.utils.transformation_utils import pose_to_tfm


def normalize_pose_selection_policy(raw: Any) -> str:
    policy = str(raw or "solver_only").strip().lower()
    if policy in {"", "solver_only", "compare_current_proxy", "choose_better_pose_error"}:
        return policy or "solver_only"
    return "solver_only"


def pose_error_key(T_used: Any, T_true: Any) -> tuple[float, float]:
    try:
        Tu = np.asarray(T_used, dtype=np.float64).reshape(4, 4)
        Tt = np.asarray(T_true, dtype=np.float64).reshape(4, 4)
        dx = float(Tu[0, 3] - Tt[0, 3])
        dy = float(Tu[1, 3] - Tt[1, 3])
        te = float(math.hypot(dx, dy))
        yaw_u = float(math.degrees(math.atan2(float(Tu[1, 0]), float(Tu[0, 0]))))
        yaw_t = float(math.degrees(math.atan2(float(Tt[1, 0]), float(Tt[0, 0]))))
        re = abs(float(((yaw_u - yaw_t + 180.0) % 360.0) - 180.0))
        return te, re
    except Exception:
        return float("inf"), float("inf")


def estimate_wins_policy(
    *,
    selection_policy: str,
    T_est: Any,
    T_current: Any,
    T_true: Any,
) -> bool:
    policy = normalize_pose_selection_policy(selection_policy)
    if policy in {"", "solver_only", "compare_current_proxy"}:
        return True

    if policy != "choose_better_pose_error":
        return True

    est_key = pose_error_key(T_est, T_true)
    cur_key = pose_error_key(T_current, T_true)
    if not math.isfinite(est_key[0]) or not math.isfinite(est_key[1]):
        return False
    if not math.isfinite(cur_key[0]) or not math.isfinite(cur_key[1]):
        return True
    if est_key[0] < cur_key[0] - 1e-9:
        return True
    if abs(est_key[0] - cur_key[0]) <= 1e-9 and est_key[1] < cur_key[1] - 1e-9:
        return True
    return False


def compute_rel_T_clean(
    base_data_dict: Mapping[Any, Any],
    *,
    ego_id: Any,
    cav_id: Any,
) -> Optional[np.ndarray]:
    try:
        ego_pose = (base_data_dict.get(ego_id) or {}).get("params", {}).get("lidar_pose_clean")
        cav_pose = (base_data_dict.get(cav_id) or {}).get("params", {}).get("lidar_pose_clean")
        if ego_pose is None or cav_pose is None:
            return None
        ego_T_world = pose_to_tfm(np.asarray([ego_pose], dtype=np.float64))[0]
        cav_T_world = pose_to_tfm(np.asarray([cav_pose], dtype=np.float64))[0]
        return np.linalg.inv(ego_T_world) @ cav_T_world
    except Exception:
        return None


def select_estimate_by_policy(
    *,
    selection_policy: str,
    base_data_dict: Mapping[Any, Any],
    ego_id: Any,
    cav_id: Any,
    T_current: Any,
    T_est: Any,
    telemetry_row: Optional[MutableMapping[str, Any]] = None,
) -> bool:
    policy = normalize_pose_selection_policy(selection_policy)
    T_true = compute_rel_T_clean(base_data_dict, ego_id=ego_id, cav_id=cav_id)
    append_selection_policy_telemetry(
        telemetry_row,
        selection_policy=policy,
        T_true=T_true,
        T_current=T_current,
        T_est=T_est,
    )
    if policy != "choose_better_pose_error":
        if isinstance(telemetry_row, MutableMapping):
            telemetry_row["pose_selection_used_estimate"] = T_est is not None
            telemetry_row["pose_selection_reason"] = "estimate_available" if T_est is not None else "no_estimate"
        return T_est is not None

    if T_est is None:
        if isinstance(telemetry_row, MutableMapping):
            telemetry_row["pose_selection_used_estimate"] = False
            telemetry_row["pose_selection_reason"] = "no_estimate"
        return False
    if T_true is None:
        if isinstance(telemetry_row, MutableMapping):
            telemetry_row["pose_selection_used_estimate"] = False
            telemetry_row["pose_selection_reason"] = "missing_clean_pose"
        return False

    use_estimate = bool(
        estimate_wins_policy(
            selection_policy=policy,
            T_est=T_est,
            T_current=T_current,
            T_true=T_true,
        )
    )
    if isinstance(telemetry_row, MutableMapping):
        telemetry_row["pose_selection_used_estimate"] = use_estimate
        telemetry_row["pose_selection_reason"] = "estimate_better" if use_estimate else "current_better_or_tie"
    return use_estimate


def append_selection_policy_telemetry(
    telemetry_row: Optional[MutableMapping[str, Any]],
    *,
    selection_policy: str,
    T_true: Any,
    T_current: Any,
    T_est: Any,
) -> None:
    if not isinstance(telemetry_row, MutableMapping):
        return
    telemetry_row["pose_selection_policy"] = normalize_pose_selection_policy(selection_policy)
    if T_true is None:
        return
    cur_te, cur_re = pose_error_key(T_current, T_true)
    telemetry_row["current_pose_te_m"] = None if not math.isfinite(cur_te) else float(cur_te)
    telemetry_row["current_pose_re_deg"] = None if not math.isfinite(cur_re) else float(cur_re)
    if T_est is not None:
        est_te, est_re = pose_error_key(T_est, T_true)
        telemetry_row["est_pose_te_m"] = None if not math.isfinite(est_te) else float(est_te)
        telemetry_row["est_pose_re_deg"] = None if not math.isfinite(est_re) else float(est_re)
