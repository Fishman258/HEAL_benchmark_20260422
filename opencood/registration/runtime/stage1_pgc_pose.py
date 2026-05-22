from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from opencood.utils.transformation_utils import pose_to_tfm, tfm_to_pose


def _wrap_angle_deg(angle: float) -> float:
    return float(((angle + 180.0) % 360.0) - 180.0)


def _delta_angle_deg(a: float, b: float) -> float:
    return float(_wrap_angle_deg(a - b))


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


def _resolve_agent_index(
    all_agent_ids: Sequence[Any],
    cav_id: Any,
    base_data_dict: Mapping[Any, Any],
) -> Optional[int]:
    cav_str = str(cav_id)
    for idx, agent_id in enumerate(all_agent_ids):
        if str(agent_id) == cav_str:
            return int(idx)
    role = _infer_dair_role_from_base(base_data_dict, cav_id)
    if role is None:
        return None
    for idx, agent_id in enumerate(all_agent_ids):
        if _normalize_agent_role(agent_id) == role:
            return int(idx)
    return None


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


@dataclass
class Stage1PGCPoseCorrector:
    """
    Apply pose override / correction from a cached pose JSON (e.g., V2VLoc PGC output).

    The cache format is expected to be:
      sample_idx -> {
        "cav_id_list": [...],
        "<pose_field>": [[x,y,z,roll,yaw,pitch], ...],
        "<confidence_field>": [conf, ...] (optional)
      }

    This corrector can run in:
      - initfree: directly apply the cached relative pose (ego->cav) to the current ego pose.
      - stable: smooth correction deltas in SE(2) over time (mirrors v2xregpp_stable).
    """

    pose_field: str = "lidar_pose_pred_np"
    confidence_field: str = "pose_confidence_np"
    min_confidence: float = 0.0
    mode: str = "initfree"  # initfree | stable
    ema_alpha: float = 0.5
    max_step_xy_m: float = 3.0
    max_step_yaw_deg: float = 10.0
    freeze_ego: bool = True

    _state: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

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
        pose_result: Mapping[str, Any],
    ) -> bool:
        self._reset_if_new_epoch(int(sample_idx))

        key = str(sample_idx)
        entry = pose_result.get(key)
        if entry is None or not isinstance(entry, Mapping):
            return False

        all_agent_ids = entry.get("cav_id_list") or entry.get("cav_ids") or []
        poses = entry.get(self.pose_field) or []
        confs = entry.get(self.confidence_field) or []
        if not isinstance(all_agent_ids, list) or not isinstance(poses, list):
            return False
        if not all_agent_ids or not poses:
            return False
        if not cav_id_list:
            return False
        ego_id = cav_id_list[0]
        if ego_id not in base_data_dict:
            return False

        mode = str(self.mode or "initfree").lower().strip()
        stable = mode == "stable"

        # Build predicted transforms (world) for the agents that exist in this frame.
        def _pose6_at(idx: int) -> Optional[np.ndarray]:
            if idx < 0 or idx >= len(poses):
                return None
            try:
                arr = np.asarray(poses[idx], dtype=np.float64).reshape(-1)
            except Exception:
                return None
            if arr.size == 6:
                return arr
            if arr.size == 3:
                full = np.zeros((6,), dtype=np.float64)
                full[[0, 1, 4]] = arr
                return full
            return None

        def _conf_at(idx: int) -> Optional[float]:
            if not isinstance(confs, list) or idx < 0 or idx >= len(confs):
                return None
            try:
                return float(confs[idx])
            except Exception:
                return None

        # Ego predicted pose is used only to derive the relative estimate.
        ego_pred_idx = _resolve_agent_index(all_agent_ids, ego_id, base_data_dict)
        ego_pose_pred = _pose6_at(int(ego_pred_idx)) if ego_pred_idx is not None else None
        if ego_pose_pred is None:
            return False

        ego_T_world_pred = pose_to_tfm(np.asarray([ego_pose_pred], dtype=np.float64))[0]
        ego_pose_current = base_data_dict[ego_id]["params"]["lidar_pose"]
        ego_T_world_current = pose_to_tfm(np.asarray([ego_pose_current], dtype=np.float64))[0]

        updated_any = False
        for cav_id in cav_id_list:
            if cav_id == ego_id and bool(self.freeze_ego):
                continue
            if cav_id not in base_data_dict:
                continue

            cav_pred_idx = _resolve_agent_index(all_agent_ids, cav_id, base_data_dict)
            if cav_pred_idx is None:
                continue
            cav_pose_pred = _pose6_at(int(cav_pred_idx))
            if cav_pose_pred is None:
                continue

            conf = _conf_at(int(cav_pred_idx))
            if conf is not None:
                base_data_dict[cav_id]["params"]["pose_confidence"] = float(conf)
                if float(conf) < float(self.min_confidence) - 1e-9:
                    # In stable mode we may still fall back to prev_delta.
                    if not stable:
                        continue
                    cav_pose_pred = None

            cav_pose_current = base_data_dict[cav_id]["params"]["lidar_pose"]
            cav_T_world_current = pose_to_tfm(np.asarray([cav_pose_current], dtype=np.float64))[0]

            rel_current_T = np.linalg.inv(ego_T_world_current) @ cav_T_world_current
            rel_T_est = None
            if cav_pose_pred is not None:
                cav_T_world_pred = pose_to_tfm(np.asarray([cav_pose_pred], dtype=np.float64))[0]
                rel_T_est = np.linalg.inv(ego_T_world_pred) @ cav_T_world_pred

            rel_T_corrected: Optional[np.ndarray]
            if not stable:
                if rel_T_est is None:
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
                                max_step_xy_m=float(self.max_step_xy_m),
                                max_step_yaw_deg=float(self.max_step_yaw_deg),
                            ):
                                delta_xyyaw = prev_delta
                            else:
                                delta_xyyaw = _smooth_se2(prev_delta, delta_xyyaw, alpha=float(self.ema_alpha))
                        self._set_prev_delta(rel_key, delta_xyyaw)
                        prev_delta = delta_xyyaw

                if prev_delta is None:
                    continue

                delta_T_se2 = pose_to_tfm(np.asarray([[prev_delta[0], prev_delta[1], prev_delta[2]]], dtype=np.float64))[0]
                rel_T_corrected = np.asarray(delta_T_se2, dtype=np.float64) @ np.asarray(rel_current_T, dtype=np.float64)

            cav_T_world_new = ego_T_world_current @ np.asarray(rel_T_corrected, dtype=np.float64)
            cav_pose_new = tfm_to_pose(cav_T_world_new)
            cav_pose = list(base_data_dict[cav_id]["params"]["lidar_pose"])
            cav_pose[0] = float(cav_pose_new[0])
            cav_pose[1] = float(cav_pose_new[1])
            cav_pose[4] = float(cav_pose_new[4])
            base_data_dict[cav_id]["params"]["lidar_pose"] = cav_pose
            updated_any = True

        return updated_any


__all__ = ["Stage1PGCPoseCorrector"]
