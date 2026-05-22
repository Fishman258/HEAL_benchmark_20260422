from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from opencood.registration.estimators.lidar_registration import (
    LidarRegistrationConfig,
    LidarRegistrationEstimator,
)
from opencood.utils.transformation_utils import pose_to_tfm, tfm_to_pose


def _wrap_angle_deg(angle: float) -> float:
    wrapped = (float(angle) + 180.0) % 360.0 - 180.0
    return float(wrapped)


def _delta_angle_deg(a: float, b: float) -> float:
    return _wrap_angle_deg(float(a) - float(b))


@dataclass
class Stage1LidarRegPoseCorrector:
    """
    Pose correction using LiDAR registration (FPFH + RANSAC/FGR + ICP).
    """

    cfg: LidarRegistrationConfig = field(default_factory=LidarRegistrationConfig)
    mode: str = "initfree"  # initfree | stable
    compare_with_current: bool = False
    compare_distance_threshold_m: float = 3.0
    min_fitness: float = 0.0
    max_inlier_rmse: float = 0.0
    # Optional cache of estimated rel_T per (sample_idx, ego_id, cav_id). This is useful
    # for OPV2V fullbench where we sweep many noise levels but the raw point clouds do not
    # change, so re-running global registration is wasteful.
    #
    # Cache file format (JSON):
    #   {"meta": {...}, "pairs": {"<sample>|<ego>|<cav>": {"T": [[4x4]], ...} | [[4x4]] | null}}
    cache_path: Optional[str] = None
    ema_alpha: float = 0.5
    max_step_xy_m: float = 3.0
    max_step_yaw_deg: float = 10.0
    freeze_ego: bool = True
    _state: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _rel_T_cache: Dict[Tuple[int, str, str], Optional[object]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._estimator = LidarRegistrationEstimator(cfg=self.cfg)
        mode = str(self.mode or "initfree").lower().strip()
        if mode not in {"initfree", "stable"}:
            mode = "initfree"
        self.mode = mode
        self.min_fitness = float(self.min_fitness or 0.0)
        self.max_inlier_rmse = float(self.max_inlier_rmse or 0.0)
        self.compare_with_current = bool(self.compare_with_current)
        self.compare_distance_threshold_m = float(self.compare_distance_threshold_m or 0.0)
        self.ema_alpha = float(np.clip(float(self.ema_alpha or 0.0), 0.0, 1.0))
        self.max_step_xy_m = float(self.max_step_xy_m or 0.0)
        self.max_step_yaw_deg = float(self.max_step_yaw_deg or 0.0)
        self.freeze_ego = bool(self.freeze_ego)
        self._load_cache()

    def _load_cache(self) -> None:
        path = str(self.cache_path or "").strip()
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            obj = json.loads(text)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        pairs = obj.get("pairs") if isinstance(obj.get("pairs"), dict) else obj
        if not isinstance(pairs, dict):
            return
        for key, val in pairs.items():
            parts = str(key).split("|")
            if len(parts) != 3:
                continue
            try:
                sample_idx = int(parts[0])
            except Exception:
                continue
            ego_id = str(parts[1])
            cav_id = str(parts[2])
            self._rel_T_cache[(int(sample_idx), ego_id, cav_id)] = val

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

    def _smooth_se2(
        self, prev: Tuple[float, float, float], cur: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
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

    def _clamp_se2_delta(self, cur: Tuple[float, float, float]) -> Tuple[float, float, float]:
        x, y, yaw = float(cur[0]), float(cur[1]), float(cur[2])
        if self.max_step_xy_m > 0.0:
            norm = float(np.hypot(x, y))
            if norm > self.max_step_xy_m + 1e-9:
                scale = float(self.max_step_xy_m) / float(norm)
                x *= scale
                y *= scale
        if self.max_step_yaw_deg > 0.0:
            if abs(float(yaw)) > self.max_step_yaw_deg + 1e-6:
                yaw = float(np.sign(yaw) * self.max_step_yaw_deg)
        return float(x), float(y), float(yaw)

    def apply(
        self,
        *,
        sample_idx: int,
        cav_id_list: Sequence[Any],
        base_data_dict: MutableMapping[Any, Dict[str, Any]],
    ) -> bool:
        self._reset_if_new_epoch(int(sample_idx))
        if not cav_id_list:
            return False
        ego_id = cav_id_list[0]
        if ego_id not in base_data_dict:
            return False

        ego_entry = base_data_dict[ego_id]
        ego_points = ego_entry.get("lidar_np")
        if ego_points is None:
            return False

        mode = str(self.mode or "initfree").lower().strip()
        stable = mode == "stable"

        ego_pose_current = ego_entry["params"]["lidar_pose"]
        ego_T_world_current = pose_to_tfm(np.asarray([ego_pose_current], dtype=np.float64))[0]
        updated_any = False

        for cav_id in cav_id_list:
            if cav_id == ego_id and self.freeze_ego:
                continue
            cav_entry = base_data_dict.get(cav_id)
            if not isinstance(cav_entry, Mapping):
                continue
            cav_points = cav_entry.get("lidar_np")
            if cav_points is None:
                continue

            cache_key = (int(sample_idx), str(ego_id), str(cav_id))
            estimate_T = None
            fitness = 0.0
            rmse = 0.0
            if self._rel_T_cache and cache_key in self._rel_T_cache:
                self._state["cache_hits"] = int(self._state.get("cache_hits", 0) or 0) + 1
                rec = self._rel_T_cache.get(cache_key)
                if rec is None:
                    continue
                if isinstance(rec, dict):
                    estimate_T = rec.get("T") or rec.get("t") or rec.get("rel_T")
                    try:
                        fitness = float(rec.get("fitness", 0.0) or 0.0)
                    except Exception:
                        fitness = 0.0
                    try:
                        rmse = float(rec.get("inlier_rmse", rec.get("rmse", 0.0)) or 0.0)
                    except Exception:
                        rmse = 0.0
                else:
                    estimate_T = rec
            else:
                estimate = self._estimator.estimate_from_points(cav_points, ego_points)
                if not estimate.success or estimate.T is None:
                    continue
                estimate_T = estimate.T
                fitness = float(estimate.extra.get("fitness", 0.0))
                rmse = float(estimate.extra.get("inlier_rmse", 0.0))

            if estimate_T is None:
                continue
            if self.min_fitness > 0.0 and fitness < self.min_fitness:
                continue
            if self.max_inlier_rmse > 0.0 and rmse > self.max_inlier_rmse:
                continue

            # `estimate.T` maps src->dst; here src=cav and dst=ego, so it is already
            # the relative transform we want (cav->ego) for pose correction.
            rel_T_est = np.asarray(estimate_T, dtype=np.float64)
            cav_pose_current = cav_entry["params"]["lidar_pose"]
            cav_T_world_current = pose_to_tfm(np.asarray([cav_pose_current], dtype=np.float64))[0]
            rel_T_current = np.linalg.inv(ego_T_world_current) @ cav_T_world_current

            if self.compare_with_current and self.compare_distance_threshold_m > 0.0:
                try:
                    delta = np.linalg.inv(np.asarray(rel_T_current, dtype=np.float64)) @ np.asarray(
                        rel_T_est, dtype=np.float64
                    )
                    delta_xy = float(np.linalg.norm(delta[:2, 3]))
                except Exception:
                    delta_xy = None
                if delta_xy is None or delta_xy > float(self.compare_distance_threshold_m):
                    continue

            if not stable:
                rel_T_corrected = rel_T_est
            else:
                rel_key = str(cav_id)
                prev_delta = self._get_prev_delta(rel_key)
                try:
                    delta_T = rel_T_est @ np.linalg.inv(rel_T_current)
                    delta_pose6 = tfm_to_pose(delta_T)
                    delta_xyyaw = (
                        float(delta_pose6[0]),
                        float(delta_pose6[1]),
                        _wrap_angle_deg(float(delta_pose6[4])),
                    )
                except Exception:
                    delta_xyyaw = None

                if delta_xyyaw is not None:
                    if prev_delta is None:
                        # Clamp the first estimate to avoid unbounded jumps on frame 0.
                        delta_xyyaw = self._clamp_se2_delta(delta_xyyaw)
                    else:
                        if not self._limit_step_se2(prev_delta, delta_xyyaw):
                            delta_xyyaw = prev_delta
                        else:
                            delta_xyyaw = self._smooth_se2(prev_delta, delta_xyyaw)
                    self._set_prev_delta(rel_key, delta_xyyaw)
                    prev_delta = delta_xyyaw

                if prev_delta is None:
                    continue
                delta_T_se2 = pose_to_tfm(
                    np.asarray([[prev_delta[0], prev_delta[1], prev_delta[2]]], dtype=np.float64)
                )[0]
                rel_T_corrected = delta_T_se2 @ rel_T_current

            cav_T_world_new = ego_T_world_current @ rel_T_corrected
            cav_pose_new = tfm_to_pose(cav_T_world_new)
            cav_pose = list(cav_entry["params"]["lidar_pose"])
            cav_pose[0] = float(cav_pose_new[0])
            cav_pose[1] = float(cav_pose_new[1])
            cav_pose[4] = float(cav_pose_new[4])
            cav_entry["params"]["lidar_pose"] = cav_pose
            updated_any = True

        return updated_any


__all__ = ["Stage1LidarRegPoseCorrector"]
