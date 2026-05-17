from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from opencood.extrinsics.late_fusion.image_matching import ImageMatchingConfig, ImageMatchingEstimator
from opencood.extrinsics.types import ExtrinsicInit
from opencood.utils.transformation_utils import pose_to_tfm, tfm_to_pose, x1_to_x2


_UE4_TO_OPENCV = np.array(
    [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)


def _wrap_angle_deg(angle: float) -> float:
    wrapped = (float(angle) + 180.0) % 360.0 - 180.0
    return float(wrapped)


def _delta_angle_deg(a: float, b: float) -> float:
    return _wrap_angle_deg(float(a) - float(b))


def _as_int_list(raw: Optional[object]) -> Optional[Sequence[int]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        values = []
        for item in raw:
            try:
                values.append(int(item))
            except Exception:
                continue
        return values if values else None
    if isinstance(raw, str):
        values = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(int(token))
            except Exception:
                continue
        return values if values else None
    try:
        return [int(raw)]
    except Exception:
        return None


def _camera_index_from_key(key: str) -> Optional[int]:
    key = str(key or "")
    if not key.startswith("camera"):
        return None
    suffix = key[len("camera") :]
    if not suffix:
        return None
    try:
        return int(suffix)
    except Exception:
        return None


def _as_numpy_image(image: Any) -> Optional[np.ndarray]:
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return image
    try:
        return np.asarray(image)
    except Exception:
        return None


def _to_4x4(matrix: np.ndarray) -> Optional[np.ndarray]:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape == (4, 4):
        return mat
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = mat
        return out
    if mat.shape == (4, 3):
        out = np.eye(4, dtype=np.float64)
        out[:, :3] = mat
        return out
    if mat.shape == (3, 3):
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = mat
        return out
    return None


def _normalize_intrinsics(K: np.ndarray) -> Optional[np.ndarray]:
    mat = np.asarray(K, dtype=np.float64)
    if mat.shape == (3, 3):
        return mat
    if mat.shape == (3, 4):
        return mat[:, :3].copy()
    if mat.shape == (4, 4):
        return mat[:3, :3].copy()
    return None


def _pick_camera_key(params: Mapping[str, Any], camera_index: int) -> Optional[str]:
    key = f"camera{int(camera_index)}"
    if key in params:
        return key
    for name, value in params.items():
        if not isinstance(name, str):
            continue
        if not name.startswith("camera"):
            continue
        if isinstance(value, Mapping) and any(k in value for k in ("intrinsic", "extrinsic", "cords")):
            return name
    return None


def _extract_camera_info(
    cav_content: Mapping[str, Any],
    camera_index: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    params = cav_content.get("params") or {}
    camera_key = _pick_camera_key(params, camera_index)
    if camera_key is None:
        return None
    cam_params = params.get(camera_key) or {}
    image_list = cav_content.get("camera_data") or []

    img_index = _camera_index_from_key(camera_key)
    if img_index is None:
        img_index = int(camera_index)
    image = None
    if isinstance(image_list, (list, tuple)) and image_list:
        if 0 <= img_index < len(image_list):
            image = image_list[int(img_index)]
        else:
            image = image_list[0]
    image_np = _as_numpy_image(image)
    if image_np is None:
        return None

    K = cam_params.get("intrinsic")
    if K is None:
        return None
    K = _normalize_intrinsics(K)
    if K is None:
        return None

    T_cam_lidar = None
    if "cords" in cam_params:
        camera_coords = cam_params.get("cords")
        lidar_pose_ref = params.get("lidar_pose_clean")
        if lidar_pose_ref is None:
            lidar_pose_ref = params.get("lidar_pose")
        if camera_coords is None or lidar_pose_ref is None:
            return None
        T_lidar_camera = x1_to_x2(camera_coords, lidar_pose_ref).astype(np.float64)
        T_lidar_camera = T_lidar_camera @ _UE4_TO_OPENCV
        try:
            T_cam_lidar = np.linalg.inv(T_lidar_camera)
        except Exception:
            T_cam_lidar = None
    elif "extrinsic" in cam_params:
        T_cam_lidar = _to_4x4(cam_params.get("extrinsic"))
    if T_cam_lidar is None:
        return None

    return image_np, K, T_cam_lidar


@dataclass
class Stage1ImageMatchPoseCorrector:
    """
    Pose correction using raw image matching (feature matching + essential matrix).

    This corrector estimates camera-to-camera extrinsics, then maps them to lidar frames
    so cooperative fusion consumes corrected poses.
    """

    cfg: ImageMatchingConfig = field(default_factory=ImageMatchingConfig)
    mode: str = "initfree"  # initfree | stable
    camera_index: int = 0
    camera_indices: Optional[Sequence[int]] = None
    try_all_cameras: bool = False
    init_source: str = "current"  # current | clean | none
    min_stability: float = 0.0
    compare_with_current: bool = False
    compare_distance_threshold_m: float = 3.0
    ema_alpha: float = 0.5
    max_step_xy_m: float = 3.0
    max_step_yaw_deg: float = 10.0
    freeze_ego: bool = True
    cache_rel_T: bool = True
    _state: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _rel_T_est_cache: Dict[Tuple[int, str], Optional[np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._estimator = ImageMatchingEstimator(cfg=self.cfg)
        mode = str(self.mode or "initfree").lower().strip()
        if mode not in {"initfree", "stable"}:
            mode = "initfree"
        self.mode = mode
        init_source = str(self.init_source or "current").lower().strip()
        if init_source not in {"current", "clean", "none"}:
            init_source = "current"
        self.init_source = init_source
        self.min_stability = float(self.min_stability or 0.0)
        self.compare_with_current = bool(self.compare_with_current)
        self.compare_distance_threshold_m = float(self.compare_distance_threshold_m or 0.0)
        self.ema_alpha = float(self.ema_alpha or 0.0)
        self.ema_alpha = float(np.clip(self.ema_alpha, 0.0, 1.0))
        self.max_step_xy_m = float(self.max_step_xy_m or 0.0)
        self.max_step_yaw_deg = float(self.max_step_yaw_deg or 0.0)
        self.freeze_ego = bool(self.freeze_ego)
        parsed_indices = _as_int_list(self.camera_indices)
        self.camera_indices = parsed_indices
        if self.cache_rel_T and self.init_source == "current" and bool(self.cfg.scale_from_init):
            # Current noisy poses affect translation scale; avoid stale cache across noise sweeps.
            self.cache_rel_T = False

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

    def _candidate_indices(self, ego: Mapping[str, Any], cav: Mapping[str, Any]) -> Sequence[int]:
        if self.camera_indices:
            return list(self.camera_indices)
        if self.try_all_cameras:
            ego_list = ego.get("camera_data") or []
            cav_list = cav.get("camera_data") or []
            count = min(len(ego_list) if isinstance(ego_list, (list, tuple)) else 0,
                        len(cav_list) if isinstance(cav_list, (list, tuple)) else 0)
            if count > 0:
                return list(range(count))
        return [int(self.camera_index)]

    def _build_init_T_cam(
        self,
        base_data_dict: Mapping[Any, Dict[str, Any]],
        ego_id: Any,
        cav_id: Any,
        rel_T_current: np.ndarray,
        T_cam_lidar_ego: np.ndarray,
        T_cam_lidar_cav: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self.init_source == "none":
            return None
        if self.init_source == "clean":
            ego_pose = base_data_dict[ego_id]["params"].get("lidar_pose_clean")
            cav_pose = base_data_dict[cav_id]["params"].get("lidar_pose_clean")
            if ego_pose is None or cav_pose is None:
                return None
            ego_T_world = pose_to_tfm(np.asarray([ego_pose], dtype=np.float64))[0]
            cav_T_world = pose_to_tfm(np.asarray([cav_pose], dtype=np.float64))[0]
            rel_T = np.linalg.inv(ego_T_world) @ cav_T_world
        else:
            rel_T = rel_T_current
        T_cam = T_cam_lidar_ego @ rel_T @ np.linalg.inv(T_cam_lidar_cav)
        return np.asarray(T_cam, dtype=np.float64)

    def _estimate_rel_T_lidar(
        self,
        *,
        src_image: np.ndarray,
        dst_image: np.ndarray,
        K_src: np.ndarray,
        K_dst: np.ndarray,
        T_cam_lidar_src: np.ndarray,
        T_cam_lidar_dst: np.ndarray,
        init_T_cam: Optional[np.ndarray],
    ) -> Optional[Tuple[np.ndarray, float]]:
        init = None
        if init_T_cam is not None:
            init = ExtrinsicInit(T_init=init_T_cam, source="init")
        est = self._estimator.estimate_from_images(
            src_image,
            dst_image,
            K_src=K_src,
            K_dst=K_dst,
            init=init,
            ctx=None,
        )
        if not est.success or est.T is None:
            return None
        stability = float(est.stability or 0.0)
        if stability < float(self.min_stability):
            return None
        rel_T_lidar = np.linalg.inv(T_cam_lidar_dst) @ np.asarray(est.T, dtype=np.float64) @ T_cam_lidar_src
        return rel_T_lidar, stability

    def apply(
        self,
        *,
        sample_idx: int,
        cav_id_list: Sequence[Any],
        base_data_dict: MutableMapping[Any, Dict[str, Any]],
        stage1_result: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        self._reset_if_new_epoch(int(sample_idx))
        if not cav_id_list:
            return False
        ego_id = cav_id_list[0]
        if ego_id not in base_data_dict:
            return False

        ego_pose = base_data_dict[ego_id]["params"]["lidar_pose"]
        ego_T_world = pose_to_tfm(np.asarray([ego_pose], dtype=np.float64))[0]

        updated_any = False

        for cav_id in cav_id_list:
            if cav_id == ego_id and self.freeze_ego:
                continue
            if cav_id not in base_data_dict:
                continue

            cav_pose_current = base_data_dict[cav_id]["params"]["lidar_pose"]
            cav_T_world_current = pose_to_tfm(np.asarray([cav_pose_current], dtype=np.float64))[0]
            rel_current_T = np.linalg.inv(ego_T_world) @ cav_T_world_current

            cache_key = (int(sample_idx), str(cav_id))
            rel_T_est_lidar = None
            if self.cache_rel_T:
                cached = self._rel_T_est_cache.get(cache_key)
                if cached is not None:
                    rel_T_est_lidar = np.asarray(cached, dtype=np.float64)
                elif cached is None and cache_key in self._rel_T_est_cache:
                    rel_T_est_lidar = None

            if rel_T_est_lidar is None and (not self.cache_rel_T or cache_key not in self._rel_T_est_cache):
                best = None
                best_score = -1.0
                candidate_indices = self._candidate_indices(base_data_dict[ego_id], base_data_dict[cav_id])
                for cam_idx in candidate_indices:
                    ego_cam = _extract_camera_info(base_data_dict[ego_id], cam_idx)
                    cav_cam = _extract_camera_info(base_data_dict[cav_id], cam_idx)
                    if ego_cam is None or cav_cam is None:
                        continue
                    ego_img, K_ego, T_cam_lidar_ego = ego_cam
                    cav_img, K_cav, T_cam_lidar_cav = cav_cam
                    init_T_cam = self._build_init_T_cam(
                        base_data_dict,
                        ego_id,
                        cav_id,
                        rel_current_T,
                        T_cam_lidar_ego,
                        T_cam_lidar_cav,
                    )
                    est_pair = self._estimate_rel_T_lidar(
                        src_image=cav_img,
                        dst_image=ego_img,
                        K_src=K_cav,
                        K_dst=K_ego,
                        T_cam_lidar_src=T_cam_lidar_cav,
                        T_cam_lidar_dst=T_cam_lidar_ego,
                        init_T_cam=init_T_cam,
                    )
                    if est_pair is None:
                        continue
                    rel_T_candidate, stability = est_pair
                    if stability > best_score:
                        best_score = stability
                        best = rel_T_candidate

                rel_T_est_lidar = best
                if self.cache_rel_T:
                    self._rel_T_est_cache[cache_key] = (
                        None if rel_T_est_lidar is None else np.asarray(rel_T_est_lidar, dtype=np.float64)
                    )

            if rel_T_est_lidar is None:
                continue

            if self.compare_with_current and self.compare_distance_threshold_m > 0.0:
                try:
                    delta = np.linalg.inv(rel_current_T) @ np.asarray(rel_T_est_lidar, dtype=np.float64)
                    delta_xy = float(np.linalg.norm(delta[:2, 3]))
                except Exception:
                    delta_xy = None
                if delta_xy is not None and delta_xy > float(self.compare_distance_threshold_m):
                    continue

            rel_T_corrected = None
            if self.mode == "stable":
                rel_key = str(cav_id)
                prev_delta = self._get_prev_delta(rel_key)
                try:
                    delta_T = np.asarray(rel_T_est_lidar, dtype=np.float64) @ np.linalg.inv(
                        np.asarray(rel_current_T, dtype=np.float64)
                    )
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
                        # Clamp the very first step as well; otherwise a bad init estimate can
                        # permanently jump the filter state.
                        delta_xyyaw = self._clamp_se2_delta(delta_xyyaw)
                    else:
                        if not self._limit_step_se2(prev_delta, delta_xyyaw):
                            delta_xyyaw = prev_delta
                        else:
                            delta_xyyaw = self._smooth_se2(prev_delta, delta_xyyaw)
                    self._set_prev_delta(rel_key, delta_xyyaw)
                    prev_delta = delta_xyyaw

                if prev_delta is not None:
                    delta_T_se2 = pose_to_tfm(
                        np.asarray([[prev_delta[0], prev_delta[1], prev_delta[2]]], dtype=np.float64)
                    )[0]
                    rel_T_corrected = np.asarray(delta_T_se2, dtype=np.float64) @ np.asarray(rel_current_T, dtype=np.float64)
                else:
                    continue
            else:
                rel_T_corrected = rel_T_est_lidar

            cav_T_world_new = ego_T_world @ np.asarray(rel_T_corrected, dtype=np.float64)
            cav_pose_new = tfm_to_pose(cav_T_world_new)
            cav_pose = list(base_data_dict[cav_id]["params"]["lidar_pose"])
            cav_pose[0] = float(cav_pose_new[0])
            cav_pose[1] = float(cav_pose_new[1])
            cav_pose[4] = float(cav_pose_new[4])
            base_data_dict[cav_id]["params"]["lidar_pose"] = cav_pose
            updated_any = True

        return updated_any


__all__ = ["Stage1ImageMatchPoseCorrector"]
