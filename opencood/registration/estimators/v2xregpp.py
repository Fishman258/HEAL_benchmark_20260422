from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from opencood.utils.path_utils import ensure_v2xreg_root_on_path, resolve_repo_path


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


def build_lidar_occ_map(
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


def estimate_occ_hint(
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
    from opencood.registration.utils.geometry import convert_6DOF_to_T  # type: ignore

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


def estimate_occ_hint_T(
    occ_src: np.ndarray,
    occ_dst: np.ndarray,
    *,
    bev_range: Sequence[float],
    rotation_max_deg: float,
    rotation_step_deg: float,
    min_peak: float,
    min_peak_ratio: float,
) -> Optional[np.ndarray]:
    hint = estimate_occ_hint(
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


def icp_refine_T(
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
class V2XRegPPRuntimeEstimator:
    config_path: str
    bbox_type: str = "detected"
    use_occ_hint: bool = False
    use_occ_pose: bool = False
    force_occ_pose: bool = False
    occ_max_delta_xy_m: float = 20.0
    occ_max_delta_yaw_deg: float = 45.0
    compare_with_current: bool = False
    compare_distance_threshold_m: float = 3.0
    apply_if_current_precision_below: float = -1.0
    min_precision_improvement: float = 0.1
    min_matched_improvement: int = 1
    icp_refine: bool = False
    icp_voxel_size_m: float = 1.0
    icp_max_corr_dist_m: float = 2.0
    icp_max_iterations: int = 30
    device: Optional[str] = None
    _cfg: Any = field(default=None, init=False, repr=False)
    _filters: Any = field(default=None, init=False, repr=False)
    _matcher: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        ensure_v2xreg_root_on_path()
        from opencood.registration.estimators.v2xregpp_runtime.config import load_config  # type: ignore
        from opencood.registration.estimators.v2xregpp_runtime.filters.pipeline import FilterPipeline  # type: ignore
        from opencood.registration.estimators.v2xregpp_runtime.matching.engine import MatchingEngine  # type: ignore

        cfg_path = resolve_repo_path(self.config_path)
        self._cfg = load_config(str(cfg_path))
        self._filters = FilterPipeline(self._cfg.filters)
        self._matcher = MatchingEngine(self._cfg.matching, device=self.device)

    @property
    def config(self):
        return self._cfg

    def _refine_payload_with_icp(
        self,
        payload: Optional[Dict[str, Any]],
        *,
        src_lidar_np: Any = None,
        dst_lidar_np: Any = None,
    ) -> Optional[Dict[str, Any]]:
        if payload is None or not self.icp_refine:
            return payload
        T_init = payload.get("T")
        if T_init is None:
            return payload
        T_icp = icp_refine_T(
            src_lidar_np,
            dst_lidar_np,
            T_init=np.asarray(T_init, dtype=np.float64),
            voxel_size_m=self.icp_voxel_size_m,
            max_corr_dist_m=self.icp_max_corr_dist_m,
            max_iterations=self.icp_max_iterations,
        )
        if T_icp is not None:
            payload = dict(payload)
            payload["T"] = T_icp
        return payload

    def estimate_rel_T(
        self,
        src_boxes,
        dst_boxes,
        occ_src,
        occ_dst,
        bev_range,
        *,
        T_current: Optional[np.ndarray] = None,
        allow_gate_reject_return: bool = False,
        src_lidar_np: Any = None,
        dst_lidar_np: Any = None,
    ) -> Optional[Dict[str, Any]]:
        ensure_v2xreg_root_on_path()
        from opencood.registration.estimators.v2xregpp_runtime.matches_to_extrinsics import Matches2Extrinsics  # type: ignore
        from opencood.registration.utils.geometry import convert_6DOF_to_T, implement_T_3dbox_object_list  # type: ignore
        from opencood.registration.estimators.box_matching import CorrespondingDetector  # type: ignore

        filtered_src, filtered_dst = self._filters.apply(src_boxes or [], dst_boxes or [])
        has_boxes = bool(filtered_src) and bool(filtered_dst)

        occ_hint = None
        T_hint = None
        if (self.use_occ_hint or self.use_occ_pose) and occ_src is not None and occ_dst is not None:
            bev_range = list(bev_range or [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5])
            occ_hint = estimate_occ_hint(
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
                return self._refine_payload_with_icp(
                    payload,
                    src_lidar_np=src_lidar_np,
                    dst_lidar_np=dst_lidar_np,
                )
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
                device=self.device,
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
            return self._refine_payload_with_icp(
                forced,
                src_lidar_np=src_lidar_np,
                dst_lidar_np=dst_lidar_np,
            )

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
                device=self.device,
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
                        device=self.device,
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
                return self._refine_payload_with_icp(
                    payload,
                    src_lidar_np=src_lidar_np,
                    dst_lidar_np=dst_lidar_np,
                )
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

                if float(self.apply_if_current_precision_below) >= 0.0:
                    cur_quality = float(cur_precision) + 3.0
                    if int(cur_matched) > 0 and float(cur_quality) > float(self.apply_if_current_precision_below):
                        if allow_gate_reject_return:
                            best["gate_reject"] = True
                            best["gate_reject_reason"] = "current_pose_good_enough"
                            return self._refine_payload_with_icp(
                                best,
                                src_lidar_np=src_lidar_np,
                                dst_lidar_np=dst_lidar_np,
                            )
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
                            return self._refine_payload_with_icp(
                                best,
                                src_lidar_np=src_lidar_np,
                                dst_lidar_np=dst_lidar_np,
                            )
                        return None

        return self._refine_payload_with_icp(
            best,
            src_lidar_np=src_lidar_np,
            dst_lidar_np=dst_lidar_np,
        )


__all__ = [
    "V2XRegPPRuntimeEstimator",
    "build_lidar_occ_map",
    "estimate_occ_hint",
    "estimate_occ_hint_T",
    "icp_refine_T",
]
