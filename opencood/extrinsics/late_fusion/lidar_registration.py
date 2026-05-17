from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from opencood.extrinsics.types import ExtrinsicEstimate, ExtrinsicInit, MethodContext


@dataclass(frozen=True)
class LidarRegistrationConfig:
    voxel_size_m: float = 1.0
    max_corr_dist_m: float = 2.0
    ransac_n: int = 4
    ransac_max_iter: int = 50000
    ransac_confidence: float = 0.999
    global_method: str = "ransac"  # ransac | fgr | teaser_gnctls | teaser_fgr | teaser_quatro
    use_fgr: bool = False
    icp_method: str = "point_to_plane"  # point_to_plane | point_to_point | gicp
    icp_max_iter: int = 50
    min_points: int = 200
    max_points: int = 60000
    teaser_noise_bound_m: float = 2.0
    teaser_mutual_filter: bool = True
    teaser_max_correspondences: int = 8000
    teaser_rotation_max_iterations: int = 10000
    teaser_rotation_cost_threshold: float = 1e-16
    teaser_rotation_gnc_factor: float = 1.4


def _downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = np.random.choice(points.shape[0], size=int(max_points), replace=False)
    return points[idx]


def _to_o3d(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    # open3d.Vector3dVector expects a contiguous float64 Nx3 array; passing float32 or
    # non-contiguous views (e.g. slicing Nx4 -> Nx3) can segfault on some builds.
    xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def _compute_fpfh(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.pipelines.registration.Feature:
    radius_normal = max(1e-3, voxel_size * 2.0)
    radius_feature = max(1e-3, voxel_size * 5.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )


def _as_feature_matrix(feat: o3d.pipelines.registration.Feature) -> np.ndarray:
    data = np.asarray(feat.data, dtype=np.float64)
    if data.ndim != 2:
        return np.zeros((0, 0), dtype=np.float64)
    return np.ascontiguousarray(data.T, dtype=np.float64)


def _feature_correspondences(
    src_feat: np.ndarray,
    dst_feat: np.ndarray,
    *,
    mutual_filter: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if src_feat.size == 0 or dst_feat.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    tree_dst = cKDTree(dst_feat)
    nn_src_to_dst = tree_dst.query(src_feat, k=1)[1].astype(np.int64)
    src_idx = np.arange(nn_src_to_dst.shape[0], dtype=np.int64)
    dst_idx = nn_src_to_dst

    if not mutual_filter:
        return src_idx, dst_idx

    tree_src = cKDTree(src_feat)
    nn_dst_to_src = tree_src.query(dst_feat, k=1)[1].astype(np.int64)
    keep = nn_dst_to_src[dst_idx] == src_idx
    return src_idx[keep], dst_idx[keep]


def _best_fit_rigid(A: np.ndarray, B: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute rigid transform (R,t) such that B ~= R @ A + t.

    Returns (R,t) or None if input is degenerate.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != 3 or B.shape[1] != 3:
        return None
    if A.shape[0] < 3 or B.shape[0] < 3 or A.shape[0] != B.shape[0]:
        return None

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    try:
        U, S, Vt = np.linalg.svd(H)
    except Exception:
        return None
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1.0
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R.astype(np.float64), t.astype(np.float64)


def _icp_point_to_point(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    init_T: np.ndarray,
    max_corr: float,
    max_iter: int,
) -> Tuple[np.ndarray, float, float]:
    """
    Lightweight CPU ICP refinement that avoids Open3D's `registration_icp`,
    which can segfault on some builds.

    Returns (T, fitness, rmse).
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(init_T, dtype=np.float64).reshape(4, 4).copy()

    if src.shape[0] < 3 or dst.shape[0] < 3:
        return T, 0.0, float("inf")

    tree = cKDTree(dst)
    max_corr = float(max_corr)
    use_gate = max_corr > 0
    prev_rmse = None

    for _ in range(int(max_iter)):
        R0 = T[:3, :3]
        t0 = T[:3, 3]
        src_T = (R0 @ src.T).T + t0[None, :]

        dists, idx = tree.query(src_T, k=1)
        if use_gate:
            mask = dists <= max_corr
            if int(mask.sum()) < 3:
                break
            A = src_T[mask]
            B = dst[idx[mask]]
            inlier_d = dists[mask]
        else:
            A = src_T
            B = dst[idx]
            inlier_d = dists

        fit = _best_fit_rigid(A, B)
        if fit is None:
            break
        R, t = fit
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = R
        delta[:3, 3] = t
        T = delta @ T

        rmse = float(np.sqrt(np.mean(np.square(inlier_d)))) if inlier_d.size else float("inf")
        if prev_rmse is not None and abs(prev_rmse - rmse) < 1e-6:
            break
        prev_rmse = rmse

    # Final fitness/rmse from last association.
    R0 = T[:3, :3]
    t0 = T[:3, 3]
    src_T = (R0 @ src.T).T + t0[None, :]
    dists, _ = tree.query(src_T, k=1)
    if use_gate:
        inliers = dists <= max_corr
        if int(inliers.sum()) > 0:
            rmse = float(np.sqrt(np.mean(np.square(dists[inliers]))))
            fitness = float(inliers.mean())
        else:
            rmse = float("inf")
            fitness = 0.0
    else:
        rmse = float(np.sqrt(np.mean(np.square(dists))))
        fitness = 1.0

    return T, float(fitness), float(rmse)


def _teaser_rotation_alg(global_method: str, teaserpp_python):
    alg = str(global_method or "").lower().strip()
    enum = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM
    if alg == "teaser_fgr":
        return enum.FGR
    if alg == "teaser_quatro":
        return enum.QUATRO
    return enum.GNC_TLS


def _coarse_with_teaser(
    *,
    cfg: LidarRegistrationConfig,
    pcd_src_down: o3d.geometry.PointCloud,
    pcd_dst_down: o3d.geometry.PointCloud,
    fpfh_src: o3d.pipelines.registration.Feature,
    fpfh_dst: o3d.pipelines.registration.Feature,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    try:
        import teaserpp_python  # type: ignore
    except Exception as e:
        return None, {"reason": "teaser_import_failed", "teaser_error": str(e)}

    src_feat = _as_feature_matrix(fpfh_src)
    dst_feat = _as_feature_matrix(fpfh_dst)
    src_idx, dst_idx = _feature_correspondences(
        src_feat,
        dst_feat,
        mutual_filter=bool(cfg.teaser_mutual_filter),
    )
    if src_idx.shape[0] < 3:
        return None, {"reason": "too_few_correspondences", "teaser_correspondences": int(src_idx.shape[0])}

    max_corr = int(cfg.teaser_max_correspondences or 0)
    if max_corr > 0 and src_idx.shape[0] > max_corr:
        rng = np.random.default_rng(3407)
        keep = rng.choice(src_idx.shape[0], size=max_corr, replace=False)
        src_idx = src_idx[keep]
        dst_idx = dst_idx[keep]

    src_xyz = np.asarray(pcd_src_down.points, dtype=np.float64)[src_idx, :3].T
    dst_xyz = np.asarray(pcd_dst_down.points, dtype=np.float64)[dst_idx, :3].T
    if src_xyz.shape[1] < 3:
        return None, {"reason": "too_few_points_for_teaser", "teaser_correspondences": int(src_xyz.shape[1])}

    params = teaserpp_python.RobustRegistrationSolver.Params()
    params.cbar2 = 1.0
    params.noise_bound = float(max(1e-3, cfg.teaser_noise_bound_m))
    params.estimate_scaling = False
    params.rotation_estimation_algorithm = _teaser_rotation_alg(cfg.global_method, teaserpp_python)
    params.rotation_gnc_factor = float(cfg.teaser_rotation_gnc_factor)
    params.rotation_max_iterations = int(cfg.teaser_rotation_max_iterations)
    params.rotation_cost_threshold = float(cfg.teaser_rotation_cost_threshold)
    params.inlier_selection_mode = teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    params.rotation_tim_graph = teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN

    try:
        solver = teaserpp_python.RobustRegistrationSolver(params)
        solver.solve(src_xyz, dst_xyz)
        sol = solver.getSolution()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(sol.rotation, dtype=np.float64)
        T[:3, 3] = np.asarray(sol.translation, dtype=np.float64).reshape(3)
        return T, {"teaser_correspondences": int(src_xyz.shape[1])}
    except Exception as e:
        return None, {"reason": "teaser_solve_failed", "teaser_error": str(e), "teaser_correspondences": int(src_xyz.shape[1])}


class LidarRegistrationEstimator:
    """
    Coarse-to-fine LiDAR registration using FPFH + RANSAC/FGR + ICP.
    Returns a source->target transform in LiDAR coordinates.
    """

    def __init__(self, *, cfg: Optional[LidarRegistrationConfig] = None) -> None:
        self._cfg = cfg or LidarRegistrationConfig()

    @property
    def config(self) -> LidarRegistrationConfig:
        return self._cfg

    def estimate_from_points(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        *,
        init: Optional[ExtrinsicInit] = None,
        ctx: Optional[MethodContext] = None,
    ) -> ExtrinsicEstimate:
        ctx = ctx or MethodContext()
        start = perf_counter()
        cfg = self._cfg

        if src_points is None or dst_points is None:
            return ExtrinsicEstimate(T=None, success=False, method="lidar_reg", extra={"reason": "missing_points"})

        src = np.asarray(src_points, dtype=np.float32).reshape(-1, src_points.shape[-1])
        dst = np.asarray(dst_points, dtype=np.float32).reshape(-1, dst_points.shape[-1])
        if src.shape[0] < cfg.min_points or dst.shape[0] < cfg.min_points:
            return ExtrinsicEstimate(
                T=None,
                success=False,
                method="lidar_reg",
                extra={"reason": "too_few_points", "src_points": int(src.shape[0]), "dst_points": int(dst.shape[0])},
            )

        src = _downsample_points(src, int(cfg.max_points))
        dst = _downsample_points(dst, int(cfg.max_points))

        pcd_src = _to_o3d(src)
        pcd_dst = _to_o3d(dst)

        voxel_size = float(cfg.voxel_size_m)
        if voxel_size > 0:
            pcd_src_down = pcd_src.voxel_down_sample(voxel_size)
            pcd_dst_down = pcd_dst.voxel_down_sample(voxel_size)
        else:
            pcd_src_down = pcd_src
            pcd_dst_down = pcd_dst

        if len(pcd_src_down.points) < cfg.min_points or len(pcd_dst_down.points) < cfg.min_points:
            return ExtrinsicEstimate(
                T=None,
                success=False,
                method="lidar_reg",
                extra={"reason": "too_few_points_downsampled"},
            )

        fpfh_src = _compute_fpfh(pcd_src_down, max(1e-3, voxel_size))
        fpfh_dst = _compute_fpfh(pcd_dst_down, max(1e-3, voxel_size))

        max_corr = max(1e-3, float(cfg.max_corr_dist_m))
        coarse_extra: Dict[str, Any] = {}
        global_method = str(cfg.global_method or "").lower().strip()
        if not global_method:
            global_method = "fgr" if bool(cfg.use_fgr) else "ransac"
        if global_method == "fgr":
            coarse = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
                pcd_src_down,
                pcd_dst_down,
                fpfh_src,
                fpfh_dst,
                o3d.pipelines.registration.FastGlobalRegistrationOption(
                    maximum_correspondence_distance=max_corr,
                ),
            )
            init_T = np.asarray(coarse.transformation, dtype=np.float64)
            coarse_extra["coarse_fitness"] = float(coarse.fitness)
            coarse_extra["coarse_inlier_rmse"] = float(coarse.inlier_rmse)
        elif global_method == "ransac":
            coarse = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                pcd_src_down,
                pcd_dst_down,
                fpfh_src,
                fpfh_dst,
                mutual_filter=True,
                max_correspondence_distance=max_corr,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=int(cfg.ransac_n),
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_corr),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                    int(cfg.ransac_max_iter), float(cfg.ransac_confidence)
                ),
            )
            init_T = np.asarray(coarse.transformation, dtype=np.float64)
            coarse_extra["coarse_fitness"] = float(coarse.fitness)
            coarse_extra["coarse_inlier_rmse"] = float(coarse.inlier_rmse)
        elif global_method in {"teaser_gnctls", "teaser_fgr", "teaser_quatro"}:
            init_T, teaser_extra = _coarse_with_teaser(
                cfg=cfg,
                pcd_src_down=pcd_src_down,
                pcd_dst_down=pcd_dst_down,
                fpfh_src=fpfh_src,
                fpfh_dst=fpfh_dst,
            )
            coarse_extra.update(teaser_extra)
            if init_T is None:
                return ExtrinsicEstimate(
                    T=None,
                    success=False,
                    method="lidar_reg",
                    extra={"reason": teaser_extra.get("reason", "teaser_failed"), **teaser_extra},
                )
        else:
            return ExtrinsicEstimate(
                T=None,
                success=False,
                method="lidar_reg",
                extra={"reason": "unknown_global_method", "global_method": global_method},
            )

        coarse_extra["coarse_method"] = global_method

        if init is not None and init.T_init is not None:
            init_T = np.asarray(init.T_init, dtype=np.float64)

        icp_max_iter = int(cfg.icp_max_iter)
        if icp_max_iter > 0:
            # Avoid Open3D ICP due to segfaults observed on some Open3D builds.
            # We only support point-to-point refinement here.
            T, fitness, rmse = _icp_point_to_point(
                np.asarray(src[:, :3], dtype=np.float64),
                np.asarray(dst[:, :3], dtype=np.float64),
                init_T=np.asarray(init_T, dtype=np.float64),
                max_corr=max_corr,
                max_iter=icp_max_iter,
            )
        else:
            T = np.asarray(init_T, dtype=np.float64)
            fitness = float(coarse_extra.get("coarse_fitness", 0.0) or 0.0)
            rmse = float(coarse_extra.get("coarse_inlier_rmse", 0.0) or 0.0)
        time_sec = float(perf_counter() - start)

        stability = float(fitness)
        return ExtrinsicEstimate(
            T=T,
            success=True,
            method="lidar_reg",
            stability=stability,
            time_sec=time_sec,
            extra={
                "fitness": float(fitness),
                "inlier_rmse": float(rmse),
                **coarse_extra,
            },
        )


__all__ = ["LidarRegistrationConfig", "LidarRegistrationEstimator"]
