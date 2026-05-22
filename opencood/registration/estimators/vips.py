from __future__ import annotations

import math
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh

from opencood.registration.utils.bbox import bbox3d_to_state7, corners_to_bbox3d_list, svd_from_corresponded_corners
from opencood.utils.path_utils import ensure_v2xreg_root_on_path
from opencood.registration.utils.types import ExtrinsicEstimate, ExtrinsicInit, MethodContext


class _CategoryEncoder:
    def __init__(self) -> None:
        self._mapping: Dict[str, int] = {}

    def encode(self, name: str) -> int:
        key = (name or "unknown").lower()
        if key not in self._mapping:
            self._mapping[key] = len(self._mapping)
        return self._mapping[key]


def _transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed = (T @ pts_h.T).T
    return transformed[:, :3]


def _transform_yaw(yaw: float, rot_mat: np.ndarray) -> float:
    dir_vec = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    rotated = rot_mat @ dir_vec
    return math.atan2(float(rotated[1]), float(rotated[0]))


class VIPSEstimator:
    """
    VIPS baseline (graph matching -> SVD).

    VIPS is initial-value-based because it relies on `world_position` computed from `T_init`.
    """

    def __init__(
        self,
        *,
        match_threshold: float = 0.5,
        match_distance_thr_m: float = 8.0,
        device: Optional[str] = None,
    ) -> None:
        self._match_threshold = float(match_threshold)
        self._match_distance_thr_m = float(match_distance_thr_m)
        self._device = torch.device(device) if device else None
        self._use_torch = bool(self._device and self._device.type == "cuda" and torch.cuda.is_available())

    @staticmethod
    def _node_similarity(n1: np.ndarray, n2: np.ndarray) -> float:
        lambda_1 = 0.5
        lambda_2 = 0.1
        f_1 = float(n1[0] == n2[0])
        # size affinity (use l/w only to match common implementations)
        f_2 = math.exp(-lambda_1 * float(np.linalg.norm(n1[4:6] - n2[4:6]) ** 2))
        f_3 = 1.0  # trajectory affinity not available
        # world position affinity (use x/y)
        f_4 = math.exp(-lambda_2 * float(np.linalg.norm(n1[7:9] - n2[7:9])))
        miu_1 = 0.5
        miu_2 = 0.5
        return f_1 * f_3 * (miu_1 * f_2 + miu_2 * f_4)

    @staticmethod
    def _edge_similarity(e1n1: np.ndarray, e1n2: np.ndarray, e2n1: np.ndarray, e2n2: np.ndarray) -> float:
        lambda_3 = 0.5
        lambda_4 = 0.1
        g_1 = float(
            ((e1n1[0] == e2n1[0]) and (e1n2[0] == e2n2[0]))
            or ((e1n2[0] == e2n1[0]) and (e1n1[0] == e2n2[0]))
        )
        # position uses x/y
        d1 = float(np.linalg.norm(e1n1[1:3] - e1n2[1:3]))
        d2 = float(np.linalg.norm(e2n1[1:3] - e2n2[1:3]))
        g_2 = math.exp(-lambda_3 * (d1 - d2) ** 2)
        g_3 = math.exp(
            -lambda_4
            * abs(
                math.sin(float(e1n1[10] - e1n2[10])) - math.sin(float(e2n1[10] - e2n2[10]))
            )
        )
        miu_3 = 0.5
        miu_4 = 0.5
        return g_1 * (miu_3 * g_2 + miu_4 * g_3)

    def _run_graph_matching(self, veh_graph: Dict[str, List], infra_graph: Dict[str, List]) -> np.ndarray:
        L1 = len(veh_graph["category"])
        L2 = len(infra_graph["category"])
        if L1 == 0 or L2 == 0:
            return np.zeros((0, 2), dtype=np.int32)
        if self._use_torch:
            try:
                return self._run_graph_matching_torch(veh_graph, infra_graph)
            except Exception:
                pass
        G1 = np.zeros((L1, 11), dtype=np.float64)
        G2 = np.zeros((L2, 11), dtype=np.float64)
        for i in range(L1):
            G1[i, 0] = float(veh_graph["category"][i])
            G1[i, 1:4] = np.asarray(veh_graph["position"][i], dtype=np.float64)
            G1[i, 4:7] = np.asarray(veh_graph["bounding_box"][i], dtype=np.float64)
            G1[i, 7:10] = np.asarray(veh_graph["world_position"][i], dtype=np.float64)
            G1[i, 10] = float(veh_graph["heading"][i][0])
        for i in range(L2):
            G2[i, 0] = float(infra_graph["category"][i])
            G2[i, 1:4] = np.asarray(infra_graph["position"][i], dtype=np.float64)
            G2[i, 4:7] = np.asarray(infra_graph["bounding_box"][i], dtype=np.float64)
            G2[i, 7:10] = np.asarray(infra_graph["world_position"][i], dtype=np.float64)
            G2[i, 10] = float(infra_graph["heading"][i][0])

        dim = L1 * L2
        # Build the affinity matrix with vectorized numpy operations.
        # Axis convention:
        #   - i, j index nodes in the vehicle graph (L1)
        #   - ip, jp index nodes in the infra graph (L2)
        #   - row corresponds to (i, ip), col corresponds to (j, jp)
        cats1 = G1[:, 0].astype(np.int64, copy=False)
        cats2 = G2[:, 0].astype(np.int64, copy=False)

        # Node similarity (L1 x L2)
        lambda_1 = 0.5
        lambda_2 = 0.1
        miu_1 = 0.5
        miu_2 = 0.5
        cat_eq = cats1[:, None] == cats2[None, :]
        dim_xy_1 = G1[:, 4:6]
        dim_xy_2 = G2[:, 4:6]
        dim_diff = dim_xy_1[:, None, :] - dim_xy_2[None, :, :]
        f_2 = np.exp(-lambda_1 * np.sum(dim_diff * dim_diff, axis=2))
        wp_xy_1 = G1[:, 7:9]
        wp_xy_2 = G2[:, 7:9]
        wp_dist = np.linalg.norm(wp_xy_1[:, None, :] - wp_xy_2[None, :, :], axis=2)
        f_4 = np.exp(-lambda_2 * wp_dist)
        node_sim = cat_eq.astype(np.float64) * (miu_1 * f_2 + miu_2 * f_4)

        # Edge similarity (L1 x L1 x L2 x L2) in (i, j, ip, jp) order.
        lambda_3 = 0.5
        lambda_4 = 0.1
        miu_3 = 0.5
        miu_4 = 0.5

        pos_xy_1 = G1[:, 1:3]
        pos_xy_2 = G2[:, 1:3]
        d1 = np.linalg.norm(pos_xy_1[:, None, :] - pos_xy_1[None, :, :], axis=2)
        d2 = np.linalg.norm(pos_xy_2[:, None, :] - pos_xy_2[None, :, :], axis=2)
        g_2 = np.exp(-lambda_3 * (d1[:, :, None, None] - d2[None, None, :, :]) ** 2)

        h1 = G1[:, 10]
        h2 = G2[:, 10]
        sin1 = np.sin(h1[:, None] - h1[None, :])
        sin2 = np.sin(h2[:, None] - h2[None, :])
        g_3 = np.exp(-lambda_4 * np.abs(sin1[:, :, None, None] - sin2[None, None, :, :]))

        # Category compatibility for edge matching, with (i, j, ip, jp) broadcasting.
        cond1 = cat_eq[:, None, :, None] & cat_eq[None, :, None, :]
        cond2 = cat_eq[:, None, None, :] & cat_eq[None, :, :, None]
        g_1 = (cond1 | cond2).astype(np.float64)

        edge_sim = g_1 * (miu_3 * g_2 + miu_4 * g_3)
        M = edge_sim.transpose(0, 2, 1, 3).reshape(dim, dim)
        np.fill_diagonal(M, node_sim.reshape(dim))

        if dim > 256:
            _, eigvecs = eigsh(M, k=1, which="LA")
            w = eigvecs[:, 0]
        else:
            _, eigvecs = np.linalg.eigh(M)
            w = eigvecs[:, -1]
        w = np.asarray(w, dtype=np.float64)
        if np.max(w) > np.min(w):
            w = (w - np.min(w)) / (np.max(w) - np.min(w))

        G = w.reshape(L1, L2)
        rows, cols = linear_sum_assignment(-G)
        kept = [(int(r), int(c)) for r, c in zip(rows, cols) if float(G[r, c]) >= self._match_threshold]
        if not kept:
            return np.zeros((0, 2), dtype=np.int32)
        return np.asarray(kept, dtype=np.int32)

    def _run_graph_matching_torch(self, veh_graph: Dict[str, List], infra_graph: Dict[str, List]) -> np.ndarray:
        L1 = len(veh_graph["category"])
        L2 = len(infra_graph["category"])
        if L1 == 0 or L2 == 0:
            return np.zeros((0, 2), dtype=np.int32)
        device = self._device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        G1 = torch.zeros((L1, 11), device=device, dtype=torch.float64)
        G2 = torch.zeros((L2, 11), device=device, dtype=torch.float64)
        for i in range(L1):
            G1[i, 0] = float(veh_graph["category"][i])
            G1[i, 1:4] = torch.as_tensor(veh_graph["position"][i], dtype=torch.float64, device=device)
            G1[i, 4:7] = torch.as_tensor(veh_graph["bounding_box"][i], dtype=torch.float64, device=device)
            G1[i, 7:10] = torch.as_tensor(veh_graph["world_position"][i], dtype=torch.float64, device=device)
            G1[i, 10] = float(veh_graph["heading"][i][0])
        for i in range(L2):
            G2[i, 0] = float(infra_graph["category"][i])
            G2[i, 1:4] = torch.as_tensor(infra_graph["position"][i], dtype=torch.float64, device=device)
            G2[i, 4:7] = torch.as_tensor(infra_graph["bounding_box"][i], dtype=torch.float64, device=device)
            G2[i, 7:10] = torch.as_tensor(infra_graph["world_position"][i], dtype=torch.float64, device=device)
            G2[i, 10] = float(infra_graph["heading"][i][0])

        dim = L1 * L2
        cats1 = G1[:, 0].to(dtype=torch.long)
        cats2 = G2[:, 0].to(dtype=torch.long)

        lambda_1 = 0.5
        lambda_2 = 0.1
        miu_1 = 0.5
        miu_2 = 0.5
        cat_eq = cats1[:, None] == cats2[None, :]
        dim_xy_1 = G1[:, 4:6]
        dim_xy_2 = G2[:, 4:6]
        dim_diff = dim_xy_1[:, None, :] - dim_xy_2[None, :, :]
        f_2 = torch.exp(-lambda_1 * torch.sum(dim_diff * dim_diff, dim=2))
        wp_xy_1 = G1[:, 7:9]
        wp_xy_2 = G2[:, 7:9]
        wp_dist = torch.linalg.norm(wp_xy_1[:, None, :] - wp_xy_2[None, :, :], dim=2)
        f_4 = torch.exp(-lambda_2 * wp_dist)
        node_sim = cat_eq.to(dtype=torch.float64) * (miu_1 * f_2 + miu_2 * f_4)

        lambda_3 = 0.5
        lambda_4 = 0.1
        miu_3 = 0.5
        miu_4 = 0.5
        pos_xy_1 = G1[:, 1:3]
        pos_xy_2 = G2[:, 1:3]
        d1 = torch.cdist(pos_xy_1, pos_xy_1, p=2)
        d2 = torch.cdist(pos_xy_2, pos_xy_2, p=2)
        g_2 = torch.exp(-lambda_3 * (d1[:, :, None, None] - d2[None, None, :, :]) ** 2)

        h1 = G1[:, 10]
        h2 = G2[:, 10]
        sin1 = torch.sin(h1[:, None] - h1[None, :])
        sin2 = torch.sin(h2[:, None] - h2[None, :])
        g_3 = torch.exp(-lambda_4 * torch.abs(sin1[:, :, None, None] - sin2[None, None, :, :]))

        cond1 = cat_eq[:, None, :, None] & cat_eq[None, :, None, :]
        cond2 = cat_eq[:, None, None, :] & cat_eq[None, :, :, None]
        g_1 = (cond1 | cond2).to(dtype=torch.float64)

        edge_sim = g_1 * (miu_3 * g_2 + miu_4 * g_3)
        M = edge_sim.permute(0, 2, 1, 3).reshape(dim, dim)
        idx = torch.arange(dim, device=device)
        M[idx, idx] = node_sim.reshape(dim)

        try:
            _, eigvecs = torch.linalg.eigh(M)
            w = eigvecs[:, -1]
        except Exception:
            M_cpu = M.detach().cpu().numpy()
            _, eigvecs = np.linalg.eigh(M_cpu)
            w = torch.as_tensor(eigvecs[:, -1], device=device, dtype=torch.float64)

        w_min = torch.min(w)
        w_max = torch.max(w)
        if float(w_max - w_min) > 0:
            w = (w - w_min) / (w_max - w_min)
        G = w.reshape(L1, L2).detach().cpu().numpy()

        rows, cols = linear_sum_assignment(-G)
        kept = [(int(r), int(c)) for r, c in zip(rows, cols) if float(G[r, c]) >= self._match_threshold]
        if not kept:
            return np.zeros((0, 2), dtype=np.int32)
        return np.asarray(kept, dtype=np.int32)

    def estimate_from_corners(
        self,
        infra_corners: np.ndarray,
        veh_corners: np.ndarray,
        *,
        init: Optional[ExtrinsicInit] = None,
        ctx: Optional[MethodContext] = None,
        bbox_type: str = "detected",
        infra_scores=None,
        veh_scores=None,
    ) -> ExtrinsicEstimate:
        infra_boxes = corners_to_bbox3d_list(infra_corners, bbox_type=bbox_type, scores=infra_scores)
        veh_boxes = corners_to_bbox3d_list(veh_corners, bbox_type=bbox_type, scores=veh_scores)
        return self.estimate(infra_boxes, veh_boxes, init=init, ctx=ctx)

    def _build_graph(self, bboxes, encoder: _CategoryEncoder, world_T: np.ndarray) -> Dict[str, List]:
        categories: List[int] = []
        positions: List[List[float]] = []
        bbox_dims: List[List[float]] = []
        world_positions: List[List[float]] = []
        headings: List[List[float]] = []
        rot = world_T[:3, :3]
        for bbox in bboxes:
            center, (l, w, h), yaw, _ = bbox3d_to_state7(bbox)
            categories.append(encoder.encode(getattr(bbox, "get_bbox_type", lambda: "unknown")()))
            positions.append(center.astype(float).tolist())
            bbox_dims.append([float(l), float(w), float(h)])
            transformed_center = _transform_points(world_T, center.reshape(1, 3))[0]
            world_positions.append(transformed_center.tolist())
            world_yaw = _transform_yaw(float(yaw), rot)
            headings.append([float(world_yaw)])
        return {
            "category": categories,
            "position": positions,
            "bounding_box": bbox_dims,
            "world_position": world_positions,
            "heading": headings,
        }

    def estimate(
        self,
        infra_boxes,
        veh_boxes,
        *,
        init: Optional[ExtrinsicInit] = None,
        ctx: Optional[MethodContext] = None,
    ) -> ExtrinsicEstimate:
        ctx = ctx or MethodContext()
        start = perf_counter()

        if not infra_boxes or not veh_boxes:
            return ExtrinsicEstimate(T=None, success=False, method="vips", extra={"reason": "empty_boxes"})

        T_init = init.T_init if init is not None else np.eye(4, dtype=np.float64)
        encoder = _CategoryEncoder()
        veh_graph = self._build_graph(veh_boxes, encoder, np.eye(4, dtype=np.float64))
        infra_graph = self._build_graph(infra_boxes, encoder, T_init)

        try:
            matches = self._run_graph_matching(veh_graph, infra_graph)  # [[veh_idx, infra_idx], ...]
        except Exception as exc:
            return ExtrinsicEstimate(
                T=None,
                success=False,
                method="vips",
                time_sec=float(perf_counter() - start),
                extra={"reason": "vips_exception", "error": str(exc), "prior_used": bool(init is not None)},
            )

        if matches.size == 0:
            return ExtrinsicEstimate(
                T=None,
                success=False,
                method="vips",
                time_sec=float(perf_counter() - start),
                extra={"reason": "no_matches", "prior_used": bool(init is not None)},
            )

        if self._match_distance_thr_m > 0:
            matches = self._filter_matches_by_distance(matches, infra_boxes, veh_boxes, T_init, self._match_distance_thr_m)
            if matches.size == 0:
                return ExtrinsicEstimate(
                    T=None,
                    success=False,
                    method="vips",
                    time_sec=float(perf_counter() - start),
                    extra={
                        "reason": "no_matches_after_distance_filter",
                        "prior_used": bool(init is not None),
                        "distance_thr_m": float(self._match_distance_thr_m),
                    },
                )

        src_corners_list = []
        dst_corners_list = []
        for veh_idx, infra_idx in matches:
            _, _, _, infra_corners = bbox3d_to_state7(infra_boxes[int(infra_idx)])
            _, _, _, veh_corners = bbox3d_to_state7(veh_boxes[int(veh_idx)])
            src_corners_list.append(infra_corners)
            dst_corners_list.append(veh_corners)

        T = svd_from_corresponded_corners(src_corners_list, dst_corners_list)
        time_sec = perf_counter() - start
        if T is None:
            return ExtrinsicEstimate(
                T=None,
                success=False,
                method="vips",
                matches=[{"veh_idx": int(i), "infra_idx": int(j)} for i, j in matches.tolist()],
                time_sec=float(time_sec),
                extra={"reason": "svd_failed", "prior_used": bool(init is not None), "num_matches": int(len(matches))},
            )

        RE = TE = None
        if ctx.T_true is not None:
            ensure_v2xreg_root_on_path()
            from opencood.registration.utils.geometry import convert_T_to_6DOF, get_RE_TE_by_compare_T_6DOF_result_true  # type: ignore

            RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(convert_T_to_6DOF(T), convert_T_to_6DOF(ctx.T_true))

        return ExtrinsicEstimate(
            T=T,
            success=True,
            method="vips",
            matches=[{"veh_idx": int(i), "infra_idx": int(j)} for i, j in matches.tolist()],
            RE=None if RE is None else float(RE),
            TE=None if TE is None else float(TE),
            time_sec=float(time_sec),
            extra={"prior_used": bool(init is not None), "num_matches": int(len(matches))},
        )

    @staticmethod
    def _filter_matches_by_distance(
        matches: np.ndarray,
        infra_boxes,
        veh_boxes,
        T_init: np.ndarray,
        thr_m: float,
    ) -> np.ndarray:
        if matches.size == 0:
            return matches
        filtered: List[List[int]] = []
        thr_m = float(thr_m)
        transformed_cache: Dict[int, np.ndarray] = {}
        for veh_idx, infra_idx in matches:
            veh_center, _, _, _ = bbox3d_to_state7(veh_boxes[int(veh_idx)])
            infra_idx_int = int(infra_idx)
            if infra_idx_int not in transformed_cache:
                infra_center, _, _, _ = bbox3d_to_state7(infra_boxes[infra_idx_int])
                transformed_center = _transform_points(T_init, infra_center.reshape(1, 3))[0]
                transformed_cache[infra_idx_int] = transformed_center
            else:
                transformed_center = transformed_cache[infra_idx_int]
            if float(np.linalg.norm(transformed_center - veh_center)) <= thr_m:
                filtered.append([int(veh_idx), infra_idx_int])
        if not filtered:
            return np.zeros((0, 2), dtype=np.int32)
        return np.asarray(filtered, dtype=np.int32)


__all__ = ["VIPSEstimator"]
