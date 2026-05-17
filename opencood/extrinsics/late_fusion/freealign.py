from __future__ import annotations

from time import perf_counter
from typing import Optional

import numpy as np

from opencood.extrinsics.bbox_utils import corners_to_bbox3d_list
from opencood.extrinsics.types import ExtrinsicEstimate, ExtrinsicInit, MethodContext
from opencood.pose.freealign_paper import FreeAlignPaperConfig, FreeAlignPaperEstimator


def _yaw_deg_from_T(T: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(float(T[1, 0]), float(T[0, 0]))))


def _wrap_angle_deg(angle: float) -> float:
    return float(((angle + 180.0) % 360.0) - 180.0)


class FreeAlignEstimator:
    """
    FreeAlign: graph-matching based pose estimation from two sets of detected boxes.

    This wrapper is intentionally lightweight: it estimates an SE(2) transform
    between two agents (source -> target) without using any prior pose.
    """

    def __init__(self, *, cfg: Optional[FreeAlignPaperConfig] = None) -> None:
        self._cfg = cfg or FreeAlignPaperConfig()
        self._estimator = FreeAlignPaperEstimator(self._cfg)

    @property
    def config(self) -> FreeAlignPaperConfig:
        return self._cfg

    def estimate_from_corners(
        self,
        src_corners: np.ndarray,
        dst_corners: np.ndarray,
        *,
        init: Optional[ExtrinsicInit] = None,
        ctx: Optional[MethodContext] = None,
        bbox_type: str = "detected",
        src_scores=None,
        dst_scores=None,
    ) -> ExtrinsicEstimate:
        src_boxes = corners_to_bbox3d_list(src_corners, bbox_type=bbox_type, scores=src_scores)
        dst_boxes = corners_to_bbox3d_list(dst_corners, bbox_type=bbox_type, scores=dst_scores)
        return self.estimate(src_boxes, dst_boxes, init=init, ctx=ctx)

    def estimate(
        self,
        src_boxes,
        dst_boxes,
        *,
        init: Optional[ExtrinsicInit] = None,
        ctx: Optional[MethodContext] = None,
    ) -> ExtrinsicEstimate:
        ctx = ctx or MethodContext()
        start = perf_counter()
        T, stability, matches, meta = self._estimator.estimate(cav_boxes=src_boxes, ego_boxes=dst_boxes, T_init=init.T_init if init else None)
        time_sec = perf_counter() - start

        RE = TE = None
        if T is not None and ctx.T_true is not None:
            yaw_est = _yaw_deg_from_T(T)
            yaw_true = _yaw_deg_from_T(ctx.T_true)
            RE = abs(_wrap_angle_deg(yaw_est - yaw_true))
            t_est = np.asarray(T, dtype=np.float64)[0:2, 3]
            t_true = np.asarray(ctx.T_true, dtype=np.float64)[0:2, 3]
            TE = float(np.linalg.norm(t_est - t_true))

        return ExtrinsicEstimate(
            T=None if T is None else np.asarray(T, dtype=np.float64),
            success=bool(T is not None),
            method="freealign",
            stability=float(stability or 0.0),
            matches=[],
            RE=None if RE is None else float(RE),
            TE=None if TE is None else float(TE),
            time_sec=float(time_sec),
            extra={
                "matches": int(matches or 0),
                **(meta or {}),
                "prior_used": bool(init is not None),
                "raw_src": int(len(src_boxes or [])),
                "raw_dst": int(len(dst_boxes or [])),
            },
        )


__all__ = ["FreeAlignEstimator"]

