from __future__ import annotations

from time import perf_counter
from typing import Optional

import numpy as np

from opencood.extrinsics.bbox_utils import corners_to_bbox3d_list
from opencood.extrinsics.path_utils import ensure_v2xreg_root_on_path, resolve_repo_path
from opencood.extrinsics.types import ExtrinsicEstimate, ExtrinsicInit, MethodContext


class V2XRegPPEstimator:
    """
    Wrapper around the parent-repo V2X-Reg++ pipeline pieces (FilterPipeline + MatchingEngine + SVD solver).

    Notes:
      - This estimator is *initial-value free* by default.
      - If `init` is provided and the matching config uses `filter_strategy=trueRetained`,
        the prior can be used to gate correspondences.
    """

    def __init__(
        self,
        *,
        config_path: str,
        matching_overrides: Optional[dict] = None,
        device: Optional[str] = None,
    ) -> None:
        ensure_v2xreg_root_on_path()
        from calib.config import load_config  # type: ignore

        cfg_path = resolve_repo_path(config_path)
        self._cfg = load_config(str(cfg_path))
        if matching_overrides:
            for key, value in matching_overrides.items():
                setattr(self._cfg.matching, key, value)

        from calib.filters.pipeline import FilterPipeline  # type: ignore
        from calib.matching.engine import MatchingEngine  # type: ignore

        self._filters = FilterPipeline(self._cfg.filters)
        self._device = device
        self._matcher = MatchingEngine(self._cfg.matching, device=device)

    @property
    def config(self):
        return self._cfg

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

    def estimate(
        self,
        infra_boxes,
        veh_boxes,
        *,
        init: Optional[ExtrinsicInit] = None,
        ctx: Optional[MethodContext] = None,
    ) -> ExtrinsicEstimate:
        ensure_v2xreg_root_on_path()
        from v2x_calib.search import Matches2Extrinsics  # type: ignore
        from v2x_calib.utils import (  # type: ignore
            convert_6DOF_to_T,
            convert_T_to_6DOF,
            get_RE_TE_by_compare_T_6DOF_result_true,
        )

        ctx = ctx or MethodContext()
        start = perf_counter()

        filtered_infra, filtered_vehicle = self._filters.apply(infra_boxes or [], veh_boxes or [])

        T_hint = init.T_init if init is not None else None
        sensor_frame = str(getattr(getattr(self._cfg, "data", None), "sensor_frame", "lidar")).lower().strip()
        sensor_combo = "camera-camera" if sensor_frame == "camera" else "lidar-lidar"
        allow_gt_fallback = bool((ctx.meta or {}).get("allow_gt_fallback", False))
        T_eval = ctx.T_true if allow_gt_fallback else None
        matches_with_score, stability = self._matcher.compute(
            filtered_infra,
            filtered_vehicle,
            T_hint=T_hint,
            T_eval=T_eval,
            sensor_combo=sensor_combo,
        )

        T = None
        RE = TE = None
        success = False
        if matches_with_score:
            solver = Matches2Extrinsics(
                filtered_infra,
                filtered_vehicle,
                matches_score_list=matches_with_score,
                svd_strategy=self._cfg.matching.svd_strategy,
                resolve_180_ambiguity=getattr(self._cfg.matching, "resolve_180_ambiguity", False),
                max_iterations=getattr(self._cfg.solver, "max_iterations", 1),
                inlier_threshold_m=getattr(self._cfg.solver, "inlier_threshold_m", 0.0),
                mad_scale=getattr(self._cfg.solver, "mad_scale", 2.5),
                min_inliers=getattr(self._cfg.solver, "min_inliers", 1),
                device=self._device,
            )
            T6 = solver.get_combined_extrinsic(matches2extrinsic_strategies=self._cfg.matching.matches2extrinsic)
            T = convert_6DOF_to_T(T6)
            success = True
            if ctx.T_true is not None:
                RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(convert_T_to_6DOF(T), convert_T_to_6DOF(ctx.T_true))

        time_sec = perf_counter() - start
        return ExtrinsicEstimate(
            T=T,
            success=success,
            method="v2xregpp",
            stability=float(stability or 0.0),
            matches=[
                {"infra_idx": int(m[0][0]), "veh_idx": int(m[0][1]), "score": float(m[1])}
                for m in matches_with_score
            ],
            RE=None if RE is None else float(RE),
            TE=None if TE is None else float(TE),
            time_sec=float(time_sec),
            extra={
                "filtered_infra": int(len(filtered_infra)),
                "filtered_vehicle": int(len(filtered_vehicle)),
                "raw_infra": int(len(infra_boxes or [])),
                "raw_vehicle": int(len(veh_boxes or [])),
                "prior_used": bool(init is not None),
            },
        )


__all__ = ["V2XRegPPEstimator"]
