from __future__ import annotations

import math
from time import perf_counter
from typing import Optional

import numpy as np

from opencood.registration.utils.bbox import (
    bbox3d_to_state7,
    bbox_list_to_array7,
    corners_to_bbox3d_list,
    svd_from_corresponded_corners,
)
from opencood.utils.path_utils import ensure_v2xreg_root_on_path
from opencood.registration.utils.types import ExtrinsicEstimate, ExtrinsicInit, MethodContext


class CBMEstimator:
    """
    CBM baseline (box matching -> SVD).

    The original CBM is an initial-value-based method; pass `init` to enable it.
    """

    def __init__(
        self,
        *,
        sigma1_deg: float = 10.0,
        sigma2_m: float = 3.0,
        sigma3_m: float = 1.0,
        absolute_dis_lim_m: float = 20.0,
        device: Optional[str] = None,
    ) -> None:
        self._cbm_args = type(
            "_CBMArgs",
            (),
            {
                "sigma1": float(math.radians(sigma1_deg)),
                "sigma2": float(sigma2_m),
                "sigma3": float(sigma3_m),
                "absolute_dis_lim": float(absolute_dis_lim_m),
            },
        )()
        self._matcher = None
        self._device = device

    def _get_matcher(self):
        if self._matcher is not None:
            return self._matcher
        ensure_v2xreg_root_on_path()
        try:
            from opencood.registration.estimators.box_matching.cbm_torch import CBM as CBMMatcher  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "CBMEstimator requires the CBM torch matcher and PyTorch runtime dependencies."
            ) from exc
        self._matcher = CBMMatcher(args=self._cbm_args, device=self._device)
        return self._matcher

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
        ctx = ctx or MethodContext()
        start = perf_counter()

        if not infra_boxes or not veh_boxes:
            return ExtrinsicEstimate(T=None, success=False, method="cbm", extra={"reason": "empty_boxes"})

        T_init = init.T_init if init is not None else np.eye(4, dtype=np.float64)

        cav_array = bbox_list_to_array7(infra_boxes)
        ego_array = bbox_list_to_array7(veh_boxes)

        matcher = self._get_matcher()
        use_torch = False
        if self._device:
            try:
                import torch

                device = torch.device(str(self._device))
                use_torch = torch.cuda.is_available() and device.type == "cuda"
            except Exception:
                use_torch = False

        if use_torch:
            import torch

            ego_tensor = torch.as_tensor(ego_array, dtype=torch.float32, device=device)
            cav_tensor = torch.as_tensor(cav_array, dtype=torch.float32, device=device)
            T_init_tensor = torch.as_tensor(T_init, dtype=torch.float32, device=device)
            matching = matcher(ego_tensor, cav_tensor, T_init_tensor)
        else:
            matching = matcher(ego_array, cav_array, T_init)
        if hasattr(matching, "detach"):
            matching = matching.detach().cpu().numpy()
        matching = np.asarray(matching, dtype=np.int32)

        if matching.size == 0:
            return ExtrinsicEstimate(
                T=None,
                success=False,
                method="cbm",
                time_sec=float(perf_counter() - start),
                extra={"reason": "no_matches", "prior_used": bool(init is not None)},
            )

        src_corners_list = []
        dst_corners_list = []
        for ego_idx, cav_idx in matching:
            _, _, _, infra_corners = bbox3d_to_state7(infra_boxes[int(cav_idx)])
            _, _, _, veh_corners = bbox3d_to_state7(veh_boxes[int(ego_idx)])
            src_corners_list.append(infra_corners)
            dst_corners_list.append(veh_corners)

        T = svd_from_corresponded_corners(src_corners_list, dst_corners_list)
        time_sec = perf_counter() - start
        if T is None:
            return ExtrinsicEstimate(
                T=None,
                success=False,
                method="cbm",
                matches=[{"veh_idx": int(i), "infra_idx": int(j)} for i, j in matching.tolist()],
                time_sec=float(time_sec),
                extra={"reason": "svd_failed", "prior_used": bool(init is not None), "num_matches": int(len(matching))},
            )

        RE = TE = None
        if ctx.T_true is not None:
            ensure_v2xreg_root_on_path()
            from opencood.registration.utils.geometry import convert_T_to_6DOF, get_RE_TE_by_compare_T_6DOF_result_true  # type: ignore

            RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(convert_T_to_6DOF(T), convert_T_to_6DOF(ctx.T_true))

        return ExtrinsicEstimate(
            T=T,
            success=True,
            method="cbm",
            matches=[{"veh_idx": int(i), "infra_idx": int(j)} for i, j in matching.tolist()],
            RE=None if RE is None else float(RE),
            TE=None if TE is None else float(TE),
            time_sec=float(time_sec),
            extra={"prior_used": bool(init is not None), "num_matches": int(len(matching))},
        )


__all__ = ["CBMEstimator"]
