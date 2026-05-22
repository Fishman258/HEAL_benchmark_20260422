import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from opencood.registration.runtime.online_box_solver import (
    _agent_index,
    _extract_boxes,
    _extract_scores,
    _resolve_stage1_entry,
)


def _prepare_centers_and_scores(
    entry: Mapping[str, Any],
    agent_idx: int,
    field: str,
    topk: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    boxes = _extract_boxes(entry, int(agent_idx), str(field))
    if boxes is None:
        return None, None
    centers = torch.mean(boxes, dim=1)
    scores = _extract_scores(entry, int(agent_idx))

    if int(topk) > 0 and int(centers.shape[0]) > int(topk):
        if isinstance(scores, torch.Tensor) and int(scores.numel()) == int(centers.shape[0]):
            keep = torch.topk(scores, k=int(topk), largest=True).indices
        else:
            keep = torch.arange(int(topk), dtype=torch.long)
        centers = centers[keep]
        if isinstance(scores, torch.Tensor) and int(scores.numel()) >= int(keep.numel()):
            scores = scores[keep]

    return centers, scores


def _build_delta_transform(
    *,
    dx_m: float,
    dy_m: float,
    dyaw_rad: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    c = math.cos(float(dyaw_rad))
    s = math.sin(float(dyaw_rad))
    T = torch.eye(4, device=device, dtype=dtype)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 3] = float(dx_m)
    T[1, 3] = float(dy_m)
    return T


def _apply_transform_xy(points_xy: torch.Tensor, T_rel: torch.Tensor) -> torch.Tensor:
    rot = T_rel[0:2, 0:2]
    trans = T_rel[0:2, 3].view(1, 2)
    return torch.matmul(points_xy, rot.T) + trans


def _nearest_neighbor_residual(
    src_xy: torch.Tensor,
    dst_xy: torch.Tensor,
    weight: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    dist = torch.cdist(src_xy, dst_xy, p=2)
    nearest, nearest_idx = torch.min(dist, dim=1)
    if isinstance(weight, torch.Tensor) and int(weight.numel()) == int(nearest.numel()):
        w = torch.clamp(weight.view(-1), min=1e-6)
        residual = torch.sum(nearest * w) / torch.sum(w)
    else:
        residual = torch.mean(nearest)
    return residual, nearest_idx


def refine_relative_pose_from_stage1_entry(
    *,
    stage1_result: Mapping[str, Any],
    sample_idx: int,
    ego_cav_id: Any,
    cav_id: Any,
    T_init: torch.Tensor,
    field: str = "pred_corner3d_np_list",
    topk: int = 60,
    use_score_weight: bool = True,
    refine_steps: int = 4,
    init_step_xy_m: float = 0.4,
    init_step_yaw_deg: float = 1.5,
    decay: float = 0.6,
    min_step_xy_m: float = 0.03,
    min_step_yaw_deg: float = 0.15,
    min_improvement_m: float = 1e-4,
    device: Optional[torch.device] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(stage1_result, Mapping):
        return None
    if not isinstance(T_init, torch.Tensor) or tuple(T_init.shape) != (4, 4):
        return None

    entry = _resolve_stage1_entry(stage1_result, int(sample_idx))
    if entry is None:
        return None

    ego_idx = _agent_index(entry, ego_cav_id)
    cav_idx = _agent_index(entry, cav_id)
    if ego_idx is None or cav_idx is None:
        return None

    centers_dst, score_dst = _prepare_centers_and_scores(entry, int(ego_idx), str(field), int(topk))
    centers_src, score_src = _prepare_centers_and_scores(entry, int(cav_idx), str(field), int(topk))
    if centers_dst is None or centers_src is None:
        return None
    if int(centers_dst.shape[0]) <= 0 or int(centers_src.shape[0]) <= 0:
        return None

    if device is None:
        device = T_init.device if isinstance(T_init, torch.Tensor) else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    dtype = T_init.dtype if isinstance(T_init, torch.Tensor) else torch.float32
    centers_dst = centers_dst.to(device=device, dtype=dtype)
    centers_src = centers_src.to(device=device, dtype=dtype)
    if isinstance(score_src, torch.Tensor):
        score_src = score_src.to(device=device, dtype=dtype)
    if isinstance(score_dst, torch.Tensor):
        score_dst = score_dst.to(device=device, dtype=dtype)

    weight = None
    if use_score_weight and isinstance(score_src, torch.Tensor):
        weight = torch.sqrt(torch.clamp(score_src.view(-1), min=1e-6))

    T_cur = T_init.to(device=device, dtype=dtype)
    src_cur = _apply_transform_xy(centers_src, T_cur)
    init_residual, _ = _nearest_neighbor_residual(src_cur, centers_dst, weight)
    best_residual = init_residual
    best_T = T_cur
    improved = False

    step_xy = max(float(init_step_xy_m), 0.0)
    step_yaw_deg = max(float(init_step_yaw_deg), 0.0)
    decay = min(max(float(decay), 1e-3), 1.0)

    for _ in range(max(1, int(refine_steps))):
        dx_choices = [0.0] if step_xy <= 0.0 else [0.0, step_xy, -step_xy]
        dy_choices = [0.0] if step_xy <= 0.0 else [0.0, step_xy, -step_xy]
        dyaw_choices = [0.0] if step_yaw_deg <= 0.0 else [0.0, math.radians(step_yaw_deg), -math.radians(step_yaw_deg)]

        local_best_residual = best_residual
        local_best_T = best_T

        for dx_m in dx_choices:
            for dy_m in dy_choices:
                for dyaw_rad in dyaw_choices:
                    if dx_m == 0.0 and dy_m == 0.0 and dyaw_rad == 0.0:
                        continue
                    T_delta = _build_delta_transform(
                        dx_m=float(dx_m),
                        dy_m=float(dy_m),
                        dyaw_rad=float(dyaw_rad),
                        device=device,
                        dtype=dtype,
                    )
                    T_cand = torch.matmul(T_delta, best_T)
                    src_cand = _apply_transform_xy(centers_src, T_cand)
                    cand_residual, _ = _nearest_neighbor_residual(src_cand, centers_dst, weight)
                    if float(cand_residual) + 1e-9 < float(local_best_residual):
                        local_best_residual = cand_residual
                        local_best_T = T_cand

        if float(best_residual) - float(local_best_residual) >= float(min_improvement_m):
            best_residual = local_best_residual
            best_T = local_best_T
            improved = True

        step_xy = max(float(min_step_xy_m), float(step_xy) * float(decay)) if step_xy > 0.0 else 0.0
        step_yaw_deg = (
            max(float(min_step_yaw_deg), float(step_yaw_deg) * float(decay))
            if step_yaw_deg > 0.0
            else 0.0
        )

    return {
        "T_rel": best_T,
        "init_residual_m": float(init_residual.detach().cpu().item()),
        "mean_residual_m": float(best_residual.detach().cpu().item()),
        "residual_improvement_m": float((init_residual - best_residual).detach().cpu().item()),
        "refined": bool(improved),
        "field": str(field),
        "sample_idx": int(sample_idx),
        "num_src_boxes": int(centers_src.shape[0]),
        "num_dst_boxes": int(centers_dst.shape[0]),
    }
