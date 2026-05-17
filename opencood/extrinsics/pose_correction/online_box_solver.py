from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch


def _resolve_stage1_entry(stage1_result: Mapping[str, Any], sample_idx: int) -> Optional[Mapping[str, Any]]:
    keys = [sample_idx, str(sample_idx)]
    if int(sample_idx) >= 0:
        keys.extend(["%06d" % int(sample_idx), "%04d" % int(sample_idx)])
    for key in keys:
        if key in stage1_result:
            value = stage1_result[key]
            if isinstance(value, Mapping):
                return value
    return None


def _agent_index(entry: Mapping[str, Any], cav_id: Any) -> Optional[int]:
    cav_ids = entry.get("cav_id_list")
    if not isinstance(cav_ids, Sequence):
        return None
    target = str(cav_id)
    for idx, raw in enumerate(cav_ids):
        if str(raw) == target:
            return int(idx)
    try:
        idx = int(cav_id)
        if 0 <= idx < len(cav_ids):
            return int(idx)
    except Exception:
        return None
    return None


def _extract_boxes(entry: Mapping[str, Any], agent_idx: int, field: str) -> Optional[torch.Tensor]:
    raw = entry.get(field)
    if not isinstance(raw, Sequence):
        return None
    if agent_idx < 0 or agent_idx >= len(raw):
        return None
    boxes = raw[agent_idx]
    if not isinstance(boxes, Sequence) or len(boxes) <= 0:
        return None
    try:
        tensor = torch.as_tensor(boxes, dtype=torch.float32)
    except Exception:
        return None
    if tensor.ndim != 3 or tensor.shape[1] < 4 or tensor.shape[2] < 2:
        return None
    return tensor[:, :, :2].contiguous()


def _extract_scores(entry: Mapping[str, Any], agent_idx: int) -> Optional[torch.Tensor]:
    raw = entry.get("pred_score_np_list")
    if not isinstance(raw, Sequence):
        return None
    if agent_idx < 0 or agent_idx >= len(raw):
        return None
    scores = raw[agent_idx]
    if not isinstance(scores, Sequence) or len(scores) <= 0:
        return None
    try:
        score_tensor = torch.as_tensor(scores, dtype=torch.float32).view(-1)
    except Exception:
        return None
    return score_tensor


def _weighted_kabsch_2d(src: torch.Tensor, dst: torch.Tensor, weight: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
    if src.ndim != 2 or dst.ndim != 2 or src.shape != dst.shape or src.shape[0] < 2 or src.shape[1] != 2:
        return None

    if weight is None:
        weight = torch.ones((src.shape[0],), device=src.device, dtype=src.dtype)
    else:
        weight = torch.clamp(weight.view(-1).to(device=src.device, dtype=src.dtype), min=1e-6)
        if weight.numel() != src.shape[0]:
            return None

    norm = torch.sum(weight)
    if float(norm) <= 0.0:
        return None
    weight = weight / norm

    src_mean = torch.sum(src * weight[:, None], dim=0)
    dst_mean = torch.sum(dst * weight[:, None], dim=0)
    src_centered = src - src_mean[None, :]
    dst_centered = dst - dst_mean[None, :]

    cov = (src_centered * weight[:, None]).T @ dst_centered
    try:
        u, _, vt = torch.linalg.svd(cov)
    except Exception:
        return None

    rot = vt.T @ u.T
    if float(torch.linalg.det(rot)) < 0.0:
        vt_fix = vt.clone()
        vt_fix[-1, :] = -vt_fix[-1, :]
        rot = vt_fix.T @ u.T

    trans = dst_mean - rot @ src_mean
    src_aligned = (rot @ src.T).T + trans[None, :]
    residual = torch.linalg.norm(src_aligned - dst, dim=1)

    T = torch.eye(4, device=src.device, dtype=src.dtype)
    T[0:2, 0:2] = rot
    T[0:2, 3] = trans

    return {
        "T": T,
        "mean_residual_m": float(torch.mean(residual).detach().cpu().item()),
        "num_matches": int(src.shape[0]),
    }


def solve_relative_pose_from_stage1_entry(
    *,
    stage1_result: Mapping[str, Any],
    sample_idx: int,
    ego_cav_id: Any,
    cav_id: Any,
    field: str = "pred_corner3d_np_list",
    min_matches: int = 3,
    max_match_distance_m: float = 6.0,
    topk: int = 60,
    use_score_weight: bool = True,
    device: Optional[torch.device] = None,
) -> Optional[Dict[str, Any]]:
    entry = _resolve_stage1_entry(stage1_result, int(sample_idx))
    if entry is None:
        return None

    ego_idx = _agent_index(entry, ego_cav_id)
    cav_idx = _agent_index(entry, cav_id)
    if ego_idx is None or cav_idx is None:
        return None

    boxes_dst = _extract_boxes(entry, ego_idx, field)
    boxes_src = _extract_boxes(entry, cav_idx, field)
    if boxes_dst is None or boxes_src is None:
        return None

    centers_dst = torch.mean(boxes_dst, dim=1)
    centers_src = torch.mean(boxes_src, dim=1)

    score_dst = _extract_scores(entry, ego_idx)
    score_src = _extract_scores(entry, cav_idx)

    if topk > 0:
        if centers_dst.shape[0] > topk:
            if score_dst is not None and score_dst.numel() == centers_dst.shape[0]:
                keep = torch.topk(score_dst, k=int(topk), largest=True).indices
            else:
                keep = torch.arange(int(topk), dtype=torch.long)
            centers_dst = centers_dst[keep]
            if score_dst is not None and score_dst.numel() >= keep.numel():
                score_dst = score_dst[keep]
        if centers_src.shape[0] > topk:
            if score_src is not None and score_src.numel() == centers_src.shape[0]:
                keep = torch.topk(score_src, k=int(topk), largest=True).indices
            else:
                keep = torch.arange(int(topk), dtype=torch.long)
            centers_src = centers_src[keep]
            if score_src is not None and score_src.numel() >= keep.numel():
                score_src = score_src[keep]

    if centers_src.shape[0] <= 0 or centers_dst.shape[0] <= 0:
        return None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    centers_src = centers_src.to(device=device)
    centers_dst = centers_dst.to(device=device)
    if score_src is not None:
        score_src = score_src.to(device=device)
    if score_dst is not None:
        score_dst = score_dst.to(device=device)

    dist = torch.cdist(centers_src, centers_dst, p=2)
    src_to_dst = torch.argmin(dist, dim=1)
    dst_to_src = torch.argmin(dist, dim=0)
    src_idx = torch.arange(centers_src.shape[0], device=device)
    mutual = dst_to_src[src_to_dst] == src_idx
    if max_match_distance_m > 0.0:
        d_best = dist[src_idx, src_to_dst]
        mutual = torch.logical_and(mutual, d_best <= float(max_match_distance_m))

    keep_src = torch.nonzero(mutual, as_tuple=False).view(-1)
    if int(keep_src.numel()) < int(min_matches):
        return None

    keep_dst = src_to_dst[keep_src]
    src_corr = centers_src[keep_src]
    dst_corr = centers_dst[keep_dst]

    weight = None
    if use_score_weight and score_src is not None and score_dst is not None:
        if score_src.numel() >= centers_src.shape[0] and score_dst.numel() >= centers_dst.shape[0]:
            weight = torch.sqrt(torch.clamp(score_src[keep_src], min=1e-6) * torch.clamp(score_dst[keep_dst], min=1e-6))

    est = _weighted_kabsch_2d(src_corr, dst_corr, weight)
    if est is None:
        return None

    return {
        "T_rel": est["T"],
        "num_matches": int(est["num_matches"]),
        "mean_residual_m": float(est["mean_residual_m"]),
        "field": str(field),
        "sample_idx": int(sample_idx),
    }
