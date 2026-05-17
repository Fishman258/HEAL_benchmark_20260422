from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _centers_xy_from_boxes(boxes: List[Any]) -> np.ndarray:
    centers: List[np.ndarray] = []
    for box in boxes or []:
        corners = np.asarray(box.get_bbox3d_8_3(), dtype=np.float64).reshape(-1, 3)
        if corners.size == 0:
            continue
        centers.append(corners.mean(axis=0)[:2])
    if not centers:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack(centers, axis=0)


def _topk_by_confidence(boxes: List[Any], k: int) -> List[Any]:
    if k <= 0 or len(boxes) <= k:
        return list(boxes)
    scored = []
    for idx, box in enumerate(boxes):
        conf = None
        if hasattr(box, "get_confidence"):
            try:
                conf = float(box.get_confidence())
            except Exception:
                conf = None
        if conf is None:
            conf = float(getattr(box, "confidence", 0.0) or 0.0)
        scored.append((conf, idx))
    scored.sort(key=lambda x: x[0], reverse=True)
    keep_idx = [idx for _, idx in scored[:k]]
    return [boxes[i] for i in keep_idx]


def _pairwise_dist(centers: np.ndarray) -> np.ndarray:
    if centers.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    diff = centers[:, None, :] - centers[None, :, :]
    return np.linalg.norm(diff, axis=2).astype(np.float32)


def _judge_box(graph1: np.ndarray, graph2: np.ndarray, anchors: List[Tuple[int, int]], cand: Tuple[int, int], max_error: float) -> bool:
    u, v = int(cand[0]), int(cand[1])
    for a_u, a_v in anchors:
        if float(np.abs(graph1[int(a_u), u] - graph2[int(a_v), v])) < float(max_error):
            continue
        return False
    return True


def _find_anchors(distance_raw: np.ndarray, *, max_error: float, graph1: np.ndarray, graph2: np.ndarray) -> List[Tuple[int, int]]:
    """
    Ported from `freealign/match/match_v7_with_detection.py`.
    """
    distance = distance_raw.copy()
    initial_index = int(np.argmin(distance))
    initial_2d = np.unravel_index(initial_index, distance.shape)
    best_anchors: List[Tuple[int, int]] = [(int(initial_2d[0]), int(initial_2d[1]))]
    distance[initial_2d] = 999.0
    distance_bak = distance.copy()

    candidates = np.where(distance < float(max_error))
    candidates_list = [(int(candidates[0][i]), int(candidates[1][i])) for i in range(len(candidates[0]))]

    for candidate in candidates_list:
        distance = distance_bak.copy()
        _ = float(distance[candidate])
        distance[int(candidate[0])] = 999.0
        distance[:, int(candidate[1])] = 999.0
        anchors: List[Tuple[int, int]] = [best_anchors[0], candidate]
        for cand in candidates_list:
            if cand == candidate:
                continue
            if float(distance[cand]) < float(max_error) and _judge_box(graph1, graph2, anchors, cand, float(max_error) * 2.0):
                anchors.append(cand)
                distance[int(cand[0])] = 999.0
                distance[:, int(cand[1])] = 999.0
        if len(anchors) > len(best_anchors):
            best_anchors = anchors

    return best_anchors


def _get_best_match(
    node_i_ego: np.ndarray,
    node_j_edge: np.ndarray,
    *,
    min_anchors: int,
    anchor_error: float,
    box_error: float,
    graph1: np.ndarray,
    graph2: np.ndarray,
) -> Tuple[List[Tuple[int, int]], float]:
    distance = np.abs(node_i_ego[:, np.newaxis] - node_j_edge).sum(axis=2)
    error = 100.0
    match = _find_anchors(distance, max_error=float(anchor_error), graph1=graph1, graph2=graph2)
    anchors = list(match)
    if len(match) < int(min_anchors):
        return match, error

    for pair in match:
        distance[int(pair[0])] = 999.0
        distance[:, int(pair[1])] = 999.0

    for _ in range(min(int(node_i_ego.shape[0]), int(node_j_edge.shape[0])) - len(match)):
        error_item = float(np.min(distance))
        if error_item >= float(box_error):
            break
        min_index = int(np.argmin(distance))
        min_2d = np.unravel_index(min_index, distance.shape)
        cand = (int(min_2d[0]), int(min_2d[1]))
        if _judge_box(graph1, graph2, anchors, cand, float(box_error)):
            match.append(cand)
            error += error_item
            distance[int(cand[0])] = 999.0
            distance[:, int(cand[1])] = 999.0
        else:
            distance[cand] = 999.0

    if len(match) == 0:
        return [], float("inf")
    return match, float(error / max(float(len(match) ** 2), 1e-9))


def _find_common_subgraph(
    graph1: np.ndarray,
    graph2: np.ndarray,
    *,
    min_anchors: int,
    anchor_error: float,
    box_error: float,
) -> Tuple[int, List[Tuple[int, int]], float]:
    """
    Ported from `freealign/match/match_v7_with_detection.py`.

    Returns:
      mcs_size, matched_pairs, best_error
    """
    best_error = 100.0
    best_match: List[Tuple[int, int]] = []
    for i in range(int(graph1.shape[0])):
        node_i_ego = graph1[i]  # (n,1)
        best_matching_ever: List[Tuple[int, int]] = []
        best_avg_err = 100.0
        for j in range(int(graph2.shape[0])):
            node_j_edge = graph2[j]  # (m,1)
            matching, avg_error = _get_best_match(
                node_i_ego,
                node_j_edge,
                min_anchors=int(min_anchors),
                anchor_error=float(anchor_error),
                box_error=float(box_error),
                graph1=graph1,
                graph2=graph2,
            )
            if len(matching) >= int(min_anchors) and float(avg_error) <= float(best_avg_err):
                best_matching_ever = matching
                best_avg_err = float(avg_error)
        if len(best_matching_ever) > int(min_anchors) and float(best_avg_err) <= float(best_error):
            best_match = best_matching_ever
            best_error = float(best_avg_err)
    return int(len(best_match)), best_match, float(best_error)


def _estimate_T_from_matches(
    ego_centers: np.ndarray,
    cav_centers: np.ndarray,
    matches: List[Tuple[int, int]],
) -> Optional[np.ndarray]:
    import cv2

    if len(matches) < 3:
        return None
    dst = np.stack([ego_centers[i] for i, _ in matches], axis=0).astype(np.float32)
    src = np.stack([cav_centers[j] for _, j in matches], axis=0).astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, dst)  # use OpenCV default (as in upstream optimize_rt)
    if M is None:
        return None
    T = np.eye(4, dtype=np.float64)
    T[0:2, 0:2] = np.asarray(M[:, :2], dtype=np.float64)
    T[0:2, 3] = np.asarray(M[:, 2], dtype=np.float64)
    return T


@dataclass(frozen=True)
class FreeAlignRepoConfig:
    """
    Reproduction of the released FreeAlign repo matching (match_v7_with_detection).

    Notes:
      - This is NOT the paper's EdgeGAT+MASS implementation; it is the repo's
        distance-graph matching variant.
    """

    max_boxes: int = 60
    min_anchors: int = 3
    anchor_error: float = 0.3
    box_error: float = 0.5
    min_nodes: int = 3

    # Optional temporal smoothing (mirrors v2xregpp_stable "delta SE(2)" smoothing).
    mode: str = "initfree"  # initfree | stable
    ema_alpha: float = 0.5
    max_step_xy_m: float = 3.0
    max_step_yaw_deg: float = 10.0

    # Optional: compare against current pose and keep the better alignment.
    compare_with_current: bool = False
    compare_distance_threshold_m: float = 3.0
    min_precision: float = 0.0
    apply_if_current_precision_below: float = -1.0
    min_precision_improvement: float = 0.0
    min_matched_improvement: int = 0


class FreeAlignRepoEstimator:
    def __init__(self, cfg: Optional[FreeAlignRepoConfig] = None) -> None:
        self.cfg = cfg or FreeAlignRepoConfig()

    def estimate(self, *, cav_boxes, ego_boxes) -> Tuple[Optional[np.ndarray], float, int, Dict[str, Any]]:
        cav_boxes = _topk_by_confidence(list(cav_boxes or []), int(self.cfg.max_boxes))
        ego_boxes = _topk_by_confidence(list(ego_boxes or []), int(self.cfg.max_boxes))
        if len(cav_boxes) < int(self.cfg.min_nodes) or len(ego_boxes) < int(self.cfg.min_nodes):
            return None, 0.0, 0, {"reason": "insufficient_boxes"}

        cav_centers = _centers_xy_from_boxes(cav_boxes)
        ego_centers = _centers_xy_from_boxes(ego_boxes)
        if cav_centers.shape[0] < int(self.cfg.min_nodes) or ego_centers.shape[0] < int(self.cfg.min_nodes):
            return None, 0.0, 0, {"reason": "empty_centers"}

        g1 = _pairwise_dist(ego_centers)[..., None]  # (n,n,1)
        g2 = _pairwise_dist(cav_centers)[..., None]  # (m,m,1)

        mcs_size, matched, best_err = _find_common_subgraph(
            g1,
            g2,
            min_anchors=int(self.cfg.min_anchors),
            anchor_error=float(self.cfg.anchor_error),
            box_error=float(self.cfg.box_error),
        )
        if int(mcs_size) < int(self.cfg.min_nodes):
            return None, 0.0, int(mcs_size), {"reason": "no_common_subgraph", "best_err": float(best_err)}

        T = _estimate_T_from_matches(ego_centers, cav_centers, matched)
        if T is None:
            return None, 0.0, int(mcs_size), {"reason": "estimate_failed", "best_err": float(best_err)}

        meta = {
            "mcs_size": int(mcs_size),
            "best_err": float(best_err),
            "min_anchors": int(self.cfg.min_anchors),
            "anchor_error": float(self.cfg.anchor_error),
            "box_error": float(self.cfg.box_error),
        }
        stability = float(mcs_size)
        return np.asarray(T, dtype=np.float64), stability, int(mcs_size), meta


__all__ = ["FreeAlignRepoConfig", "FreeAlignRepoEstimator"]
