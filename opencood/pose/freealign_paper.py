from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


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


def _pairwise_dist(centers: np.ndarray, *, device: Optional[torch.device] = None) -> np.ndarray:
    if centers.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        centers_t = torch.as_tensor(centers, dtype=torch.float32, device=device)
        diff = centers_t[:, None, :] - centers_t[None, :, :]
        D = torch.linalg.norm(diff, dim=2)
        return D.detach().cpu().numpy().astype(np.float32)
    diff = centers[:, None, :] - centers[None, :, :]
    return np.linalg.norm(diff, axis=2).astype(np.float32)


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


def _pick_anchor_indices(centers: np.ndarray, topk: int) -> np.ndarray:
    if centers.size == 0:
        return np.zeros((0,), dtype=np.int64)
    n = int(centers.shape[0])
    if int(topk) <= 0 or int(topk) >= n:
        return np.arange(n, dtype=np.int64)
    radius = np.linalg.norm(centers, axis=1)
    return np.argsort(-radius)[: int(topk)].astype(np.int64)


def _estimate_T_from_matches(
    ego_centers: np.ndarray,
    cav_centers: np.ndarray,
    matches: List[Tuple[int, int]],
    *,
    affine_method: str = "lmeds",
    ransac_reproj_threshold: float = 1.0,
    max_iters: int = 2000,
    confidence: float = 0.99,
    refine_iters: int = 10,
    refit_rigid: bool = True,
    min_inliers: int = 3,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    import cv2

    if len(matches) < 3:
        return None, {"reason": "insufficient_matches"}
    dst = np.stack([ego_centers[i] for i, _ in matches], axis=0).astype(np.float32)
    src = np.stack([cav_centers[j] for _, j in matches], axis=0).astype(np.float32)

    method_key = str(affine_method or "lmeds").lower().strip()
    if method_key in {"lmeds", "lmes", "lms", "lmed"}:
        method = cv2.LMEDS
    elif method_key in {"ransac"}:
        method = cv2.RANSAC
    else:
        raise ValueError(f"Unsupported affine_method: {affine_method!r}")

    M, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=method,
        ransacReprojThreshold=float(ransac_reproj_threshold),
        maxIters=int(max_iters),
        confidence=float(confidence),
        refineIters=int(refine_iters),
    )
    if M is None:
        return None, {"reason": "estimateAffinePartial2D_failed"}

    inlier_mask = None
    if inliers is not None:
        inlier_mask = np.asarray(inliers).reshape(-1) > 0
        if inlier_mask.shape[0] != src.shape[0]:
            inlier_mask = None
    inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if inlier_mask is None or inlier_count < int(min_inliers):
        src_fit = src.astype(np.float64)
        dst_fit = dst.astype(np.float64)
    else:
        src_fit = src[inlier_mask].astype(np.float64)
        dst_fit = dst[inlier_mask].astype(np.float64)

    if not bool(refit_rigid):
        T = np.eye(4, dtype=np.float64)
        T[0:2, 0:2] = np.asarray(M[:, :2], dtype=np.float64)
        T[0:2, 3] = np.asarray(M[:, 2], dtype=np.float64)
        meta = {
            "inliers": int(inliers.sum()) if inliers is not None else None,
            "inlier_count": int(inlier_count),
            "affine_method": str(method_key),
            "refit_rigid": False,
        }
        return T, meta

    # Refit a rigid SE(2) transform using inliers (if any).
    src_cent = src_fit.mean(axis=0)
    dst_cent = dst_fit.mean(axis=0)
    X = src_fit - src_cent
    Y = dst_fit - dst_cent
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = dst_cent - R @ src_cent
    T = np.eye(4, dtype=np.float64)
    T[0:2, 0:2] = R
    T[0:2, 3] = t
    meta = {
        "inliers": int(inliers.sum()) if inliers is not None else None,
        "inlier_count": int(inlier_count),
        "affine_method": str(method_key),
        "refit_rigid": True,
    }
    return T, meta


class EdgeGATLite(nn.Module):
    """
    A lightweight edge-feature learner for fully-connected graphs.

    Input: pairwise distance matrix D (N,N)
    Output: edge feature tensor W (N,N,K)

    This follows the FreeAlign paper's intent: enrich invariant edge cues by
    injecting global geometric context via node-aggregated messages on a
    fully-connected graph.
    """

    def __init__(self, *, hidden_dim: int = 64, out_dim: int = 16):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.refine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        """
        Args:
            D: (N,N) float tensor, pairwise distance
        Returns:
            W: (N,N,K) edge feature tensor
        """
        if D.ndim != 2:
            raise ValueError(f"Expected D to be 2D (N,N), got {tuple(D.shape)}")
        N = int(D.shape[0])
        if N == 0:
            return D.new_zeros((0, 0, int(self.refine_mlp[-1].out_features)))

        d_in = D.unsqueeze(-1)  # (N,N,1)
        e0 = self.edge_mlp(d_in)  # (N,N,H)
        # Node context: aggregate outgoing edges.
        h = e0.mean(dim=1)  # (N,H)
        h_i = h.unsqueeze(1).expand(N, N, -1)
        h_j = h.unsqueeze(0).expand(N, N, -1)
        e1 = self.refine_mlp(torch.cat([e0, h_i, h_j], dim=-1))  # (N,N,K)
        # Symmetrize to reduce order sensitivity.
        e1 = 0.5 * (e1 + e1.transpose(0, 1))
        return e1


class EdgeGAT(nn.Module):
    """
    EdgeGAT-style edge feature learner with attention over fully-connected edges.

    Input: pairwise distance matrix D (N,N)
    Output: edge feature tensor W (N,N,K)
    """

    def __init__(self, *, hidden_dim: int = 64, out_dim: int = 16, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.num_heads = max(1, int(num_heads))
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})")
        self.head_dim = self.hidden_dim // self.num_heads
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attn_proj = nn.Linear(self.hidden_dim, self.num_heads, bias=False)
        self.out_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else None

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        if D.ndim != 2:
            raise ValueError(f"Expected D to be 2D (N,N), got {tuple(D.shape)}")
        N = int(D.shape[0])
        if N == 0:
            return D.new_zeros((0, 0, int(self.out_dim)))

        d_in = D.unsqueeze(-1)  # (N,N,1)
        e0 = self.edge_mlp(d_in)  # (N,N,H)
        if self.dropout is not None:
            e0 = self.dropout(e0)

        attn_logits = self.attn_proj(e0)  # (N,N,heads)
        attn = torch.softmax(attn_logits, dim=1)  # normalize over neighbors

        v = self.value_proj(e0)  # (N,N,H)
        v = v.view(N, N, self.num_heads, self.head_dim)
        attn = attn.unsqueeze(-1)  # (N,N,heads,1)
        h = (attn * v).sum(dim=1)  # (N,heads,head_dim)
        h = h.reshape(N, self.hidden_dim)

        h_i = h.unsqueeze(1).expand(N, N, -1)
        h_j = h.unsqueeze(0).expand(N, N, -1)
        e1 = self.out_mlp(torch.cat([e0, h_i, h_j], dim=-1))
        # Symmetrize to reduce order sensitivity.
        e1 = 0.5 * (e1 + e1.transpose(0, 1))
        return e1


@dataclass(frozen=True)
class FreeAlignPaperConfig:
    ckpt_path: Optional[str] = None
    device: str = "cpu"
    hidden_dim: int = 64
    out_dim: int = 16
    use_gnn: bool = False
    gnn_type: str = "lite"  # lite | egat
    gnn_heads: int = 4
    gnn_dropout: float = 0.0
    max_boxes: int = 60
    anchor_topk: int = 10
    seed_strategy: str = "topk_radius"  # "topk_radius" | "all"
    min_nodes: int = 5
    anchor_max_count: int = 4
    sim_threshold: float = 0.6
    anchor_threshold: Optional[float] = None
    box_threshold: Optional[float] = None
    anchor_consistency_multiplier: float = 2.0
    full_consistency: bool = True
    p: float = 1.0
    selection_mode: str = "max_size_then_min_eps"  # "min_eps_then_max_size" | "max_size_then_min_eps"

    affine_method: str = "lmeds"  # "lmeds" | "ransac"
    ransac_reproj_threshold: float = 1.0
    max_iters: int = 2000
    confidence: float = 0.99
    refine_iters: int = 10
    refit_rigid: bool = True
    min_inliers: int = 3

    # Optional temporal smoothing (mirrors v2xregpp_stable "delta SE(2)" smoothing).
    # "initfree": apply per-frame estimate directly; "stable": smooth correction deltas over time.
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

    # Optional temporal alignment (search over ego time buffer).
    # time_buffer = number of historical steps (l in [t, t-τ, ..., t-lτ]); 0 disables.
    time_buffer: int = 0
    time_stride: int = 1
    time_use_clean_pose: bool = True


class FreeAlignPaperEstimator:
    def __init__(self, cfg: FreeAlignPaperConfig):
        self.cfg = cfg
        self.device = torch.device(str(cfg.device))
        self.net = None
        if bool(cfg.use_gnn):
            gnn_type = str(getattr(cfg, "gnn_type", "lite") or "lite").lower().strip()
            if gnn_type in {"egat", "edgegat", "full"}:
                self.net = EdgeGAT(
                    hidden_dim=int(cfg.hidden_dim),
                    out_dim=int(cfg.out_dim),
                    num_heads=int(cfg.gnn_heads),
                    dropout=float(cfg.gnn_dropout),
                )
            else:
                self.net = EdgeGATLite(hidden_dim=int(cfg.hidden_dim), out_dim=int(cfg.out_dim))
            self.net.eval()
            self.net.to(self.device)
            if cfg.ckpt_path:
                state = torch.load(str(cfg.ckpt_path), map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self.net.load_state_dict(state, strict=True)

    @torch.no_grad()
    def _edge_features(self, centers_xy: np.ndarray) -> np.ndarray:
        D = _pairwise_dist(centers_xy, device=self.device)  # (N,N) float32
        if not bool(self.cfg.use_gnn) or self.net is None:
            return D[..., None].astype(np.float32)  # (N,N,1)
        Dt = torch.from_numpy(D).to(self.device)
        W = self.net(Dt).detach().cpu().numpy().astype(np.float32)  # (N,N,K)
        return W

    @staticmethod
    def _cdist_edges(a: np.ndarray, b: np.ndarray, *, device: Optional[torch.device] = None) -> np.ndarray:
        """
        a: (N,K), b: (M,K) -> (N,M) L2 distance
        """
        if device is not None and device.type == "cuda" and torch.cuda.is_available():
            at = torch.as_tensor(a, dtype=torch.float32, device=device)
            bt = torch.as_tensor(b, dtype=torch.float32, device=device)
            d = torch.cdist(at, bt, p=2)
            return d.detach().cpu().numpy().astype(np.float32)
        a2 = (a * a).sum(axis=1, keepdims=True)
        b2 = (b * b).sum(axis=1, keepdims=True).T
        ab = a @ b.T
        d2 = np.maximum(a2 + b2 - 2.0 * ab, 0.0)
        return np.sqrt(d2, dtype=np.float32)

    def _build_anchor_list(self, Wi: np.ndarray, Wj: np.ndarray, seed_i: int, seed_j: int) -> List[Tuple[int, int]]:
        n = int(Wi.shape[0])
        m = int(Wj.shape[0])
        anchors: List[Tuple[int, int]] = [(int(seed_i), int(seed_j))]
        used_i = {int(seed_i)}
        used_j = {int(seed_j)}

        anchor_thr = float(self.cfg.sim_threshold if self.cfg.anchor_threshold is None else self.cfg.anchor_threshold)
        cons_thr = anchor_thr * float(self.cfg.anchor_consistency_multiplier)

        # Compare edge features from the seed anchor to all other nodes.
        Di = self._cdist_edges(Wi[int(seed_i)], Wj[int(seed_j)], device=self.device)  # (n,m)
        Di[int(seed_i), :] = np.inf
        Di[:, int(seed_j)] = np.inf

        cand = np.argwhere(Di <= float(anchor_thr))
        if cand.size == 0:
            return anchors
        vals = Di[cand[:, 0], cand[:, 1]]
        order = np.argsort(vals, kind="mergesort")
        for idx in order.tolist():
            if len(anchors) >= int(self.cfg.anchor_max_count):
                break
            u, v = int(cand[int(idx), 0]), int(cand[int(idx), 1])
            if u in used_i or v in used_j:
                continue
            ok_all = True
            for ai, aj in anchors:
                if float(np.linalg.norm(Wi[int(ai), int(u)] - Wj[int(aj), int(v)])) > cons_thr:
                    ok_all = False
                    break
            if not ok_all:
                continue
            anchors.append((u, v))
            used_i.add(u)
            used_j.add(v)

        return anchors

    def _expand_subgraph(self, Wi: np.ndarray, Wj: np.ndarray, anchors: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        n = int(Wi.shape[0])
        m = int(Wj.shape[0])
        used_i = {i for i, _ in anchors}
        used_j = {j for _, j in anchors}
        box_thr = float(self.cfg.sim_threshold if self.cfg.box_threshold is None else self.cfg.box_threshold)

        # Candidate mask: must be consistent with ALL anchors.
        ok = np.ones((n, m), dtype=bool)
        for ai, aj in anchors:
            Dij = self._cdist_edges(Wi[:, int(ai)], Wj[:, int(aj)], device=self.device)  # (n,m)
            ok &= Dij <= box_thr
        for i in used_i:
            ok[int(i), :] = False
        for j in used_j:
            ok[:, int(j)] = False

        cand = np.argwhere(ok)
        if cand.size == 0:
            return list(anchors)

        # Score candidates by mean discrepancy to anchors.
        scores = []
        for u, v in cand.tolist():
            diffs = []
            for ai, aj in anchors:
                diffs.append(float(np.linalg.norm(Wi[int(u), int(ai)] - Wj[int(v), int(aj)])))
            scores.append((float(np.mean(diffs) if diffs else 0.0), int(u), int(v)))
        scores.sort(key=lambda x: x[0])

        matches = list(anchors)
        for _, u, v in scores:
            if u in used_i or v in used_j:
                continue
            if bool(self.cfg.full_consistency):
                ok_all = True
                for mi, mj in matches:
                    if float(np.linalg.norm(Wi[int(u), int(mi)] - Wj[int(v), int(mj)])) > box_thr:
                        ok_all = False
                        break
                if not ok_all:
                    continue
            matches.append((u, v))
            used_i.add(u)
            used_j.add(v)
        return matches

    def _discrepancy(self, Wi: np.ndarray, Wj: np.ndarray, matches: List[Tuple[int, int]]) -> float:
        r = int(len(matches))
        if r <= 1:
            return float("inf")
        diffs = []
        for a in range(r):
            ui, vi = matches[a]
            for b in range(a + 1, r):
                uj, vj = matches[b]
                diffs.append(float(np.linalg.norm(Wi[int(ui), int(uj)] - Wj[int(vi), int(vj)])))
        if not diffs:
            return float("inf")
        p = float(self.cfg.p)
        if p <= 0:
            p = 1.0
        denom = float(max(r, 1) ** (1.0 / p))
        return float(np.sum(diffs) / max(denom, 1e-9))

    def mass_search(self, Wi: np.ndarray, Wj: np.ndarray, ego_centers: np.ndarray, cav_centers: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
        n = int(Wi.shape[0])
        m = int(Wj.shape[0])
        if n < int(self.cfg.min_nodes) or m < int(self.cfg.min_nodes):
            return [], float("inf")

        seed_strategy = str(self.cfg.seed_strategy or "topk_radius").lower().strip()
        if seed_strategy == "all":
            ego_anchor_idx = np.arange(n, dtype=np.int64)
            cav_anchor_idx = np.arange(m, dtype=np.int64)
        elif seed_strategy in {"topk", "topk_radius"}:
            ego_anchor_idx = _pick_anchor_indices(ego_centers, int(self.cfg.anchor_topk))
            cav_anchor_idx = _pick_anchor_indices(cav_centers, int(self.cfg.anchor_topk))
        else:
            raise ValueError(f"Unsupported seed_strategy: {self.cfg.seed_strategy!r}")

        best: List[Tuple[int, int]] = []
        best_eps = float("inf")
        best_size = 0
        mode = str(self.cfg.selection_mode or "min_eps_then_max_size").lower().strip()
        for seed_i in ego_anchor_idx.tolist():
            for seed_j in cav_anchor_idx.tolist():
                anchors = self._build_anchor_list(Wi, Wj, int(seed_i), int(seed_j))
                matches = self._expand_subgraph(Wi, Wj, anchors)
                if len(matches) < int(self.cfg.min_nodes):
                    continue
                eps = self._discrepancy(Wi, Wj, matches)
                size = int(len(matches))
                if mode in {"max_size", "max_size_then_min_eps"}:
                    if size > best_size or (size == best_size and eps < best_eps - 1e-12):
                        best = matches
                        best_eps = float(eps)
                        best_size = int(size)
                elif mode in {"min_eps", "min_eps_then_max_size"}:
                    if eps < best_eps - 1e-12 or (abs(eps - best_eps) <= 1e-12 and size > best_size):
                        best = matches
                        best_eps = float(eps)
                        best_size = int(size)
                else:
                    raise ValueError(f"Unsupported selection_mode: {self.cfg.selection_mode!r}")
        return best, float(best_eps)

    def estimate(self, cav_boxes, ego_boxes, T_init: Optional[np.ndarray]):
        del T_init  # FreeAlign is prior-free
        max_boxes = int(self.cfg.max_boxes)
        cav_boxes = _topk_by_confidence(list(cav_boxes or []), max_boxes)
        ego_boxes = _topk_by_confidence(list(ego_boxes or []), max_boxes)

        if len(cav_boxes) < int(self.cfg.min_nodes) or len(ego_boxes) < int(self.cfg.min_nodes):
            return None, 0.0, 0, {"reason": "insufficient_boxes"}

        cav_centers = _centers_xy_from_boxes(cav_boxes)
        ego_centers = _centers_xy_from_boxes(ego_boxes)
        if cav_centers.shape[0] < int(self.cfg.min_nodes) or ego_centers.shape[0] < int(self.cfg.min_nodes):
            return None, 0.0, 0, {"reason": "empty_centers"}

        Wi = self._edge_features(ego_centers)
        Wj = self._edge_features(cav_centers)
        matches, eps = self.mass_search(Wi, Wj, ego_centers=ego_centers, cav_centers=cav_centers)
        if len(matches) < int(self.cfg.min_nodes):
            return None, 0.0, 0, {"reason": "no_common_subgraph"}

        T, meta_extra = _estimate_T_from_matches(
            ego_centers,
            cav_centers,
            matches,
            affine_method=str(self.cfg.affine_method),
            ransac_reproj_threshold=float(self.cfg.ransac_reproj_threshold),
            max_iters=int(self.cfg.max_iters),
            confidence=float(self.cfg.confidence),
            refine_iters=int(self.cfg.refine_iters),
            refit_rigid=bool(self.cfg.refit_rigid),
            min_inliers=int(self.cfg.min_inliers),
        )
        if T is None:
            return None, 0.0, int(len(matches)), meta_extra
        meta = {
            "eps": float(eps),
            "matches": int(len(matches)),
            "sim_threshold": float(self.cfg.sim_threshold),
            "anchor_threshold": float(self.cfg.sim_threshold if self.cfg.anchor_threshold is None else self.cfg.anchor_threshold),
            "box_threshold": float(self.cfg.sim_threshold if self.cfg.box_threshold is None else self.cfg.box_threshold),
            "anchor_max_count": int(self.cfg.anchor_max_count),
            "anchor_topk": int(self.cfg.anchor_topk),
            "selection_mode": str(self.cfg.selection_mode),
            **(meta_extra or {}),
        }
        meta["match_pairs"] = [{"ego_idx": int(i), "cav_idx": int(j)} for i, j in matches]
        stability = float(len(matches))
        return T, stability, int(len(matches)), meta
