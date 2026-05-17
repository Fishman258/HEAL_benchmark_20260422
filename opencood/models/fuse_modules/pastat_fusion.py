# -*- coding: utf-8 -*-

import math
from typing import Optional, Tuple

import torch
from torch import nn

from opencood.models.fuse_modules.fuse_utils import regroup as Regroup
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


def _build_se2_affine(
    dx: torch.Tensor,
    dy: torch.Tensor,
    dtheta: torch.Tensor,
    *,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Build a 2x3 affine matrix for `torch.nn.functional.affine_grid`.

    dx, dy are in feature-map pixels, dtheta in radians.
    """
    if dx.ndim != 1 or dy.ndim != 1 or dtheta.ndim != 1:
        raise ValueError("dx/dy/dtheta must be rank-1 tensors (B,)")
    if not (dx.shape == dy.shape == dtheta.shape):
        raise ValueError("dx/dy/dtheta must have the same shape")
    B = int(dx.shape[0])
    dtype = dx.dtype
    device = dx.device

    Hf = float(max(int(H), 1))
    Wf = float(max(int(W), 1))

    cos = torch.cos(dtheta)
    sin = torch.sin(dtheta)

    M = torch.zeros((B, 2, 3), device=device, dtype=dtype)
    M[:, 0, 0] = cos
    M[:, 0, 1] = -sin * (Hf / Wf)
    M[:, 1, 0] = sin * (Wf / Hf)
    M[:, 1, 1] = cos
    M[:, 0, 2] = 2.0 * dx / Wf
    M[:, 1, 2] = 2.0 * dy / Hf
    return M


class ConfidenceEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 1) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        self.proj = nn.Linear(1, self.embed_dim, bias=True)

    def forward(self, conf_norm: torch.Tensor, *, H: int, W: int) -> torch.Tensor:
        """
        Args:
            conf_norm: (B,L) normalized confidence in [0,1]
        Returns:
            ce_map: (B,L,embed_dim,H,W)
        """
        if conf_norm.ndim != 2:
            raise ValueError(f"Expected conf_norm shape (B,L), got {tuple(conf_norm.shape)}")
        B, L = conf_norm.shape
        x = conf_norm.reshape(B * L, 1)
        x = self.proj(x)
        x = x.reshape(B, L, self.embed_dim, 1, 1)
        return x.expand(B, L, self.embed_dim, int(H), int(W))


class FSAPairwiseAligner(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: int = 128,
        mlp_hidden: int = 128,
        trans_limit: float = 8.0,
        rot_limit_deg: float = 10.0,
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        hidden_channels = int(hidden_channels)
        mlp_hidden = int(mlp_hidden)
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if hidden_channels <= 0 or mlp_hidden <= 0:
            raise ValueError("hidden_channels/mlp_hidden must be positive")

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, 3),
        )

        self.trans_limit = float(trans_limit)
        self.rot_limit = float(rot_limit_deg) * math.pi / 180.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            dx, dy: (B,) in pixels
            dtheta: (B,) in radians
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x shape (B,C,H,W), got {tuple(x.shape)}")
        h = self.conv(x)
        h = self.pool(h).flatten(1)
        out = self.mlp(h)
        dx = torch.tanh(out[:, 0]) * self.trans_limit
        dy = torch.tanh(out[:, 1]) * self.trans_limit
        dtheta = torch.tanh(out[:, 2]) * self.rot_limit
        return dx, dy, dtheta


class PASTATFusion(nn.Module):
    """
    Pose-Aware Spatio-Temporal Alignment Transformer (PASTAT) fusion module.

    This module follows V2VLoc (AAAI'26) high-level design:
      1) Coarse alignment by poses (affine_matrix),
      2) Confidence Embedding (CE) and Feature Spatial Alignment (FSA),
      3) Transformer fusion (ViT-style, via V2XTransformer baseline in this repo).
    """

    def __init__(self, args: dict) -> None:
        super().__init__()
        from opencood.models.sub_modules.v2xvit_basic import V2XTransformer

        transformer_args = args.get("transformer")
        if transformer_args is None:
            raise ValueError("PASTATFusion requires args['transformer']")
        self.fusion_net = V2XTransformer(transformer_args)

        ce_args = args.get("confidence_embedding") or {}
        ce_dim = int(ce_args.get("embed_dim", 1))
        self.ce = ConfidenceEmbedding(embed_dim=int(ce_dim))

        fsa_args = args.get("fsa") or {}
        self.use_pose_confidence = bool(args.get("use_pose_confidence", True))
        self.drop_confidence_channel = bool(args.get("drop_confidence_channel", True))

        # NOTE: Do NOT lazily create trainable modules in forward. This breaks DDP
        # because parameters created after wrapping won't be synchronized.
        feat_dim = args.get("feat_dim")
        if feat_dim is None:
            # Default to the ViT embedding dim used by V2X-ViT style encoder.
            try:
                feat_dim = transformer_args["encoder"]["cav_att_config"]["dim"]
            except Exception as e:  # pragma: no cover
                raise ValueError("PASTATFusion requires args['feat_dim'] or transformer.encoder.cav_att_config.dim") from e
        self.feat_dim = int(feat_dim)
        if self.feat_dim <= 0:
            raise ValueError("PASTATFusion feat_dim must be positive")

        self.fsa = FSAPairwiseAligner(
            in_channels=int(self.feat_dim) * 2 + int(ce_dim),
            hidden_channels=int(fsa_args.get("hidden_channels", 128)),
            mlp_hidden=int(fsa_args.get("mlp_hidden", 128)),
            trans_limit=float(fsa_args.get("trans_limit", 8.0)),
            rot_limit_deg=float(fsa_args.get("rot_limit_deg", 10.0)),
        )

    def forward(self, x: torch.Tensor, record_len: torch.Tensor, affine_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (sum(n_cav), C, H, W) if use_pose_confidence==False
               (sum(n_cav), C+1, H, W) if use_pose_confidence==True, last channel is pose confidence map.
            record_len: (B,)
            affine_matrix: (B, L, L, 2, 3)
        Returns:
            fused ego feature: (B, C, H, W)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x rank-4, got {tuple(x.shape)}")

        conf_map: Optional[torch.Tensor] = None
        if self.use_pose_confidence:
            if x.shape[1] < 2:
                raise ValueError("x must have at least 2 channels when use_pose_confidence is True")
            conf_map = x[:, -1:, :, :]
            if self.drop_confidence_channel:
                x = x[:, :-1, :, :]

        _, C, H, W = x.shape
        B, L = affine_matrix.shape[:2]
        if int(C) != int(self.feat_dim):
            raise ValueError(f"PASTATFusion expected feature channels C={self.feat_dim}, got C={int(C)}")

        feat_rg, mask = Regroup(x, record_len, int(L))  # (B,L,C,H,W)
        if conf_map is None:
            conf_map = torch.ones((x.shape[0], 1, H, W), device=x.device, dtype=x.dtype)
        conf_rg, _ = Regroup(conf_map, record_len, int(L))  # (B,L,1,H,W)

        # Confidence Embedding (CE) uses per-agent confidence scalars (Eq.7).
        # Do NOT derive scalars from warped maps: coarse alignment translates feature maps
        # by large inter-agent baselines, so sampling a single pixel after warping can
        # incorrectly turn confidence into zeros due to padding.
        conf_scalar = conf_rg[:, :, 0, 0, 0].clamp(min=0.0) * mask.to(conf_rg.dtype)  # (B,L)
        denom = conf_scalar.sum(dim=1, keepdim=True).clamp(min=1e-6)
        conf_norm = conf_scalar / denom  # (B,L)
        ce_map = self.ce(conf_norm, H=int(H), W=int(W))  # (B,L,Ce,H,W)

        warped_feat = []
        for b in range(int(B)):
            ego = 0
            warped_feat.append(warp_affine_simple(feat_rg[b], affine_matrix[b, ego], (int(H), int(W))))
        feat = torch.stack(warped_feat, dim=0)  # (B,L,C,H,W)

        ego_feat = feat[:, 0]  # (B,C,H,W)
        aligned = [ego_feat.unsqueeze(1)]

        for j in range(1, int(L)):
            neigh = feat[:, j]
            inp = torch.cat([ego_feat, neigh, ce_map[:, j]], dim=1)
            dx, dy, dtheta = self.fsa(inp)
            M = _build_se2_affine(dx, dy, dtheta, H=int(H), W=int(W))
            neigh_aligned = warp_affine_simple(neigh, M, (int(H), int(W)))
            aligned.append(neigh_aligned.unsqueeze(1))

        feat_aligned = torch.cat(aligned, dim=1)  # (B,L,C,H,W)

        prior = torch.zeros((int(B), int(L), 3, int(H), int(W)), device=x.device, dtype=x.dtype)
        x_in = torch.cat([feat_aligned, prior], dim=2).permute(0, 1, 3, 4, 2).contiguous()  # (B,L,H,W,C+3)

        spatial_correction_matrix = torch.eye(4, device=x.device, dtype=x.dtype).view(1, 1, 4, 4).repeat(int(B), int(L), 1, 1)
        fused = self.fusion_net(x_in, mask, spatial_correction_matrix)  # (B,H,W,C)
        return fused.permute(0, 3, 1, 2).contiguous()
