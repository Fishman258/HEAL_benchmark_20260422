import torch
from torch import nn

from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CavAttention(nn.Module):
    """
    Vanilla CAV attention.
    """
    def __init__(self, dim, heads, dim_head=64, dropout=0.1, pose_pe=None):
        super().__init__()
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        # Optional pose-aware bias for agent-wise attention.
        pose_cfg = pose_pe or {}
        self.use_pose_pe = bool(pose_cfg.get("enabled", False))
        if self.use_pose_pe:
            hidden_dim = int(pose_cfg.get("hidden_dim", max(32, dim_head * 2)))
            self.pose_trans_scale = float(pose_cfg.get("translation_scale", 50.0))
            in_dim = 9  # (dx,dy,dz) + sin/cos for (roll,yaw,pitch)
            self.pose_mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, heads),
            )
            self.pose_dropout = nn.Dropout(float(pose_cfg.get("dropout", 0.0)))

    def _pose_bias(self, pairwise_t_matrix: torch.Tensor) -> torch.Tensor:
        """
        Build a head-wise bias from full 6DoF pairwise transforms.

        Args:
            pairwise_t_matrix: (B, L, L, 4, 4) where [i,j] maps i->j.
        Returns:
            bias: (B, heads, L, L)
        """
        # Import here to avoid importing heavy utils on module import.
        from opencood.utils.transformation_utils import tfm_to_pose_torch

        B, L = pairwise_t_matrix.shape[:2]
        tfm = pairwise_t_matrix.reshape(-1, 4, 4).to(dtype=torch.float32)
        pose_deg = tfm_to_pose_torch(tfm, dof=6)  # (B*L*L, 6) => x,y,z,roll,yaw,pitch in degree

        trans = pose_deg[:, 0:3] / max(self.pose_trans_scale, 1e-6)
        angles = torch.deg2rad(pose_deg[:, 3:6])
        ang_sin = torch.sin(angles)
        ang_cos = torch.cos(angles)
        feats = torch.cat([trans, ang_sin, ang_cos], dim=-1)  # (N, 9)

        bias = self.pose_mlp(feats)  # (N, heads)
        bias = self.pose_dropout(bias)
        bias = bias.view(B, L, L, self.heads).permute(0, 3, 1, 2).contiguous()
        return bias

    def forward(self, x, mask, prior_encoding=None, pairwise_t_matrix=None):
        # x: (B, L, H, W, C) -> (B, H, W, L, C)
        # mask: (B, L)
        x = x.permute(0, 2, 3, 1, 4)
        # mask: (B, 1, H, W, L, 1)
        mask = mask.unsqueeze(1)

        # qkv: [(B, H, W, L, C_inner) *3]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # q: (B, M, H, W, L, C)
        q, k, v = map(lambda t: rearrange(t, 'b h w l (m c) -> b m h w l c',
                                          m=self.heads), qkv)

        # attention, (B, M, H, W, L, L)
        att_map = torch.einsum('b m h w i c, b m h w j c -> b m h w i j',
                               q, k) * self.scale
        if self.use_pose_pe and (pairwise_t_matrix is not None):
            # (B, M, H, W, L, L) += (B, M, 1, 1, L, L)
            pose_bias = self._pose_bias(pairwise_t_matrix).to(dtype=att_map.dtype, device=att_map.device)
            att_map = att_map + pose_bias[:, :, None, None, :, :]
        # add mask
        att_map = att_map.masked_fill(mask == 0, -float('inf'))
        # softmax
        att_map = self.attend(att_map)

        # out:(B, M, H, W, L, C_head)
        out = torch.einsum('b m h w i j, b m h w j c -> b m h w i c', att_map,
                           v)
        out = rearrange(out, 'b m h w l c -> b h w l (m c)',
                        m=self.heads)
        out = self.to_out(out)
        # (B L H W C)
        out = out.permute(0, 3, 1, 2, 4)
        return out


class BaseEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CavAttention(dim,
                                          heads=heads,
                                          dim_head=dim_head,
                                          dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask):
        for attn, ff in self.layers:
            x = attn(x, mask=mask) + x
            x = ff(x) + x
        return x


class BaseTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        dim = args['dim']
        depth = args['depth']
        heads = args['heads']
        dim_head = args['dim_head']
        mlp_dim = args['mlp_dim']
        dropout = args['dropout']
        max_cav = args['max_cav']

        self.encoder = BaseEncoder(dim, depth, heads, dim_head, mlp_dim,
                                   dropout)

    def forward(self, x, mask):
        # B, L, H, W, C
        output = self.encoder(x, mask)
        # B, H, W, C
        output = output[:, 0]

        return 
