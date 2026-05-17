"""
Multi-scale window transformer
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

from einops import rearrange
from opencood.models.sub_modules.split_attn import SplitAttn


def _rope_precompute_coeffs_1d(
    positions: torch.Tensor,
    *,
    feat_dim: int,
    freq_base: float,
    freq_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute split-RoPE coefficients.

    Returns:
        cos, sin with shape (seqlen, feat_dim//2).
    """
    if positions.dim() != 1:
        raise ValueError(f"`positions` must be 1D, got shape {tuple(positions.shape)}")
    if feat_dim % 2 != 0:
        raise ValueError(f"`feat_dim` must be even, got {feat_dim}")

    half = feat_dim // 2
    freqs = freq_scale * (
        freq_base ** (-torch.arange(half, device=positions.device, dtype=torch.float32) / half)
    )  # (half,)
    angles = positions.to(torch.float32).unsqueeze(-1) * freqs.unsqueeze(0)  # (seqlen, half)
    return torch.cos(angles), torch.sin(angles)


def _rope_apply_coeffs_split(
    feats: torch.Tensor,
    *,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """
    Apply split-RoPE to `feats` with last-dim `feat_dim`.

    `cos` / `sin` must be broadcastable to feats[..., :feat_dim//2].
    """
    feat_dim = feats.shape[-1]
    if feat_dim % 2 != 0:
        raise ValueError(f"`feats` last-dim must be even, got {feat_dim}")

    half = feat_dim // 2
    x_in = feats[..., :half]
    y_in = feats[..., half:]

    cos = cos.to(dtype=feats.dtype, device=feats.device)
    sin = sin.to(dtype=feats.dtype, device=feats.device)

    if not inverse:
        x_out = cos * x_in + sin * y_in
        y_out = -sin * x_in + cos * y_in
    else:
        x_out = cos * x_in - sin * y_in
        y_out = sin * x_in + cos * y_in

    return torch.cat((x_out, y_out), dim=-1)


def _vrope_position_ids_2d(h: int, w: int, half_head_dim: int, *, device: torch.device) -> torch.Tensor:
    """
    Generate VRoPE position ids for a 2D grid (h, w).

    Ported from https://github.com/CASIA-IVA-Lab/VRoPE (generate_nd_positions)
    with time_shape=1 and half_time_dim=0.

    Returns:
        position_ids: (h*w, half_head_dim) long tensor.
    """
    nd_len = int(h * w)
    # nd_positions for dim-0 (h)
    pos_h = torch.arange(h, device=device, dtype=torch.long).unsqueeze(0).unsqueeze(-1).expand(1, -1, w)
    pos_h = torch.stack([pos_h.flatten(), pos_h.flip(dims=(1,)).flatten()], dim=0).view(2, 1, nd_len)

    # nd_positions for dim-1 (w)
    pos_w = torch.arange(w, device=device, dtype=torch.long).unsqueeze(0).unsqueeze(-1).expand(h, -1, 1)
    pos_w = torch.stack([pos_w.flatten(), pos_w.flip(dims=(1,)).flatten()], dim=0).view(1, 2, nd_len)

    # Apply symmetric operations (2^n patterns, n=2 => 4 patterns).
    two_pow = (pos_h + pos_w).reshape(-1, nd_len)  # (4, nd_len)
    # Channel allocation (repeat patterns across channels).
    position_ids = torch.stack([two_pow[i % two_pow.shape[0]] for i in range(half_head_dim)], dim=-1)
    return position_ids  # (nd_len, half_head_dim)


def get_relative_distances(window_size):
    indices = torch.tensor(np.array(
        [[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class BaseWindowAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, drop_out, window_size,
                 relative_pos_embedding, pos_encoding: str = "rpe"):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.pos_encoding = (pos_encoding or "rpe").lower()
        self.relative_pos_embedding = relative_pos_embedding if self.pos_encoding == "rpe" else False

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.pos_encoding == "rpe":
            if self.relative_pos_embedding:
                self.relative_indices = get_relative_distances(window_size) + window_size - 1
                self.pos_embedding = nn.Parameter(
                    torch.randn(2 * window_size - 1, 2 * window_size - 1)
                )
            else:
                self.pos_embedding = nn.Parameter(torch.randn(window_size**2, window_size**2))

        elif self.pos_encoding == "prope":
            if dim_head % 4 != 0:
                raise ValueError(f"PRoPE requires dim_head % 4 == 0, got {dim_head}")
            # Use PRoPE's 2D split-RoPE (x/y) on the last half of head dim.
            # For BEV tokens we do not have camera matrices, so we apply x/y RoPE only.
            positions_x = torch.tile(torch.arange(window_size), (window_size,))
            positions_y = torch.repeat_interleave(torch.arange(window_size), window_size)
            seg_dim = dim_head // 4
            cos_x, sin_x = _rope_precompute_coeffs_1d(
                positions_x, feat_dim=seg_dim, freq_base=100.0, freq_scale=1.0
            )
            cos_y, sin_y = _rope_precompute_coeffs_1d(
                positions_y, feat_dim=seg_dim, freq_base=100.0, freq_scale=1.0
            )
            self.register_buffer("prope_cos_x", cos_x, persistent=False)
            self.register_buffer("prope_sin_x", sin_x, persistent=False)
            self.register_buffer("prope_cos_y", cos_y, persistent=False)
            self.register_buffer("prope_sin_y", sin_y, persistent=False)

        elif self.pos_encoding == "vrope":
            if dim_head % 2 != 0:
                raise ValueError(f"VRoPE requires dim_head % 2 == 0, got {dim_head}")
            half = dim_head // 2
            position_ids = _vrope_position_ids_2d(window_size, window_size, half, device=torch.device("cpu"))
            freqs = 10000.0 ** (-torch.arange(half, dtype=torch.float32) / half)  # (half,)
            angles = position_ids.to(torch.float32) * freqs.unsqueeze(0)  # (tokens, half)
            self.register_buffer("vrope_cos", torch.cos(angles), persistent=False)
            self.register_buffer("vrope_sin", torch.sin(angles), persistent=False)

        else:
            raise ValueError(f"Unknown pos_encoding={pos_encoding!r}; expected rpe/prope/vrope")

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        b, l, h, w, c, m = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        new_h = h // self.window_size
        new_w = w // self.window_size

        # q : (b, l, m, new_h*new_w, window_size^2, c_head)
        q, k, v = map(
            lambda t: rearrange(t,
                                'b l (new_h w_h) (new_w w_w) (m c) -> b l m (new_h new_w) (w_h w_w) c',
                                m=m, w_h=self.window_size,
                                w_w=self.window_size), qkv)

        if self.pos_encoding == "prope":
            # Apply PRoPE x/y split-RoPE to q/k in-place.
            tokens = self.window_size * self.window_size
            dim_head = q.shape[-1]
            if dim_head % 4 != 0:
                raise RuntimeError(f"PRoPE requires dim_head % 4 == 0, got {dim_head}")
            seg_dim = dim_head // 4
            q_flat = q.reshape(-1, tokens, dim_head)
            k_flat = k.reshape(-1, tokens, dim_head)

            q_first = q_flat[..., : dim_head // 2]
            q_x = q_flat[..., dim_head // 2 : dim_head // 2 + seg_dim]
            q_y = q_flat[..., dim_head // 2 + seg_dim :]
            k_first = k_flat[..., : dim_head // 2]
            k_x = k_flat[..., dim_head // 2 : dim_head // 2 + seg_dim]
            k_y = k_flat[..., dim_head // 2 + seg_dim :]

            q_x = _rope_apply_coeffs_split(q_x, cos=self.prope_cos_x, sin=self.prope_sin_x)
            q_y = _rope_apply_coeffs_split(q_y, cos=self.prope_cos_y, sin=self.prope_sin_y)
            k_x = _rope_apply_coeffs_split(k_x, cos=self.prope_cos_x, sin=self.prope_sin_x)
            k_y = _rope_apply_coeffs_split(k_y, cos=self.prope_cos_y, sin=self.prope_sin_y)

            q = torch.cat([q_first, q_x, q_y], dim=-1).reshape_as(q)
            k = torch.cat([k_first, k_x, k_y], dim=-1).reshape_as(k)

        elif self.pos_encoding == "vrope":
            tokens = self.window_size * self.window_size
            dim_head = q.shape[-1]
            half = dim_head // 2
            q_flat = q.reshape(-1, tokens, dim_head)
            k_flat = k.reshape(-1, tokens, dim_head)

            q = _rope_apply_coeffs_split(
                q_flat, cos=self.vrope_cos, sin=self.vrope_sin
            ).reshape_as(q)
            k = _rope_apply_coeffs_split(
                k_flat, cos=self.vrope_cos, sin=self.vrope_sin
            ).reshape_as(k)

        # b l m h window_size window_size
        dots = torch.einsum('b l m h i c, b l m h j c -> b l m h i j',
                            q, k, ) * self.scale
        if self.pos_encoding == "rpe":
            # consider prior knowledge of the local window
            if self.relative_pos_embedding:
                dots += self.pos_embedding[self.relative_indices[:, :, 0],
                                           self.relative_indices[:, :, 1]]
            else:
                dots += self.pos_embedding

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b l m h i j, b l m h j c -> b l m h i c', attn, v)
        # b l h w c
        out = rearrange(out,
                        'b l m (new_h new_w) (w_h w_w) c -> b l (new_h w_h) (new_w w_w) (m c)',
                        m=self.heads, w_h=self.window_size,
                        w_w=self.window_size,
                        new_w=new_w, new_h=new_h)
        out = self.to_out(out)

        return out


class PyramidWindowAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads, drop_out, window_size,
                 relative_pos_embedding, fuse_method='naive', pos_encoding: str = "rpe"):
        super().__init__()

        assert isinstance(window_size, list)
        assert isinstance(heads, list)
        assert isinstance(dim_heads, list)
        assert len(dim_heads) == len(heads)

        self.pwmsa = nn.ModuleList([])

        for (head, dim_head, ws) in zip(heads, dim_heads, window_size):
            self.pwmsa.append(BaseWindowAttention(dim,
                                                  head,
                                                  dim_head,
                                                  drop_out,
                                                  ws,
                                                  relative_pos_embedding,
                                                  pos_encoding=pos_encoding))
        self.fuse_mehod = fuse_method
        if fuse_method == 'split_attn':
            self.split_attn = SplitAttn(256)
        elif fuse_method == 'split_attn128':
            self.split_attn = SplitAttn(128)
        elif fuse_method == 'split_attn64':
            self.split_attn = SplitAttn(64)

    def forward(self, x):
        output = None
        # naive fusion will just sum up all window attention output and do a
        # mean
        if self.fuse_mehod == 'naive':
            for wmsa in self.pwmsa:
                output = wmsa(x) if output is None else output + wmsa(x)
            return output / len(self.pwmsa)

        elif self.fuse_mehod.startswith('split_attn'):
            window_list = []
            for wmsa in self.pwmsa:
                window_list.append(wmsa(x))
            return self.split_attn(window_list)
