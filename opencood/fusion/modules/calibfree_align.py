"""
Calibration-free (no external pose) alignment utilities for intermediate fusion.

This is inspired by CBR's "match by feature similarity instead of calibration":
we estimate an SE(2) warp (yaw + x/y translation on the BEV feature grid) between
ego and a neighbor using phase correlation on downsampled BEV features.

The output is an affine matrix compatible with `warp_affine_simple` (i.e. the
matrix maps output coords -> input coords in normalized [-1, 1] space).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple


def _to_yaw_candidates(cfg: object) -> List[float]:
    if cfg is None:
        return [0.0]
    if isinstance(cfg, (int, float)):
        max_deg = float(cfg)
        if max_deg <= 0.0:
            return [0.0]
        step = 5.0
        n = int(math.floor(max_deg / step))
        out = [float(k * step) for k in range(-n, n + 1)]
        if 0.0 not in out:
            out.append(0.0)
        return sorted(out)
    if isinstance(cfg, (list, tuple)):
        out = [float(v) for v in cfg]
        if not out:
            return [0.0]
        if 0.0 not in out:
            out.append(0.0)
        return sorted(set(out))
    if isinstance(cfg, dict):
        max_deg = float(cfg.get("max_deg", 0.0) or 0.0)
        step = float(cfg.get("step_deg", 0.0) or 0.0)
        if max_deg <= 0.0 or step <= 0.0:
            return [0.0]
        n = int(math.floor(max_deg / step))
        out = [float(k * step) for k in range(-n, n + 1)]
        if 0.0 not in out:
            out.append(0.0)
        return sorted(out)
    return [0.0]


def _rot_affine_matrix(theta_rad: float, *, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    c = float(math.cos(theta_rad))
    s = float(math.sin(theta_rad))
    a00 = c
    a01 = -s * (float(H) / float(W))
    a10 = s * (float(W) / float(H))
    a11 = c
    return torch.tensor([[a00, a01, 0.0], [a10, a11, 0.0]], device=device, dtype=dtype)


@dataclass(frozen=True)
class CalibFreeAlignConfig:
    enabled: bool = False
    downsample: int = 8
    phase_norm: bool = True
    soft_argmax: bool = False
    temperature: float = 0.2
    yaw_search: object = None

    @staticmethod
    def from_dict(d: Optional[Dict[str, object]]) -> "CalibFreeAlignConfig":
        d = dict(d or {})
        return CalibFreeAlignConfig(
            enabled=bool(d.get("enabled", False)),
            downsample=max(1, int(d.get("downsample", 8) or 1)),
            phase_norm=bool(d.get("phase_norm", True)),
            soft_argmax=bool(d.get("soft_argmax", False)),
            temperature=float(d.get("temperature", 0.2) or 0.2),
            yaw_search=d.get("yaw_search", None),
        )


class CalibFreeAligner(nn.Module):
    def __init__(self, cfg: CalibFreeAlignConfig):
        super().__init__()
        self.cfg = cfg
        self.yaw_candidates_deg = _to_yaw_candidates(cfg.yaw_search)

    def _phase_corr(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = 1e-6
        a = a.to(dtype=torch.float32)
        b = b.to(dtype=torch.float32)

        Fa = torch.fft.fft2(a, dim=(-2, -1))
        Fb = torch.fft.fft2(b, dim=(-2, -1))
        R = (Fa * torch.conj(Fb)).sum(dim=1)
        if self.cfg.phase_norm:
            R = R / (torch.abs(R) + eps)
        corr = torch.fft.ifft2(R, dim=(-2, -1)).real
        corr = torch.fft.fftshift(corr, dim=(-2, -1))
        corr_score = corr.abs()

        H, W = int(corr.shape[-2]), int(corr.shape[-1])
        peak = corr_score.reshape(1, -1).max(dim=-1).values

        if not self.cfg.soft_argmax:
            idx = corr_score.reshape(1, -1).argmax(dim=-1)
            y0 = int(torch.div(idx, W, rounding_mode="trunc").item())
            x0 = int((idx % W).item())

            def _parabola_offset(v_m1: float, v_0: float, v_p1: float) -> float:
                denom = (v_m1 - 2.0 * v_0 + v_p1)
                if abs(denom) < 1e-9:
                    return 0.0
                delta = 0.5 * (v_m1 - v_p1) / denom
                if not math.isfinite(delta):
                    return 0.0
                return float(max(-0.5, min(0.5, delta)))

            y_m1 = (y0 - 1) % H
            y_p1 = (y0 + 1) % H
            x_m1 = (x0 - 1) % W
            x_p1 = (x0 + 1) % W

            v_y_m1 = float(corr_score[0, y_m1, x0].item())
            v_y_0 = float(corr_score[0, y0, x0].item())
            v_y_p1 = float(corr_score[0, y_p1, x0].item())
            v_x_m1 = float(corr_score[0, y0, x_m1].item())
            v_x_0 = v_y_0
            v_x_p1 = float(corr_score[0, y0, x_p1].item())

            dy = _parabola_offset(v_y_m1, v_y_0, v_y_p1)
            dx = _parabola_offset(v_x_m1, v_x_0, v_x_p1)

            shift_y = torch.tensor([float(y0) + dy - float(H // 2)], device=a.device, dtype=torch.float32)
            shift_x = torch.tensor([float(x0) + dx - float(W // 2)], device=a.device, dtype=torch.float32)
            return shift_x, shift_y, peak

        temp = float(self.cfg.temperature)
        if temp <= 0.0:
            temp = 0.2
        weights = F.softmax(corr_score.reshape(1, -1) / temp, dim=-1)

        ys = torch.arange(H, device=a.device, dtype=torch.float32) - float(H // 2)
        xs = torch.arange(W, device=a.device, dtype=torch.float32) - float(W // 2)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_y = grid_y.reshape(1, -1)
        grid_x = grid_x.reshape(1, -1)
        shift_y = (weights * grid_y).sum(dim=-1)
        shift_x = (weights * grid_x).sum(dim=-1)
        return shift_x, shift_y, peak

    @torch.no_grad()
    def estimate_affine(self, ego_feat: torch.Tensor, cav_feat: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if ego_feat.ndim != 3 or cav_feat.ndim != 3:
            raise ValueError("expect ego_feat/cav_feat as (C,H,W)")
        _, H, W = int(ego_feat.shape[0]), int(ego_feat.shape[1]), int(ego_feat.shape[2])
        device = ego_feat.device
        dtype = ego_feat.dtype

        ds = int(self.cfg.downsample)
        if ds > 1:
            ego_small = F.avg_pool2d(ego_feat.unsqueeze(0), kernel_size=ds, stride=ds)
            cav_small = F.avg_pool2d(cav_feat.unsqueeze(0), kernel_size=ds, stride=ds)
        else:
            ego_small = ego_feat.unsqueeze(0)
            cav_small = cav_feat.unsqueeze(0)

        ego_small = F.normalize(ego_small, dim=1)
        cav_small = F.normalize(cav_small, dim=1)

        Hs, Ws = int(ego_small.shape[-2]), int(ego_small.shape[-1])
        best_peak = None
        best_theta = 0.0
        best_dx = 0.0
        best_dy = 0.0

        for yaw_deg in self.yaw_candidates_deg:
            theta = float(math.radians(float(yaw_deg)))
            M_rot = _rot_affine_matrix(theta, H=Hs, W=Ws, device=device, dtype=ego_small.dtype).unsqueeze(0)
            cav_rot = warp_affine_simple(cav_small, M_rot, (Hs, Ws))
            shift_x, shift_y, peak = self._phase_corr(ego_small, cav_rot)
            peak_val = float(peak.item())
            if (best_peak is None) or (peak_val > best_peak):
                best_peak = peak_val
                best_theta = theta
                best_dx = -float(shift_x.item())
                best_dy = -float(shift_y.item())

        c = float(math.cos(best_theta))
        s = float(math.sin(best_theta))
        tx_small = c * best_dx - s * best_dy
        ty_small = s * best_dx + c * best_dy

        scale_x = float(W) / float(Ws)
        scale_y = float(H) / float(Hs)
        tx = tx_small * scale_x
        ty = ty_small * scale_y

        tnx = 2.0 * tx / float(W)
        tny = 2.0 * ty / float(H)

        M_full = _rot_affine_matrix(best_theta, H=H, W=W, device=device, dtype=dtype)
        M_full[0, 2] = float(tnx)
        M_full[1, 2] = float(tny)
        return M_full, float(best_peak or 0.0)


__all__ = ["CalibFreeAlignConfig", "CalibFreeAligner"]
