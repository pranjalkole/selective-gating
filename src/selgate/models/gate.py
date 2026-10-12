"""The selective gate: g = sigma(f_gate([x, L(x)])).

L(x) is a *cheap hand-crafted* illumination estimate (no extra network), so the
gate stays interpretable: dark -> g near 1 (enhance), bright -> g near 0
(pass through).  f_gate is 3-4 conv layers (~20K params).

Granularity variants (ablation axis):
  * pixel   — full-resolution per-pixel gate (default)
  * patch   — gate computed at stride ``s`` and bilinearly upsampled
  * global  — single image-level scalar gate (degenerate baseline)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

GRANULARITIES = ("pixel", "patch", "global")


def illumination_map(x: torch.Tensor, mode: str = "max") -> torch.Tensor:
    """Cheap illumination estimate L(x) in [0,1], shape (B,1,H,W).

    modes:
      max        — per-pixel RGB max (Retinex-style, free)
      luma       — Rec.709 luminance
      luma_blur  — 3x3-averaged luminance (smoother gate signal)
    """
    if mode == "max":
        lum, _ = x.max(dim=1, keepdim=True)
    elif mode == "luma":
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    elif mode == "luma_blur":
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        lum = F.avg_pool2d(F.pad(lum, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    else:
        raise ValueError(f"unknown illumination mode: {mode}")
    return lum.clamp(0.0, 1.0)


class GateNet(nn.Module):
    """f_gate: 4 -> width -> ... -> 1 conv stack over [x, L(x)]."""

    def __init__(
        self,
        granularity: str = "pixel",
        width: int = 32,
        depth: int = 4,
        illum_mode: str = "max",
        stride: int = 8,
    ):
        super().__init__()
        if granularity not in GRANULARITIES:
            raise ValueError(f"granularity must be one of {GRANULARITIES}, got {granularity!r}")
        if depth < 2:
            raise ValueError("depth must be >= 2")
        self.granularity = granularity
        self.illum_mode = illum_mode
        self.stride = stride

        layers = [nn.Conv2d(4, width, 3, padding=1), nn.SiLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1), nn.SiLU(inplace=True)]
        layers.append(nn.Conv2d(width, 1, 3, padding=1))  # logits
        self.convs = nn.Sequential(*layers)

        # start near "pass through the raw input" (bias < 0), not random mid-gate
        nn.init.zeros_(self.convs[-1].weight)
        nn.init.constant_(self.convs[-1].bias, -1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (g, logits) both (B,1,H,W); g = sigmoid(logits) at input resolution."""
        L = illumination_map(x, self.illum_mode)
        h, w = x.shape[-2:]

        if self.granularity == "pixel":
            logits = self.convs(torch.cat([x, L], dim=1))
        elif self.granularity == "patch":
            pooled = F.avg_pool2d(torch.cat([x, L], 1), self.stride, stride=self.stride)
            logits = self.convs(pooled)
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        else:  # global
            pooled = F.adaptive_avg_pool2d(torch.cat([x, L], 1), 1)
            logits = self.convs(pooled)
            logits = F.interpolate(logits, size=(h, w), mode="nearest")

        g = torch.sigmoid(logits)
        return g, logits


def gate_param_count(net: GateNet) -> int:
    return sum(p.numel() for p in net.parameters())
