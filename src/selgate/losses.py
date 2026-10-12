"""Training objective.

    L = L1(out, gt) + 0.2 · (1 − SSIM(out, gt))
        + λ_s · TV(g)                      # spatial smoothness of the gate
        + λ_i · L_illum(g, L(x))           # monotonicity prior: dark → open

The illumination prior is what keeps the gate *interpretable* instead of
degenerating into a second enhancer; without it (λ_i = 0) it is ablation #2.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.gate import illumination_map


# --------------------------------------------------------------------------- SSIM
def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
) -> torch.Tensor:
    """Mean SSIM over a batch of (B,C,H,W) images in [0,1]."""
    c = img1.shape[1]
    kernel = _gaussian_kernel(window_size).to(img1.device, img1.dtype)
    kernel = kernel.expand(c, 1, window_size, window_size)
    pad = window_size // 2

    def filt(im):
        im = F.pad(im, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(im, kernel, groups=c)

    mu1, mu2 = filt(img1), filt(img2)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sig1 = (filt(img1 * img1) - mu1_sq).clamp(min=0)
    sig2 = (filt(img2 * img2) - mu2_sq).clamp(min=0)
    sig12 = filt(img1 * img2) - mu12

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    cs = (2 * sig12 + c2) / (sig1 + sig2 + c2 + 1e-12)
    ssim_map = (2 * mu12 + c1) / (mu1_sq + mu2_sq + c1 + 1e-12) * cs
    return ssim_map.mean()


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    return 1.0 - ssim(img1, img2)


# ---------------------------------------------------------------------- TV / illum
def tv_loss(g: torch.Tensor) -> torch.Tensor:
    """Mean total variation of the gate map (anisotropic)."""
    dh = (g[..., 1:, :] - g[..., :-1, :]).abs().mean()
    dw = (g[..., :, 1:] - g[..., :, :-1]).abs().mean()
    return dh + dw


def illum_prior_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    mode: str = "bce",
    illum_mode: str = "max",
) -> torch.Tensor:
    """Prior: the gate should be OPEN where the input is DARK.

    bce   — BCEWithLogits against target (1 − L(x))   (default; pixel-faithful)
    corr  — 1 − Pearson(g, 1 − L(x))                  (rank/monotonicity flavour)
    """
    target = (1.0 - illumination_map(x.detach(), illum_mode)).clamp(0.0, 1.0)
    if mode == "bce":
        return F.binary_cross_entropy_with_logits(logits, target)
    if mode == "corr":
        g = torch.sigmoid(logits)
        a = g.flatten(1) - g.flatten(1).mean(dim=1, keepdim=True)
        b = target.flatten(1) - target.flatten(1).mean(dim=1, keepdim=True)
        corr = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)
        return (1.0 - corr).mean()
    raise ValueError(f"unknown illum prior mode: {mode}")


# ------------------------------------------------------------------------- criterion
class SelGateCriterion(nn.Module):
    def __init__(
        self,
        ssim_weight: float = 0.2,
        tv_lambda: float = 0.01,
        illum_lambda: float = 0.1,
        illum_mode: str = "bce",
        gate_illum_mode: str = "max",
    ):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.tv_lambda = tv_lambda
        self.illum_lambda = illum_lambda
        self.illum_mode = illum_mode
        self.gate_illum_mode = gate_illum_mode

    def forward(
        self,
        out: torch.Tensor,
        gt: torch.Tensor,
        x: torch.Tensor,
        gate: torch.Tensor = None,
        gate_logits: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        l1 = F.l1_loss(out, gt)
        sim = ssim_loss(out, gt)
        rec = l1 + self.ssim_weight * sim

        zero = rec.new_zeros(())
        tv = tv_loss(gate) if (gate is not None and self.tv_lambda > 0) else zero
        ill = (
            illum_prior_loss(gate_logits, x, self.illum_mode, self.gate_illum_mode)
            if (gate_logits is not None and self.illum_lambda > 0)
            else zero
        )

        total = rec + self.tv_lambda * tv + self.illum_lambda * ill
        return {"total": total, "l1": l1, "ssim_loss": sim, "tv": tv, "illum": ill}
