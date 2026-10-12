"""Standard full-reference metrics: PSNR / SSIM / LPIPS (optional)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..losses import ssim as ssim_torch

_lpips_model = None


def psnr(pred: torch.Tensor, gt: torch.Tensor, data_range: float = 1.0) -> float:
    """PSNR in dB averaged over the batch. Tensors (B,C,H,W) in [0,1]."""
    mse = F.mse_loss(pred, gt, reduction="none").flatten(1).mean(1)
    val = 10.0 * torch.log10(data_range ** 2 / (mse + 1e-12))
    return float(val.mean())


def _get_lpips(device):
    global _lpips_model
    if _lpips_model is None:
        import lpips  # optional dependency

        _lpips_model = lpips.LPIPS(net="alex", verbose=False)
    return _lpips_model.to(device)


def lpips_dist(pred: torch.Tensor, gt: torch.Tensor, device=None) -> Optional[float]:
    """Mean LPIPS distance; returns None if the optional package is missing."""
    try:
        model = _get_lpips(device or pred.device)
    except Exception as e:  # noqa: BLE001
        print(f"[selgate] LPIPS unavailable ({e}); skipping")
        return None
    # lpips expects [-1,1]
    with torch.no_grad():
        d = model(pred.to(next(model.parameters()).device) * 2 - 1,
                  gt.to(next(model.parameters()).device) * 2 - 1)
    return float(d.mean())


def evaluate_pair(
    pred: torch.Tensor,
    gt: torch.Tensor,
    with_lpips: bool = True,
) -> Dict[str, float]:
    """pred/gt: (1,C,H,W) or (C,H,W) in [0,1]."""
    if pred.dim() == 3:
        pred, gt = pred.unsqueeze(0), gt.unsqueeze(0)
    out = {"psnr": psnr(pred, gt), "ssim": float(ssim_torch(pred, gt))}
    if with_lpips:
        v = lpips_dist(pred, gt, device=pred.device)
        if v is not None:
            out["lpips"] = v
    return out
