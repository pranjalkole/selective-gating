"""GatedRetinexformer:  x_out = g ⊙ F_θ(x) + (1 − g) ⊙ x"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .gate import GateNet


class SelGateModel(nn.Module):
    """Backbone + selective output gate.

    ``gate=None`` (or ``model.gate_enabled=False``) gives the control model:
    plain Retinexformer, x_out = F_θ(x).
    """

    def __init__(
        self,
        backbone: nn.Module,
        gate: Optional[GateNet] = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.gate = gate
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        # keep frozen backbone in eval mode (BN/dropout-free anyway, but be explicit)
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    @property
    def gate_enabled(self) -> bool:
        return self.gate is not None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        enhanced = self.backbone(x)
        if self.gate is None:
            return {"out": enhanced, "gate": torch.ones_like(x[:, :1])}
        g, logits = self.gate(x)
        out = g * enhanced + (1.0 - g) * x
        return {"out": out, "gate": g, "gate_logits": logits, "enhanced": enhanced}


def gate_stats(g: torch.Tensor) -> Dict[str, float]:
    """Collapse monitor: fraction of gate mass at the extremes, mean, std."""
    with torch.no_grad():
        return {
            "gate_mean": float(g.mean()),
            "gate_std": float(g.std()),
            "frac_open": float((g > 0.9).float().mean()),    # ~ pass enhanced
            "frac_closed": float((g < 0.1).float().mean()),  # ~ pass input
        }
