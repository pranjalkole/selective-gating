"""Loss unit tests: SSIM, TV, illumination prior, combined criterion."""

import pytest
import torch

from selgate.losses import (
    SelGateCriterion,
    illum_prior_loss,
    ssim,
    ssim_loss,
    tv_loss,
)


def test_ssim_identity_and_ordering():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 48, 48)
    assert ssim(x, x).item() == pytest.approx(1.0, abs=1e-4)

    y = x + 0.05 * torch.randn_like(x)
    z = x + 0.30 * torch.randn_like(x)
    assert ssim(x, y).item() > ssim(x, z).item()
    assert ssim_loss(x, x).item() < 1e-4


def test_tv_zero_for_constant_gate():
    g = torch.full((1, 1, 16, 16), 0.4)
    assert tv_loss(g).item() == pytest.approx(0.0, abs=1e-7)

    g2 = torch.rand(1, 1, 16, 16)
    assert tv_loss(g2).item() > 0


def test_illum_prior_prefers_dark_open():
    """BCE loss is lower when the gate is open on dark and closed on bright."""
    x = torch.rand(1, 3, 16, 16)
    # ideal logits: positive where dark, negative where bright
    L = x.max(dim=1, keepdim=True).values
    good_logits = (1 - L) * 4 - 2  # dark -> +2, bright -> -2
    bad_logits = -good_logits
    good = illum_prior_loss(good_logits, x, mode="bce")
    bad = illum_prior_loss(bad_logits, x, mode="bce")
    assert good < bad


def test_illum_prior_corr_mode():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 16, 16)
    L = x.max(dim=1, keepdim=True).values
    good_logits = (1 - L) * 6 - 3
    bad_logits = -(1 - L) * 6 + 3
    assert illum_prior_loss(good_logits, x, mode="corr") < illum_prior_loss(bad_logits, x, mode="corr")


def test_criterion_gate_off_ignores_gate_terms():
    crit = SelGateCriterion(ssim_weight=0.2, tv_lambda=0.01, illum_lambda=0.1)
    out = torch.rand(1, 3, 32, 32)
    gt = torch.rand(1, 3, 32, 32)
    x = torch.rand(1, 3, 32, 32)

    with_gate = crit(out, gt, x, gate=torch.rand(1, 1, 32, 32),
                     gate_logits=torch.randn(1, 1, 32, 32))
    without = crit(out, gt, x)
    assert without["tv"].item() == 0.0 and without["illum"].item() == 0.0
    assert with_gate["total"] > 0 and without["total"] > 0
    # decomposition holds
    rec = without["l1"] + 0.2 * without["ssim_loss"]
    assert torch.allclose(without["total"], rec, atol=1e-6)


def test_criterion_weights_scale_gate_terms():
    crit_a = SelGateCriterion(tv_lambda=0.0, illum_lambda=0.0)
    crit_b = SelGateCriterion(tv_lambda=1.0, illum_lambda=1.0)
    out = gt = x = torch.rand(1, 3, 32, 32)
    gate = torch.rand(1, 1, 32, 32)
    logits = torch.randn(1, 1, 32, 32)
    a = crit_a(out, gt, x, gate, logits)["total"]
    b = crit_b(out, gt, x, gate, logits)["total"]
    assert b > a
