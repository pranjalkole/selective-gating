"""Model assembly: backbone, gate on/off, frozen mode, forward contract."""

import torch

from selgate.models import GateNet, SelGateModel, build_backbone, count_params


def _tiny_backbone():
    return build_backbone(n_feat=16, stage=1, num_blocks=(1, 1, 1))


def test_backbone_forward_shape():
    bb = _tiny_backbone()
    x = torch.rand(1, 3, 32, 48)
    y = bb(x)
    assert y.shape == x.shape
    assert count_params(bb) > 0


def test_gate_off_matches_backbone():
    bb = _tiny_backbone()
    bb.eval()
    model = SelGateModel(bb, gate=None)
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        out = model(x)["out"]
        ref = bb(x)
    assert torch.allclose(out, ref)
    assert model.gate_enabled is False


def test_gated_forward_contract():
    bb = _tiny_backbone()
    gate = GateNet(granularity="pixel", width=8, depth=3)
    model = SelGateModel(bb, gate)
    x = torch.rand(1, 3, 40, 56)

    res = model(x)
    assert set(res) >= {"out", "gate", "gate_logits", "enhanced"}
    assert res["out"].shape == x.shape
    assert res["gate"].shape == (1, 1, 40, 56)

    # out must lie between the input and the raw enhancement (convex blend)
    g = res["gate"]
    lo = torch.minimum(x, res["enhanced"])
    hi = torch.maximum(x, res["enhanced"])
    assert (res["out"] >= lo - 1e-5).all() and (res["out"] <= hi + 1e-5).all()


def test_frozen_backbone_has_no_grad():
    bb = _tiny_backbone()
    gate = GateNet(granularity="pixel", width=8, depth=3)
    model = SelGateModel(bb, gate, freeze_backbone=True)

    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.gate.parameters())

    x = torch.rand(1, 3, 32, 32)
    res = model(x)
    res["out"].sum().backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert any(p.grad is not None for p in model.gate.parameters())

    # frozen backbone stays in eval() even after model.train()
    model.train()
    assert not model.backbone.training
    assert model.gate.training


def test_default_retinexformer_param_count():
    """Official LOL-v1 config (n_feat=40, stage=1, blocks=[1,2,2]) ≈ 1.6M params."""
    bb = build_backbone(n_feat=40, stage=1, num_blocks=(1, 2, 2))
    n = count_params(bb)
    assert 1_000_000 <= n <= 3_000_000, f"unexpected backbone size: {n}"
