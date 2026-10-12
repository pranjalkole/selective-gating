"""End-to-end smoke: 3 training iterations on the fake LOL dataset (CPU)."""

import json

import torch

from selgate.train import build_model, load_model_from_checkpoint, train


def test_train_smoke_creates_checkpoint(tiny_cfg):
    from pathlib import Path

    run_dir = Path(tiny_cfg.train.out_dir) / "unit_smoke"
    ckpt = train(tiny_cfg)

    assert ckpt.exists()
    assert (run_dir / "history.json").exists()
    assert (run_dir / "best.pt").exists()

    with open(run_dir / "history.json") as f:
        hist = json.load(f)
    assert len(hist) >= 1
    assert "total" in hist[0] and "gate_mean" in hist[0]  # gate stats logged

    # checkpoint round-trips
    model = load_model_from_checkpoint(ckpt, tiny_cfg)
    x = torch.rand(1, 3, 16, 16)
    out = model(x)["out"]
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_train_smoke_gate_off(tiny_cfg):
    tiny_cfg.set_path("model.gate.enabled", False)
    tiny_cfg.set_path("name", "unit_smoke_off")
    ckpt = train(tiny_cfg)
    model = load_model_from_checkpoint(ckpt, tiny_cfg)
    assert model.gate is None
    res = model(torch.rand(1, 3, 16, 16))
    assert torch.allclose(res["gate"], torch.ones(1, 1, 16, 16))


def test_build_model_respects_freeze(tiny_cfg):
    tiny_cfg.set_path("model.freeze_backbone", True)
    model = build_model(tiny_cfg)
    assert model.freeze_backbone
    assert all(not p.requires_grad for p in model.backbone.parameters())
