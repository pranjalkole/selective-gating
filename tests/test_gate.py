"""Gate unit tests: shapes, granularity variants, param budget, range."""

import pytest
import torch

from selgate.models.gate import GRANULARITIES, GateNet, gate_param_count, illumination_map


def test_illumination_modes_shape_and_range():
    x = torch.rand(2, 3, 32, 48)
    for mode in ("max", "luma", "luma_blur"):
        L = illumination_map(x, mode)
        assert L.shape == (2, 1, 32, 48)
        assert 0.0 <= L.min() and L.max() <= 1.0
    # max is at least as bright as luma for sRGB values
    assert (illumination_map(x, "max") >= illumination_map(x, "luma") - 1e-6).all()


@pytest.mark.parametrize("gran", GRANULARITIES)
def test_gate_output_shape(gran):
    x = torch.rand(2, 3, 40, 64)
    net = GateNet(granularity=gran, width=8, depth=3, stride=8)
    g, logits = net(x)
    assert g.shape == logits.shape == (2, 1, 40, 64)
    assert g.min() >= 0.0 and g.max() <= 1.0
    assert torch.allclose(g, torch.sigmoid(logits))


def test_pixel_gate_param_budget():
    """Default pixel gate should stay ~20K params (the paper's claim)."""
    net = GateNet(granularity="pixel", width=32, depth=4)
    n = gate_param_count(net)
    assert 15_000 <= n <= 30_000, f"expected ~20K params, got {n}"


def test_global_gate_is_spatially_constant():
    x = torch.rand(1, 3, 32, 32)
    net = GateNet(granularity="global", width=8, depth=3)
    g, _ = net(x)
    assert torch.allclose(g, g.mean().expand_as(g), atol=1e-6)


def test_gate_init_starts_near_pass_through():
    """Zero final weights + bias -1 -> uniform logits -1 -> g = sigmoid(-1) ≈ 0.27.

    The gate therefore starts as a neutral (slightly closed) blend and only
    opens/closes where the illumination prior pushes it during training.
    """
    net = GateNet(granularity="pixel", width=8, depth=3)
    g, logits = net(torch.rand(1, 3, 16, 16))
    assert torch.allclose(logits, torch.full_like(logits, -1.0))
    assert torch.allclose(g, torch.full_like(g, torch.sigmoid(torch.tensor(-1.0))))


def test_gate_responds_to_input_once_trained_weights():
    """With non-zero output weights the gate must vary with illumination."""
    torch.manual_seed(0)
    net = GateNet(granularity="pixel", width=8, depth=3)
    torch.nn.init.normal_(net.convs[-1].weight, std=0.5)  # simulate trained weights
    with torch.no_grad():
        net.convs[-1].bias.fill_(0.0)
    dark = torch.zeros(1, 3, 16, 16)
    bright = torch.ones(1, 3, 16, 16)
    g_dark, _ = net(dark)
    g_bright, _ = net(bright)
    assert not torch.allclose(g_dark, g_bright), "gate ignores its input!"
