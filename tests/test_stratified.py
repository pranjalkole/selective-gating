"""Region-stratified protocol: colour maths, terciles, and metric sanity."""

import numpy as np
import pytest

from selgate.eval.stratified import (
    delta_e2000,
    luminance,
    run_stratified_eval,
    srgb_to_lab,
    stratified_metrics,
    tercile_masks,
)
from selgate.eval.loe import loe


def test_srgb_to_lab_reference_values():
    # sRGB white -> L=100, a=b=0
    lab = srgb_to_lab(np.array([1.0, 1.0, 1.0]))
    assert lab[0] == pytest.approx(100.0, abs=0.1)
    assert lab[1] == pytest.approx(0.0, abs=0.1)
    assert lab[2] == pytest.approx(0.0, abs=0.1)
    # black -> L=0
    assert srgb_to_lab(np.array([0.0, 0.0, 0.0]))[0] == pytest.approx(0.0, abs=0.1)
    # mid grey ~ L=53
    assert srgb_to_lab(np.array([0.5, 0.5, 0.5]))[0] == pytest.approx(53.6, abs=1.0)


def test_delta_e2000_properties():
    rng = np.random.default_rng(0)
    lab1 = np.stack([rng.uniform(0, 100, 100), rng.uniform(-60, 60, 100),
                     rng.uniform(-60, 60, 100)], axis=-1)
    assert np.all(delta_e2000(lab1, lab1) == 0)          # identity
    de = delta_e2000(lab1, lab1 + 5.0)                     # symmetric-ish, positive
    assert np.all(de >= 0)
    assert np.all(delta_e2000(lab1, lab1 + 0.0) == 0)


def test_tercile_masks_partition():
    rng = np.random.default_rng(0)
    lum = rng.uniform(0, 1, size=(50, 70))
    masks = tercile_masks(lum)
    total = sum(m.astype(int) for m in masks.values())
    assert np.all(total == 1)                              # every pixel in exactly one stratum
    for m in masks.values():
        assert 0.2 < m.mean() < 0.5                        # terciles, not degenerate


def test_stratified_metrics_identical_images():
    rng = np.random.default_rng(0)
    img = (rng.uniform(0, 255, (64, 64, 3))).astype(np.uint8)
    m = stratified_metrics(img, img.copy())
    for s in ("dark", "mid", "bright", "overall"):
        assert m[s]["psnr"] > 60
        assert m[s]["delta_e2000"] == pytest.approx(0.0, abs=1e-6)


def test_stratified_detects_dark_only_corruption():
    """Degrade ONLY the dark half; bright-stratum fidelity must stay high."""
    yy = np.linspace(0, 1, 64)[:, None].repeat(64, axis=1)  # vertical gradient
    gt = (np.clip(yy, 0, 1) * 255).astype(np.uint8)[..., None].repeat(3, axis=2)
    pred = gt.copy()
    dark = gt.mean(axis=2) < np.quantile(gt.mean(axis=2), 1 / 3)
    pred[dark] = np.clip(pred[dark].astype(np.int16) + 40, 0, 255).astype(np.uint8)

    m = stratified_metrics(gt, pred)
    assert m["dark"]["psnr"] < 30          # corrupted
    assert m["bright"]["psnr"] > 50        # untouched
    assert m["dark"]["psnr"] < m["mid"]["psnr"] < m["bright"]["psnr"] + 1e-9


def test_run_stratified_eval_writes_csv(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(0)
    gt_dir, pred_dir = tmp_path / "gt", tmp_path / "pred"
    gt_dir.mkdir(), pred_dir.mkdir()
    for i in range(3):
        img = (rng.uniform(0, 255, (32, 32, 3))).astype(np.uint8)
        Image.fromarray(img).save(gt_dir / f"{i}.png")
        noisy = np.clip(img.astype(np.int16) + rng.integers(-20, 20, img.shape), 0, 255).astype(np.uint8)
        Image.fromarray(noisy).save(pred_dir / f"{i}.png")

    summary = run_stratified_eval(gt_dir, pred_dir, tmp_path / "out")
    assert set(summary) == {"dark", "mid", "bright", "overall"}
    assert (tmp_path / "out" / "stratified_summary.csv").exists()
    assert (tmp_path / "out" / "stratified_per_image.csv").exists()
    # a noisy copy must be finite and in a sane range
    assert 5 < summary["overall"]["psnr"] < 60


def test_loe_identical_is_zero_and_sensitive():
    rng = np.random.default_rng(0)
    img = (rng.uniform(0, 255, (48, 48, 3))).astype(np.uint8)
    assert loe(img, img.copy()) == 0.0

    inverted = 255 - img  # complete order reversal -> high LOE
    assert loe(img, inverted) > 0.4
