import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _gradient_image(w: int, h: int, seed: int, bright: bool = False) -> np.ndarray:
    """Deterministic textured image with a broad luminance range."""
    rng = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    base = 0.15 + 0.8 * (0.6 * xx + 0.4 * yy) if bright else 0.05 + 0.6 * (0.6 * xx + 0.4 * yy)
    noise = 0.1 * rng.standard_normal((h, w))
    lum = np.clip(base + noise, 0, 1)
    tint = rng.uniform(0.85, 1.15, size=3)
    rgb = np.clip(lum[..., None] * tint, 0, 1)
    return (rgb * 255).astype(np.uint8)


@pytest.fixture()
def fake_lol(tmp_path: Path) -> Path:
    """40 train pairs (our485) + 15 test pairs (eval15), 32x32."""
    root = tmp_path / "LOL-v1"
    for d in ("our485/low", "our485/high", "eval15/low", "eval15/high"):
        (root / d).mkdir(parents=True)

    for i in range(40):
        low = _gradient_image(32, 32, seed=i, bright=False)
        high = np.clip(low.astype(np.float32) * 1.6 + 20, 0, 255).astype(np.uint8)
        Image.fromarray(low).save(root / "our485/low" / f"{i + 1:04d}.png")
        Image.fromarray(high).save(root / "our485/high" / f"{i + 1:04d}.png")

    for i in range(15):
        low = _gradient_image(32, 32, seed=1000 + i, bright=False)
        high = np.clip(low.astype(np.float32) * 1.6 + 20, 0, 255).astype(np.uint8)
        Image.fromarray(low).save(root / "eval15/low" / f"{i + 1:03d}.png")
        Image.fromarray(high).save(root / "eval15/high" / f"{i + 1:03d}.png")
    return root


@pytest.fixture()
def tiny_cfg(fake_lol):
    from selgate.config import Config

    return Config.from_dict(
        {
            "name": "unit_smoke",
            "seed": 0,
            "data": {
                "root": str(fake_lol),
                "split_file": str(fake_lol.parent / "split_lol_v1.json"),
                "val_fraction": 0.1,
                "split_seed": 0,
                "crop_size": 16,
                "batch_size": 2,
                "num_workers": 0,
                "augment": True,
            },
            "model": {
                "backbone": {"n_feat": 16, "stage": 1, "num_blocks": [1, 1, 1], "pretrained": None},
                "gate": {"enabled": True, "granularity": "pixel", "width": 8,
                         "depth": 3, "illum": "max", "stride": 4},
                "freeze_backbone": False,
            },
            "train": {
                "iters": 3,
                "lr": 1.0e-3,
                "lr_min": 1.0e-6,
                "amp": False,
                "grad_clip": 1.0,
                "log_every": 1,
                "val_every": 3,
                "max_val_images": 2,
                "out_dir": str(tmp_path_out(fake_lol)),
                "device": "cpu",
            },
            "loss": {"ssim_weight": 0.2, "tv_lambda": 0.01, "illum_lambda": 0.1,
                     "illum_mode": "bce", "gate_illum_mode": "max"},
        }
    )


def tmp_path_out(fake_lol) -> Path:
    p = fake_lol.parent / "runs"
    p.mkdir(exist_ok=True)
    return p
