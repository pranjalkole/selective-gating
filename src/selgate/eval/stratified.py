"""Region-stratified evaluation — the novel protocol contribution.

Test pixels are split *per image* into dark / mid / bright terciles by the
ground-truth luminance.  For each stratum we report PSNR (fidelity) and
ΔE2000 (colour fidelity).  The headline claim is about the **bright stratum**:
"selective gating improves bright-region fidelity at no dark-region cost".

Everything here is numpy-only (no cv2 / skimage) so it runs anywhere,
including an internet-disabled Kaggle kernel.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

STRATA = ("dark", "mid", "bright")

# ------------------------------------------------------------------------- colour
_SRGB2XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(...,3) sRGB in [0,1] -> CIELAB (D65)."""
    c = rgb.astype(np.float64)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _SRGB2XYZ.T
    xyz = xyz / _D65
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def delta_e2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference (Sharma et al. 2005 formulation)."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = (C1 + C2) / 2
    Cbar7 = Cbar ** 7
    G = 0.5 * (1 - np.sqrt(Cbar7 / (Cbar7 + 25 ** 7)))

    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    with np.errstate(invalid="ignore"):
        dhp = np.where(
            C1p * C2p == 0,
            0.0,
            np.where(
                np.abs(h2p - h1p) <= 180,
                h2p - h1p,
                np.where(h2p - h1p > 180, h2p - h1p - 360, h2p - h1p + 360),
            ),
        )
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)

    Lbar = (L1 + L2) / 2
    Cbarp = (C1p + C2p) / 2
    with np.errstate(invalid="ignore"):
        hbar = np.where(
            C1p * C2p == 0,
            h1p + h2p,
            np.where(
                np.abs(h1p - h2p) <= 180,
                (h1p + h2p) / 2,
                np.where(h1p + h2p < 360, (h1p + h2p + 360) / 2, (h1p + h2p - 360) / 2),
            ),
        )

    T = (
        1
        - 0.17 * np.cos(np.radians(hbar - 30))
        + 0.24 * np.cos(np.radians(2 * hbar))
        + 0.32 * np.cos(np.radians(3 * hbar + 6))
        - 0.20 * np.cos(np.radians(4 * hbar - 63))
    )
    d_theta = 30 * np.exp(-(((hbar - 275) / 25) ** 2))
    Cp7 = Cbarp ** 7
    R_C = 2 * np.sqrt(Cp7 / (Cp7 + 25 ** 7))
    S_L = 1 + 0.015 * (Lbar - 50) ** 2 / np.sqrt(20 + (Lbar - 50) ** 2)
    S_C = 1 + 0.045 * Cbarp
    S_H = 1 + 0.015 * Cbarp * T
    R_T = -np.sin(np.radians(2 * d_theta)) * R_C

    term_l = dLp / S_L
    term_c = dCp / S_C
    term_h = dHp / S_H
    de = np.sqrt(term_l ** 2 + term_c ** 2 + term_h ** 2 + R_T * term_c * term_h)
    return np.nan_to_num(de, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------- stratification
def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luma of gamma-encoded sRGB, (H,W)."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def tercile_masks(lum: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-image dark/mid/bright masks by GT-luminance terciles."""
    t1, t2 = np.quantile(lum, [1 / 3, 2 / 3])
    return {
        "dark": lum <= t1,
        "mid": (lum > t1) & (lum <= t2),
        "bright": lum > t2,
    }


def _psnr_from_mse(mse: float) -> float:
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def stratified_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """gt/pred: (H,W,3) uint8 or float in [0,1]. Returns per-stratum + overall."""
    if gt.dtype != np.uint8:
        gt = np.clip(gt * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if pred.dtype != np.uint8:
        pred = np.clip(pred * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: gt {gt.shape} vs pred {pred.shape}")

    gt_f, pred_f = gt.astype(np.float64) / 255.0, pred.astype(np.float64) / 255.0
    masks = tercile_masks(luminance(gt_f))

    diff2 = ((gt_f - pred_f) ** 2).mean(axis=-1)  # per-pixel MSE over channels
    lab_gt, lab_pred = srgb_to_lab(gt_f), srgb_to_lab(pred_f)
    de = delta_e2000(lab_gt, lab_pred)

    out: Dict[str, Dict[str, float]] = {}
    for name, m in masks.items():
        out[name] = {
            "psnr": _psnr_from_mse(float(diff2[m].mean())),
            "delta_e2000": float(de[m].mean()),
            "pixel_frac": float(m.mean()),
        }
    out["overall"] = {
        "psnr": _psnr_from_mse(float(diff2.mean())),
        "delta_e2000": float(de.mean()),
        "pixel_frac": 1.0,
    }
    return out


# ------------------------------------------------------------------------ runner
def _load(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _find(name: str, d: Path) -> Optional[Path]:
    p = d / name
    if p.exists():
        return p
    stem = Path(name).stem
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        p = d / (stem + ext)
        if p.exists():
            return p
    return None


def run_stratified_eval(
    gt_dir: str | Path,
    pred_dir: str | Path,
    out_dir: Optional[str | Path] = None,
    prefix: str = "stratified",
) -> Dict[str, Dict[str, float]]:
    """Evaluate a folder of predictions against a folder of ground truths.

    Writes ``<prefix>_per_image.csv`` and ``<prefix>_summary.csv`` into out_dir
    (if given) and returns the mean-over-images summary.
    """
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    rows: List[Dict] = []
    for gt_path in sorted(gt_dir.iterdir()):
        if gt_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        pred_path = _find(gt_path.name, pred_dir)
        if pred_path is None:
            print(f"[selgate] WARNING: no prediction for {gt_path.name}")
            continue
        m = stratified_metrics(_load(gt_path), _load(pred_path))
        for stratum, vals in m.items():
            rows.append({"image": gt_path.name, "stratum": stratum, **vals})

    if not rows:
        raise RuntimeError(f"no matched image pairs between {gt_dir} and {pred_dir}")

    summary: Dict[str, Dict[str, float]] = {}
    for stratum in (*STRATA, "overall"):
        sel = [r for r in rows if r["stratum"] == stratum]
        summary[stratum] = {
            "psnr": float(np.mean([r["psnr"] for r in sel])),
            "delta_e2000": float(np.mean([r["delta_e2000"] for r in sel])),
            "n_images": len(sel),
        }

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{prefix}_per_image.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(out_dir / f"{prefix}_summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["stratum", "psnr", "delta_e2000", "n_images"])
            for s, v in summary.items():
                w.writerow([s, f"{v['psnr']:.4f}", f"{v['delta_e2000']:.4f}", v["n_images"]])

    return summary
