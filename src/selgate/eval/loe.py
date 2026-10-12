"""Lightness Order Error (LOE) — guards against over-enhancement reordering.

LOE counts pairs of pixels whose relative lightness order flips between the
reference (input) and the enhanced image.  Full O(N²) is prohibitive, so we
sample a fixed number of random pixel pairs (deterministic given a seed),
following common practice.
"""

from __future__ import annotations

import numpy as np


def _lightness(rgb: np.ndarray) -> np.ndarray:
    """(H,W,3) -> (H,W) lightness = max channel (as in the Zero-DCE LOE)."""
    return rgb.max(axis=-1).astype(np.float64)


def loe(
    reference: np.ndarray,
    enhanced: np.ndarray,
    n_samples: int = 20000,
    seed: int = 0,
) -> float:
    """LOE in [0,1] (lower is better). Inputs (H,W,3) uint8 or float in [0,1]."""
    if reference.shape != enhanced.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {enhanced.shape}")
    if reference.dtype != np.uint8:
        reference = np.clip(reference * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if enhanced.dtype != np.uint8:
        enhanced = np.clip(enhanced * 255.0 + 0.5, 0, 255).astype(np.uint8)

    l_ref = _lightness(reference).ravel()
    l_enh = _lightness(enhanced).ravel()
    n = l_ref.size
    if n < 2:
        return 0.0

    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_samples)
    j = rng.integers(0, n, size=n_samples)
    keep = i != j
    i, j = i[keep], j[keep]
    if i.size == 0:
        return 0.0

    order_ref = l_ref[i] > l_ref[j]
    order_enh = l_enh[i] > l_enh[j]
    return float(np.mean(order_ref != order_enh))


def loe_pair(reference: np.ndarray, enhanced: np.ndarray, **kw) -> float:
    return loe(reference, enhanced, **kw)
