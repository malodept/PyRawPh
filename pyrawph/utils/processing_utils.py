from __future__ import annotations

from typing import Tuple

import numpy as np


def percentile_stretch(
    x: np.ndarray,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Robust normalization to [0,1] using percentiles.
    Returns float32.
    """
    xf = x.astype(np.float32, copy=False)
    lo, hi = np.nanpercentile(xf, (p_lo, p_hi))
    y = (xf - lo) / (hi - lo + eps)
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


def normalized_difference(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    (a - b) / (a + b + eps) in float32
    """
    af = a.astype(np.float32, copy=False)
    bf = b.astype(np.float32, copy=False)
    return (af - bf) / (af + bf + eps)


def make_rgb(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    stretch: Tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """
    Build an RGB HxWx3 float32 image in [0,1] from 2D bands.
    """
    p_lo, p_hi = float(stretch[0]), float(stretch[1])
    rr = percentile_stretch(r, p_lo=p_lo, p_hi=p_hi)
    gg = percentile_stretch(g, p_lo=p_lo, p_hi=p_hi)
    bb = percentile_stretch(b, p_lo=p_lo, p_hi=p_hi)
    return np.stack([rr, gg, bb], axis=-1)