from __future__ import annotations

import numpy as np


def rolling_csd_metrics(series: np.ndarray, window: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Computes rolling variance and lag-1 autocorrelation (AC1) to detect bifurcation proximity."""
    n = len(series)
    if n < window + 2:
        raise ValueError("series shorter than window + 2")
    variances = np.zeros(n - window)
    ac1s = np.zeros(n - window)
    for i in range(n - window):
        seg = series[i : i + window]
        variances[i] = np.var(seg)
        s0, s1 = seg[:-1] - np.mean(seg[:-1]), seg[1:] - np.mean(seg[1:])
        denom = np.sqrt(np.sum(s0 ** 2) * np.sum(s1 ** 2)) + 1e-12
        ac1s[i] = np.sum(s0 * s1) / denom
    return variances, ac1s
