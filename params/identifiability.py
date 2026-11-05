from __future__ import annotations

from typing import Callable
import numpy as np


def profile_likelihood_1d(loss_fn: Callable[[np.ndarray], float],
                          theta_hat: np.ndarray,
                          param_idx: int,
                          grid: np.ndarray) -> tuple[np.ndarray, bool]:
    """Evaluates loss across a grid for parameter `param_idx`; flags flat profiles."""
    scores = np.zeros_like(grid, dtype=float)
    for i, val in enumerate(grid):
        trial = theta_hat.copy()
        trial[param_idx] = val
        scores[i] = loss_fn(trial)
    curvature = float(np.max(scores) - np.min(scores))
    is_identifiable = curvature > 1e-3
    return scores, is_identifiable
