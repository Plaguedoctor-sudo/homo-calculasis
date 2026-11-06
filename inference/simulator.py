from __future__ import annotations
import numpy as np
from params.clamps import evaluate_clamped_observables
from params.space import ParameterSpace

def simulate_batch(thetas: np.ndarray, space: ParameterSpace,
                   active_clamps: tuple[str, ...] | None = None,
                   noise_sd: float = 0.015, seed: int = 0) -> np.ndarray:
    """Vectorized rollout of N virtual patients returning (N, D_obs) biomarker matrix."""
    rng = np.random.default_rng(seed)
    n = thetas.shape[0]
    out = [
        evaluate_clamped_observables(thetas[i], space, active_clamps=active_clamps,
                                     noise_sd=noise_sd, rng=rng)
        for i in range(n)
    ]
    return np.asarray(out, dtype=float)

def jacobian_rank(space: ParameterSpace, active_clamps: tuple[str, ...] | None = None,
                  eps: float = 1e-4) -> tuple[int, float]:
    """Computes local sensitivity Jacobian d(x)/d(theta) at prior mean to verify full rank."""
    theta0 = space.mean.copy()
    x0 = evaluate_clamped_observables(theta0, space, active_clamps=active_clamps, noise_sd=0.0)
    J = np.zeros((len(x0), space.dim), dtype=float)
    for j in range(space.dim):
        step = theta0.copy()
        step[j] += eps
        x_step = evaluate_clamped_observables(step, space, active_clamps=active_clamps, noise_sd=0.0)
        J[:, j] = (x_step - x0) / eps
    rank = int(np.linalg.matrix_rank(J, tol=1e-3))
    svals = np.linalg.svd(J, compute_uv=False)
    cond = float(svals[0] / max(svals[-1], 1e-12))
    return rank, cond
