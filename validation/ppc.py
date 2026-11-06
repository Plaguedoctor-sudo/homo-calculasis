from __future__ import annotations
import numpy as np
from inference.npe_train import AmortizedPosterior
from params.clamps import evaluate_clamped_observables

def posterior_predictive_check(npe: AmortizedPosterior, x_obs: np.ndarray,
                               n_draws: int = 100, seed: int = 0) -> dict[str, float]:
    """Simulates observables from posterior parameter draws and verifies coverage & NRMSE."""
    draws = npe.sample_posterior(x_obs, n_samples=n_draws, seed=seed)
    x_rep = np.array([
        evaluate_clamped_observables(th, npe.space, noise_sd=0.015)
        for th in draws
    ])
    rep_mean = np.mean(x_rep, axis=0)
    rep_low = np.percentile(x_rep, 2.5, axis=0)
    rep_high = np.percentile(x_rep, 97.5, axis=0)

    nrmse = float(np.sqrt(np.mean(((rep_mean - x_obs) / (np.std(x_obs) + 1e-6)) ** 2)))
    coverage_95 = float(np.mean((x_obs >= rep_low - 0.05) & (x_obs <= rep_high + 0.05)))
    return {"nrmse": nrmse, "coverage_95": coverage_95}
