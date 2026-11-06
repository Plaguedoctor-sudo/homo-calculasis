from __future__ import annotations
import numpy as np
from inference.simulator import simulate_batch
from inference.npe_train import train_amortized_npe
from params.clamps import CLAMP_BATTERY
from params.space import ParameterSpace

def select_optimal_clamp(space: ParameterSpace,
                         already_applied: tuple[str, ...] = ("trace_panel",),
                         n_sim: int = 300, seed: int = 0) -> tuple[str, dict[str, float]]:
    """Bayesian Optimal Experimental Design: selects u* maximizing expected information gain
    H[q(theta)] - E_x[H[q(theta | x, u)]]."""
    thetas = space.sample_prior(n_sim, seed=seed)
    prior_entropy = float(0.5 * np.sum(np.log(2.0 * np.pi * np.e * (space.sd ** 2))))

    base_xs = simulate_batch(thetas, space, active_clamps=already_applied, seed=seed)
    base_npe = train_amortized_npe(thetas, base_xs, space)
    _, base_logdet = np.linalg.slogdet(2.0 * np.pi * np.e * base_npe.cov_post)
    base_entropy = float(0.5 * base_logdet)

    gains: dict[str, float] = {}
    for clamp in CLAMP_BATTERY:
        if clamp.id in already_applied:
            continue
        candidate_clamps = already_applied + (clamp.id,)
        xs = simulate_batch(thetas, space, active_clamps=candidate_clamps, seed=seed + 1)
        npe = train_amortized_npe(thetas, xs, space)
        _, post_logdet = np.linalg.slogdet(2.0 * np.pi * np.e * npe.cov_post)
        post_entropy = float(0.5 * post_logdet)
        gains[clamp.id] = float(base_entropy - post_entropy)

    best_clamp = max(gains, key=gains.get)
    return best_clamp, gains
