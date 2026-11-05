from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AllostaticPrior:
    mu_prior: float = 0.0      # expected body-budget deficit
    pi_prior: float = 1.0      # prior rigidity
    pi_sensory: float = 1.0    # combined interoceptive precision

    def posterior_belief(self, y_channels: np.ndarray) -> tuple[float, float]:
        """Precision-weighted Bayesian update over [visceral, nociceptive, immune, metabolic]."""
        y_mean = float(np.mean(y_channels))
        pi_post = self.pi_prior + self.pi_sensory
        mu_post = (self.pi_prior * self.mu_prior + self.pi_sensory * y_mean) / pi_post
        prediction_error = self.pi_sensory * (y_mean - self.mu_prior)
        return float(mu_post), float(prediction_error)

    def variational_free_energy(self, y_channels: np.ndarray) -> float:
        """F = D_KL[q || p] (complexity) - E_q[ln p(y | b)] (accuracy)."""
        mu_post, _ = self.posterior_belief(y_channels)
        pi_post = self.pi_prior + self.pi_sensory
        kl_complexity = 0.5 * (
            np.log(pi_post / self.pi_prior)
            + (1.0 / pi_post + (mu_post - self.mu_prior) ** 2) * self.pi_prior
            - 1.0
        )
        inaccuracy = 0.5 * self.pi_sensory * float(np.mean((y_channels - mu_post) ** 2))
        return float(kl_complexity + inaccuracy)
