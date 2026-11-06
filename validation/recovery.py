from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from inference.npe_train import AmortizedPosterior

@dataclass
class RecoveryReport:
    r2_per_param: dict[str, float]
    mean_r2: float
    contraction_per_param: dict[str, float]

    def assert_gate(self, min_mean_r2: float = 0.75) -> None:
        if self.mean_r2 < min_mean_r2:
            raise AssertionError(f"T1 recovery gate failed: mean R^2={self.mean_r2:.3f} < {min_mean_r2}")

def evaluate_recovery(npe: AmortizedPosterior, true_thetas: np.ndarray,
                      test_xs: np.ndarray) -> RecoveryReport:
    """Computes out-of-sample coefficient of determination R^2 and posterior contraction."""
    pred_means, post_cov = npe.predict_posterior(test_xs)
    post_var = np.diag(post_cov)
    prior_var = npe.space.sd ** 2

    r2_map: dict[str, float] = {}
    contraction_map: dict[str, float] = {}
    for j, name in enumerate(npe.space.names):
        y_true = true_thetas[:, j]
        y_pred = pred_means[:, j]
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
        r2_map[name] = max(0.0, 1.0 - ss_res / ss_tot)
        contraction_map[name] = float(1.0 - post_var[j] / prior_var[j])

    mean_r2 = float(np.mean(list(r2_map.values())))
    return RecoveryReport(
        r2_per_param=r2_map,
        mean_r2=mean_r2,
        contraction_per_param=contraction_map,
    )
