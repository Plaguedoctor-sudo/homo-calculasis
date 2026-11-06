from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class HierarchicalPopulation:
    pop_mean: np.ndarray             # mu_pop (D_theta,)
    pop_cov: np.ndarray              # Sigma_pop (D_theta, D_theta)
    shrunk_thetas: np.ndarray        # Partially pooled patient parameters (N, D_theta)
    eigenvalues: np.ndarray          # Sorted eigenvalues of Sigma_pop (subtype variance)
    eigenvectors: np.ndarray         # Principal subtype axes (D_theta, D_theta)

    @property
    def effective_rank(self) -> float:
        """Shannon participation ratio of eigenvalues; > 1.5 confirms multi-biotype structure."""
        p = self.eigenvalues / (np.sum(self.eigenvalues) + 1e-12)
        return float(np.exp(-np.sum(p * np.log(p + 1e-12))))

def fit_hierarchical_eb(patient_means: np.ndarray,
                        patient_noise_cov: np.ndarray) -> HierarchicalPopulation:
    """Empirical Bayes multilevel pooling: shrinks individual patient posteriors toward
    population hyper-parameters (mu_pop, Sigma_pop) and extracts subtype eigenstructure."""
    n, d = patient_means.shape
    pop_mean = np.mean(patient_means, axis=0)
    sample_cov = np.cov(patient_means, rowvar=False)
    # Subtract observation estimation noise to isolate true between-patient covariance Sigma_pop
    raw_pop_cov = sample_cov - 0.5 * patient_noise_cov
    eigvals, eigvecs = np.linalg.eigh(raw_pop_cov)
    eigvals_clipped = np.maximum(eigvals, 1e-4)
    pop_cov = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    # James-Stein / precision-weighted partial pooling per patient
    inv_pop = np.linalg.inv(pop_cov)
    inv_noise = np.linalg.inv(patient_noise_cov + 1e-6 * np.eye(d))
    post_cov = np.linalg.inv(inv_pop + inv_noise)

    shrunk = np.zeros_like(patient_means)
    for i in range(n):
        shrunk[i] = post_cov @ (inv_pop @ pop_mean + inv_noise @ patient_means[i])

    order = np.argsort(eigvals_clipped)[::-1]
    return HierarchicalPopulation(
        pop_mean=pop_mean,
        pop_cov=pop_cov,
        shrunk_thetas=shrunk,
        eigenvalues=eigvals_clipped[order],
        eigenvectors=eigvecs[:, order],
    )
