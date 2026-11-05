"""Minimal competitive ensemble. Two functional groups: butyrate producers,
proteolytic. SCFA is the butyrate-producer output and gates barrier integrity.
"""

from dataclasses import dataclass

import numpy as np

# Scaled competitive interaction preserving baseline equilibrium [0.6, 0.4]
A = np.array([[0.75, 1.125], [0.5, 1.0]])
R = np.array([0.9, 0.7])


@dataclass
class Microbiota:
    n: np.ndarray = None
    rng: np.random.Generator = None

    def __post_init__(self):
        if self.n is None:
            self.n = np.array([0.6, 0.4])
        if self.rng is None:
            self.rng = np.random.default_rng(0)

    @property
    def scfa(self) -> float:
        return float(self.n[0] / (0.5 + self.n[0]))

    def step(self, dt: float, perturb: np.ndarray | None = None) -> None:
        r = R if perturb is None else R * perturb
        growth = 0.01 * self.n * (r - A @ self.n)
        noise = np.sqrt(np.abs(self.n)) * self.rng.standard_normal(2) * 0.002
        self.n = np.clip(self.n + dt * growth + noise * np.sqrt(dt), 0.0, None)
