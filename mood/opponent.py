from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class OpponentProcess:
    """Axiom A4: Slow b-process grows with repeated activation, producing tolerance and after-reaction."""
    b: float = 0.0
    tau_b: float = 12.0
    kappa: float = 1.2

    def step(self, a_primary: float, dt: float) -> float:
        b_ss = self.kappa * a_primary
        self.b += dt * (b_ss - self.b) / self.tau_b
        return float(a_primary - self.b)


@dataclass
class HelplessnessController:
    """Reformulated learned helplessness (1978): estimated controllability c_hat gates effort u."""
    c_hat: float = 0.5
    eta: float = 0.15
    beta: float = 4.0

    def step(self, action_succeeded: bool) -> float:
        outcome = 1.0 if action_succeeded else 0.0
        self.c_hat += self.eta * (outcome - self.c_hat)
        return self.effort

    @property
    def effort(self) -> float:
        return float(1.0 / (1.0 + np.exp(-self.beta * (self.c_hat - 0.3))))
