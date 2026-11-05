from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DorsalHornGate:
    """Axiom A5: Melzack-Wall gate control with central NMDA windup."""
    w_c: float = 1.2
    w_a: float = 0.4
    w_i: float = 1.5
    s_windup: float = 0.0
    tau_s: float = 4.0
    f_crit: float = 0.5
    beta_windup: float = 0.6

    def step(self, c_fiber: float, a_beta: float, d_rvm: float,
             mg_free: float = 1.0, qa_ratio: float = 1.0, dt: float = 0.1) -> float:
        y_gate = (self.w_c * c_fiber + self.w_a * a_beta) / (1.0 + self.w_i * a_beta + d_rvm)
        self.s_windup += dt * (0.5 * y_gate - self.s_windup / self.tau_s)
        if c_fiber > self.f_crit:
            nmda_gain = (qa_ratio / (1.0 + qa_ratio)) / (0.3 + mg_free)
            self.s_windup *= (1.0 + self.beta_windup * nmda_gain * dt)
        return float(y_gate + self.s_windup)
