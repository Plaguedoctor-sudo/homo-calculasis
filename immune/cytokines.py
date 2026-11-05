"""IL-6 / TNF-a / CRP with the alpha7nAChR cholinergic brake (Tracey)."""

from dataclasses import dataclass

K_ACH = 1.8


@dataclass
class Cytokines:
    il6: float = 0.3
    tnf: float = 0.3
    crp: float = 0.3
    lps_challenge: float = 0.0

    def step(self, dt: float, lps: float, ach: float) -> None:
        drive = lps + self.lps_challenge
        t_tnf = 0.1 + drive
        self.tnf += dt * (t_tnf - self.tnf * (1.0 + K_ACH * ach)) / 1.0
        self.tnf = max(self.tnf, 0.0)

        t_il6 = 0.1 + 1.2 * self.tnf
        self.il6 += dt * (t_il6 - self.il6) / 2.0
        self.il6 = max(self.il6, 0.0)

        self.crp += dt * (0.1 + 0.8 * self.il6 - self.crp) / 12.0
        self.crp = max(self.crp, 0.0)

    @property
    def pyretic(self) -> float:
        return self.il6 / (1.0 + self.il6)
