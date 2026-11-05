"""Intestinal permeability and LPS translocation. SCFA tightens, cortisol leaks."""

from dataclasses import dataclass


@dataclass
class Barrier:
    perm: float = 0.3
    lps_lumen: float = 1.0

    def step(self, dt: float, scfa: float, cortisol: float = 0.5) -> None:
        target = 0.05 + 0.6 * cortisol / (0.5 + cortisol) - 0.35 * scfa
        target = min(max(target, 0.02), 1.0)
        self.perm += dt * (target - self.perm) / 2.0

    @property
    def lps_plasma(self) -> float:
        return self.lps_lumen * self.perm
