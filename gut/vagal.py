"""Vagal afferent (interoceptive) and efferent (anti-inflammatory) limbs."""

from dataclasses import dataclass


@dataclass
class Vagal:
    afferent: float = 0.5
    efferent: float = 0.5
    severed: bool = False

    def step(self, dt: float, scfa: float, lps: float, pyretic: float = 0.0) -> None:
        target = 0.2 + 0.5 * scfa + 0.3 * lps / (1.0 + lps) - 0.3 * pyretic
        target = min(max(target, 0.0), 1.0)
        self.afferent += dt * (target - self.afferent) / 1.5
        if self.severed:
            self.efferent += dt * (0.0 - self.efferent) / 1.5
        else:
            self.efferent += dt * (target - self.efferent) / 1.5
