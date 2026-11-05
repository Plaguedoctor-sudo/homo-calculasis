from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CorticalCriticality:
    ei_ratio: float = 1.0      # healthy balanced E/I == 1.0 -> branching ratio sigma == 1.0

    @property
    def branching_ratio(self) -> float:
        return float(self.ei_ratio)

    def avalanche_exponent(self) -> float:
        """At criticality (sigma=1.0), power-law exponent alpha == 1.5."""
        return float(1.5 + 0.8 * abs(self.branching_ratio - 1.0))


@dataclass
class CircadianClock:
    phase: float = 0.0         # radians
    omega: float = 2.0 * np.pi / 24.0
    desync_load: float = 0.0

    def step(self, dt: float, light_phase: float, phase_shift_jetlag: float = 0.0) -> float:
        error = np.sin(light_phase - self.phase + phase_shift_jetlag)
        self.phase = (self.phase + dt * (self.omega + 0.25 * error)) % (2.0 * np.pi)
        self.desync_load += dt * (abs(error) - 0.1 * self.desync_load)
        self.desync_load = max(0.0, self.desync_load)
        return float(self.desync_load)
