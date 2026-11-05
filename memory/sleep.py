from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SleepGlymphatic:
    metabolite_load: float = 0.2
    synaptic_weight_sum: float = 1.0
    prod_rate: float = 0.15
    k_clear_awake: float = 0.05
    k_clear_sws: float = 0.65
    sws_downscale_rate: float = 0.25

    def step(self, dt: float, sws_density: float, wake_potentiation: float = 0.1) -> None:
        """Glymphatic clearance scales with SWS density; synaptic weights downscale during SWS."""
        k_eff = self.k_clear_awake + (self.k_clear_sws - self.k_clear_awake) * sws_density
        self.metabolite_load += dt * (self.prod_rate - k_eff * self.metabolite_load)
        self.metabolite_load = max(0.0, self.metabolite_load)

        pot = wake_potentiation * (1.0 - sws_density)
        down = self.sws_downscale_rate * sws_density * self.synaptic_weight_sum
        self.synaptic_weight_sum = max(0.2, self.synaptic_weight_sum + dt * (pot - down))
