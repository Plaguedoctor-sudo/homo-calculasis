from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Exposome:
    """Macro-environmental inputs: photoperiod, PM2.5, ambient temperature, acoustic noise."""
    lux: float = 2500.0            # daytime illuminance (lux); < 500 in winter/indoor deprivation
    daylength_h: float = 12.0      # hours of daylight
    pm25_ug_m3: float = 8.0        # WHO guideline <= 5.0 ug/m^3
    temp_c: float = 22.0           # thermoneutral reference = 22.0 C
    noise_db: float = 35.0         # nocturnal ambient noise (dB)

    @property
    def photoperiod_drive(self) -> float:
        """ipRGC melanopsin drive in [0.1, 1.5] scaling raphe 5-HT release and SCN coupling."""
        lux_term = self.lux / (500.0 + self.lux)
        day_term = np.clip(self.daylength_h / 12.0, 0.3, 1.5)
        return float(lux_term * day_term)

    @property
    def pm25_inflammatory_drive(self) -> float:
        """Exogenous alveolar/olfactory microglial cytokine drive above 5 ug/m^3 threshold."""
        excess = max(0.0, self.pm25_ug_m3 - 5.0)
        return float(0.04 * excess / (1.0 + 0.02 * excess))

    @property
    def thermal_allostatic_cost(self) -> float:
        """Quadratic bioenergetic demand outside thermoneutral zone (22 C)."""
        return float(0.0025 * ((self.temp_c - 22.0) ** 2))

    @property
    def sws_noise_penalty(self) -> float:
        """Fraction of slow-wave sleep preserved under nocturnal acoustic pollution."""
        excess_db = max(0.0, self.noise_db - 40.0)
        return float(1.0 / (1.0 + 0.05 * excess_db))
