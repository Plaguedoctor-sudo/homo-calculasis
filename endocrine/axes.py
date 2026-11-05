from __future__ import annotations
from dataclasses import dataclass

@dataclass
class HPTAxis:
    """Hypothalamic-Pituitary-Thyroid axis with Selenium-dependent DIO2 T4->T3 conversion."""
    tsh: float = 1.0
    t4_plasma: float = 1.0
    t3_brain: float = 1.0

    def step(self, dt: float, se_free: float, cortisol: float = 0.5, il6: float = 0.3) -> None:
        # DIO2 requires selenocysteine; cortisol and IL-6 inhibit conversion (low-T3 syndrome)
        dio2_activity = (se_free / (0.4 + se_free)) / (1.0 + 0.4 * cortisol + 0.4 * il6)
        t3_target = 1.4 * self.t4_plasma * dio2_activity
        self.t3_brain += dt * (t3_target - self.t3_brain) / 6.0
        self.t3_brain = max(0.1, self.t3_brain)

        # Negative feedback of T3 on pituitary TSH
        tsh_target = 1.5 / (0.5 + self.t3_brain)
        self.tsh += dt * (tsh_target - self.tsh) / 4.0
        self.t4_plasma += dt * (0.8 * self.tsh - 0.8 * self.t4_plasma) / 12.0
