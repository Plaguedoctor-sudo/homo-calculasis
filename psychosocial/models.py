from __future__ import annotations
from dataclasses import dataclass

@dataclass
class PsychosocialState:
    """Formalizes Social Baseline, Social Safety/CTRA, Social Rank, Scarcity, and Joiner IPS."""
    relational_support: float = 0.8       # [0, 1]; 1.0 = strong co-regulation (Coan)
    social_rejection_threat: float = 0.0  # targeted exclusion / humiliation (Slavich)
    social_defeat_rank: float = 0.1       # chronic subordination / entrapment (Gilbert/Price)
    socioeconomic_scarcity: float = 0.1   # financial / resource unpredictability (Mullainathan)
    thwarted_belonging: float = 0.1       # Joiner factor 1
    perceived_burden: float = 0.1         # Joiner factor 2
    reappraisal_skill: float = 0.5        # cognitive control effector [0, 1]

    @property
    def social_baseline_multiplier(self) -> float:
        """Isolation acts as a bioenergetic amplifier on threat and homeostatic load."""
        return float(1.0 + 0.8 * max(0.0, 1.0 - self.relational_support))

    @property
    def ctra_inflammatory_drive(self) -> float:
        """Beta-adrenergic Conserved Transcriptional Response to Adversity (sterile inflammation)."""
        raw = self.social_rejection_threat * self.social_baseline_multiplier
        buffered = raw * (1.0 - 0.5 * self.reappraisal_skill)
        return float(max(0.0, 0.6 * buffered))

    def effective_reward_sensitivity(self, rho_base: float = 1.0) -> float:
        """Involuntary Defeat Strategy downregulates striatal D2 / reward gain."""
        return float(rho_base / (1.0 + 0.9 * self.social_defeat_rank))

    @property
    def bandwidth_tax(self) -> float:
        """Fraction of prefrontal executive capacity consumed by material scarcity."""
        return float(self.socioeconomic_scarcity / (1.0 + self.socioeconomic_scarcity))

    @property
    def interpersonal_mood_drift(self) -> float:
        """Joiner synergistic drift term pushing mood landscape toward depressive well."""
        tb = max(0.0, self.thwarted_belonging)
        pb = max(0.0, self.perceived_burden)
        return float(0.15 * tb + 0.15 * pb + 0.50 * (tb * pb))
