from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class InteroceptiveChannel:
    pi_intero: float = 1.0     # objective sensory precision (acetylcholine-modulated)
    pi_prior: float = 1.0      # top-down prior precision (rigidity)
    sensibility: float = 1.0   # subjective self-report gain
    crit_window_open: bool = True

    def calibrate_early_life(self, caregiving_reliability: float, stress_load: float) -> float:
        """DOHaD / Bowlby attachment calibration during critical window."""
        if not self.crit_window_open:
            return self.pi_prior
        self.pi_prior = np.clip(1.0 + 1.2 * stress_load - 0.5 * caregiving_reliability, 0.2, 4.0)
        self.crit_window_open = False
        return float(self.pi_prior)

    def heartbeat_accuracy(self, vagal_afferent: float, noise_sd: float = 0.2) -> float:
        """Objective accuracy scales with pi_intero, independent of baseline mood."""
        snr = (self.pi_intero * vagal_afferent) / (noise_sd + 1e-6)
        return float(snr / (1.0 + snr))

    def metacognitive_awareness(self, accuracy: float) -> float:
        """Garfinkel et al. correspondence between objective accuracy and sensibility."""
        return float(1.0 - abs(accuracy - (self.sensibility / (1.0 + self.sensibility))))


def kamin_blocking_step(pi_cs: float, predictedness: float, alpha: float = 0.3) -> float:
    """Axiom A8: A fully predicted conditioned stimulus drives Pi_CS -> 0."""
    return float(max(0.0, pi_cs * (1.0 - alpha * np.clip(predictedness, 0.0, 1.0))))


def categorize_emotion(y_intero: float, concept_granularity: float, pi_prior: float) -> dict[str, float]:
    """Constructed emotion (Barrett): granular concepts partition arousal without altering y_intero."""
    raw_distress = y_intero * pi_prior
    somatic_unlabeled = raw_distress / (1.0 + concept_granularity)
    differentiated_affect = raw_distress - somatic_unlabeled
    return {
        "y_intero_unchanged": float(y_intero),
        "somatic_unlabeled": float(somatic_unlabeled),
        "differentiated_affect": float(differentiated_affect),
    }
