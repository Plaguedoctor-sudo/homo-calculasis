from __future__ import annotations


def pag_rvm_drive(vagal_hrv: float, serotonin: float, noradrenaline: float) -> float:
    """Descending conditioned pain modulation (DNIC) scales with HRV vagal tone and monoamines."""
    return float(0.8 * vagal_hrv + 0.5 * serotonin + 0.5 * noradrenaline)


def inferred_pain_report(nociceptive_input: float, prior_expectation: float,
                         pi_noci: float = 1.0, pi_prior: float = 1.0) -> float:
    """Predictive processing of pain (Wiech & Buchel): posterior pain shifts with prior precision."""
    return float((pi_noci * nociceptive_input + pi_prior * prior_expectation) / (pi_noci + pi_prior))
