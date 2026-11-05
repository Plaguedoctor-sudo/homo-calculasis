from __future__ import annotations


def open_hidden_response(drug_effect: float, expectation_sign: float, pi_prior: float, is_open: bool) -> float:
    """Axiom A7: Open-label administration adds precision-weighted expectation prior;
    expectation_sign = +1 (placebo) or -1 (nocebo)."""
    prior_contrib = (expectation_sign * 0.4 * pi_prior) if is_open else 0.0
    return float(drug_effect + prior_contrib)
