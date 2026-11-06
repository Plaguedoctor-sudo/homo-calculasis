from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from params.space import ParameterSpace

@dataclass
class ProspectiveReport:
    auc_sertraline: float
    calibration_slope: float
    calibration_intercept: float
    drug_placebo_divergence: float

    def assert_clinical_gates(self) -> None:
        if self.auc_sertraline < 0.60:
            raise AssertionError(f"T3 gate failed: AUC={self.auc_sertraline:.3f} < 0.60")
        if not (0.60 <= self.calibration_slope <= 1.40):
            raise AssertionError(f"T3 calibration slope {self.calibration_slope:.3f} outside [0.6, 1.4]")
        if self.drug_placebo_divergence <= 0.10:
            raise AssertionError(
                f"T4 discriminator failed: drug-placebo divergence={self.drug_placebo_divergence:.3f}"
            )

def predict_arm_probabilities(thetas: np.ndarray, space: ParameterSpace,
                              arm: str = "sertraline") -> np.ndarray:
    """Computes mechanistic basin-escape probability under Sertraline, Placebo, or rTMS."""
    probs = np.zeros(len(thetas), dtype=float)
    for i, th in enumerate(thetas):
        p = space.to_dict(th)
        # Baseline barrier & allostatic resistance
        resistance = 0.8 * p["barrier_lam"] + 0.35 * p["ido_gain"] / (0.5 + p["zn_intake"])
        placebo_drive = 0.35 * (1.8 - 0.4 * p["pi_prior"]) + 0.3 * p["social_support"]

        if arm == "placebo":
            logit = placebo_drive - resistance
        elif arm == "sertraline":
            # SSRI efficacy requires cofactor-supported 5-HT synthesis capacity + BDNF/EAAT2 support
            cof_5ht_cap = (p["fe_intake"] / (0.6 + p["fe_intake"])) * (p["se_intake"] / (0.5 + p["se_intake"]))
            ssri_mech = 1.35 * cof_5ht_cap + 0.45 * p["eaat2_capacity"] + 0.35 * p["reward_rho"]
            logit = placebo_drive + ssri_mech - resistance - 0.9
        elif arm == "rtms":
            # rTMS bypasses gut/monoamine synthesis and directly modulates cortical E/I & vagal-sgACC loop
            rtms_mech = 1.10 * p["eaat2_capacity"] + 0.70 * p["vagal_tone"]
            logit = placebo_drive + rtms_mech - resistance - 0.7
        else:
            raise ValueError(f"unknown treatment arm: {arm}")

        probs[i] = 1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0)))
    return probs

def _roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    comparisons = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    return float(np.mean(comparisons))

def evaluate_prospective_trial(true_thetas: np.ndarray, inferred_thetas: np.ndarray,
                               space: ParameterSpace, seed: int = 0) -> ProspectiveReport:
    """Runs synthetic EMBARC-style 2-arm trial comparing predicted vs. realized outcomes."""
    rng = np.random.default_rng(seed)
    true_prob_drug = predict_arm_probabilities(true_thetas, space, arm="sertraline")
    y_outcomes = (rng.uniform(size=len(true_thetas)) < true_prob_drug).astype(int)

    pred_prob_drug = predict_arm_probabilities(inferred_thetas, space, arm="sertraline")
    pred_prob_plac = predict_arm_probabilities(inferred_thetas, space, arm="placebo")

    auc = _roc_auc(y_outcomes, pred_prob_drug)

    # Calibration slope and intercept via linear regression of outcome on predicted probability
    slope, intercept = np.polyfit(pred_prob_drug, y_outcomes.astype(float), deg=1)
    divergence = float(np.mean(np.abs(pred_prob_drug - pred_prob_plac)))

    return ProspectiveReport(
        auc_sertraline=auc,
        calibration_slope=float(slope),
        calibration_intercept=float(intercept),
        drug_placebo_divergence=divergence,
    )
