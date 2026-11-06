from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from params.space import ParameterSpace

@dataclass(frozen=True)
class ClampProtocol:
    id: str
    target_subsystem: str
    perturbation_magnitude: float

CLAMP_BATTERY: tuple[ClampProtocol, ...] = (
    ClampProtocol("trace_panel",          "cofactors",     1.0),
    ClampProtocol("tryptophan_depletion", "serotonin_5ht", 0.85),
    ClampProtocol("lps_challenge",        "immune_kyn",    2.5),
    ClampProtocol("cold_pressor_qst",     "pain_gate",     1.8),
    ClampProtocol("oral_glucose_lnaa",    "metabolic_bbb", 2.0),
    ClampProtocol("sleep_deprivation",    "glymphatic",    0.8),
)

def evaluate_clamped_observables(theta: np.ndarray, space: ParameterSpace,
                                 active_clamps: tuple[str, ...] | None = None,
                                 noise_sd: float = 0.01,
                                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Computes the clinical biomarker readout vector x given patient
    parameters theta and active experimental clamps."""
    p = space.to_dict(space.clip(theta))
    clamps = set(active_clamps if active_clamps is not None else (
        "trace_panel", "tryptophan_depletion", "lps_challenge",
        "cold_pressor_qst", "oral_glucose_lnaa", "sleep_deprivation"
    ))

    # 1. Cofactor panel readouts (Serum Fe, Zn, Mg, Se & T3 conversion)
    panel_gain = 1.0 if "trace_panel" in clamps else 0.25
    fe_obs = panel_gain * p["fe_intake"]
    zn_obs = panel_gain * p["zn_intake"]
    mg_obs = panel_gain * p["mg_intake"]
    se_obs = panel_gain * p["se_intake"]

    # 2. Tryptophan depletion clamp -> 5-HT synthesis & PRT response bias
    trp_challenge = 0.25 if "tryptophan_depletion" in clamps else 1.0
    da_syn = p["fe_intake"] / (0.6 + p["fe_intake"])
    serotonin_5ht = (p["fe_intake"] / (0.8 + p["fe_intake"])) * trp_challenge
    prt_bias = p["reward_rho"] * (0.6 * da_syn + 0.4 * serotonin_5ht) * (0.5 + 0.5 * p["social_support"])

    # 3. LPS challenge clamp -> IL-6, QA/KA ratio, and vagal HRV damping
    lps_load = 2.5 if "lps_challenge" in clamps else 0.2
    il6_level = (0.3 + lps_load * (1.2 - 0.4 * p["scfa_production"])) / (1.0 + 0.7 * p["vagal_tone"])
    qa_ka_ratio = (p["ido_gain"] * il6_level) / (0.4 + 0.8 * p["zn_intake"])

    # 4. Cold pressor / QST clamp -> Nociceptive threshold & DNIC
    cold_stim = 1.8 if "cold_pressor_qst" in clamps else 0.0
    dnic = (0.6 * p["pag_rvm_gain"] + 0.4 * p["vagal_tone"]) if "cold_pressor_qst" in clamps else 0.0
    pain_report = (cold_stim / (0.8 + dnic + 0.3 * p["mg_intake"])) * (
        0.5 * p["pi_intero"] + 0.5 * p["pi_prior"]
    )

    # 5. Oral glucose / mixed meal clamp -> HOMA-IR, LAT1 uptake, ATP flux
    meal_load = 2.0 if "oral_glucose_lnaa" in clamps else 0.5
    homa_ir_obs = meal_load * p["insulin_resistance"]
    lat1_trp_influx = 1.5 / (0.6 + 0.4 * p["insulin_resistance"])

    # 6. Sleep deprivation / PSG clamp -> Metabolite accumulation, retrieval, E/I sigma
    sleep_drive = 1.8 if "sleep_deprivation" in clamps else 0.6
    metabolite_load = sleep_drive / (0.4 + p["sws_clearance"])
    glu_spillover = (0.3 + 0.4 * qa_ka_ratio) / (0.4 + p["eaat2_capacity"])
    t3_brain = (p["se_intake"] / (0.5 + p["se_intake"])) / (1.0 + 0.2 * il6_level)

    # 7. Interoceptive heartbeat detection, prior rigidity, mood barrier depth & psychosocial evaluation
    hb_acc = (p["pi_intero"] * p["vagal_tone"]) / (0.5 + p["pi_intero"] * p["vagal_tone"])
    open_hidden_gap = 0.5 * p["pi_prior"]
    barrier_depth = p["barrier_lam"] * (1.2 - 0.5 * p["social_support"])
    social_eval = 0.8 * p["social_support"]

    x = np.array([
        fe_obs, zn_obs, mg_obs, se_obs,
        serotonin_5ht, prt_bias, il6_level, qa_ka_ratio,
        dnic, pain_report, homa_ir_obs, lat1_trp_influx,
        metabolite_load, glu_spillover, t3_brain,
        hb_acc, open_hidden_gap, barrier_depth,
        social_eval
    ], dtype=float)

    if rng is not None and noise_sd > 0.0:
        x = x + rng.normal(0.0, noise_sd, size=x.shape)
    return x
