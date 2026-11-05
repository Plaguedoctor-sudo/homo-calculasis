import pytest
from control.requisite import assert_requisite, variety
from core.whole_body_runner import WholeBodyPatient
from endocrine.axes import HPTAxis
from environment.exposome import Exposome
from neuroglia.astrocyte import TripartiteSynapse
from nutrition.bioenergetics import MitochondrialBioenergetics
from nutrition.macronutrients import MacronutrientState
from psychosocial.models import PsychosocialState


def test_expanded_requisite_variety():
    r, e = variety()
    assert r == 23 and e == 21
    assert_requisite()


def test_fernstrom_wurtman_lnaa_competition():
    high_prot_low_carb = MacronutrientState(carb_intake=0.2, protein_lnaa=3.0)
    carb_buffered = MacronutrientState(carb_intake=1.8, protein_lnaa=1.0)
    assert carb_buffered.bbb_trp_uptake_factor(1.0) > high_prot_low_carb.bbb_trp_uptake_factor(1.0)


def test_ketogenic_bypass_under_insulin_resistance():
    m_glyc = MitochondrialBioenergetics()
    m_keto = MitochondrialBioenergetics()
    atp_glyc = m_glyc.step(1.0, glucose=1.0, ketone_bhb=0.05, insulin_resistance=4.0, cortisol=0.5, il6=0.3, gpx_activity=1.0)
    atp_keto = m_keto.step(1.0, glucose=1.0, ketone_bhb=2.50, insulin_resistance=4.0, cortisol=0.5, il6=0.3, gpx_activity=1.0)
    assert atp_keto > 1.3 * atp_glyc


def test_social_rejection_drives_sterile_inflammation():
    p_safe = WholeBodyPatient()
    p_rej = WholeBodyPatient()
    p_rej.psy.social_rejection_threat = 2.5
    p_rej.psy.relational_support = 0.1
    for _ in range(100):
        p_safe.step(0.1)
        p_rej.step(0.1)
    assert p_rej.axis.cyto.il6 > 1.4 * p_safe.axis.cyto.il6
    assert p_rej.axis.trp.qa_ka > p_safe.axis.trp.qa_ka


def test_joiner_interpersonal_synergy():
    tb_only = PsychosocialState(thwarted_belonging=1.0, perceived_burden=0.0).interpersonal_mood_drift
    pb_only = PsychosocialState(thwarted_belonging=0.0, perceived_burden=1.0).interpersonal_mood_drift
    both = PsychosocialState(thwarted_belonging=1.0, perceived_burden=1.0).interpersonal_mood_drift
    assert both > (tb_only + pb_only) * 2.0


def test_astrocytic_eaat2_and_selenium_hpt():
    glia = TripartiteSynapse()
    for _ in range(100):
        glia.step(0.1, synaptic_drive=1.2, qa=2.5, il6=2.0, gs_activity=0.2, atp_flux=0.4, mg_free=0.2)
    assert glia.ei_ratio_shift > 1.15
    assert glia.bdnf < 0.7

    hpt_norm = HPTAxis()
    hpt_low_se = HPTAxis()
    for _ in range(200):
        hpt_norm.step(0.2, se_free=1.0)
        hpt_low_se.step(0.2, se_free=0.05)
    assert hpt_low_se.t3_brain < 0.5 * hpt_norm.t3_brain
