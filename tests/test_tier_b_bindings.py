import numpy as np
import pytest

from core.circadian_criticality import CircadianClock, CorticalCriticality
from interoception.allostatic_prior import AllostaticPrior
from interoception.placebo import open_hidden_response
from interoception.precision import InteroceptiveChannel, categorize_emotion, kamin_blocking_step
from memory.encode_consolidate import MemorySystem
from memory.sleep import SleepGlymphatic
from mood.early_warning import rolling_csd_metrics
from mood.opponent import HelplessnessController, OpponentProcess
from pain.descending import inferred_pain_report, pag_rvm_drive
from pain.gate import DorsalHornGate
from params.identifiability import profile_likelihood_1d


def test_blocking_precision():
    pi = 1.0
    for _ in range(25):
        pi = kamin_blocking_step(pi, predictedness=1.0, alpha=0.25)
    assert pi < 1e-3


def test_heartbeat_load_and_vocab_shift():
    ch = InteroceptiveChannel(pi_intero=2.0)
    assert ch.heartbeat_accuracy(0.8) > InteroceptiveChannel(pi_intero=0.3).heartbeat_accuracy(0.8)
    low_gran = categorize_emotion(1.0, concept_granularity=0.2, pi_prior=1.0)
    high_gran = categorize_emotion(1.0, concept_granularity=3.0, pi_prior=1.0)
    assert low_gran["y_intero_unchanged"] == high_gran["y_intero_unchanged"]
    assert high_gran["differentiated_affect"] > low_gran["differentiated_affect"]


def test_early_care_and_open_hidden_placebo():
    ch = InteroceptiveChannel()
    pi_stressed = ch.calibrate_early_life(caregiving_reliability=0.1, stress_load=1.5)
    assert pi_stressed > 1.5
    assert open_hidden_response(1.0, +1.0, pi_stressed, is_open=True) > open_hidden_response(1.0, +1.0, pi_stressed, is_open=False)
    assert open_hidden_response(1.0, -1.0, pi_stressed, is_open=True) < open_hidden_response(1.0, +1.0, pi_stressed, is_open=False)


def test_pain_gate_expectation_and_hrv_dnic():
    gate = DorsalHornGate()
    d_low = pag_rvm_drive(vagal_hrv=0.1, serotonin=0.5, noradrenaline=0.5)
    d_high = pag_rvm_drive(vagal_hrv=1.2, serotonin=0.5, noradrenaline=0.5)
    assert gate.step(0.8, 0.2, d_high) < gate.step(0.8, 0.2, d_low)
    assert inferred_pain_report(0.5, prior_expectation=1.5, pi_noci=1.0, pi_prior=2.0) > 0.8


def test_sleep_clearance_and_sws_density():
    sg_good = SleepGlymphatic(metabolite_load=1.0, synaptic_weight_sum=1.5)
    sg_frag = SleepGlymphatic(metabolite_load=1.0, synaptic_weight_sum=1.5)
    for _ in range(50):
        sg_good.step(0.1, sws_density=0.9)
        sg_frag.step(0.1, sws_density=0.1)
    assert sg_good.metabolite_load < sg_frag.metabolite_load
    assert sg_good.synaptic_weight_sum < sg_frag.synaptic_weight_sum
    mem = MemorySystem()
    enc_acc, ret_acc = mem.retrieval_accuracy(metabolite_load=sg_frag.metabolite_load)
    assert ret_acc < enc_acc


def test_criticality_and_phase_shift():
    crit = CorticalCriticality(ei_ratio=1.0)
    lesioned = CorticalCriticality(ei_ratio=1.4)
    assert abs(crit.branching_ratio - 1.0) < abs(lesioned.branching_ratio - 1.0)
    clock = CircadianClock()
    for _ in range(40):
        clock.step(0.5, light_phase=0.0, phase_shift_jetlag=2.5)
    assert clock.desync_load > 0.5


def test_opponent_helplessness_and_identifiability():
    opp = OpponentProcess()
    first = opp.step(1.0, dt=0.5)
    for _ in range(40):
        later = opp.step(1.0, dt=0.5)
    assert later < first
    hc = HelplessnessController()
    for _ in range(20):
        hc.step(action_succeeded=False)
    assert hc.effort < 0.35
    _, ident_ok = profile_likelihood_1d(lambda th: float((th[0] - 2.0) ** 2), np.array([2.0, 1.0]), 0, np.linspace(0, 4, 20))
    _, ident_flat = profile_likelihood_1d(lambda th: float((th[0] - 2.0) ** 2), np.array([2.0, 1.0]), 1, np.linspace(0, 4, 20))
    assert ident_ok and not ident_flat
