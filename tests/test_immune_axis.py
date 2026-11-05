import numpy as np
import pytest

from immune.axis import ImmuneAxis


@pytest.fixture
def baseline():
    a = ImmuneAxis()
    a.run(400.0)
    return a


def test_tryptophan_budget():
    a = ImmuneAxis()
    t0 = a.trp.total()
    a.run(200.0)
    assert abs(a.trp.total() - (t0 + 200.0 * a.trp.intake)) < 1e-4


@pytest.mark.gono
def test_lps_raises_qa_ka(baseline):
    before = baseline.trp.qa_ka
    baseline.cyto.lps_challenge = 3.0
    baseline.run(200.0)
    after = baseline.trp.qa_ka
    print(f"\nLPS: qa_ka {before:.3f} -> {after:.3f}")
    assert after > 1.4 * before


@pytest.mark.gono
def test_zinc_lowers_qa_ka(baseline):
    before = baseline.trp.qa_ka
    baseline.cof.zn.intake = 10.0
    baseline.run(400.0)
    after = baseline.trp.qa_ka
    print(f"\nZn: qa_ka {before:.3f} -> {after:.3f}")
    assert after < 0.8 * before


def test_arms_are_independent(baseline):
    """QA and KA must move separately, else the ratio is just kynurenine."""
    q0, k0 = baseline.trp.qa, baseline.trp.ka
    baseline.cyto.lps_challenge = 3.0
    baseline.run(200.0)
    assert baseline.trp.qa / q0 > 1.2
    assert baseline.trp.ka / k0 < 1.3


def test_cholinergic_brake(baseline):
    hi = ImmuneAxis(); hi.run(300.0); hi.cyto.lps_challenge = 3.0; hi.run(60.0)
    lo = ImmuneAxis(); lo.run(300.0); lo.vagal.severed = True; lo.run(60.0)
    lo.cyto.lps_challenge = 3.0; lo.run(60.0)
    assert hi.cyto.tnf < lo.cyto.tnf


def test_vagotomy_raises_tnf(baseline):
    before = baseline.cyto.tnf
    baseline.vagal.severed = True
    baseline.run(200.0)
    assert baseline.cyto.tnf > before


def test_scfa_tightens_barrier(baseline):
    before = baseline.barrier.perm
    baseline.mb.n = np.array([1.2, 0.4])
    baseline.run(200.0)
    assert baseline.barrier.perm < before


def test_dysbiosis_leaks_lps(baseline):
    before = baseline.barrier.lps_plasma
    baseline.mb.n = np.array([0.1, 1.0])
    baseline.run(300.0)
    assert baseline.barrier.lps_plasma > before


def test_tryptophan_depletion_cuts_serotonin(baseline):
    before = baseline.trp.serotonin
    baseline.trp.intake = 0.05
    baseline.run(300.0)
    assert baseline.trp.serotonin < 0.6 * before


def test_qa_ka_finite_and_positive(baseline):
    baseline.cyto.lps_challenge = 10.0
    baseline.cof.zn.intake = 0.0
    baseline.run(300.0)
    assert np.isfinite(baseline.trp.qa_ka)
    assert baseline.trp.qa_ka > 0.0


def test_no_negative_states(baseline):
    baseline.trp.intake = 0.0
    baseline.run(400.0)
    for f in ("trp_plasma", "trp_brain", "serotonin", "kyn", "hk", "qa", "ka"):
        assert getattr(baseline.trp, f) >= 0.0
