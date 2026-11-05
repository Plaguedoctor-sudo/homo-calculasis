import numpy as np
import pytest

from cofactors import enzyme_capacity as ec
from cofactors.model import CofactorModel
from cofactors.pools import Element, ElementParams


def run(m: CofactorModel, hours: float, dt: float = 0.05) -> None:
    for _ in range(int(hours / dt)):
        m.step(dt)


# --- conservation -------------------------------------------------------

@pytest.mark.parametrize("attr", ["fe", "zn", "cu", "mg", "se"])
def test_element_budget(attr):
    m = CofactorModel()
    el = getattr(m, attr)
    el.intake = 1.0
    t0 = el.total()
    run(m, 200.0)
    assert abs(el.total() - (t0 + 200.0 * 1.0)) < 1e-6


# --- regulatory return --------------------------------------------------

def test_iron_store_recovers_after_depletion():
    m = CofactorModel()
    m.fe.store = 0.05
    m.fe.plasma = 0.2
    m.fe.intake = 1.0
    run(m, 500.0)
    assert m.fe.store > 0.7, "store did not recover — absorption not upregulated"


def test_high_intake_plateaus_no_runaway():
    m = CofactorModel()
    m.fe.intake = 10.0
    run(m, 2000.0)
    first = m.fe.store
    run(m, 500.0)
    assert abs(m.fe.store - first) / first < 0.15, "store never plateaued"


# --- antagonism ---------------------------------------------------------

def test_zinc_load_suppresses_copper():
    base = CofactorModel(); base.zn.intake = 1.0; base.cu.intake = 1.0
    run(base, 400.0)
    loaded = CofactorModel(); loaded.zn.intake = 12.0; loaded.cu.intake = 1.0
    run(loaded, 400.0)
    assert loaded.cu.free(loaded.mt, 0.3) < 0.8 * base.cu.free(base.mt, 0.3)


def test_mt_rises_with_zinc():
    m = CofactorModel(); m.zn.intake = 10.0
    run(m, 200.0)
    assert m.mt > 0.5


# --- enzyme capacity ----------------------------------------------------

def test_th_capacity_monotone_and_bounded():
    vals = [ec.dopamine_synthesis(f, 1.0) for f in np.linspace(0, 10, 60)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert max(vals) <= ec.TH.v_max + 1e-12
    assert min(vals) >= 0.0


def test_zero_iron_halves_capacity():
    assert ec.dopamine_synthesis(0.0, 1.0) < 0.05 * ec.TH.v_max


def test_negative_cofactor_rejected():
    with pytest.raises(ValueError):
        ec.dopamine_synthesis(-0.1, 1.0)


def test_low_methylation_degrades_bh4():
    assert ec.bh4_regeneration(0.1) < ec.bh4_regeneration(1.0)


# --- integration --------------------------------------------------------

def test_iron_deficiency_reaches_enzyme_output():
    m = CofactorModel()
    m.fe.store = 0.02; m.fe.plasma = 0.05; m.fe.intake = 0.02
    run(m, 300.0)
    assert m.outputs()["dopamine_synthesis"] < 0.5


def test_inflammation_blocks_absorption():
    m = CofactorModel(); m.il6 = 8.0; m.fe.intake = 1.0
    run(m, 200.0)
    assert m.h > 0.5
    assert m.fe.store < 1.0
