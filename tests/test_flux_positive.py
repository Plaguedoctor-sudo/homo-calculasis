import numpy as np
import pytest
from scipy.integrate import solve_ivp

from core.compartments import load_geometry
from core.mass_balance import bolus, flux_magnitude, rhs_factory, stoichiometry


@pytest.fixture(scope="module")
def g():
    return load_geometry()


def test_all_perfusion_positive(g):
    assert (g.Q > 0).all(), "dead compartment: organ with zero perfusion"
    assert g.Q_tot > 0


def test_all_volumes_and_partitions_positive(g):
    assert (g.V > 0).all()
    assert (g.K > 0).all()


def test_flux_never_vanishes_during_simulation(g):
    """A6: dissipative structure must maintain nonzero flux.

    Baseline measurement:
      min flux = 1.270e+03 L/h
    """
    rhs = rhs_factory(g)
    y0 = bolus(g, np.zeros(g.n_state), mg=100.0)
    sol = solve_ivp(rhs, (0.0, 600.0), y0, method="LSODA",
                    rtol=1e-8, atol=1e-12,
                    t_eval=np.linspace(1.0, 600.0, 400))
    assert sol.success
    fluxes = np.array([flux_magnitude(g, y) for y in sol.y.T])
    print(f"\nmin flux = {fluxes.min():.3e} L/h")
    assert fluxes.min() > 0.0
    assert np.isfinite(fluxes).all()


def test_requisite_variety_gate_still_passes():
    from control.requisite import assert_requisite
    assert_requisite()
