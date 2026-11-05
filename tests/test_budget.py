import numpy as np
import pytest
from scipy.integrate import solve_ivp

from core.compartments import Clearance, load_geometry
from core.mass_balance import dose, rhs_factory, stoichiometry


@pytest.fixture(scope="module")
def g():
    return load_geometry()


def test_still_conservative_with_clearance(g):
    S = stoichiometry(g, Clearance())
    assert np.abs(g.V @ S).max() < 1e-10


def test_conservative_with_zero_clearance(g):
    S = stoichiometry(g, Clearance(cl_int_hepatic=0, cl_renal=0,
                                   k_bile=0, k_feces_unabsorbed=0))
    assert np.abs(g.V @ S).max() < 1e-10


def test_budget_closes_over_24h(g):
    """Mass in == body + lumen + sinks. Nothing vanishes.

    Baseline measurement:
      budget drift = ~1e-12
    """
    rhs = rhs_factory(g, Clearance())
    y0 = dose(g, mg=100.0)
    m0 = g.total_mass(y0)

    sol = solve_ivp(rhs, (0.0, 24 * 3600), y0, method="LSODA",
                    rtol=1e-10, atol=1e-14)
    assert sol.success
    assert (sol.y >= -1e-10).all()

    drift = abs(g.total_mass(sol.y[:, -1]) - m0) / m0
    print(f"\nbudget drift = {drift:.3e}")
    assert drift < 1e-6


def test_body_mass_declines(g):
    rhs = rhs_factory(g, Clearance())
    sol = solve_ivp(rhs, (0.0, 48 * 3600), dose(g, 100.0),
                    method="LSODA", rtol=1e-9, atol=1e-13)
    body = [g.body_mass(y) for y in sol.y.T]
    peak = max(body)
    assert peak > 50.0
    assert body[-1] < 0.5 * peak


def test_sinks_accumulate_monotonically(g):
    rhs = rhs_factory(g, Clearance())
    sol = solve_ivp(rhs, (0.0, 24 * 3600), dose(g, 100.0),
                    method="LSODA", rtol=1e-9, atol=1e-13,
                    t_eval=np.linspace(60, 24 * 3600, 200))
    s = g.n_tissue + 4
    for j in range(3):
        series = sol.y[s + j]
        assert (np.diff(series) >= -1e-12).all(), f"sink {j} decreased"


def test_lumen_depletes_monotonically(g):
    rhs = rhs_factory(g, Clearance())
    sol = solve_ivp(rhs, (0.0, 12 * 3600), dose(g, 100.0),
                    method="LSODA", rtol=1e-9, atol=1e-13,
                    t_eval=np.linspace(1, 12 * 3600, 200))
    lumen = sol.y[g.idx_lumen]
    assert (np.diff(lumen) <= 1e-12).all()
    assert lumen[-1] < 0.01 * lumen[0]


def test_hepatic_clearance_scales_decline(g):
    def end_mass(cl_int):
        rhs = rhs_factory(g, Clearance(cl_int_hepatic=cl_int))
        sol = solve_ivp(rhs, (0.0, 24 * 3600), dose(g, 100.0),
                        method="LSODA", rtol=1e-9, atol=1e-13)
        return g.body_mass(sol.y[:, -1])

    assert end_mass(2.0) < end_mass(0.2)


def test_zero_clearance_leaves_no_elimination(g):
    cl = Clearance(cl_int_hepatic=0, cl_renal=0, k_feces_unabsorbed=0, k_bile=0)
    rhs = rhs_factory(g, cl)
    sol = solve_ivp(rhs, (0.0, 48 * 3600), dose(g, 100.0),
                    method="LSODA", rtol=1e-9, atol=1e-13)
    assert g.sink_mass(sol.y[:, -1]) < 1e-6
