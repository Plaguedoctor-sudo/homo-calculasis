import numpy as np
import pytest
from scipy.integrate import solve_ivp

from core.compartments import load_geometry
from core.mass_balance import assert_conservative, bolus, rhs_factory, stoichiometry


@pytest.fixture(scope="module")
def g():
    return load_geometry()


def test_stoichiometry_is_conservative(g):
    """Structural: V^T S == 0. Holds for any params, any integrator."""
    S = stoichiometry(g)
    assert np.abs(g.V @ S).max() < 1e-10


def test_all_diagonal_entries_nonpositive(g):
    S = stoichiometry(g)
    assert (np.diag(S) <= 0).all()


def test_numerical_closure_over_60s(g):
    """Mass drift under 1e-6 relative with zero clearance.

    Baseline measurement:
      mass drift over 3600 s = 5.039e-13
    """
    rhs = rhs_factory(g)
    y0 = bolus(g, np.zeros(g.n_state), mg=100.0)
    m0 = g.total_mass(y0)

    sol = solve_ivp(rhs, (0.0, 3600.0), y0, method="LSODA",
                    rtol=1e-9, atol=1e-12, dense_output=False)
    assert sol.success

    for y in sol.y.T:
        assert (y >= -1e-12).all(), "negative concentration"
    m_end = g.total_mass(sol.y[:, -1])
    drift = abs(m_end - m0) / m0
    print(f"\nmass drift over 3600 s = {drift:.3e}")
    assert drift < 1e-6


def test_flow_fractions_validated():
    import pathlib, tempfile, yaml
    from core.compartments import GEOM, load_geometry as lg
    spec = yaml.safe_load(GEOM.read_text())
    spec["organs"][0]["flow_frac"] += 0.1
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(spec, f)
        p = pathlib.Path(f.name)
    with pytest.raises(ValueError, match="must be 1.0"):
        lg(p)
