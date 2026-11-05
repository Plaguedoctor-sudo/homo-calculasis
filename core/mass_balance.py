"""Circulatory mass balance as a linear stoichiometry matrix.

dC/dt = S @ C.  Conservation holds iff V^T S == 0, asserted at build.
Loop: tissues -> venous -> lung -> arterial -> tissues.
"""

from __future__ import annotations

import numpy as np

from .compartments import Geometry


def stoichiometry(g: Geometry) -> np.ndarray:
    n, art, ven, lung = g.n_tissue, g.idx_art, g.idx_ven, g.idx_lung
    S = np.zeros((g.n_state, g.n_state))

    for k in range(n):
        S[k, art] += g.Q[k] / g.V[k]              # inflow from arterial
        S[k, k] -= g.Q[k] / (g.V[k] * g.K[k])     # outflow to venous

    S[lung, ven] += g.Q_tot / g.V[lung]
    S[lung, lung] -= g.Q_tot / (g.V[lung] * g.K[lung])

    S[art, lung] += g.Q_tot / g.V[art]
    S[art, art] -= g.Q_tot / g.V[art]

    for k in range(n):
        S[ven, k] += g.Q[k] / (g.V[ven] * g.K[k])
    S[ven, ven] -= g.Q_tot / g.V[ven]

    return S


def assert_conservative(g: Geometry, tol: float = 1e-10) -> np.ndarray:
    """A1/A6 structural gate: volume-weighted column sums must vanish."""
    S = stoichiometry(g)
    resid = np.abs(g.V @ S).max()
    if resid > tol:
        raise AssertionError(f"stoichiometry not conservative: max|V^T S|={resid:.3e}")
    return S


def rhs_factory(g: Geometry, clearance: np.ndarray | None = None):
    S = assert_conservative(g)
    if clearance is None:
        return lambda t, y: S @ y
    C = np.zeros(g.n_state)
    C[: g.n_tissue] = clearance
    return lambda t, y: S @ y - C * y


def flux_magnitude(g: Geometry, y: np.ndarray) -> float:
    """Total absolute transcapillary flux. Must stay > 0 (A6)."""
    inflow = np.abs(g.Q * y[g.idx_art])
    outflow = np.abs(g.Q * y[: g.n_tissue] / g.K[: g.n_tissue])
    return float(inflow.sum() + outflow.sum())


def bolus(g: Geometry, y: np.ndarray, mg: float) -> np.ndarray:
    y = y.copy()
    y[g.idx_ven] += mg / g.V[g.idx_ven]
    return y
