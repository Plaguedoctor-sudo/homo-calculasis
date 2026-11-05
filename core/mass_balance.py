from __future__ import annotations

import numpy as np

from .compartments import Clearance, Geometry


def stoichiometry(g: Geometry, cl: Clearance | None = None) -> np.ndarray:
    cl = cl or Clearance(0.0, 0.0, 0.0, 0.0, 0.0)
    n = g.n_tissue
    gut, liver, kidney = g.idx_gut, g.idx_liver, g.idx_kidney
    art, ven, lung, lumen = g.idx_art, g.idx_ven, g.idx_lung, g.idx_lumen
    m_sink, u_sink, f_sink = g.idx_sink_metab, g.idx_sink_urine, g.idx_sink_feces
    S = np.zeros((g.n_state, g.n_state))

    for k in range(n):
        S[k, art] += g.Q[k] / g.V[k]
        S[k, k] -= g.Q[k] / (g.V[k] * g.K[k])

    # Portal rerouting: gut outflow -> liver
    S[liver, gut] += g.Q[gut] / (g.V[liver] * g.K[gut])
    q_liver_out = g.Q[liver] + g.Q[gut]
    # Remove standard hepatic-artery-only outflow placed in loop, replace with combined outflow
    S[liver, liver] += g.Q[liver] / (g.V[liver] * g.K[liver])
    S[liver, liver] -= q_liver_out / (g.V[liver] * g.K[liver])

    for k in range(n):
        if k in (gut, liver):
            continue
        S[ven, k] += g.Q[k] / (g.V[ven] * g.K[k])
    S[ven, liver] += q_liver_out / (g.V[ven] * g.K[liver])
    S[ven, ven] -= g.Q_tot / g.V[ven]

    S[lung, ven] += g.Q_tot / g.V[lung]
    S[lung, lung] -= g.Q_tot / (g.V[lung] * g.K[lung])
    S[art, lung] += g.Q_tot / g.V[art]
    S[art, art] -= g.Q_tot / g.V[art]

    # Volume-weighted boundary transfers (guarantees V^T @ S == 0)
    S[gut, lumen] += cl.k_absorb * (g.V[lumen] / g.V[gut])
    S[f_sink, lumen] += cl.k_feces_unabsorbed * (g.V[lumen] / g.V[f_sink])
    S[lumen, lumen] -= (cl.k_absorb + cl.k_feces_unabsorbed)

    S[lumen, liver] += cl.k_bile * (g.V[liver] / g.V[lumen])
    S[liver, liver] -= cl.k_bile

    S[m_sink, liver] += cl.cl_int_hepatic * (g.V[liver] / g.V[m_sink])
    S[liver, liver] -= cl.cl_int_hepatic

    S[u_sink, kidney] += cl.cl_renal * (g.V[kidney] / g.V[u_sink])
    S[kidney, kidney] -= cl.cl_renal
    return S


def assert_conservative(g: Geometry, cl: Clearance | None = None, tol: float = 1e-10) -> np.ndarray:
    S = stoichiometry(g, cl)
    resid = np.abs(g.V @ S).max()
    if resid > tol:
        raise AssertionError(f"stoichiometry not conservative: max|V^T S|={resid:.3e}")
    return S


def rhs_factory(g: Geometry, cl: Clearance | None = None):
    S = assert_conservative(g, cl)
    return lambda t, y: S @ y


def flux_magnitude(g: Geometry, y: np.ndarray) -> float:
    inflow = np.abs(g.Q * y[g.idx_art])
    outflow = np.abs(g.Q * y[: g.n_tissue] / g.K[: g.n_tissue])
    return float(inflow.sum() + outflow.sum())


def bolus(g: Geometry, y: np.ndarray, mg: float) -> np.ndarray:
    y = y.copy()
    y[g.idx_ven] += mg / g.V[g.idx_ven]
    return y


def dose(g: Geometry, mg: float) -> np.ndarray:
    y = np.zeros(g.n_state)
    y[g.idx_lumen] = mg / g.V[g.idx_lumen]
    return y
