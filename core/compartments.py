"""Frozen body geometry. No fitted parameters live here (A1, A6)."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import yaml

GEOM = pathlib.Path(__file__).with_name("organs.yaml")
FLOW_TOL = 1e-9


@dataclass(frozen=True)
class Geometry:
    names: tuple[str, ...]
    V: np.ndarray        # litres, all compartments, incl. blood + lung
    Q: np.ndarray        # L/h, tissue perfusion
    K: np.ndarray        # partition coefficients
    Q_tot: float         # cardiac output, L/h
    idx_lung: int
    idx_art: int
    idx_ven: int

    @property
    def n_tissue(self) -> int:
        return len(self.names)

    @property
    def n_state(self) -> int:
        return self.n_tissue + 3

    def total_mass(self, y: np.ndarray) -> float:
        return float(np.dot(self.V, y))


def load_geometry(path: pathlib.Path = GEOM) -> Geometry:
    with path.open() as f:
        spec = yaml.safe_load(f)

    bw = float(spec["body_weight_kg"])
    q_tot = float(spec["cardiac_output_L_per_h"])
    organs = spec["organs"]
    blood = spec["blood"]

    flow_frac = np.array([float(o["flow_frac"]) for o in organs])
    if abs(flow_frac.sum() - 1.0) > FLOW_TOL:
        raise ValueError(
            f"tissue flow fractions sum to {flow_frac.sum():.12f}, must be 1.0"
        )

    vol_frac = np.array([float(o["vol_frac"]) for o in organs])
    blood_L = (
        float(blood["arterial_L"])
        + float(blood["venous_L"])
        + float(blood["lung_tissue_L"])
    )
    if vol_frac.sum() * bw + blood_L > bw * 1.02:
        raise ValueError("organ + blood volume exceeds body weight")

    names = tuple(o["name"] for o in organs)
    V_tissue = vol_frac * bw
    Q = flow_frac * q_tot
    K = np.array([float(o["partition"]) for o in organs])

    V = np.concatenate([V_tissue, [float(blood["lung_tissue_L"]),
                                   float(blood["arterial_L"]),
                                   float(blood["venous_L"])]])
    K_full = np.concatenate([K, [float(blood["lung_partition"]), 1.0, 1.0]])

    n = len(names)
    return Geometry(
        names=names, V=V, Q=Q, K=K_full, Q_tot=q_tot,
        idx_lung=n, idx_art=n + 1, idx_ven=n + 2,
    )
