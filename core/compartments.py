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
    V: np.ndarray
    Q: np.ndarray
    K: np.ndarray
    Q_tot: float
    idx_gut: int
    idx_liver: int
    idx_kidney: int
    idx_lung: int
    idx_art: int
    idx_ven: int
    idx_lumen: int
    idx_sink_metab: int
    idx_sink_urine: int
    idx_sink_feces: int

    @property
    def n_tissue(self) -> int:
        return len(self.names)

    @property
    def n_state(self) -> int:
        return self.n_tissue + 7

    @property
    def n_sink(self) -> int:
        return 3

    def body_mass(self, y: np.ndarray) -> float:
        return float(np.dot(self.V[: self.n_tissue + 3], y[: self.n_tissue + 3]))

    def sink_mass(self, y: np.ndarray) -> float:
        s = self.n_tissue + 4
        return float(np.dot(self.V[s:], y[s:]))

    def lumen_mass(self, y: np.ndarray) -> float:
        return float(self.V[self.idx_lumen] * y[self.idx_lumen])

    def total_mass(self, y: np.ndarray) -> float:
        return self.body_mass(y) + self.lumen_mass(y) + self.sink_mass(y)


@dataclass(frozen=True)
class Clearance:
    cl_int_hepatic: float = 0.8
    cl_renal: float = 0.25
    k_absorb: float = 1.2
    k_feces_unabsorbed: float = 0.15
    k_bile: float = 0.05


def load_geometry(path: pathlib.Path = GEOM) -> Geometry:
    with path.open() as f:
        spec = yaml.safe_load(f)
    bw = float(spec["body_weight_kg"])
    q_tot = float(spec["cardiac_output_L_per_h"])
    organs = spec["organs"]
    blood = spec["blood"]

    flow_frac = np.array([float(o["flow_frac"]) for o in organs])
    if abs(flow_frac.sum() - 1.0) > FLOW_TOL:
        raise ValueError(f"tissue flow fractions sum to {flow_frac.sum():.12f}, must be 1.0")

    vol_frac = np.array([float(o["vol_frac"]) for o in organs])
    blood_L = (float(blood["arterial_L"]) + float(blood["venous_L"]) + float(blood["lung_tissue_L"]))
    if vol_frac.sum() * bw + blood_L > bw * 1.02:
        raise ValueError("organ + blood volume exceeds body weight")

    names = tuple(o["name"] for o in organs)
    V_tissue = vol_frac * bw
    Q = flow_frac * q_tot
    K = np.array([float(o["partition"]) for o in organs])

    n = len(names)
    lumen = float(spec.get("lumen_L", 0.50))
    sink = float(spec.get("sink_L", 1.0))

    V = np.concatenate([
        V_tissue,
        np.array([float(blood["lung_tissue_L"]), float(blood["arterial_L"]),
                  float(blood["venous_L"]), lumen, sink, sink, sink])
    ])
    K_full = np.concatenate([K, np.array([float(blood["lung_partition"]), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])])

    return Geometry(
        names=names, V=V, Q=Q, K=K_full, Q_tot=q_tot,
        idx_gut=names.index("gut"), idx_liver=names.index("liver"),
        idx_kidney=names.index("kidney"), idx_lung=n, idx_art=n + 1,
        idx_ven=n + 2, idx_lumen=n + 3, idx_sink_metab=n + 4,
        idx_sink_urine=n + 5, idx_sink_feces=n + 6,
    )
