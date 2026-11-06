from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ParamSpec:
    name: str
    low: float
    high: float
    prior_mean: float
    prior_sd: float
    module: str

CORE_16_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("fe_intake",          0.10, 2.50, 1.00, 0.45, "cofactors"),
    ParamSpec("zn_intake",          0.10, 2.50, 1.00, 0.45, "cofactors"),
    ParamSpec("mg_intake",          0.10, 2.50, 1.00, 0.45, "cofactors"),
    ParamSpec("se_intake",          0.10, 2.50, 1.00, 0.45, "cofactors"),
    ParamSpec("scfa_production",    0.10, 2.00, 0.80, 0.35, "gut"),
    ParamSpec("vagal_tone",         0.10, 2.00, 0.80, 0.35, "control"),
    ParamSpec("ido_gain",           0.20, 3.00, 1.20, 0.55, "immune"),
    ParamSpec("pi_intero",          0.20, 3.00, 1.00, 0.50, "interoception"),
    ParamSpec("pi_prior",           0.20, 3.50, 1.10, 0.60, "interoception"),
    ParamSpec("reward_rho",         0.20, 2.00, 1.00, 0.40, "mood"),
    ParamSpec("barrier_lam",        0.10, 1.80, 0.60, 0.30, "mood"),
    ParamSpec("sws_clearance",      0.15, 1.50, 0.65, 0.25, "memory"),
    ParamSpec("pag_rvm_gain",       0.20, 2.50, 1.00, 0.45, "pain"),
    ParamSpec("insulin_resistance", 0.50, 4.00, 1.30, 0.65, "nutrition"),
    ParamSpec("social_support",     0.05, 1.00, 0.65, 0.22, "psychosocial"),
    ParamSpec("eaat2_capacity",     0.30, 2.50, 1.30, 0.45, "neuroglia"),
)

class ParameterSpace:
    """Manages prior sampling, physiological bounding, and vector-to-dict mapping."""
    def __init__(self, specs: tuple[ParamSpec, ...] = CORE_16_SPECS):
        self.specs = specs
        self.names = tuple(s.name for s in specs)
        self.low = np.array([s.low for s in specs], dtype=float)
        self.high = np.array([s.high for s in specs], dtype=float)
        self.mean = np.array([s.prior_mean for s in specs], dtype=float)
        self.sd = np.array([s.prior_sd for s in specs], dtype=float)

    @property
    def dim(self) -> int:
        return len(self.specs)

    def sample_prior(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        raw = rng.normal(loc=self.mean, scale=self.sd, size=(n, self.dim))
        return np.clip(raw, self.low, self.high)

    def clip(self, theta: np.ndarray) -> np.ndarray:
        return np.clip(theta, self.low, self.high)

    def to_dict(self, theta: np.ndarray) -> dict[str, float]:
        return {name: float(val) for name, val in zip(self.names, theta)}
