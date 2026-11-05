"""Trace-element pools: saturable absorption, storage, excretion.

Units are relative; baselines are 1.0. Absolute calibration is an S1 task.
Iron has NO regulated excretion — loss is fixed sloughing only. That is why
overload is a distinct failure mode and why `slough` is not a tuning knob.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ElementParams:
    a_max: float = 1.0        # max absorption, units/h
    k_rep: float = 1.0        # store-repression half-saturation
    k_st: float = 0.9         # plasma -> store
    k_mob: float = 0.15       # store -> plasma
    k_use: float = 0.4        # functional utilization (returns to store)
    k_renal: float = 0.2      # urinary loss
    slough: float = 0.01      # fixed loss (iron: obligate)
    hepcidin_sensitive: bool = False


@dataclass
class Element:
    name: str
    lumen: float = 1.0
    plasma: float = 1.0
    store: float = 1.0
    excreta: float = 0.0
    intake: float = 0.0
    p: ElementParams | None = None

    def __post_init__(self):
        if self.p is None:
            self.p = ElementParams()

    def total(self) -> float:
        return self.lumen + self.plasma + self.store + self.excreta

    def free(self, mt: float = 0.0, kd: float = 1.0) -> float:
        """MT-bound fraction removed. kd small => tight binding."""
        return self.plasma / (1.0 + mt / kd)

    def rhs(self, h: float = 0.0, mt: float = 0.0, kd: float = 1.0) -> dict:
        p = self.p
        free = self.free(mt, kd)
        block = max(0.0, 1.0 - 1.2 * h) if p.hepcidin_sensitive else 1.0
        lumen_avail = self.lumen / (1.0 + self.lumen)
        absorb = p.a_max * lumen_avail / (p.k_rep + 0.1 * self.store) * block
        if self.name == "zn":
            absorb += 0.02 * self.lumen / (1.0 + 0.005 * self.lumen)
        absorb = min(self.lumen * 10.0, absorb)

        to_store = p.k_st * self.plasma
        mob = p.k_mob * self.store
        use = p.k_use * free
        renal = p.k_renal * self.plasma
        slough_loss = p.slough * self.store

        return {
            "lumen": self.intake - absorb,
            "plasma": absorb - to_store + mob - use - renal,
            "store": to_store - mob + use - slough_loss,
            "excreta": renal + slough_loss,
        }

    def step(self, dt: float, **kw) -> None:
        d = self.rhs(**kw)
        self.lumen = max(0.0, self.lumen + dt * d["lumen"])
        self.plasma = max(0.0, self.plasma + dt * d["plasma"])
        self.store = max(0.0, self.store + dt * d["store"])
        self.excreta += dt * d["excreta"]
