from dataclasses import dataclass, field

from . import enzyme_capacity as ec
from .pools import Element, ElementParams
from .regulators import hepcidin, metallothionein

KD_CU = 0.3     # MT binds Cu tightly
KD_ZN = 2.0     # MT binds Zn loosely


@dataclass
class CofactorModel:
    fe: Element = field(default_factory=lambda: Element(
        "fe", p=ElementParams(hepcidin_sensitive=True, slough=0.02, k_renal=0.01)))
    zn: Element = field(default_factory=lambda: Element("zn"))
    cu: Element = field(default_factory=lambda: Element("cu", p=ElementParams(k_renal=0.05)))
    mg: Element = field(default_factory=lambda: Element("mg", p=ElementParams(k_renal=0.5)))
    se: Element = field(default_factory=lambda: Element("se", p=ElementParams(k_renal=0.3)))
    il6: float = 0.0
    h: float = 0.0
    mt: float = 0.0
    methylation: float = 1.0

    def step(self, dt: float) -> None:
        self.h = hepcidin(self.fe.store, self.il6, self.h, dt)
        self.mt = metallothionein(self.zn.free(self.mt, KD_ZN), self.il6, self.mt, dt)

        self.fe.step(dt, h=self.h)
        self.zn.step(dt, mt=self.mt, kd=KD_ZN)
        self.cu.step(dt, mt=self.mt, kd=KD_CU)
        self.mg.step(dt)
        self.se.step(dt)

    def outputs(self) -> dict:
        bh4 = ec.bh4_regeneration(self.methylation)
        fe_free = self.fe.free(self.mt, KD_ZN)
        cu_free = self.cu.free(self.mt, KD_CU)
        return {
            "dopamine_synthesis": ec.dopamine_synthesis(fe_free, bh4),
            "serotonin_synthesis": ec.serotonin_synthesis(fe_free, bh4),
            "noradrenaline_synthesis": ec.noradrenaline_synthesis(cu_free),
            "gpx_activity": ec.capacity(ec.GPX, self.se.free()),
        }
