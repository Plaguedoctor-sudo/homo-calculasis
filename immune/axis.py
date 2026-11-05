"""Couples cofactors, gut, immune, and the tryptophan split."""

from dataclasses import dataclass, field

from cofactors.model import CofactorModel
from gut.barrier import Barrier
from gut.microbiota import Microbiota
from gut.vagal import Vagal
from .cytokines import Cytokines
from .kynurenine import TrpModel


@dataclass
class ImmuneAxis:
    cof: CofactorModel = field(default_factory=CofactorModel)
    mb: Microbiota = field(default_factory=Microbiota)
    barrier: Barrier = field(default_factory=Barrier)
    vagal: Vagal = field(default_factory=Vagal)
    cyto: Cytokines = field(default_factory=Cytokines)
    trp: TrpModel = field(default_factory=TrpModel)
    cortisol: float = 0.5

    def step(self, dt: float) -> None:
        self.mb.step(dt)
        self.barrier.step(dt, self.mb.scfa, self.cortisol)
        self.vagal.step(dt, self.mb.scfa, self.barrier.lps_plasma,
                        self.cyto.pyretic)
        self.cyto.step(dt, self.barrier.lps_plasma, self.vagal.efferent)
        self.cof.il6 = self.cyto.il6
        self.cof.step(dt)

        out = self.cof.outputs()
        ido = 0.2 + 1.5 * self.cyto.il6 / (1.0 + self.cyto.il6)
        kmo = 0.3 + 1.2 * self.cyto.il6 / (1.0 + self.cyto.il6)
        kat = 0.2 + 1.2 * self.cof.zn.free(self.cof.mt, 2.0) / (
            1.0 + self.cof.zn.free(self.cof.mt, 2.0))
        self.trp.step(dt, ido, kmo, kat, out["serotonin_synthesis"])

    def run(self, hours: float, dt: float = 0.05) -> None:
        for _ in range(int(hours / dt)):
            self.step(dt)
