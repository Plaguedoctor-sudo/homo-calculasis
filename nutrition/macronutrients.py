from __future__ import annotations
from dataclasses import dataclass

@dataclass
class MacronutrientState:
    """Dietary macronutrient partitioning, LAT1 amino-acid competition, and PUFA membrane tone."""
    carb_intake: float = 1.0       # relative glycemic intake
    protein_lnaa: float = 1.0      # plasma BCAA/LNAA pool (Val, Leu, Ile, Phe, Tyr)
    omega3_epa_dha: float = 1.0    # membrane EPA + DHA index
    omega6_aa: float = 2.0         # membrane arachidonic acid (AA) index
    ketone_bhb: float = 0.1        # beta-hydroxybutyrate (mM equivalent)
    insulin_resistance: float = 1.0  # 1.0 = normal sensitivity; > 2.0 = metabolic syndrome

    @property
    def effective_insulin(self) -> float:
        """Effective insulin signaling clearing LNAA into muscle."""
        raw_ins = 0.3 + 0.7 * self.carb_intake
        return float(raw_ins / (0.5 + 0.5 * self.insulin_resistance))

    def lnaa_plasma(self) -> float:
        """Muscle BCAA clearance reduces plasma LNAA pool proportionally to insulin sensitivity."""
        return float(self.protein_lnaa / (0.4 + 0.6 * self.effective_insulin))

    def bbb_trp_uptake_factor(self, trp_plasma: float = 1.0, km: float = 0.5, ki: float = 1.0) -> float:
        """Fernstrom-Wurtman LAT1 transporter saturation with competitive LNAA inhibition."""
        lnaa = self.lnaa_plasma()
        apparent_km = km * (1.0 + lnaa / ki)
        # Normalized so baseline (trp=1, lnaa=1, km=0.5, ki=1) returns ~1.0
        raw = trp_plasma / (apparent_km + trp_plasma)
        return float(raw * 2.0)

    @property
    def spm_resolution_gain(self) -> float:
        """Specialized pro-resolving mediator (resolvins/protectins) gain from EPA+DHA / AA ratio."""
        ratio = self.omega3_epa_dha / (0.2 + self.omega6_aa)
        return float(0.5 + 1.0 * (ratio / (0.5 + ratio)))
