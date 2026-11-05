"""Tryptophan split. 5-HT arm vs kynurenine arm; KYN -> QA (excitotoxic)
or KA (protective). The QA/KA ratio is the readout.

Conserved: intake -> plasma -> {brain -> 5-HT -> sink, KYN -> QA/KA -> sink}.
"""

from dataclasses import dataclass


@dataclass
class TrpModel:
    trp_plasma: float = 1.0
    trp_brain: float = 0.3
    serotonin: float = 1.0
    kyn: float = 1.0
    hk: float = 0.5
    qa: float = 1.0
    ka: float = 1.0
    s_5ht: float = 0.0
    s_qa: float = 0.0
    s_ka: float = 0.0
    intake: float = 1.0
    k_uptake: float = 0.4
    k_tph: float = 0.05
    k_ido: float = 0.05
    k_kmo: float = 0.3
    k_kat: float = 0.3
    k_deg: float = 0.2

    def total(self) -> float:
        return (self.trp_plasma + self.trp_brain + self.serotonin + self.kyn
                + self.hk + self.qa + self.ka + self.s_5ht + self.s_qa + self.s_ka)

    def step(self, dt: float, ido: float, kmo: float, kat: float, tph: float) -> None:
        upt = self.k_uptake * self.trp_plasma
        ido_f = self.k_ido * ido * self.trp_plasma
        tph_f = self.k_tph * tph * self.trp_brain
        kmo_f = self.k_kmo * kmo * self.kyn
        kat_f = self.k_kat * kat * (self.kyn / (0.5 + self.kyn))
        hk_f = 0.5 * self.hk
        deg_5ht = self.k_deg * self.serotonin
        deg_qa = self.k_deg * self.qa
        deg_ka = self.k_deg * self.ka

        self.trp_plasma += dt * (self.intake - upt - ido_f)
        self.trp_brain += dt * (upt - tph_f)
        self.serotonin += dt * (tph_f - deg_5ht)
        self.s_5ht += dt * deg_5ht
        self.kyn += dt * (ido_f - kmo_f - kat_f)
        self.hk += dt * (kmo_f - hk_f)
        self.qa += dt * (hk_f - deg_qa)
        self.s_qa += dt * deg_qa
        self.ka += dt * (kat_f - deg_ka)
        self.s_ka += dt * deg_ka

        for f in ("trp_plasma", "trp_brain", "serotonin", "kyn", "hk", "qa", "ka"):
            setattr(self, f, max(0.0, getattr(self, f)))

    @property
    def qa_ka(self) -> float:
        return self.qa / (self.ka + 1e-6)
