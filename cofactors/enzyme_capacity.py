"""Enzyme capacity from cofactor occupancy. Saturating, bounded, monotone."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Enzyme:
    name: str
    v_max: float
    k_cof: float
    k_sub: float
    cofactor: str


TH = Enzyme("tyrosine_hydroxylase", v_max=1.0, k_cof=0.6, k_sub=0.3, cofactor="fe")
TPH = Enzyme("tryptophan_hydroxylase", v_max=1.0, k_cof=0.8, k_sub=0.4, cofactor="fe")
DBH = Enzyme("dopamine_beta_hydroxylase", v_max=1.0, k_cof=1.0, k_sub=0.5, cofactor="cu")
GS = Enzyme("glutamine_synthetase", v_max=1.0, k_cof=0.9, k_sub=0.5, cofactor="mn")
GPX = Enzyme("glutathione_peroxidase", v_max=1.0, k_cof=0.7, k_sub=0.3, cofactor="se")


def capacity(e: Enzyme, cof_free: float, bh4: float = 1.0, sub: float = 1.0) -> float:
    """Michaelis-Menten in cofactor and substrate, times BH4 saturation.

    Strictly in [0, v_max]; monotone in cof_free.
    """
    if cof_free < 0.0:
        raise ValueError("negative cofactor")
    cof_term = cof_free / (e.k_cof + cof_free)
    sub_term = sub / (e.k_sub + sub)
    return e.v_max * cof_term * sub_term * (bh4 / (0.5 + bh4))


def dopamine_synthesis(fe_free: float, bh4: float, tyr: float = 1.0) -> float:
    return capacity(TH, fe_free, bh4, tyr)


def serotonin_synthesis(fe_free: float, bh4: float, trp: float = 1.0) -> float:
    return capacity(TPH, fe_free, bh4, trp)


def noradrenaline_synthesis(cu_free: float, da: float = 1.0) -> float:
    return capacity(DBH, cu_free, 1.0, da)


def bh4_regeneration(methylation: float) -> float:
    """BH4 depends on the one-carbon pool; low methylation degrades regeneration."""
    return min(1.0, max(0.0, methylation))
