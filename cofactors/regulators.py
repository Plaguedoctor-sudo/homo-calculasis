"""Hepcidin and metallothionein. Both are IL-6 driven — the immune bridge."""

TAU_H = 6.0
TAU_MT = 3.0


def hepcidin(store: float, il6: float, h: float, dt: float) -> float:
    """Inflammation and iron load both raise hepcidin; hepcidin blocks ferroportin."""
    target = il6 / (2.0 + il6) + store / (2.0 + store)
    target = min(target, 1.0)
    return h + dt * (target - h) / TAU_H


def metallothionein(zn_free: float, il6: float, mt: float, dt: float) -> float:
    """Zn and IL-6 both induce MT. MT then sequesters Cu (see Kd_CU)."""
    target = zn_free / (3.0 + zn_free) + il6 / (2.0 + il6)
    return mt + dt * (min(target, 1.0) - mt) / TAU_MT
