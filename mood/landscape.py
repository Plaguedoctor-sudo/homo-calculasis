from dataclasses import dataclass

import numpy as np

from load import Load


@dataclass
class Params:
    a: float = 1.0
    b: float = 1.0
    c: float = 0.45
    lam: float = 0.55
    sigma: float = 0.02
    dt: float = 1e-3
    t_max: float = 60.0


def dU(m: float, c: float, barrier: float, a: float = 1.0, b: float = 1.0) -> float:
    # U(m) = a m^4/4 - b m^2/2 + c m + lam*L*m ; dU/dm:
    return a * m**3 - b * m + c + barrier


def run_arm(start: float, drive: float, load: Load, p: Params,
            seed: int, target: float, sign: int) -> tuple[int, np.ndarray]:
    """Integrate one arm until m crosses `target` in direction `sign`."""
    rng = np.random.default_rng(seed)
    m = start
    traj = [m]
    steps = int(p.t_max / p.dt)
    for i in range(steps):
        dt_eff = p.dt * load.capacity() if drive < 0 else p.dt
        noise = p.sigma * np.sqrt(dt_eff) * rng.standard_normal()
        m = m - dU(m, drive, load.barrier_gain(p.lam)) * dt_eff + noise
        traj.append(m)
        if drive > 0:
            load.integrate(abs(m - start) * 0.03, dt_eff)
        if sign * (m - target) < 0.0:
            return i, np.asarray(traj)
    return steps, np.asarray(traj)


def onset(p: Params, seed: int = 0) -> tuple[int, float]:
    load = Load()
    steps, _ = run_arm(start=1.0, drive=+p.c, load=load, p=p,
                       seed=seed, target=0.0, sign=+1)
    return steps, load.L


def recovery(p: Params, L0: float, seed: int = 0) -> int:
    load = Load(L=L0)
    steps, _ = run_arm(start=-1.0, drive=-p.c, load=load, p=p,
                       seed=seed, target=0.0, sign=-1)
    return steps
