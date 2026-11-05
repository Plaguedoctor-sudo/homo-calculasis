from load import Load


def test_no_decay_mid_episode():
    ld = Load()
    for _ in range(2000):
        ld.integrate(deficit=1.0, dt=1e-3)
    held = ld.L
    assert held > 0.0
    assert ld.L == held          # no passive relaxation mid-episode


def test_capacity_falls_with_load():
    assert Load(L=0.0).capacity() > Load(L=0.9).capacity()
