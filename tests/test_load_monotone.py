from load import Load


def test_monotone_under_deficit():
    ld = Load()
    prev = 0.0
    for _ in range(5000):
        cur = ld.integrate(deficit=0.5, dt=1e-3)
        assert cur >= prev
        prev = cur
    assert 0.0 < ld.L <= 1.0


def test_saturates():
    ld = Load()
    for _ in range(200_000):
        ld.integrate(deficit=10.0, dt=1e-3)
    assert ld.L <= 1.0
