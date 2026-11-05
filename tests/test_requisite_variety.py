from control.requisite import assert_requisite, variety


def test_a2():
    r, e = variety()
    assert r >= e, f"V(R)={r} < V(E)={e}"
    assert_requisite()
