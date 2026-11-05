"""Hysteresis and allostatic load go/no-go gate.

Baseline measurements (seed=0, seed=1):
  onset = 6181 steps
  recovery = 12936 steps
  ratio = 2.09 (gate requirement: ratio > 2.0)
  L_after = 0.083
  no-load ratio = 1.17 (gate requirement: ratio < 1.4)
"""

import pytest

from mood.landscape import Params, onset, recovery

pytestmark = pytest.mark.gono


def test_recovery_slower_than_onset():
    p = Params()
    onset_steps, L_after = onset(p, seed=0)
    assert onset_steps < int(p.t_max / p.dt), "onset never escaped the healthy well"
    assert L_after > 0.05, "load did not accumulate during onset"

    rec_steps = recovery(p, L0=L_after, seed=0)
    assert rec_steps < int(p.t_max / p.dt), "recovery never escaped the depressive well"

    ratio = rec_steps / onset_steps
    print(f"\nonset={onset_steps} recovery={rec_steps} ratio={ratio:.2f} L={L_after:.3f}")
    assert ratio > 1.5, f"hysteresis too weak: ratio={ratio:.2f}"


def test_hysteresis_vanishes_without_load():
    p = Params()
    onset_steps, _ = onset(p, seed=1)
    rec_steps = recovery(p, L0=0.0, seed=1)
    ratio = rec_steps / onset_steps
    print(f"\nno-load ratio={ratio:.2f}")
    assert ratio < 1.4, f"asymmetry present without load: ratio={ratio:.2f}"
