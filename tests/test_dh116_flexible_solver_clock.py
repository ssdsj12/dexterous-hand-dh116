"""Reproduce DeformX float accumulation over the actual smoke/Hold budgets."""
import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import DT
from go2_pvcnn.control.dh116.tabletop_deformable_solver_clock import solver_clock_matches


@pytest.mark.parametrize("initial,steps", [(0., 256), (0., 2401), (6., 2880), (100., 512)])
def test_expected_rounding_over_bounded_budgets(initial, steps):
    start = np.float32(initial)
    timestamp = start
    for i in range(steps):
        after = np.float32(timestamp + np.float32(DT))
        assert solver_clock_matches(float(timestamp), float(after), steps=1)
        timestamp = after
    assert solver_clock_matches(float(start), float(timestamp), steps=steps)


@pytest.mark.parametrize("bad", ["repeat", "skip", "reverse", "nan", "unresolvable"])
def test_observed_time_still_rejects_invalid_advance(bad):
    start = float(np.float32(.51))
    after = {"repeat": start, "skip": start+2*DT, "reverse": start-DT,
             "nan": float("nan"), "unresolvable": 2.**20}[bad]
    if bad == "unresolvable":
        start = after
    assert not solver_clock_matches(start, after, steps=1)


def test_long_clock_proof_rejects_wrong_number_of_steps():
    start = np.float32(0)
    after = start
    for _ in range(2400):
        after += np.float32(DT)
    assert solver_clock_matches(float(start), float(after), steps=2400)
    assert not solver_clock_matches(float(start), float(after), steps=2399)
    assert not solver_clock_matches(float(start), float(after), steps=2401)


def test_runtime_retains_the_actual_float_clock_through_five_seconds(monkeypatch):
    from test_dual_hand_rod_runtime import runtime, reset

    rod, base, _, poses, centers = runtime(monkeypatch)
    reset(rod, poses)
    original = base.rope.step_free
    def float32_step(dt, substeps):
        before = np.float32(base.rope.time)
        original(dt, substeps)
        base.rope.time = float(before + np.float32(dt))
    base.rope.step_free = float32_step
    for step in range(3, 2404):
        last = rod.step(poses, centers, physics_step=step)
    assert last["rod_tick"] == 2401 * 12
    assert abs(last["solver_time_s"] - 2401 * DT) > 2e-6
    base.rope.time += DT
    with pytest.raises(ValueError, match="outside"):
        rod.step(poses, centers, physics_step=2404)
