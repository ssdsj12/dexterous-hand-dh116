import numpy as np
import pytest

from scripts import run_dh116_pinch_index_only_preload as trim


def state():
    left = np.array([.28, .47, .70, 1.15, 1.15, 1.15, 0., 0., 0., 0., 0., 0.])
    right = left.copy()
    right[2] = .74
    return {side: (values.copy(), values) for side, values in (("left", left), ("right", right))}


def test_left_index_only_is_exact_after_the_same_45_windows():
    baseline = np.zeros((45, 24))
    baseline[-1, 3] = -.03
    measured = state()
    actions, targets = trim.compile_index(baseline, measured)
    np.testing.assert_array_equal(actions[:45], baseline)
    np.testing.assert_array_equal(actions[47:], np.zeros((8, 24)))
    delta = actions[45:47].sum(axis=0) * trim.seating.prelift.WINDOW_SCALE
    np.testing.assert_array_equal(np.flatnonzero(delta), [2])
    assert delta[2] == pytest.approx(.0001, abs=1e-12)
    np.testing.assert_array_equal(targets["right"], measured["right"][1])


def test_index_only_rejects_wrist_or_right_target_drift():
    measured = state()
    _, frozen = trim.compile_index(np.zeros((45, 24)), measured)
    frozen["right"][7] += 1e-6
    with pytest.raises(ValueError, match="archive"):
        trim.compile_index(np.zeros((45, 24)), measured, nominal_targets=frozen)


def test_index_only_retains_original_joint_limit():
    measured = state()
    measured["left"][1][2] = trim.seating.prelift.active_limits()[1][2]
    with pytest.raises(ValueError, match="joint limits"):
        trim.compile_index(np.zeros((45, 24)), measured)


def test_index_only_uses_the_original_early_gates(monkeypatch):
    calls = []
    plan = {}
    def record(row, frozen, **kwargs):
        assert frozen is plan and row["physics_step"] == kwargs["sdk_step"]
        calls.append((kwargs["sdk_step"], kwargs["gate_key"], kwargs["compiler"]))

    def upstream(sim, owners, trace, output, *, plan, sdk_steps, after):
        assert sdk_steps == 442
        for step in (202, 282, 361, 362):
            after({"physics_step": step})

    monkeypatch.setattr(trim.seating.preload, "measure_gate", record)
    monkeypatch.setattr(trim.seating.closure, "acquire", upstream)
    trim.acquire(None, None, None, None, plan=plan)
    assert calls == [
        (202, "bilateral_preload_gate", trim.seating.preload.compile_preload),
        (282, "asymmetric_preload_gate", trim.seating.prior.compile_asymmetric),
        (362, "left_index_only_gate", trim.compile_index),
    ]
