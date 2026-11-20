import numpy as np
import pytest

from scripts import run_dh116_pinch_fixed_wrist_seating as seating


def states():
    issued=np.zeros(12)
    issued[:6]=[.28,.47,.70,1.15,1.15,1.15]
    return {side:(issued.copy(),issued.copy()) for side in ("left","right")}


def test_only_new_left_abduction_and_index_release_are_issued():
    pre=np.zeros((45,24))
    pre[0,14]=.1
    measured=states()
    action, goals=seating.compile_seating(pre,measured)
    np.testing.assert_array_equal(action[:45],pre)
    np.testing.assert_array_equal(action[47:],0)
    delta=action[45:47].sum(axis=0)*seating.prelift.WINDOW_SCALE
    np.testing.assert_array_equal(np.flatnonzero(delta),[0,2])
    np.testing.assert_allclose(delta[[0,2]],[-.003,-.0005])
    np.testing.assert_array_equal(goals["right"],measured["right"][1])


def test_seating_does_not_clip_an_invalid_target():
    measured=states()
    measured["left"][1][0]=0.
    with pytest.raises(ValueError,match="joint limits"):
        seating.compile_seating(np.zeros((45,24)),measured)


def test_tracking_box_contains_actual_issued_and_target_for_new_axes():
    actual=np.zeros(12)
    command=actual.copy()
    command[[0,2,7,9]]=[.01,-.02,.00005,.0001]
    target=command.copy()
    target[[0,2]]=[-.015,.015]
    reference,bounds=seating.seating_domain(actual,command,target)
    controlled=np.array([0,2,6,7,8,9,10,11])
    for q in (actual,command,target):
        delta=reference[controlled]-actual[controlled]
        fractions=np.divide(q[controlled]-actual[controlled],delta,out=np.zeros(8),where=delta!=0)
        assert np.all(fractions>=bounds[0]) and np.all(fractions<=bounds[1])


def test_archive_roundoff_cannot_move_held_channels():
    measured=states()
    pre=np.zeros((45,24))
    _, archived=seating.compile_seating(pre,measured)
    measured["right"][1][1] += 1e-10
    measured["left"][1][7] += 1e-10
    action,goal=seating.compile_seating(pre,measured,archived)
    np.testing.assert_array_equal(action[45:,6:],0)
    assert goal["left"][7]==measured["left"][1][7]
    assert goal["right"][1]==measured["right"][1][1]


def test_acquisition_keeps_all_three_contact_gates(monkeypatch):
    plan = {"action_schedule": np.zeros((55, 24))}
    observed = []

    def measure(row, frozen_plan, **gate):
        assert frozen_plan is plan
        assert row["physics_step"] == gate["sdk_step"]
        observed.append((gate["sdk_step"], gate["gate_key"], gate.get("certify")))

    def upstream(sim, owners, trace, output, *, plan, sdk_steps, after):
        assert sdk_steps == 442
        for step in (201, 202, 203, 281, 282, 361, 362, 363, 442):
            after({"physics_step": step})
        return "receipt"

    monkeypatch.setattr(seating.preload, "measure_gate", measure)
    monkeypatch.setattr(seating.closure, "acquire", upstream)
    assert seating.acquire(None, None, None, None, plan=plan) == "receipt"
    assert observed == [
        (202, "bilateral_preload_gate", None),
        (282, "asymmetric_preload_gate", None),
        (362, "fixed_wrist_seating_gate", seating.certify_seating),
    ]
