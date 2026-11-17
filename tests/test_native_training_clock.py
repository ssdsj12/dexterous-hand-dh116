"""CPU-only clock mathematics, never SDK precision or native qualification."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

DT = 1 / 480
MODES = ("fp32_dt_float64_sum", "float64_dt_float64_sum")
BOUNDARIES = (0, 1, 240, 241, 258, 2402, 9600)


def api():
    from go2_pvcnn.control.dh116 import tabletop_native_training_clock as clock
    return clock


def accumulated(step, mode):
    term = float(np.float32(DT)) if mode == MODES[0] else DT
    stamp = 0.
    for _ in range(step):
        stamp += term
    return stamp


def rod_row(step=None, *, valid=False):
    return dict(run_uuid="run-cpu", env_id=0, episode_uid="cpu:episode:1",
        reset_id=0 if step is None else 1, rod_uid="rod-cpu", physics_step=step,
        rod_tick=0 if step is None else 12 * max(0, step - 2),
        timestamp_s=None if step is None else step * DT,
        solver_time_s=0. if step is None else .5, clock_valid=valid)


def objects(step, mode=MODES[0], row=None):
    sim = SimpleNamespace(current_time_step_index=step,
        current_time=accumulated(step, mode), get_physics_dt=lambda: DT)
    state = rod_row(step, valid=True) if row is None else row
    rod = SimpleNamespace(clock=lambda: state)
    return sim, rod, state


def test_old_reset_source_remains_bounded_without_mutation():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_session import reset_clock_preflight
    sim = SimpleNamespace(current_time_step_index=258)
    with pytest.raises(ValueError, match="bounded_mesh_reset_clock"):
        reset_clock_preflight(SimpleNamespace(sim=sim))
    assert sim.current_time_step_index == 258


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("step", BOUNDARIES)
def test_accepts_actual_accumulated_timestamp_without_rewriting(step, mode):
    clock = api()
    policy = clock.ClockPolicyV2(max_sdk_step=9600, timestamp_mode=mode)
    stamp = accumulated(step, mode)
    result = clock.validate_sdk_clock(step=step, timestamp_s=stamp, dt_s=DT, policy=policy)
    assert type(result) is clock.SdkClockV2
    assert result.sdk_step_abs == step
    assert result.timestamp_s == stamp
    assert result.dt_s == DT
    assert 0 < result.tolerance_s < DT / 2
    assert result.policy_sha256 == policy.sha256
    with pytest.raises(FrozenInstanceError):
        result.timestamp_s = 0.


def test_fp32_accumulation_past_old_absolute_tolerance_is_not_normalized():
    clock = api()
    stamp = accumulated(9600, MODES[0])
    assert abs(stamp - 9600 * DT) > 2e-7
    result = clock.validate_sdk_clock(step=9600, timestamp_s=stamp, dt_s=DT,
        policy=clock.ClockPolicyV2(max_sdk_step=9600, timestamp_mode=MODES[0]))
    assert result.timestamp_s != 9600 * DT
    assert result.timestamp_s == stamp


@pytest.mark.parametrize("value", [True, False, 0, -1, 1., None, "9600", float("nan")])
def test_policy_requires_positive_exact_integer_bound(value):
    with pytest.raises(ValueError):
        api().ClockPolicyV2(max_sdk_step=value, timestamp_mode=MODES[0])


@pytest.mark.parametrize("value", [None, "", "native", "fp32", True, 0])
def test_policy_rejects_unverified_mode_names(value):
    with pytest.raises(ValueError):
        api().ClockPolicyV2(max_sdk_step=9600, timestamp_mode=value)


def test_policy_identity_is_owned_stable_and_mode_bound():
    clock = api()
    a = clock.ClockPolicyV2(max_sdk_step=9600, timestamp_mode=MODES[0])
    assert len(a.sha256) == 64 and all(c in "0123456789abcdef" for c in a.sha256)
    assert a.sha256 == clock.ClockPolicyV2(max_sdk_step=9600, timestamp_mode=MODES[0]).sha256
    assert a.sha256 != clock.ClockPolicyV2(max_sdk_step=9600, timestamp_mode=MODES[1]).sha256
    assert a.sha256 != clock.ClockPolicyV2(max_sdk_step=9601, timestamp_mode=MODES[0]).sha256
    with pytest.raises(FrozenInstanceError):
        a.max_sdk_step = 1


@pytest.mark.parametrize("step", [True, False, -1, 1., None, "258", float("nan"), 9601])
def test_rejects_bad_or_out_of_policy_step(step):
    clock = api()
    with pytest.raises(ValueError):
        clock.validate_sdk_clock(step=step, timestamp_s=0., dt_s=DT,
            policy=clock.ClockPolicyV2(max_sdk_step=9600, timestamp_mode=MODES[0]))


@pytest.mark.parametrize("value", [True, False, None, "0.5", float("nan"),
    float("inf"), -0.1])
def test_rejects_bad_actual_timestamp(value):
    clock = api()
    with pytest.raises(ValueError):
        clock.validate_sdk_clock(step=240, timestamp_s=value, dt_s=DT,
            policy=clock.ClockPolicyV2(max_sdk_step=9600, timestamp_mode=MODES[0]))


@pytest.mark.parametrize("value", [True, False, None, "480", float("nan"),
    float("inf"), 0., -DT, 1/240, float(np.float32(DT))])
def test_rejects_non_480hz_actual_dt(value):
    clock = api()
    with pytest.raises(ValueError):
        clock.validate_sdk_clock(step=1, timestamp_s=DT, dt_s=value,
            policy=clock.ClockPolicyV2(max_sdk_step=9600, timestamp_mode=MODES[0]))


@pytest.mark.parametrize("mode", MODES)
def test_rejects_shift_outside_derived_envelope(mode):
    clock = api()
    policy = clock.ClockPolicyV2(max_sdk_step=9600, timestamp_mode=mode)
    result = clock.validate_sdk_clock(step=2402, timestamp_s=accumulated(2402, mode),
        dt_s=DT, policy=policy)
    for sign in (-1, 1):
        with pytest.raises(ValueError, match="envelope"):
            clock.validate_sdk_clock(step=2402,
                timestamp_s=2402*DT + sign*2*result.tolerance_s, dt_s=DT, policy=policy)


def test_float64_policy_rejects_fp32_callback_drift():
    clock = api()
    with pytest.raises(ValueError, match="envelope"):
        clock.validate_sdk_clock(step=258, timestamp_s=accumulated(258, MODES[0]), dt_s=DT,
            policy=clock.ClockPolicyV2(max_sdk_step=9600, timestamp_mode=MODES[1]))


@pytest.mark.parametrize("mode", MODES)
def test_envelope_requires_usable_sub_half_step_bound(mode):
    clock = api()
    step = 1_000_000_000
    with pytest.raises(ValueError, match="envelope"):
        clock.validate_sdk_clock(step=step, timestamp_s=step*DT, dt_s=DT,
            policy=clock.ClockPolicyV2(max_sdk_step=step, timestamp_mode=mode))


def test_summation_bound_rejects_nu_at_one():
    clock = api()
    step = 2**53
    with pytest.raises(ValueError, match="envelope"):
        clock.validate_sdk_clock(step=step, timestamp_s=step*DT, dt_s=DT,
            policy=clock.ClockPolicyV2(max_sdk_step=step, timestamp_mode=MODES[1]))


@pytest.mark.parametrize("policy", [None, {}, True, "fp32_dt_float64_sum"])
def test_validate_requires_exact_policy(policy):
    with pytest.raises(ValueError):
        api().validate_sdk_clock(step=0, timestamp_s=0., dt_s=DT, policy=policy)


def test_registered_rod_clock_is_owned_and_source_fields_are_preserved():
    clock = api()
    sim, rod, state = objects(258)
    before = deepcopy(vars(sim))
    packet = clock.read_clock_v2(sim, rod,
        policy=clock.ClockPolicyV2(max_sdk_step=9600, timestamp_mode=MODES[0]))
    assert packet["sdk_physics_step"] == 258
    assert packet["sdk_timestamp_s"] == sim.current_time
    assert packet["sdk_timestamp_s"] != state["timestamp_s"]
    assert packet["sdk_dt_s"] == DT
    assert packet["rod"] == state
    assert packet["clock_policy_sha256"]
    assert packet["sdk_tolerance_s"] > 0
    state["solver_time_s"] = 99.
    assert packet["rod"]["solver_time_s"] == .5
    assert vars(sim) == before


def test_constructor_unknown_fields_remain_unknown_not_sdk_zero():
    clock = api()
    sim, rod, state = objects(2, row=rod_row())
    packet = clock.read_clock_v2(sim, rod,
        policy=clock.ClockPolicyV2(max_sdk_step=9600, timestamp_mode=MODES[0]))
    assert packet["sdk_physics_step"] == 2
    assert packet["rod"]["physics_step"] is None
    assert packet["rod"]["timestamp_s"] is None
    assert packet["rod"]["clock_valid"] is False
    assert packet["rod"] == state


def test_poisoned_registered_clock_stays_false():
    clock = api()
    sim, rod, state = objects(258, row=rod_row(258, valid=False))
    assert clock.read_clock_v2(sim, rod, policy=clock.ClockPolicyV2(
        max_sdk_step=9600, timestamp_mode=MODES[0]))["rod"]["clock_valid"] is False
    assert state["clock_valid"] is False


@pytest.mark.parametrize("patch", [
    {"physics_step": 257}, {"physics_step": True}, {"rod_tick": None},
    {"rod_tick": True}, {"reset_id": True}, {"env_id": -1},
    {"solver_time_s": float("nan")}, {"solver_time_s": -1.},
    {"clock_valid": 1}, {"timestamp_s": 9.}, {"episode_uid": ""},
    {"run_uuid": None}, {"extra": 1}])
def test_bad_registered_rod_clock_is_rejected_without_mutation(patch):
    clock = api()
    state = rod_row(258, valid=True)
    state.update(patch)
    sim, rod, state = objects(258, row=state)
    before = deepcopy(state)
    with pytest.raises(ValueError):
        clock.read_clock_v2(sim, rod, policy=clock.ClockPolicyV2(
            max_sdk_step=9600, timestamp_mode=MODES[0]))
    # NaN uses identity to avoid a misleading dictionary NaN-equality test.
    assert state.keys() == before.keys()
    assert sim.current_time == accumulated(258, MODES[0])


@pytest.mark.parametrize("patch", [
    {"physics_step": 0}, {"timestamp_s": 0.}, {"clock_valid": True},
    {"rod_tick": 12}, {"reset_id": 1}, {"solver_time_s": .01}])
def test_constructor_missing_clock_is_not_repaired(patch):
    clock = api()
    state = rod_row()
    state.update(patch)
    sim, rod, _ = objects(2, row=state)
    with pytest.raises(ValueError):
        clock.read_clock_v2(sim, rod, policy=clock.ClockPolicyV2(
            max_sdk_step=9600, timestamp_mode=MODES[0]))


def test_episode_elapsed_uses_actual_reset_endpoint_difference():
    clock = api()
    policy = clock.ClockPolicyV2(max_sdk_step=9600, timestamp_mode=MODES[0])
    sim, rod, state = objects(258)
    reset = clock.read_clock_v2(sim, rod, policy=policy)
    # The CPU SDK endpoint advances by actual cumulative additions.
    for _ in range(8):
        sim.current_time_step_index += 1
        sim.current_time += float(np.float32(DT))
    state.update(rod_row(266, valid=True))
    after = clock.read_clock_v2(sim, rod, policy=policy)
    elapsed = after["sdk_timestamp_s"] - reset["sdk_timestamp_s"]
    assert elapsed == 8*float(np.float32(DT))
    assert after["sdk_physics_step"] == 266
    assert reset["sdk_physics_step"] == 258


def test_reader_does_not_retain_borrowed_sim_or_rod_in_traceback():
    clock = api()
    sim, rod, _ = objects(258, row=rod_row(257, valid=True))
    try:
        clock.read_clock_v2(sim, rod, policy=clock.ClockPolicyV2(
            max_sdk_step=9600, timestamp_mode=MODES[0]))
    except ValueError as error:
        frame = error.__traceback__
        while frame:
            if frame.tb_frame.f_code.co_name == "read_clock_v2":
                assert frame.tb_frame.f_locals["sim"] is None
                assert frame.tb_frame.f_locals["rod"] is None
            frame = frame.tb_next
    else:
        pytest.fail("mismatched clock accepted")


@pytest.mark.parametrize("change", ["step", "timestamp", "dt"])
def test_rod_capture_cannot_join_two_different_sdk_endpoints(change):
    clock = api()
    sim, _, state = objects(258)
    before = sim.current_time
    def capture():
        if change == "step":
            sim.current_time_step_index += 1
            sim.current_time += float(np.float32(DT))
        elif change == "timestamp":
            sim.current_time += 1e-15
        else:
            sim.get_physics_dt = lambda: 1/240
        return state
    rod = SimpleNamespace(clock=capture)
    with pytest.raises(ValueError):
        clock.read_clock_v2(sim, rod, policy=clock.ClockPolicyV2(
            max_sdk_step=9600, timestamp_mode=MODES[0]))
    if change == "timestamp":
        assert sim.current_time != before
