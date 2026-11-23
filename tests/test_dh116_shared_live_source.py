"""Source-bound live context references; no grasp qualification from fixtures."""
from dataclasses import replace
import pytest
import torch

from test_native_teacher_context import backend_profile
from test_dh116_shared_teacher_source import reader


def test_reset_reference_context_tracks_actual_epoch_and_previous_command(backend_profile):
    from go2_pvcnn.control.dh116.tabletop_shared_teacher_live_source import SharedSmokeReferences
    from go2_pvcnn.control.dh116.tabletop_native_teacher_context import validate_context
    source, runtime, _, _ = reader(backend_profile)
    refs = SharedSmokeReferences(source, grasp="pinch", episode_cap_s=8., wall_cap_s=180.)
    for i in range(2):
        refs.reset_lane(i, distance_mean_m=.02)
    contexts = refs.contexts()
    for i, c in enumerate(contexts):
        validate_context(c,backend_profile=source.profile)
        assert c.metadata.identities[0].episode_uid == runtime.episodes[i]
        assert not c.values[:,12:36].any()
    a = torch.linspace(-.2,.2,48).reshape(2,24)
    refs.set_previous_request(a, "actual-recorded-archive-window")
    contexts = refs.contexts()
    torch.testing.assert_close(contexts[1].values[:,12:36],a[1:2])
    runtime.resets[0] += 1
    runtime.episodes = ("new-ep0",runtime.episodes[1])
    with pytest.raises(ValueError,match="Reset"):
        refs.contexts()
    refs.reset_lane(0,distance_mean_m=.01)
    contexts = refs.contexts()
    assert not contexts[0].values[:,12:36].any()
    torch.testing.assert_close(contexts[1].values[:,12:36],a[1:2])

def test_raw_sdk_float32_dt_and_nominal_policy_time_are_separate(backend_profile):
    from go2_pvcnn.control.dh116.tabletop_shared_teacher_live_source import lane_clock
    source, runtime, _, _ = reader(backend_profile)
    dt = float(torch.tensor(1/480,dtype=torch.float32))
    source.sim.get_physics_dt = lambda: dt
    runtime.physics_step = source.sim.current_time_step_index = 10
    source.sim.current_time = 10*dt
    b = source._boundary()
    assert b.sdk_time_s == 0.020833334419876337
    assert b.sdk_dt_s == dt
    assert b.policy_time_s == 10/480
    assert lane_clock(source,0)["timestamp_s"] == 10/480
    # Real one-tick delay cannot be normalized away as float32 dt representation.
    source.sim.current_time -= dt
    with pytest.raises(ValueError):
        source._boundary()

def test_sdk_nominal_dt_getter_with_float32_timer_callback(backend_profile):
    source, runtime, _, _ = reader(backend_profile)
    source.sim.get_physics_dt = lambda: 1/480
    runtime.physics_step = source.sim.current_time_step_index = 10
    source.sim.current_time = 0.020833334419876337
    b = source._boundary()
    assert b.sdk_dt_s == 1/480
    assert b.sdk_timer_dt_s == float(torch.tensor(1/480,dtype=torch.float32))
    assert b.policy_time_s == 10/480
