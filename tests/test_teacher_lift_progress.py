"""Measured prelift progression and fixed-Reset Lift potential."""
import importlib

import pytest

from test_supported_lift_observation import reference, sample


def api():
    name = "go2_pvcnn.control.dh116.tabletop_teacher_lift_progress"
    assert importlib.util.find_spec(name), "measured lift progression is missing"
    return importlib.import_module(name)


def row(rod, tick, *, palm_shift=0.):
    rod.update(physics_step=tick, rod_tick=12 * (tick - 2),
               timestamp_s=tick / 480, solver_time_s=(tick - 2) / 480)
    rod["rod_mesh_contact_pairs"].update(physics_step=tick, rod_tick=rod["rod_tick"])
    poses = {f"{side}.base_link": ([.43 + palm_shift, y, .40], [0., 0., 0., 1.])
             for side, y in [("left", .18), ("right", -.18)]}
    return dict(rod=rod, sdk_hand_geometry=dict(physics_step=tick, poses=poses))


def test_measured_pre_hold_and_lift_reward_follow_the_wire():
    rod = sample()
    progress = api().SupportedLiftProgress(reference(rod), grasp="tripod", prelift_stable_s=2/480)
    first = progress.observe(row(rod, 3))
    assert first.phase.phase == "PreliftHold" and first.phase.prelift_stable_s == 0
    assert first.relative_node_speed_m_s is None
    assert progress.observe(row(rod, 4)).phase.prelift_stable_s == 0
    assert progress.observe(row(rod, 5)).phase.prelift_stable_s == 1/480
    assert progress.observe(row(rod, 6)).phase.phase == "Lift"
    rod["node_positions"][0, :, 2] += .015
    raised = progress.observe(row(rod, 7))
    assert raised.lift_potential == pytest.approx(2.)
    assert raised.phase.phase == "Lift" and not raised.phase.success_candidate
    assert not raised.phase.final_hold_s
    lost = progress.observe(row(sample(missing_palm="left.finger14_link"), 8))
    assert lost.phase.phase == "Close" and lost.lift_potential == 0
    assert api().lift_reward_delta(raised, lost, gamma=.995, terminated=False) == pytest.approx(-2.)


def test_moving_palms_clear_prelift_progress_even_with_opposing_force():
    rod = sample()
    progress = api().SupportedLiftProgress(reference(rod), grasp="tripod", prelift_stable_s=1/480)
    progress.observe(row(rod, 3))
    progress.observe(row(rod, 4))
    moved = progress.observe(row(rod, 5, palm_shift=.001))
    assert moved.observation.bilateral
    assert moved.palm_speed_m_s == pytest.approx(.48)
    assert moved.phase.phase == "PreliftHold" and moved.phase.prelift_stable_s == 0
    assert moved.lift_potential == 0


def test_stale_measured_palm_row_cannot_advance_phase():
    rod = sample()
    progress = api().SupportedLiftProgress(reference(rod), grasp="tripod", prelift_stable_s=1/480)
    progress.observe(row(rod, 3))
    bad = row(rod, 4)
    bad["sdk_hand_geometry"]["physics_step"] = 3
    with pytest.raises(ValueError, match="same physical"):
        progress.observe(bad)
    assert progress.observe(row(rod, 4)).phase.physics_tick == 4


def test_unchanged_positive_lift_potential_retains_discount():
    rod = sample()
    progress = api().SupportedLiftProgress(reference(rod), grasp="tripod", prelift_stable_s=1/480)
    for tick in range(3, 6):
        progress.observe(row(rod, tick))
    rod["node_positions"][0, :, 2] += .03
    pre = progress.observe(row(rod, 6))
    post = progress.observe(row(rod, 7))
    assert api().lift_reward_delta(pre, post, gamma=.995, terminated=False) == pytest.approx(-.02)
    assert api().lift_reward_delta(pre, post, gamma=.995, terminated=True) == pytest.approx(-4.)
