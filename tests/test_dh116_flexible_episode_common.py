"""Original common-state observations at the real flexible episode boundary."""
import numpy as np
import pytest
import torch

from test_dh116_dual_hand_episode import setup


def start(monkeypatch, tmp_path):
    episode, rod, hands, sim, seed, owners, rows = setup(monkeypatch, tmp_path)
    for h in hands:
        h.data.body_link_pose_w = h.link_poses[..., [0, 1, 2, 6, 3, 4, 5]]
        h.data.body_lin_vel_w = torch.zeros((1,23,3))
        h.data.body_ang_vel_w = torch.zeros((1,23,3))
        h.root_physx_view.get_link_velocities = lambda: torch.zeros((1,23,6))
        original = h.update
        def update(dt, h=h, original=original):
            original(dt)
            h.data.body_link_pose_w = h.link_poses[..., [0, 1, 2, 6, 3, 4, 5]]
        h.update = update
    episode.reset(seed, episode_uid="run-cpu:state", owner_resets=owners)
    return episode, rod, hands, sim, rows


def test_uses_measured_common_slots_and_same_clock_rod_without_imputation(monkeypatch, tmp_path):
    episode, rod, _, sim, rows = start(monkeypatch, tmp_path)
    sample = episode.pre_action_state()
    state = sample["common"]
    assert state["valid"].sum() == 66
    assert np.isnan(state["values"][~state["valid"]]).all()
    assert state["values"][155] == pytest.approx(.04/.05, abs=1e-6)
    assert state["identity"]["physics_step"] == sample["rod_boundary"]["rod_snapshot"]["physics_step"] == 2
    assert state["identity"]["policy_step"] == 0
    assert state["identity"]["episode_uid"] == "run-cpu:state"
    assert state["identity"]["rod_tick"] == 0
    assert not sample["teacher_admission"]
    assert sim.current_time_step_index == 2 and rod.rope.steps == 0 and len(rows) == 1
    episode.control(torch.zeros((1,24)))
    later = episode.pre_action_state()
    assert later["common"]["identity"]["physics_step"] == 10
    assert later["common"]["identity"]["policy_step"] == 1
    assert later["rod_boundary"]["rod_snapshot"]["rod_tick"] == 8*12


def test_stale_cache_and_foreign_clock_are_rejected(monkeypatch, tmp_path):
    episode, rod, hands, sim, _ = start(monkeypatch, tmp_path)
    hands[0].data.body_link_pose_w[0, 0, 0] += .01
    with pytest.raises(ValueError, match="cache"):
        episode.pre_action_state()
    hands[0].update(1/480)
    sim.current_time_step_index += 1
    with pytest.raises(ValueError, match="clock"):
        episode.pre_action_state()
    assert rod.rope.steps == 0


def test_pre_action_result_is_owned_and_no_sample_survives_terminal(monkeypatch, tmp_path):
    episode, rod, hands, sim, _ = start(monkeypatch, tmp_path)
    first = episode.pre_action_state()
    first["common"]["values"][:24] = 99
    first["rod_boundary"]["rod_snapshot"]["node_positions"][:] = 99
    assert episode.pre_action_state()["common"]["values"][0] != 99
    assert episode.pre_action_state()["rod_boundary"]["rod_snapshot"]["node_positions"].max() < 1
    step = sim.step
    def fail(*, render):
        step(render=render)
        hands[1].measured[0, hands[1].joint_names.index("finger12")] = -.05
    sim.step = fail
    assert episode.control(torch.zeros((1,24)))["termination_reason"] == "right:joint_limit"
    with pytest.raises(RuntimeError, match="active"):
        episode.pre_action_state()
