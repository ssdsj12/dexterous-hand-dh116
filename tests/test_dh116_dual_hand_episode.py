"""Reset and physical exchange through one owner with the original adapters."""
import hashlib

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import DT
from go2_pvcnn.control.dh116.tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode
from test_dual_hand_reset import setup as reset_setup


def setup(monkeypatch, tmp_path, *, invalid_clearance=False):
    events, rod, hands, bindings, sim, seed, owners = reset_setup(monkeypatch)
    for hand in hands:
        hand.root_physx_view.get_coms = lambda: np.zeros((1, 23, 7))
        composer = hand.permanent_wrench_composer
        set_all = composer.set_forces_and_torques
        def set_wrench(forces, torques, body_ids=None, *, composer=composer, set_all=set_all, **kw):
            if body_ids is None:
                set_all(forces, torques, **kw)
            else:
                composer.composed_force_as_torch[:, body_ids] = forces
                composer.composed_torque_as_torch[:, body_ids] = torques
                composer.active = True
        composer.set_forces_and_torques = set_wrench
        write = hand.write_data_to_sim
        def submit(hand=hand, write=write):
            if rod.rope.resets:
                hand.backend[:] = hand.data.joint_pos_target
            else:
                write()
        hand.write_data_to_sim = submit
    sim.is_playing = lambda: True
    def step(*, render):
        assert render is False
        sim.current_time_step_index += 1
        sim.current_time += DT
    sim.step = step
    offset = -.9 if invalid_clearance else -.05
    inspected = dict(pad_bindings=rod.bindings, table_top_m=.30,
        shapes=[(i, 0, np.array([[0., 0., offset]])) for i in range(2)])
    artifact = tmp_path / "solver.so"
    artifact.write_bytes(b"CPU double")
    rows = []
    episode = DualHandPhysicalEpisode(sim, rod, bindings,
        backend_artifact=artifact, backend_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        inspect_geometry=lambda: dict(inspected, identity=[rod.clock()]),
        physics_budget_end=18, record=rows.append)
    return episode, rod, hands, sim, seed, owners, rows


def test_reset_then_two_intervals_keep_measured_rod_and_action_identity(monkeypatch, tmp_path):
    episode, rod, hands, sim, seed, owners, rows = setup(monkeypatch, tmp_path)
    receipt = episode.reset(seed, episode_uid="run-cpu:a", owner_resets=owners)
    assert receipt["reset"]["after_clock"]["physics_step"] == 2
    assert receipt["rod_boundary"]["rod_tick"] == 0
    assert not receipt["rod_boundary"]["rod_snapshot"]["contact_valid"].any()
    assert rod.rope.resets == 1 and rod.rope.steps == 0
    action = torch.zeros((1, 24))
    for _ in range(2):
        result = episode.control(action)
        assert result["physical_samples"] == 8 and result["training_reward"] is None
    assert [r["kind"] for r in rows] == ["reset"] + ["physical"] * 16
    for step, row in enumerate(rows[1:], 3):
        assert row["exchange"]["physics_step"] == row["rod_boundary"]["rod_snapshot"]["physics_step"] == step
        assert row["rod_boundary"]["rod_snapshot"]["episode_uids"] == ["run-cpu:a"]
        assert not row["teacher_admission"]
    assert sim.current_time_step_index == 18 and rod.rope.steps == 16
    with pytest.raises(RuntimeError, match="budget"):
        episode.control(action)
    assert sim.current_time_step_index == 18


def test_initial_penetration_stops_after_reset_and_before_any_physics(monkeypatch, tmp_path):
    episode, rod, _, sim, seed, owners, rows = setup(monkeypatch, tmp_path, invalid_clearance=True)
    with pytest.raises(ValueError, match="clearance"):
        episode.reset(seed, episode_uid="run-cpu:a", owner_resets=owners)
    assert rod.rope.steps == 0 and sim.current_time_step_index == 2
    assert len(rows) == 1 and rows[0]["kind"] == "reset"
    with pytest.raises(RuntimeError, match="active"):
        episode.control(torch.zeros((1,24)))


def test_terminal_recorded_before_the_exchange_stops(monkeypatch, tmp_path):
    episode, rod, hands, sim, seed, owners, rows = setup(monkeypatch, tmp_path)
    episode.reset(seed, episode_uid="run-cpu:a", owner_resets=owners)
    step = sim.step
    def fail(*, render):
        step(render=render)
        hands[1].measured[0, hands[1].joint_names.index("finger12")] = -.05
    sim.step = fail
    result = episode.control(torch.zeros((1,24)))
    assert result["termination_reason"] == "right:joint_limit"
    assert len(rows) == 2 and rows[1]["exchange"]["guard_failure"] == "right:joint_limit"
    assert sim.current_time_step_index == 3 and rod.rope.steps == 1
    with pytest.raises(RuntimeError, match="active"):
        episode.control(torch.zeros((1,24)))


def test_recorder_error_closes_exchange_without_replaying_physics(monkeypatch, tmp_path):
    episode, rod, _, sim, seed, owners, rows = setup(monkeypatch, tmp_path)
    episode.reset(seed, episode_uid="run-cpu:a", owner_resets=owners)
    def fail(row):
        raise OSError("evidence storage unavailable")
    episode._record = fail
    with pytest.raises(OSError, match="storage"):
        episode.control(torch.zeros((1,24)))
    assert sim.current_time_step_index == 3 and rod.rope.steps == 1
    with pytest.raises(RuntimeError, match="active"):
        episode.control(torch.zeros((1,24)))
