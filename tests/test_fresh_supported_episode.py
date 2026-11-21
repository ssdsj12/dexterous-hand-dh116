"""Actual episode Reset -> common state -> seven-route exchange, CPU only."""
import hashlib
from copy import deepcopy
import numpy as np
import pytest
import torch

from test_fresh_supported_physical_exchange import prepared
from go2_pvcnn.control.dh116.tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode
from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import HAND_NAMES


def episode_case(monkeypatch,tmp_path):
    c,module,plan,_ = prepared(monkeypatch)
    for hand in c.hands:
        hand.root_physx_view.get_coms = lambda: np.zeros((1,23,7),np.float32)
        hand.root_physx_view.get_link_velocities = lambda: torch.zeros(1,23,6)
        hand.data.body_lin_vel_w = torch.zeros(1,23,3)
        hand.data.body_ang_vel_w = torch.zeros(1,23,3)
        previous = hand.update
        def update(dt,hand=hand,previous=previous):
            previous(dt)
            hand.data.body_link_pose_w = hand.link_poses[..., [0,1,2,6,3,4,5]].clone()
        hand.update = update
        hand.update(1/480)
    source_map = {key:deepcopy(c.rod.bindings[key]) for key in HAND_NAMES}
    inspected = dict(pad_bindings=source_map,table_top_m=.30,
        shapes=[(side,index,np.array([[0.,0.,-.01]])) for side in (0,1) for index in range(23)])
    artifact = tmp_path/"CPU-only-solver.so"
    artifact.write_bytes(b"explicit CPU physical API fixture")
    rows,readback = [],[]
    episode = DualHandPhysicalEpisode(c.sim,c.rod,c.bindings,
        backend_artifact=artifact,backend_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        inspect_geometry=lambda:dict(inspected,identity=[c.rod.clock()]),
        physics_budget_end=18,record=rows.append)
    return c,module,plan,episode,rows,readback


def test_episode_supported_reset_records_source_then_exposes_same_clock66D_and_all8_physics_rows(monkeypatch,tmp_path):
    c,module,plan,episode,rows,readback = episode_case(monkeypatch,tmp_path)
    before = (c.sim.current_time_step_index,c.rope.time,c.rope.steps,c.rope.resets)
    result = episode.reset(c.seed,episode_uid="cpu17:episode",owner_resets=c.owners,
        supported_mesh_plan=plan,record_reset_readback=readback.append)
    assert (c.sim.current_time_step_index,c.rope.time,c.rope.steps) == before[:3]
    assert c.rope.resets == before[3]+1
    assert len(readback) == len(rows) == 1 and readback[0]["complete"]
    assert result["reset"]["mesh_state_readback"]["complete"]
    assert len(result["reset"]["pad_poses_xyzw"]) == 14
    pre = episode.pre_action_state()
    assert pre["common"]["valid"].sum() == 66
    assert np.isnan(pre["common"]["values"][~pre["common"]["valid"]]).all()
    assert pre["common"]["identity"]["physics_step"] == 2
    report = episode.control(torch.zeros(1,24))
    assert report["physical_samples"] == 8 and report["termination_reason"] is None
    assert [row["kind"] for row in rows] == ["reset"]+["physical"]*8
    for i,row in enumerate(rows[1:],3):
        assert row["exchange"]["physics_step"] == row["rod_boundary"]["rod_snapshot"]["physics_step"] == i
        assert row["exchange"]["rod"]["mesh_names"] == result["reset"]["mesh_state_readback"]["mesh_names"]
        assert row["exchange"]["issued_composer_wrench"]["right"]["forces"].shape == (1,7,3)
        assert row["teacher_admission"] is False
    assert episode.pre_action_state()["common"]["identity"]["physics_step"] == 10
    assert c.rope.steps == 8


def test_supported_episode_requires_an_independent_readback_recorder_before_reset(monkeypatch,tmp_path):
    c,module,plan,episode,rows,readback = episode_case(monkeypatch,tmp_path)
    before = (c.rope.resets,c.rope.steps,c.sim.current_time_step_index)
    with pytest.raises(ValueError,match="readback|recorder"):
        episode.reset(c.seed,episode_uid="cpu17:no-sink",owner_resets=c.owners,supported_mesh_plan=plan)
    assert (c.rope.resets,c.rope.steps,c.sim.current_time_step_index) == before
    assert not rows and not readback
