import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_native_teacher_swept_projection import (
    WristSweep, preaction_target,
)


def fixture(*, clearance, lever):
    vertices = {"source:base_link": np.array([[0., 0., -lever]])}
    link = np.eye(4)
    link[2, 3] = clearance + lever
    return WristSweep(root="/hand/base_link", edges=[], vertices=vertices,
        body_paths={"source:base_link": "/hand/base_link"}, table_top_m=0.), link


def test_safe_endpoints_can_have_an_unsafe_swept_middle():
    sweep, _ = fixture(clearance=.00205, lever=.1)
    from scipy.spatial.transform import Rotation

    xyz = np.array([0., 0., .10205])
    measured_angle = np.array([0., .05, 0.])
    link = np.eye(4)
    link[:3, :3] = Rotation.from_euler("XYZ", measured_angle).as_matrix()
    link[:3, 3] = xyz
    rotation = np.eye(3)
    q = np.zeros(11)
    bound = sweep.prove(
        measured_link_frames={"source:base_link": link},
        anchor_rotation=rotation, start_xyz=xyz, start_angles=measured_angle,
        start_q=q, final_q=q, target_xyz=xyz, target_angles=-measured_angle,
    )
    assert min(bound["start_m"], bound["end_m"]) > .0021
    assert bound["status"] == "measured_violation"


def test_path_uses_hard_floor_and_terminal_reserve_separately():
    sweep, link = fixture(clearance=.00211, lever=.1)
    goal = link[:3, 3] + [0., 0., .0001]
    proof = sweep.prove(
        measured_link_frames={"source:base_link": link},
        anchor_rotation=np.eye(3), start_xyz=link[:3, 3], start_angles=np.zeros(3),
        start_q=np.zeros(11), final_q=np.zeros(11), target_xyz=goal, target_angles=np.zeros(3),
    )
    assert proof["start_m"] < .00215 < proof["end_m"]
    assert proof["proved"]


def test_candidate_rate_limits_include_three_axis_norm_even_for_vertical_correction():
    old = np.zeros(3)
    theta = np.zeros(3)
    nominal_xyz = np.array([1.8e-5, 0., .001])
    nominal_rot = np.array([.0005, .001, -.0004])
    checked = []

    def certify(xyz, angles):
        checked.append((xyz.copy(), angles.copy()))
        passed = bool(xyz[2] > 0. and np.linalg.norm(angles) <= .0009)
        return dict(proved=passed, start_m=.0022, end_m=.0023)

    decision = preaction_target(nominal_xyz=nominal_xyz, nominal_angles=nominal_rot,
        last_xyz=old, last_angles=theta, measured_angles=theta, seed_xyz=old,
        certify=certify)
    assert decision["accepted"]
    assert np.linalg.norm(decision["xyz"]) <= .01 / 480 + 1e-12
    assert np.linalg.norm(decision["angles"]) <= .5 / 480 + 1e-12
    assert all(np.linalg.norm(xyz) <= .01 / 480 + 1e-12 for xyz,_ in checked)


def test_no_certificate_does_not_supply_a_target():
    zero = np.zeros(3)
    result = preaction_target(nominal_xyz=zero, nominal_angles=zero, last_xyz=zero,
        last_angles=zero, measured_angles=zero, seed_xyz=zero,
        certify=lambda *_: dict(proved=False,start_m=.0022,end_m=.00205))
    assert not result["accepted"]
    assert "xyz" not in result and "angles" not in result


def test_upper_bound_can_prove_that_no_rate_limited_destination_recovers_clearance():
    sweep, link = fixture(clearance=.00204, lever=.1)
    xyz = link[:3, 3].copy()
    boundary = sweep.boundary(measured_link_frames={"source:base_link": link},
        anchor_rotation=np.eye(3), start_xyz=xyz, start_angles=np.zeros(3),
        start_q=np.zeros(11), final_q=np.zeros(11))
    upper = boundary.endpoint_upper_bound(xyz, np.zeros(3))
    assert .00204 + .01 / 480 <= upper < .0021
    for delta in ((.5 / 480, 0., 0.), (0., .5 / 480, 0.), (0., 0., .5 / 480)):
        target = xyz + [0., 0., .01 / 480]
        point = boundary.prove(target, np.array(delta))
        assert point["end_m"] <= upper


def test_scope_denies_before_composer_sdk_or_solver_write(monkeypatch):
    import torch
    from test_dh116_flexible_physical_exchange import setup
    from go2_pvcnn.control.dh116 import tabletop_deformable_physical_exchange as exchange
    from go2_pvcnn.control.dh116.tabletop_swept_projection_scope import swept_projection_scope

    runtime, sim, base, writes = setup(monkeypatch)
    original = exchange.project_policy_wrist_targets, exchange.BimanualStepConnection
    trace = []
    action = torch.zeros((1, 24))

    def reject(pre, request, nominal, state):
        assert pre["identity"][0]["physics_step"] == 2
        assert torch.equal(request["action24"].float(), action)
        return (dict(accepted=True), dict(accepted=False, reason="no_certified_original_limit_target"))

    with pytest.raises(RuntimeError, match="wrist_sweep_rejected:right"):
        with swept_projection_scope(reject, record=trace.append):
            runtime.control(action, observe=lambda _: None)
    assert not writes and sim.current_time_step_index == 2 and base.rope.steps == 0
    assert trace[0]["physics_step"] == 2 and trace[0]["rejected_sides"] == ["right"]
    assert original == (exchange.project_policy_wrist_targets, exchange.BimanualStepConnection)


def test_scope_preserves_raw_action_and_integrates_from_accepted_wrist_target(monkeypatch):
    import torch
    from test_dh116_flexible_physical_exchange import setup
    from go2_pvcnn.control.dh116 import tabletop_deformable_physical_exchange as exchange
    from go2_pvcnn.control.dh116.tabletop_swept_projection_scope import swept_projection_scope

    runtime, sim, _, _ = setup(monkeypatch)
    nominal_project = exchange.project_policy_wrist_targets
    seen, rows, trace = [], [], []
    action = torch.zeros((1, 24))
    action[0, 22] = .25

    def clip_wrist(pre, request, nominal, state):
        seen.append(state["angles_rad"].clone())
        assert torch.equal(request["action24"].float(), action)
        assert torch.equal(nominal["projected_angles"][0, :, 1], request["requested_angles_rad"][0, :, 1])
        angles = (state["angles_rad"][0] + nominal["projected_angles"][0]) * .5
        proof = dict(proved=True, start_m=.01, end_m=.01, margin_lower_m=.0079)
        return tuple(dict(accepted=True, xyz=nominal["projected_root_poses"][0, i, :3].numpy(),
                          angles=angles[i].numpy(), certificate=proof) for i in range(2))

    with swept_projection_scope(clip_wrist, record=trace.append):
        runtime.control(action, observe=lambda row: rows.append(row))
        runtime.control(action, observe=lambda row: rows.append(row))
    assert sim.current_time_step_index == 18 and len(trace) == len(rows) == 16
    assert exchange.project_policy_wrist_targets is nominal_project
    for i, row in enumerate(rows):
        assert torch.equal(row["action_readback"]["request"]["action24"].float(), action)
        final = row["wrist_projection"]["projected_angles"]
        if i + 1 < len(seen):
            assert torch.equal(final, seen[i + 1])
        np.testing.assert_array_equal(final.numpy(),
            row["action_readback"]["submission"]["projected_angles"].numpy())


def test_projector_cannot_issue_overrate_wrist_target_even_with_a_pass_label(monkeypatch):
    import torch
    from test_dh116_flexible_physical_exchange import setup
    from go2_pvcnn.control.dh116.tabletop_swept_projection_scope import swept_projection_scope

    runtime, sim, base, writes = setup(monkeypatch)
    trace = []

    def propose(pre, request, nominal, state):
        xyz = nominal["projected_root_poses"][0, :, :3].numpy().copy()
        xyz[0, 2] += .001
        return tuple(dict(accepted=True, xyz=xyz[idx], angles=np.zeros(3),
                          certificate=dict(proved=True, start_m=.01, end_m=.01, margin_lower_m=.0079))
                     for idx in range(2))

    with pytest.raises(ValueError, match="original wrist contract"):
        with swept_projection_scope(propose, record=trace.append):
            runtime.control(torch.zeros(1, 24), observe=lambda _: None)
    assert not writes and base.rope.steps == 0 and sim.current_time_step_index == 2
    assert trace[0]["accepted"] is False and trace[0]["binding_failure"]
