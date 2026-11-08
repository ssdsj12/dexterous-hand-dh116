import pytest
import torch
from types import SimpleNamespace

from go2_pvcnn.control.dh116.tabletop_bimanual_coordination import (
    BimanualCoordinator,
    CoordinationState,
    coordination_state_from_native,
    static_support_allocation,
)


def state(n=1):
    scalar = torch.zeros(n, dtype=torch.float64)
    hands = torch.zeros(n, 2, dtype=torch.float64)
    palm = torch.zeros(n, 2, 3, dtype=torch.float64)
    palm[:, 0, 0] = -.1
    palm[:, 1, 0] = .1
    return CoordinationState(
        palm_world_m=palm.clone(), palm_offset_m=torch.zeros_like(palm),
        clearance_m=hands + .01, grasp_points_w=palm.clone(),
        object_com_w=torch.zeros(n, 3, dtype=torch.float64),
        object_lift_m=scalar.clone(), opposed=torch.ones(n, 2, dtype=torch.bool),
        contact_valid=torch.ones(n, 2, dtype=torch.bool), support_n=hands + 1.,
        object_weight_n=scalar + 1., joint_tracking_error_rad=hands.clone(),
        mimic_error_rad=hands.clone(), slip_m=scalar.clone(),
        slip_valid=torch.zeros(n, dtype=torch.bool),
        slip_applicable=torch.zeros(n, dtype=torch.bool),
        live=torch.ones(n, dtype=torch.bool), episode_id=torch.zeros(n, dtype=torch.long),
    )


def actions(n=1):
    return torch.linspace(-.8, .8, 24, dtype=torch.float64).repeat(n, 1)


def test_z0_preserves_physical_actions_and_z1_changes_only_right_z():
    raw = actions(2)
    original = raw.clone()
    unchanged, info = BimanualCoordinator(2).project(raw)
    assert torch.equal(unchanged, original)
    assert info['mode'] == 'Z0'
    neutral, info = BimanualCoordinator(2, mode='Z1').project(raw)
    mask = torch.arange(24) != 17
    assert torch.equal(neutral[:, mask], original[:, mask])
    assert torch.equal(neutral[:, 17], torch.zeros(2, dtype=raw.dtype))
    assert torch.equal(raw, original)
    assert info['controller_output_source'] == 'privileged_simulation_intervention'


def test_support_allocation_uses_com_moment_and_rejects_outside_span():
    s = state(2)
    s.object_com_w[0, 0] = -.05
    s.object_com_w[1, 0] = .3
    required, valid = static_support_allocation(
        s.grasp_points_w, s.object_com_w, s.object_weight_n)
    assert valid.tolist() == [True, False]
    assert required[0].tolist() == pytest.approx([.75, .25])
    assert not torch.isfinite(required[1]).any()


def test_z2_restricts_only_right_z_and_stops_descent_at_grasp_height_or_table():
    s = state(2)
    raw = actions(2)
    raw[:, 17] = -.7
    c = BimanualCoordinator(2, mode='Z2')
    result, diag = c.project(raw, s, dt_s=1/480)
    keep = torch.arange(24) != 17
    assert torch.equal(result[:, keep], raw[:, keep])
    assert (result[:, 17] == 0).all()
    assert diag['right_relative_height_limited'].all()
    s.palm_world_m[:, 1, 2] += .002
    s.clearance_m[1, 1] = 0.
    result, diag = c.project(raw, s, dt_s=1/480)
    assert result[0, 17] == raw[0, 17]
    assert result[1, 17] == 0
    assert diag['table_limited'].tolist() == [False, True]


def test_z3_requires_real_bilateral_support_and_continuous_stability():
    s = state()
    raw = actions()
    c = BimanualCoordinator(1, mode='Z3', resume_dwell_s=.02)
    issued, diag = c.project(raw, s, dt_s=.01)
    assert (issued[:, [14, 17]] <= 0).all()
    issued, diag = c.project(raw, s, dt_s=.01)
    assert (issued[:, [14, 17]] > 0).all()
    assert diag['resume_ready'].item()
    s.opposed[0, 1] = False
    issued, diag = c.project(raw, s, dt_s=.01)
    assert (issued[:, [14, 17]] <= 0).all()
    assert diag['cancel_upward_target'].all()
    s.opposed[0, 1] = True
    issued, diag = c.project(raw, s, dt_s=.01)
    assert not diag['resume_ready'].item()
    assert (issued[:, [14, 17]] <= 0).all()


def test_z3_admits_subweight_pickup_but_loss_of_either_contact_pauses_immediately():
    s = state(2)
    # Each hand is measured to support only 0.08 N of a 1 N cable. The native
    # opposed-pad contract already requires bearing pad loads above 0.05 N.
    s.support_n[:] = .08
    c = BimanualCoordinator(2, mode='Z3', resume_dwell_s=.02)
    issued, diag = c.project(actions(2), s, dt_s=.02)
    assert diag['resume_ready'].all()
    assert (issued[:, [14, 17]] > 0).all()
    assert not diag['floating_support_ready'].any()
    assert diag['pickup_support_ready'].all()
    s.opposed[0, 1] = False
    s.support_n[1, 0] = 0.
    issued, diag = c.project(actions(2), s, dt_s=1/480)
    assert not diag['resume_ready'].any()
    assert (issued[:, [14, 17]] <= 0).all()
    assert diag['cancel_upward_target'].all()


def test_z3_full_weight_is_required_at_native_hold_height_and_z2_retains_old_gate():
    s = state(2)
    s.support_n[:] = .08
    s.object_lift_m[1] = .03
    c = BimanualCoordinator(2, mode='Z3', resume_dwell_s=0.)
    issued, diag = c.project(actions(2), s, dt_s=1/480)
    assert diag['support_ready'].tolist() == [True, False]
    assert diag['at_native_hold_height'].tolist() == [False, True]
    assert (issued[1, [14, 17]] <= 0).all()
    raw = actions(2)
    raw[:, 17] = .2
    issued, diag = BimanualCoordinator(2, mode='Z2', resume_dwell_s=0.).project(raw, s, dt_s=1/480)
    assert (issued[:, 17] == 0).all()
    assert not diag['support_ready'].any()


def test_unknown_or_excess_slip_and_tracking_failure_pause_both_hands():
    s = state(4)
    c = BimanualCoordinator(4, mode='Z3', resume_dwell_s=0.)
    s.slip_applicable[:2] = True
    s.slip_valid[1] = True
    s.slip_m[1] = .00051
    s.joint_tracking_error_rad[2, 1] = .031
    s.contact_valid[3, 1] = False
    issued, diag = c.project(actions(4), s, dt_s=1/480)
    assert (issued[:, [14, 17]] <= 0).all()
    assert not diag['support_ready'].any()
    assert diag['cancel_upward_target'].all()


def test_per_episode_reset_clears_stability_and_relative_grasp_reference():
    s = state(2)
    c = BimanualCoordinator(2, mode='Z3', resume_dwell_s=.02)
    c.project(actions(2), s, dt_s=.02)
    s.episode_id[0] += 1
    s.palm_world_m[0, :, 2] += .04
    issued, diag = c.project(actions(2), s, dt_s=.01)
    assert diag['episode_changed'].tolist() == [True, False]
    assert diag['resume_ready'].tolist() == [False, True]
    assert diag['relative_grasp_error_m'][0].tolist() == pytest.approx([0., 0.])


def test_z3_preserves_fingers_rotations_and_independent_xyz_budget():
    s = state()
    s.palm_world_m[0, 0, 2] += .004
    c = BimanualCoordinator(1, mode='Z3', resume_dwell_s=0.)
    issued, diag = c.project(actions(), s, dt_s=1/480)
    assert torch.equal(issued[:, :12], actions()[:, :12])
    assert torch.equal(issued[:, 18:], actions()[:, 18:])
    # Native Cartesian speed limiting still shares the translation budget.
    assert (issued[:, 12:18].reshape(1, 2, 3).norm(dim=-1) <= 1 + 1e-12).all()
    assert set(diag['objective']) == {'lift', 'contact', 'slip', 'joint', 'sync', 'total'}


def test_inactive_rows_are_zero_and_bad_measured_rows_cannot_move_up():
    s = state(2)
    s.live[0] = False
    s.palm_world_m[1, 1, 0] = float('nan')
    issued, diag = BimanualCoordinator(2, mode='Z3', resume_dwell_s=0.).project(
        actions(2), s, dt_s=1/480)
    assert not issued[0].any()
    assert (issued[1, [14, 17]] <= 0).all()
    assert not diag['measurement_valid'][1]


def test_zero_is_a_diagnostic_not_an_unbounded_replacement_action():
    with pytest.raises(ValueError, match='normalized'):
        BimanualCoordinator(1, mode='Z1').project(actions() * 2)
    with pytest.raises(ValueError, match='positive'):
        BimanualCoordinator(1, mode='Z3').project(actions(), state(), dt_s=0.)


def test_native_builder_keeps_no_contact_unknown_slip_and_target_frame_offset():
    from go2_pvcnn.control.dh116.tabletop_evaluation_telemetry import EvaluationTelemetry

    roots = torch.zeros(1, 2, 7, dtype=torch.float64)
    roots[:, :, 3] = 1.
    roots[:, :, 2] = .08
    queued = roots.new_zeros(1, 2, 3)
    queued[:, :, 2] = .004
    targets = roots.clone()
    targets[:, :, 2] += .003
    world = SimpleNamespace(n=1, roots=roots, palms=queued, root_targets=targets,
        wp=roots.new_tensor([[0., 0., .06]]), vertical_components=roots.new_zeros(1, 2, 2),
        object_weight_n=roots.new_ones(1),
        wire=SimpleNamespace(data=SimpleNamespace(root_com_pos_w=roots.new_tensor([[.001, 0., .06]]))))
    measurement = dict(
        effective_pad_loads_n=roots.new_zeros(1, 2, 5),
        effective_pad_vectors_w=roots.new_zeros(1, 2, 5, 3),
        effective_pad_valid=torch.ones(1, 2, 5, dtype=torch.bool),
        effective_pad_points_w=roots.new_zeros(1, 2, 5, 3),
        rejected_contact_loads_n=roots.new_zeros(1, 2),
        vertical_hand_force_n=roots.new_zeros(1, 2),
        hand_clearance=roots.new_full((1, 2), .01), bottom=roots.new_tensor([.06]),
        q=roots.new_zeros(1, 12), mimic_errors=roots.new_zeros(1, 10),
    )
    task = SimpleNamespace(m=measurement, command=roots.new_zeros(1, 12),
        done=torch.zeros(1, dtype=torch.bool))
    telemetry = EvaluationTelemetry(1, 'cpu', lambda packet: None)
    telemetry.reference_bottom_m[:] = .06
    # Old terminal evidence is deliberately stale and cannot leak through reset.
    telemetry._last = dict(slip_m=roots.new_tensor([.009]), slip_valid=torch.tensor([True]),
        slip_applicable=torch.tensor([True]))
    env = SimpleNamespace(evaluation_telemetry=telemetry,
        diagnostic_episode_ids=torch.ones(1, dtype=torch.long))
    measured = coordination_state_from_native(world, task, env)
    assert not measured.opposed.any()
    assert measured.contact_valid.all()
    assert not measured.slip_applicable.any()
    assert not measured.slip_valid.any()
    assert torch.isnan(measured.slip_m).all()
    assert measured.palm_offset_m[:, :, 2].tolist()[0] == pytest.approx([.001, .001])
    assert measured.object_lift_m.item() == 0.
    assert measured.object_com_w.tolist() == [[.001, 0., .06]]
    raw = actions()
    raw[:, 17] = -.5
    issued, diag = BimanualCoordinator(1, mode='Z2').project(raw, measured, dt_s=1/480)
    assert issued[0, 17] == -.5
    assert not diag['support_ready'].any()
    world.wire.data.root_com_pos_w = None
    local_com = roots.new_tensor([[.001, 0., 0., 1., 0., 0., 0.]])
    world.wire.root_physx_view = SimpleNamespace(get_coms=lambda: local_com)
    world.wq = roots.new_tensor([[0., 0., 0., 1.]])
    measured = coordination_state_from_native(world, task, env)
    assert measured.object_com_w.tolist()[0] == pytest.approx([-.001, 0., .06])
