"""Static wrench proposals are never native force measurements."""
import importlib
import importlib.util

import numpy as np
import pytest

MODULE = 'go2_pvcnn.control.dh116.tabletop_tripod_support_allocation'


def allocate(**changes):
    assert importlib.util.find_spec(MODULE) is not None, 'missing COM-aware wrench proposal'
    values = dict(points_w_m=[[-1., 0., 0.], [1., 0., 0.]], object_com_w_m=[.5, 0., 0.],
        directions_w=[[0., 0., 1.], [0., 0., 1.]], capacities_n=[10., 10.],
        object_weight_n=10., force_to_joint_torque_nm_per_n=[[.01, 0.], [0., .01]],
        original_effort_limit_nm=[.2, .2], hand_ids=[0, 1])
    values.update(changes)
    return importlib.import_module(MODULE).allocate_support_wrench(**values)


def test_two_real_points_allocate_by_com_moment_arm_without_equal_share_assumption():
    out = allocate()
    assert out['feasible']
    assert out['target_hand_fz_n'] == pytest.approx([2.5, 7.5])
    assert out['contact_magnitude_n'] == pytest.approx([2.5, 7.5])
    assert out['force_residual_n'] == pytest.approx([0., 0., 0.], abs=1e-9)
    assert out['moment_residual_nm'] == pytest.approx([0., 0., 0.], abs=1e-9)
    assert out['evidence_kind'] == 'allocation_proposal_not_measured_force'
    assert out['support_bearing_attested'] is False


@pytest.mark.parametrize('changes', [
    dict(object_com_w_m=[2., 0., 0.]), dict(capacities_n=[10., 6.]),
    dict(original_effort_limit_nm=[.02, .02]),
    dict(directions_w=[[1., 0., 0.], [-1., 0., 0.]]),
])
def test_infeasible_com_directions_capacity_or_effort_is_not_clipped_into_balance(changes):
    out = allocate(**changes)
    assert not out['feasible']
    assert out['target_hand_fz_n'] is None
    assert out['contact_force_w_n'] is None


def test_missing_or_nonfinite_motor_evidence_stays_unknown():
    out = allocate(force_to_joint_torque_nm_per_n=None)
    assert not out['feasible'] and out['reason'] == 'unknown_motor_capacity'
    out = allocate(original_effort_limit_nm=[float('nan'), .2])
    assert not out['feasible'] and out['reason'] == 'invalid_measurement'


def test_general_three_point_wrench_equilibrium_and_joint_projection():
    out = allocate(points_w_m=[[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.]],
        object_com_w_m=[0., 0., .2], directions_w=[[0., 0., 2.]] * 3,
        capacities_n=[10.] * 3, hand_ids=[0, 0, 1],
        force_to_joint_torque_nm_per_n=[[.01, .01, 0.], [0., 0., .01]])
    assert out['feasible']
    assert out['contact_magnitude_n'] == pytest.approx([2.5, 2.5, 5.])
    assert out['target_joint_torque_nm'] == pytest.approx([.05, .05])


@pytest.mark.parametrize('changes', [
    dict(points_w_m=[[False, 0., 0.], [1., 0., 0.]]),
    dict(object_com_w_m=np.array([np.bool_(False), 0., 0.], dtype=object)),
    dict(directions_w=np.array([[False, False, True], [False, False, True]])),
    dict(capacities_n=[10., np.bool_(True)]),
    dict(object_weight_n=True),
    dict(object_weight_n=np.bool_(True)),
    dict(force_to_joint_torque_nm_per_n=[[np.bool_(False), 0.], [0., .01]]),
    dict(original_effort_limit_nm=np.array([True, True])),
])
def test_boolean_measurements_never_become_numeric_capacity(changes):
    out = allocate(**changes)
    assert not out['feasible'] and out['reason'] == 'invalid_measurement'
    assert out['contact_force_w_n'] is None


def test_boolean_nested_inside_an_object_array_is_rejected_before_float_coercion():
    capacities = np.empty(2, dtype=object)
    capacities[:] = [np.array([np.bool_(True)]), 10.]
    out = allocate(capacities_n=capacities)
    assert not out['feasible'] and out['reason'] == 'invalid_measurement'


@pytest.mark.parametrize('changes', [
    dict(object_com_w_m=[2., 0., 0.]), dict(capacities_n=[10., 6.]),
    dict(original_effort_limit_nm=[.02, .02]),
    dict(directions_w=[[1., 0., 0.], [-1., 0., 0.]]),
])
def test_infeasible_wrench_has_bounded_residual_evidence_without_a_proposal(changes):
    out = allocate(**changes)
    assert not out['feasible'] and out['reason'] == 'equilibrium_infeasible'
    for field in ('target_hand_fz_n', 'contact_force_w_n', 'contact_magnitude_n',
                  'target_joint_torque_nm'):
        assert out[field] is None
    assert out['force_residual_n'] is not None
    assert out['moment_residual_nm'] is not None
    assert np.isfinite(out['force_residual_n']).all()
    assert np.isfinite(out['moment_residual_nm']).all()
    diagnostic = out.get('infeasibility_diagnostic')
    assert diagnostic is not None
    assert diagnostic['evidence_kind'] == 'bounded_minimum_wrench_residual_not_command'
    assert diagnostic['normalized_linf_residual'] > 0.
    magnitude = np.asarray(diagnostic['bounded_candidate_contact_magnitude_n'])
    assert np.all(magnitude >= 0.)
    assert np.all(magnitude <= changes.get('capacities_n', [10., 10.]))
    torque = np.asarray(diagnostic['bounded_candidate_joint_torque_nm'])
    assert np.all(np.abs(torque) <= np.asarray(changes.get('original_effort_limit_nm', [.2, .2])) + 1e-9)


def test_capacity_shortfall_certificate_has_explicit_physical_units_and_normalization():
    out = allocate(capacities_n=[10., 6.])
    diagnostic = out.get('infeasibility_diagnostic')
    assert diagnostic is not None
    assert diagnostic['force_scale_n'] == 10.
    assert diagnostic['characteristic_lever_arm_m'] == 1.5
    assert diagnostic['moment_scale_nm'] == 15.
    # The bounded minimax solution [3, 6] N balances its normalized force
    # shortfall 1/10 against its normalized moment error 1.5/15.
    assert diagnostic['bounded_candidate_contact_magnitude_n'] == pytest.approx([3., 6.])
    assert diagnostic['normalized_linf_residual'] == pytest.approx(.1)
    assert out['force_residual_n'] == pytest.approx([0., 0., -1.])
    assert out['moment_residual_nm'] == pytest.approx([0., 1.5, 0.])


def test_com_outside_support_certificate_cannot_hide_tipping_with_force_clipping():
    out = allocate(object_com_w_m=[2., 0., 0.])
    diagnostic = out.get('infeasibility_diagnostic')
    assert diagnostic is not None
    assert diagnostic['normalized_linf_residual'] == pytest.approx(.25)
    assert out['force_residual_n'] == pytest.approx([0., 0., -2.5])
    assert out['moment_residual_nm'] == pytest.approx([0., 7.5, 0.])


def test_coincident_contacts_report_force_residual_without_inventing_a_length_scale():
    out = allocate(points_w_m=[[0., 0., 0.]] * 2, object_com_w_m=[0., 0., 0.],
                   capacities_n=[2., 2.])
    diagnostic = out.get('infeasibility_diagnostic')
    assert diagnostic is not None
    assert diagnostic['characteristic_lever_arm_m'] == 0.
    assert diagnostic['moment_scale_nm'] is None
    assert diagnostic['moment_rows_identically_zero'] is True
    assert diagnostic['normalized_linf_residual'] == pytest.approx(.6)
    assert out['moment_residual_nm'] == pytest.approx([0., 0., 0.])


def test_finite_overflow_in_derived_moment_is_rejected_instead_of_crashing():
    out = allocate(points_w_m=[[1e308, 0., 0.], [-1e308, 0., 0.]],
                   object_com_w_m=[-1e308, 0., 0.])
    assert not out['feasible'] and out['reason'] == 'invalid_measurement'


def test_large_finite_direction_ray_is_normalized_without_overflow():
    out = allocate(directions_w=[[0., 0., 1e308]] * 2)
    assert out['feasible']
    assert out['target_hand_fz_n'] == pytest.approx([2.5, 7.5])
