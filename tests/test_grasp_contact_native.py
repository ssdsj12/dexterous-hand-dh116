"""Arithmetic fixtures do not attest native contact or hardware feedback."""
import importlib
import torch
import pytest


def api():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_grasp_contact_native')


def fixture():
    poses = torch.tensor([[[0., 0., 0., 1., 0., 0., 0.]],
                          [[0., 0., 0., 1., 0., 0., 0.]]], dtype=torch.float64)
    row = dict(center_local_m=[0., 0., 0.], tangent_u=[1., 0., 0.],
               tangent_v=[0., 0., 1.], half_widths_m=[.004, .0035],
               planes_local=[[0., -1., 0., 0.]], surface_tolerance_m=.001,
               normal_cosine_min=.8)
    raw = (torch.tensor([.2, .3]), torch.tensor([[.001, 0., 0.], [0., 0., 0.]]),
           torch.tensor([[0., 1., 0.], [0., 1., 0.]]), torch.tensor([-.0001, -.0002]),
           torch.tensor([1, 1]), torch.tensor([1, 0]))
    return poses, row, raw


def test_new_native_contact_adapter_exists():
    assert callable(getattr(api(), 'reduce_pad_contact_buffers', None))


def test_raw_slot_owner_and_actual_normal_are_preserved():
    poses, row, raw = fixture()
    result = api().reduce_pad_contact_buffers([(0, 0, 0)], [raw], [poses, poses],
                                              {(0, 0): row}, 2)
    assert result[0]['raw_contacts'][0]['raw_slot'] == 1
    assert result[1]['raw_contacts'][0]['raw_slot'] == 0
    assert result[0]['raw_contacts'][0]['normal_on_hand_w'] == [0., 1., 0.]
    pad = result[0]['pads'][0]
    assert pad['accepted_load_n'] == pytest.approx(.3)
    assert pad['position_w_m'] == [0., 0., 0.]
    assert pad['normal_source'] == 'reduced_native_force_direction'


def test_wrong_normal_is_distinct_from_missing_contact_and_off_pad():
    poses, row, raw = fixture()
    raw = list(raw)
    raw[2] = torch.tensor([[0., -1., 0.], [0., -1., 0.]])
    result = api().reduce_pad_contact_buffers([(0, 0, 0)], [tuple(raw)],
                                              [poses, poses], {(0, 0): row}, 2)
    pad = result[0]['pads'][0]
    assert pad['load_n'] > .05 and pad['accepted_load_n'] == 0
    assert pad['wrong_normal_count'] == 1
    assert result[0]['raw_contacts'][0]['position_in_pad'] is True
    raw[1] = torch.tensor([[.02, 0., 0.], [.02, 0., 0.]])
    result = api().reduce_pad_contact_buffers([(0, 0, 0)], [tuple(raw)],
                                              [poses, poses], {(0, 0): row}, 2)
    assert result[0]['pads'][0]['wrong_normal_count'] == 0
    assert result[0]['pads'][0]['load_n'] == 0


def test_invalid_or_overlapping_native_intervals_never_become_no_contact():
    poses, row, raw = fixture()
    raw = list(raw); raw[5] = torch.tensor([0, 0])
    result = api().reduce_pad_contact_buffers([(0, 0, 0)], [tuple(raw)],
                                              [poses, poses], {(0, 0): row}, 2)
    assert all(not r['measurement_valid'] for r in result)
    assert all(not r['pads'][0]['valid'] for r in result)


def test_nan_contact_keeps_raw_invalidity_without_fabricating_a_contact():
    poses, row, raw = fixture()
    raw = list(raw); raw[1][1, 0] = float('nan')
    result = api().reduce_pad_contact_buffers([(0, 0, 0)], [tuple(raw)],
                                              [poses, poses], {(0, 0): row}, 2)
    assert result[0]['measurement_valid'] is False
    assert result[0]['raw_contacts'][0]['position_w_m'][0] is None
    assert result[0]['pads'][0]['accepted_load_n'] == 0
    assert result[1]['measurement_valid'] is True


def test_unused_slots_and_signed_release_force_do_not_create_compression():
    poses, row, raw = fixture()
    raw = list(raw); raw[0] = torch.tensor([-.2, -.3])
    result = api().reduce_pad_contact_buffers([(0, 0, 0)], [tuple(raw)],
                                              [poses, poses], {(0, 0): row}, 2)
    assert result[0]['measurement_valid'] is True
    assert result[0]['raw_contacts'][0]['normal_scalar_n'] < 0
    assert result[0]['pads'][0]['accepted_load_n'] == 0


def test_invalid_actual_link_pose_remains_unknown_without_manifold_points():
    poses, row, raw = fixture()
    poses[0, 0, 3:] = 0.
    raw = list(raw); raw[4] = torch.tensor([0, 0])
    result = api().reduce_pad_contact_buffers([(0, 0, 0)], [tuple(raw)],
                                              [poses, poses], {(0, 0): row}, 2)
    assert result[0]['measurement_valid'] is False
    assert result[0]['pads'][0]['valid'] is False
    assert result[1]['measurement_valid'] is True


def test_probe_cannot_issue_a_production_evaluation_certificate():
    assert api().validate_contact_probe_options(dict(enabled=True, diagnostics=True,
        checkpoint='frozen.pt', resume=None, num_envs=8, maximum=120)) == 1
    for change in ({'resume': 'old.pt'}, {'diagnostics': False}, {'maximum': 1200},
                   {'num_envs': 200}, {'checkpoint': None}):
        value = dict(enabled=True, diagnostics=True, checkpoint='frozen.pt',
                     resume=None, num_envs=8, maximum=120); value.update(change)
        with pytest.raises(ValueError): api().validate_contact_probe_options(value)


def test_contact_control_tick_uses_actual_substep_wrap_after_arbitrary_warmup():
    clock = None
    for step, substep in enumerate(range(1, 9), 37):
        clock = api().advance_contact_clock(clock, step, substep, 8)
        assert clock['control_step'] == 0
    clock = api().advance_contact_clock(clock, 45, 1, 8)
    assert clock['control_step'] == 1
    with pytest.raises(ValueError): api().advance_contact_clock(clock, 45, 2, 8)
    with pytest.raises(ValueError): api().advance_contact_clock(clock, 47, 3, 8)


def test_invalid_torque_remains_raw_unknown_without_aborting_contact_records():
    from test_grasp_contact_diagnostics import frame, row
    from go2_pvcnn.control.dh116.tabletop_grasp_contact_diagnostics import GraspContactDiagnostics
    estimate = torch.zeros(1, 12); estimate[0, 7] = float('nan')
    mask = torch.ones(1, 12, dtype=torch.bool); mask[0, 7] = False
    result = api().torque_row_fields(dict(estimated_torque_nm=estimate,
        effort_limit_nm=torch.full((1, 12), .2), torque_valid=mask,
        torque_source=api().TORQUE_SOURCE), 0)
    assert result['estimated_torque_nm'] is None
    assert result['raw_torque_diagnostics']['estimated_torque_nm'][7] is None
    r = row(); r.update(result)
    output = GraspContactDiagnostics('fixture').observe(frame(0, r))[0]
    assert output['source_validity']['contact'] is True
    assert output['source_validity']['torque'] is False
