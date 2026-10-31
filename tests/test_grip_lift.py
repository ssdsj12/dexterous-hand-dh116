"""Bounded joint closure and measured lift; no commanded-force qualification."""
import importlib
import importlib.util
import json
import math

import pytest
import torch


@pytest.mark.parametrize('extra', [
    ['--solve-articulation-contact-last'], ['--velocity-iterations','64']])
def test_diagnostic_cli_accepts_explicit_solver_probe_before_starting_isaac(tmp_path, monkeypatch, extra):
    import importlib,json,sys
    driver=importlib.import_module('scripts.run_dh116_grip_lift')
    candidate=tmp_path/'candidate.json'
    candidate.write_text(json.dumps({'candidates':[{'geometry':dict(penetration=0.,object_in_hand=0.,self=0.,minimum_clearance_m=.001)}]}))
    class ReachedLauncher(Exception): pass
    def stop_before_launch(): raise ReachedLauncher
    monkeypatch.setattr(driver,'awake_launcher',stop_before_launch)
    monkeypatch.setattr(sys,'argv',['driver','--candidates',str(candidate),'--output',str(tmp_path/'out'),*extra])
    with pytest.raises(ReachedLauncher):driver.entry()

from go2_pvcnn.control.dh116.linear_coupling import active_limits


def module():
    name = 'go2_pvcnn.control.dh116.tabletop_grip_lift'
    assert importlib.util.find_spec(name) is not None, 'grip/lift helpers are missing'
    return importlib.import_module(name)


def bounds(dtype=torch.float64):
    return tuple(torch.as_tensor(a, dtype=dtype).repeat(2) for a in active_limits())


def angles(n=2, dtype=torch.float64):
    low, high = bounds(dtype)
    return ((low + high) / 2).repeat(n, 1)


@pytest.mark.parametrize('batched_reference', [False, True])
def test_closing_changes_only_the_requested_thumb_and_index_coordinates(batched_reference):
    seed = angles()
    reference = seed.clone() + .01
    reference = reference if batched_reference else reference[0]
    margin = torch.tensor([.02, .04], dtype=seed.dtype)
    spread = torch.tensor([.01, .03], dtype=seed.dtype)
    expected = reference.expand_as(seed).clone()
    expected[:, [1, 2, 7, 8]] += margin[:, None]
    expected[:, [0, 6]] += spread[:, None]
    original_seed, original_reference = seed.clone(), reference.clone()
    actual = module().closing_targets(seed, reference, margin, thumb_spread=spread)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(seed, original_seed)
    torch.testing.assert_close(reference, original_reference)


def test_closing_saturates_at_the_actual_coupled_limits_and_preserves_dtype():
    seed = angles(dtype=torch.float32)
    low, high = bounds(seed.dtype)
    closed = module().closing_targets(seed, high, torch.full((2,), .3),
                                      thumb_spread=torch.full((2,), .3))
    assert closed.dtype == seed.dtype and closed.device == seed.device
    assert ((closed >= low) & (closed <= high)).all()
    torch.testing.assert_close(closed, high.expand_as(seed))


@pytest.mark.parametrize('spread', [.4, .6, .8])
def test_large_thumb_opposition_is_allowed_but_clipped_to_actual_angles(spread):
    seed = angles()
    low, high = bounds()
    closed = module().closing_targets(seed, seed, torch.zeros(2),
                                      thumb_spread=torch.full((2,), spread, dtype=seed.dtype))
    expected = seed.clone()
    expected[:, [0, 6]] = (seed[:, [0, 6]] + spread).minimum(high[[0, 6]])
    torch.testing.assert_close(closed, expected)
    assert ((closed >= low) & (closed <= high)).all()


@pytest.mark.parametrize('bad', ['nan', 'outside', 'shape'])
def test_closing_rejects_invalid_seed(bad):
    seed = angles()
    if bad == 'nan':
        seed[0, 0] = float('nan')
    elif bad == 'outside':
        seed[0, 1] = 0.  # The affine thumb follower forbids this master value.
    else:
        seed = seed[:, :11]
    with pytest.raises(ValueError):
        module().closing_targets(seed, angles()[0], torch.zeros(2))


@pytest.mark.parametrize('argument,maximum', [('margin', .3), ('thumb_spread', .8)])
@pytest.mark.parametrize('bad', [-.001, 'above_limit', float('nan'), float('inf')])
def test_closing_rejects_invalid_offsets(argument, maximum, bad):
    kwargs = dict(margin=torch.zeros(2), thumb_spread=torch.zeros(2))
    kwargs[argument][0] = maximum + .001 if bad == 'above_limit' else bad
    with pytest.raises(ValueError):
        module().closing_targets(angles(), angles()[0], **kwargs)


def test_closing_validates_reference_and_offset_shapes():
    with pytest.raises(ValueError):
        module().closing_targets(angles(), torch.zeros(11), torch.zeros(2))
    reference = angles(); reference[0, 0] = float('nan')
    with pytest.raises(ValueError):
        module().closing_targets(angles(), reference, torch.zeros(2))
    with pytest.raises(ValueError):
        module().closing_targets(angles(), angles(), torch.zeros(2, 1))


def test_program_interpolates_then_holds_and_returns_a_bounded_lift_goal():
    seed = angles(5)
    closed = module().closing_targets(seed, seed, torch.full((5,), .1))
    elapsed = torch.tensor([-1., 0., 1., 2., 100.], dtype=seed.dtype)
    actual, z = module().advance_program(seed, seed, closed, elapsed, 1.)
    expected = seed + (closed - seed) * torch.tensor([0., 0., .5, 1., 1.])[:, None]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(z, torch.tensor([0., 0., 0., 0., .04], dtype=seed.dtype))
    held, z = module().advance_program(closed, seed, closed, elapsed.new_full((5,), 100.), 1 / 60)
    torch.testing.assert_close(held, closed)
    torch.testing.assert_close(z, z.new_full((5,), .04))


def test_program_rate_limits_each_joint_without_mutating_inputs():
    seed = angles()
    low, high = bounds()
    closed = high.expand_as(seed).clone()
    command = low.expand_as(seed).clone()
    before = command.clone()
    actual, z = module().advance_program(command, seed, closed,
                                         seed.new_tensor([3., 4.]), .01)
    assert (actual - command).abs().max() <= .003 + 1e-12
    assert ((actual >= low) & (actual <= high)).all()
    torch.testing.assert_close(command, before)
    torch.testing.assert_close(z, z.new_tensor([.004, .012]))


@pytest.mark.parametrize('kwargs', [
    {'dt': 0.}, {'dt': -1.}, {'dt': float('nan')}, {'close_s': 0.},
    {'close_s': float('inf')}, {'lift_start_s': 1.9}, {'lift_start_s': float('nan')},
    {'lift_speed_m_s': -.001}, {'lift_speed_m_s': .011},
    {'lift_height_m': -.001}, {'lift_height_m': .051},
])
def test_program_rejects_invalid_timing_and_lift_bounds(kwargs):
    options = dict(dt=1 / 60); options.update(kwargs)
    with pytest.raises(ValueError):
        module().advance_program(angles(), angles(), angles(), torch.zeros(2), **options)


def test_program_rejects_nonfinite_state_and_elapsed():
    bad = angles(); bad[0, 0] = float('inf')
    with pytest.raises(ValueError):
        module().advance_program(bad, angles(), angles(), torch.zeros(2), .01)
    with pytest.raises(ValueError):
        module().advance_program(angles(), angles(), angles(), torch.tensor([0., float('nan')]), .01)


def measurement(n=2):
    pose = torch.zeros(n, 7, dtype=torch.float64)
    pose[:, 2] = .339; pose[:, 3] = 1.
    return [pose, torch.zeros(n, 6, dtype=torch.float64),
            torch.tensor([[20., 0.]], dtype=torch.float64).repeat(n, 1),
            torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)]


def test_large_force_or_only_one_bearing_hand_can_qualify_but_no_contact_cannot():
    state = module().LiftDwell(3, 'cpu')
    sample = measurement(3)
    sample[2][1] = torch.tensor([0., .06])
    sample[2][2] = 0.
    for _ in range(29):
        assert not state.update(*sample, 1 / 60).any()
    assert state.update(*sample, 1 / 60).tolist() == [True, True, False]
    assert state.ever_qualified.tolist() == [True, True, False]
    torch.testing.assert_close(state.bottom, torch.full((3,), .331, dtype=torch.float64))
    torch.testing.assert_close(state.maximum_height_m, torch.full((3,), .031, dtype=torch.float64))


def test_height_is_the_oriented_rod_bottom_instead_of_center_height():
    state = module().LiftDwell(2, 'cpu')
    sample = measurement()
    sample[0][:, 2] = .34
    sample[0][0, 3:] = torch.tensor([math.sqrt(.5), math.sqrt(.5), 0., 0.])
    for _ in range(30):
        actual = state.update(*sample, 1 / 60)
    assert actual.tolist() == [False, True]
    torch.testing.assert_close(state.bottom, torch.tensor([.09, .332], dtype=torch.float64), atol=1e-8, rtol=0)


@pytest.mark.parametrize('bad', ['zero_quaternion', 'nonunit_quaternion', 'nan_quaternion',
                                  'nan_position', 'nonfinite_velocity', 'nonfinite_load', 'negative_load',
                                  'invalid', 'reset', 'speed', 'insufficient_height'])
def test_bad_evidence_breaks_only_the_affected_candidates_current_dwell(bad):
    state = module().LiftDwell(2, 'cpu')
    sample = measurement()
    if bad == 'zero_quaternion': sample[0][0, 3:] = 0.
    elif bad == 'nonunit_quaternion': sample[0][0, 3] = 1.01
    elif bad == 'nan_quaternion': sample[0][0, 3] = float('nan')
    elif bad == 'nan_position': sample[0][0, 0] = float('nan')
    elif bad == 'nonfinite_velocity': sample[1][0, 0] = float('inf')
    elif bad == 'nonfinite_load': sample[2][0, 0] = float('nan')
    elif bad == 'negative_load': sample[2][0] = sample[2].new_tensor([-1., 20.])
    elif bad == 'invalid': sample[3][0] = False
    elif bad == 'reset': sample[4][0] = True
    elif bad == 'speed': sample[1][0, 0] = .03
    else: sample[0][0, 2] = .337
    for _ in range(30):
        actual = state.update(*sample, 1 / 60)
    assert actual.tolist() == [False, True]
    assert state.hold_s[0] == 0. and not state.last_good[0]


def test_reset_and_invalid_clear_current_success_but_preserve_history():
    state = module().LiftDwell(2, 'cpu')
    sample = measurement()
    for _ in range(30): state.update(*sample, 1 / 60)
    sample[4][0] = True
    assert state.update(*sample, 1 / 60).tolist() == [False, True]
    assert state.ever_qualified.tolist() == [True, True]
    assert state.longest_hold_s[0] == pytest.approx(.5)
    sample[4][0] = False; sample[3][1] = False
    for _ in range(29): assert not state.update(*sample, 1 / 60).any()
    assert state.update(*sample, 1 / 60).tolist() == [True, False]
    assert state.ever_qualified.tolist() == [True, True]


def test_contact_loss_clears_current_success_without_auto_latching_it():
    state = module().LiftDwell(2, 'cpu')
    sample = measurement()
    for _ in range(30): state.update(*sample, 1 / 60)
    sample[2].zero_()
    assert not state.update(*sample, 1 / 60).any()
    assert state.ever_qualified.all() and not state.hold_s.any()


def test_summary_is_strict_json_even_after_invalid_pose():
    state = module().LiftDwell(2, 'cpu')
    sample = measurement(); sample[0][0, 3:] = 0.
    state.update(*sample, 1 / 60)
    summary = state.summary()
    json.dumps(summary, allow_nan=False)
    assert summary['qualified'] == [False, False]
    assert summary['ever_qualified'] == [False, False]


def test_dwell_validates_shapes_flags_and_dt_before_mutating_history():
    state = module().LiftDwell(2, 'cpu')
    sample = measurement(); state.update(*sample, .01)
    before = state.hold_s.clone()
    with pytest.raises(ValueError): state.update(*sample, 0.)
    malformed = measurement(); malformed[3] = torch.ones(2)
    with pytest.raises(ValueError): state.update(*malformed, .01)
    malformed = measurement(); malformed[0] = torch.zeros(2, 6)
    with pytest.raises(ValueError): state.update(*malformed, .01)
    torch.testing.assert_close(state.hold_s, before)


def readback():
    pose, velocity, hand_load, _, _ = measurement()
    full_pose = torch.zeros(2, 2, 17, 7, dtype=torch.float64)
    full_pose[..., 3] = 1.
    return dict(pose=pose, velocity=velocity, q=angles(), qd=torch.zeros(2, 12),
                hand_load=hand_load, full_pose=full_pose, full_qd=torch.zeros(2, 2, 23),
                joint_failure=torch.zeros(2, dtype=torch.long),
                adapter_bits=torch.zeros(2, dtype=torch.long),
                clearance=torch.full((2, 2), .001, dtype=torch.float64))


@pytest.mark.parametrize('flattened', [False, True])
def test_valid_readback_accepts_full_or_flattened_link_axes_without_force_upper_bound(flattened):
    packet = readback()
    if flattened:
        packet['full_pose'] = packet['full_pose'].reshape(2, 34, 7)
        packet['full_qd'] = packet['full_qd'].reshape(2, -1)
    packet['clearance'][0, 0] = -.001
    packet['pose'][0, 3] = 1.0005
    packet['full_pose'][1, ..., 3] = 1.0005
    assert module().valid_readback(**packet).tolist() == [True, True]


@pytest.mark.parametrize('key,index,value', [
    ('pose', (0, 0), float('nan')),
    ('velocity', (0, 1), float('inf')),
    ('q', (0, 3), float('nan')),
    ('qd', (0, 4), float('inf')),
    ('hand_load', (0, 0), -.001),
    ('hand_load', (0, 1), float('nan')),
    ('full_pose', (0, 1, 16, 0), float('nan')),
    ('full_pose', (0, 1, 16, 3), 0.),
    ('full_pose', (0, 0, 3, 3), 1.002),
    ('pose', (0, 3), 0.),
    ('pose', (0, 3), 1.002),
    ('full_qd', (0, 1, 22), float('nan')),
    ('joint_failure', (0,), 1),
    ('adapter_bits', (0,), 4),
    ('clearance', (0, 1), -.00101),
    ('clearance', (0, 0), float('nan')),
])
def test_valid_readback_rejects_bad_physical_state_only_in_affected_slot(key, index, value):
    packet = readback(); packet[key][index] = value
    assert module().valid_readback(**packet).tolist() == [False, True]


@pytest.mark.parametrize('key,shape', [('pose', (2, 6)), ('velocity', (2, 5)),
                                    ('q', (2, 11)), ('qd', (2, 13)),
                                    ('hand_load', (2, 1)), ('full_pose', (2, 34, 6)),
                                    ('full_qd', (3, 4)), ('joint_failure', (2, 1)),
                                    ('adapter_bits', (3,)), ('clearance', (2,))])
def test_valid_readback_rejects_malformed_shapes(key, shape):
    packet = readback(); packet[key] = torch.zeros(shape)
    with pytest.raises(ValueError):
        module().valid_readback(**packet)
