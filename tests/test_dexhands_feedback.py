"""CPU contract tests: FK proximity is distinct from measured normal contact."""
import importlib
import importlib.util

import pytest
import torch


MODULE = 'go2_pvcnn.control.dh116.tabletop_dexhands_feedback'
FIELDS = {
    'pad_relative': (4, 3), 'object_quat': (4,), 'bottom': (),
    'grouped_object_loads': (2, 6), 'hand_object_loads': (2,),
    'object_velocity': (6,), 'wrist_velocity': (2, 6),
}


def feedback():
    assert importlib.util.find_spec(MODULE) is not None, 'DexHands feedback module is missing'
    return importlib.import_module(MODULE)


def measured(n=1, dtype=torch.float64):
    m = {key: torch.zeros((n, *shape), dtype=dtype) for key, shape in FIELDS.items()}
    m['object_quat'][:, 0] = 1.
    m['bottom'].fill_(.30)
    m['pad_relative'][:, :, 0] = .008
    return m


def loaded(n=1, dtype=torch.float64):
    m = measured(n, dtype)
    m['grouped_object_loads'][:, :, :2] = 1.
    m['hand_object_loads'][:] = 2.
    return m


def rates(m, **kwargs):
    target = m['hand_object_loads'].new_full((m['bottom'].shape[0], 2), 2.)
    return feedback().reward_rates(m, target, **kwargs)


def test_module_exists_before_behavior_tests():
    feedback()


def test_feature_order_finite_cylinder_caps_inside_and_corner():
    m = measured()
    m['pad_relative'][0] = torch.tensor([
        [.016, .0, .0], [0., .258, .0], [0., .0, .0], [.016, .258, .0],
    ], dtype=torch.float64)
    m['grouped_object_loads'][0] = torch.tensor([
        [0., 1., 80., 90., 100., 110.], [.05, 3., 120., 130., 140., 150.],
    ], dtype=torch.float64)
    features = feedback().geometry_features(m)
    assert features.shape == (1, 20)
    torch.testing.assert_close(features[0, :4], torch.tensor([1., 1., 1., 2**.5], dtype=torch.float64))
    torch.testing.assert_close(features[0, 4:8], torch.zeros(4, dtype=torch.float64))
    torch.testing.assert_close(features[0, 8:12], torch.tensor([0., 1.032, 0., 1.032], dtype=torch.float64))
    torch.testing.assert_close(features[0, 12:16], torch.log1p(torch.tensor([0., 1., .05, 3.], dtype=torch.float64)))
    torch.testing.assert_close(features[0, 16:], torch.tensor([0., 1., 0., 1.], dtype=torch.float64))


def test_radial_world_z_removes_axial_component_for_tilted_local_y_axis():
    m = measured()
    # +90 degrees around X maps cylinder local Y onto world Z.
    m['object_quat'][0] = torch.tensor([2**-.5, 2**-.5, 0., 0.])
    m['pad_relative'][0, :, :] = torch.tensor([.008, 0., .125])
    f = feedback().geometry_features(m)
    torch.testing.assert_close(f[0, :8], torch.zeros(8, dtype=torch.float64), atol=1e-6, rtol=0.)
    torch.testing.assert_close(f[0, 8:12], torch.full((4,), .5, dtype=torch.float64), atol=1e-6, rtol=0.)


def test_custom_dimensions_keep_distance_scale_at_eight_mm():
    m = measured()
    m['pad_relative'][0, :, :] = torch.tensor([.024, .5, -.008])
    f = feedback().geometry_features(m, radius_m=.016, length_m=2.)
    expected = ((.024**2 + .008**2)**.5 - .016) / .008
    torch.testing.assert_close(f[0, :4], torch.full((4,), expected, dtype=torch.float64))
    torch.testing.assert_close(f[0, 4:8], torch.full((4,), -1., dtype=torch.float64))
    torch.testing.assert_close(f[0, 8:12], torch.full((4,), .5, dtype=torch.float64))


@pytest.mark.parametrize('dtype,scale', [
    (torch.float32, 1e-38), (torch.float32, 1e38),
    (torch.float64, 1e-300), (torch.float64, 1e300),
])
def test_tiny_huge_and_sign_flipped_quaternions_are_normalized(dtype, scale):
    m = loaded(dtype=dtype)
    m['object_quat'][:] = torch.tensor([.25, .5, .75, 1.], dtype=dtype)
    expected = feedback().geometry_features(m)
    m['object_quat'] *= scale
    torch.testing.assert_close(feedback().geometry_features(m), expected)
    m['object_quat'].neg_()
    torch.testing.assert_close(feedback().geometry_features(m), expected)
    assert all(torch.isfinite(v).all() for v in rates(m).values())


def test_fk_on_surface_does_not_create_measured_contacts():
    m = measured()
    features = feedback().geometry_features(m)
    assert not features[:, :4].any()
    assert not features[:, 12:].any()
    r = rates(m)
    assert r['dex_reach'].item() == pytest.approx(6.)
    assert r['dex_contact'].item() == r['dex_lift'].item() == 0.


def test_reach_improves_with_distance_and_each_of_four_pads_matters():
    m = measured(6)
    m['pad_relative'][1, :, 0] += .008
    for pad in range(4):
        m['pad_relative'][pad + 2, pad, 0] += .008
    reach = rates(m)['dex_reach']
    assert reach[0] == pytest.approx(6.)
    assert reach[1] < reach[2] < reach[0]
    torch.testing.assert_close(reach[2:], reach[2].expand(4))
    expected = 3. * ((3. + torch.exp(torch.tensor(-1.))) / 4. + torch.exp(torch.tensor(-1.)))
    assert reach[2] == pytest.approx(expected.item())


def test_contact_credits_each_digit_and_extra_joint_support_with_per_digit_saturation():
    m = loaded(6)
    for digit in range(4):
        m['grouped_object_loads'][digit + 1].reshape(-1)[(digit // 2) * 6 + digit % 2] = 0.
    m['grouped_object_loads'][5, :, :2] = 100.
    contact = rates(m)['dex_contact']
    torch.testing.assert_close(contact, torch.tensor([4., 2.25, 2.25, 2.25, 2.25, 4.], dtype=torch.float64))
    m['grouped_object_loads'][:, :, :2] = .25
    torch.testing.assert_close(rates(m)['dex_contact'], torch.ones(6, dtype=torch.float64))


@pytest.mark.parametrize('requested_n',[2.,8.,12.])
def test_first_new_contact_improves_feedback_without_waiting_for_four_contacts(requested_n):
    m=measured(3)
    m['grouped_object_loads'][1,0,0]=.1
    m['grouped_object_loads'][2,0,0]=.2
    m['hand_object_loads'][:,0]=torch.tensor([0.,.1,.2])
    r=feedback().reward_rates(m,torch.full((3,2),requested_n,dtype=torch.float64))
    assert 0<r['dex_contact'][1]<r['dex_contact'][2]
    combined=sum(r[key] for key in ('dex_contact','dex_force','dex_balance'))
    assert ((combined[1:]-combined[:-1])>.01).all()


def test_lift_uses_actual_bottom_with_half_mm_deadzone_and_thirty_mm_saturation():
    m = loaded(6)
    m['bottom'][:] = torch.tensor([.299, .30, .3005, .3155, .3305, .50], dtype=torch.float64)
    m['object_velocity'][:, 2] = .01
    torch.testing.assert_close(rates(m)['dex_lift'], torch.tensor([0., 0., 0., 30., 60., 60.], dtype=torch.float64))


def test_empty_or_unilateral_ascent_has_no_lift_even_when_fk_is_near():
    m = loaded(4)
    m['bottom'][:] = .4
    m['hand_object_loads'][:] = torch.tensor([[0., 0.], [2., 0.], [0., 2.], [.05, 2.]], dtype=torch.float64)
    m['object_velocity'][:, 2] = 1.
    m['wrist_velocity'][:, :, 2] = 10.
    assert not rates(m)['dex_lift'].any()


def test_lift_bilateral_load_gate_is_soft_above_contact_threshold():
    m = loaded(3)
    m['bottom'][:] = .3305
    m['hand_object_loads'][:] = torch.tensor([[.05, 2.], [.525, 2.], [1., 2.]], dtype=torch.float64)
    torch.testing.assert_close(rates(m)['dex_lift'], torch.tensor([0., 30., 60.], dtype=torch.float64))


def test_wrist_height_orientation_and_velocity_do_not_supply_lift_credit():
    m = loaded(2)
    m['bottom'][:] = .30
    m['wrist_pose'] = torch.full((2, 2, 7), 1000., dtype=torch.float64)
    m['wrist_velocity'][1] = 1000.
    m['object_quat'][1] = torch.tensor([1., 1., 1., 1.])
    m['pad_relative'][1] = 10.
    assert not rates(m)['dex_lift'].any()


def test_force_tracking_is_bounded_real_load_error_for_each_requested_target():
    m = loaded(4)
    target = torch.tensor([[2., 12.]] * 4, dtype=torch.float64)
    m['hand_object_loads'][:] = torch.tensor([[2., 12.], [1., 6.], [0., 0.], [1e100, 1e100]], dtype=torch.float64)
    force = feedback().reward_rates(m, target)['dex_force']
    torch.testing.assert_close(force, torch.tensor([0., -2., -4., -4.], dtype=torch.float64))


def test_balance_cost_starts_above_max_min_ratio_one_point_five():
    m = loaded(4)
    m['grouped_object_loads'][:, :, :2] = torch.tensor([
        [[1., 1.], [0., 0.]], [[1.5, 1.], [1.5, 1.]],
        [[3., 1.], [3., 1.]], [[1., 0.], [0., 1.]],
    ], dtype=torch.float64)
    torch.testing.assert_close(rates(m)['dex_balance'], torch.tensor([0., 0., -1., -1.], dtype=torch.float64))


def test_balance_ratio_boundary_and_one_point_six_to_one_are_distinguished():
    m = loaded(4)
    m['grouped_object_loads'][:, :, :2] = torch.tensor([
        [[1.49, 1.], [1., 1.49]], [[1.5, 1.], [1., 1.5]],
        [[1.6, 1.], [1., 1.6]], [[1., 1.6], [1.6, 1.]],
    ], dtype=torch.float64)
    cost = rates(m)['dex_balance']
    torch.testing.assert_close(cost[:2], torch.zeros(2, dtype=torch.float64))
    assert (cost[2:] < 0.).all()
    torch.testing.assert_close(cost[2:], torch.full((2,), -.1, dtype=torch.float64))


def test_action_change_cost_is_optional_bounded_and_does_not_change_other_rates():
    m = loaded(3)
    a = torch.ones(3, 26, dtype=torch.float64)
    p = a.clone()
    p[1] = 0.
    p[2] = -1.
    base = rates(m)
    assert not base['dex_action_rate'].any()
    assert not rates(m, actions=a)['dex_action_rate'].any()
    changed = rates(m, actions=a, previous_actions=p)
    torch.testing.assert_close(changed['dex_action_rate'], torch.tensor([0., -.125, -.5], dtype=torch.float64))
    for key in base:
        if key != 'dex_action_rate':
            torch.testing.assert_close(changed[key], base[key])
    a.fill_(1e100)
    torch.testing.assert_close(rates(m, actions=a, previous_actions=-a)['dex_action_rate'], torch.full((3,), -.5, dtype=torch.float64))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_rates_are_finite_n_vectors_stateless_and_do_not_mutate_measurements(dtype):
    m = loaded(3, dtype)
    before = {k: v.clone() for k, v in m.items()}
    first = rates(m)
    assert set(first) == {'dex_reach', 'dex_contact', 'dex_lift', 'dex_force', 'dex_balance', 'dex_action_rate'}
    for k, v in first.items():
        assert v.shape == (3,) and v.dtype == dtype and v.device.type == 'cpu'
        assert torch.isfinite(v).all()
        torch.testing.assert_close(v, rates(m)[k])
    for key in m:
        torch.testing.assert_close(m[key], before[key])
    m.pop('wrist_velocity')
    for k, v in rates(m).items():
        torch.testing.assert_close(v, first[k])


def test_contract_documents_feature_scales_rate_limits_and_single_dt_integration():
    c = feedback().CONTRACT
    assert c['feature_size'] == 20 and len(c['feature_order']) == 20
    assert c['distance_scale_m'] == .008 and c['bearing_threshold_n'] == .05
    assert c['balance_max_min_ratio'] == 1.5
    assert c['table_top_m'] == .30 and c['lift_deadzone_m'] == .0005
    assert c['rate_bounds_per_s'] == {
        'dex_reach': [0., 6.], 'dex_contact': [0., 4.], 'dex_lift': [0., 60.],
        'dex_force': [-4., 0.], 'dex_balance': [-2., 0.], 'dex_action_rate': [-.5, 0.],
    }
    assert c['dt_application'] == 'caller_multiplies_rates_by_dt_once'
    assert c['stage_flow'] is None and c['pose_latch'] is False


@pytest.mark.parametrize('key', list(FIELDS))
@pytest.mark.parametrize('api', ['geometry_features', 'reward_rates'])
def test_bad_measurement_shape_rejected(key, api):
    m = measured(2)
    m[key] = torch.zeros(2, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match=key):
        if api == 'geometry_features':
            feedback().geometry_features(m)
        else:
            rates(m)


@pytest.mark.parametrize('key', list(FIELDS))
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_measurements_rejected(key, bad):
    m = measured(2)
    m[key].reshape(-1)[0] = bad
    with pytest.raises(ValueError, match=key):
        feedback().geometry_features(m)
    with pytest.raises(ValueError, match=key):
        rates(m)


@pytest.mark.parametrize('key', ['grouped_object_loads', 'hand_object_loads'])
def test_negative_normal_load_is_rejected_instead_of_becoming_fake_contact(key):
    m = measured()
    m[key].reshape(-1)[0] = -.001
    with pytest.raises(ValueError, match=key):
        feedback().geometry_features(m)
    with pytest.raises(ValueError, match=key):
        rates(m)


def test_zero_quaternion_is_rejected_in_both_apis():
    m = measured(2)
    m['object_quat'][1] = 0.
    with pytest.raises(ValueError, match='object_quat'):
        feedback().geometry_features(m)
    with pytest.raises(ValueError, match='object_quat'):
        rates(m)


@pytest.mark.parametrize('key', [k for k in FIELDS if k != 'wrist_velocity'])
def test_missing_required_measurement_is_rejected(key):
    m = measured()
    del m[key]
    with pytest.raises(ValueError, match=key):
        feedback().reward_rates(m, torch.full((1, 2), 2., dtype=torch.float64))


@pytest.mark.parametrize('key,value', [
    ('radius_m', 0.), ('length_m', -1.), ('radius_m', float('nan')),
    ('length_m', float('inf')),
])
def test_invalid_cylinder_dimensions_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        feedback().geometry_features(measured(), **{key: value})


@pytest.mark.parametrize('target', [torch.zeros(1, 1), torch.full((1, 2), float('nan')),
                                  torch.full((1, 2), 1.99), torch.full((1, 2), 12.01)])
def test_force_target_requires_n_by_two_and_finite_two_to_twelve_n(target):
    with pytest.raises(ValueError, match='requested_force'):
        feedback().reward_rates(measured(), target)


@pytest.mark.parametrize('kwargs', [
    {'actions': torch.zeros(1, 0)}, {'actions': torch.zeros(2, 26)},
    {'actions': torch.full((1, 26), float('nan'))},
    {'previous_actions': torch.full((1, 26), float('inf'))},
    {'actions': torch.zeros(1, 26), 'previous_actions': torch.zeros(1, 20)},
])
def test_invalid_action_data_rejected(kwargs):
    with pytest.raises(ValueError, match='actions'):
        rates(measured(), **kwargs)
