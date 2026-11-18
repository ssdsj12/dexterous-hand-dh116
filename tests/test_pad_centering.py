"""Native recovery geometry is never an effective-load measurement (CPU only)."""
import copy
import importlib
import importlib.util
from types import SimpleNamespace as NS

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_effective_pad_runtime as runtime


def module():
    name = 'go2_pvcnn.control.dh116.tabletop_pad_centering'
    assert importlib.util.find_spec(name) is not None, 'optional pad centering helper is missing'
    return importlib.import_module(name)


def tensor(x):
    return torch.tensor(x, dtype=torch.float64)


def sampler():
    links = ['base_link'] + ['finger'+s for s in (
        '11_link', '12_link', '13_link', '14_link', '21_link', '22_link', '23_link',
        '31_link', '32_link', '33_link', '41_link', '42_link', '43_link',
        '51_link', '52_link', '53_link')]
    hands = []
    for h in range(2):
        poses = torch.zeros(2, 17, 7, dtype=torch.float64); poses[..., 3] = 1.
        poses[..., 0] = torch.arange(2)[:, None]*2+h*.25
        hands.append(NS(body_names=links, data=NS(body_link_pose_w=poses)))
    s = runtime.PadContactSampler.__new__(runtime.PadContactSampler)
    s.n, s.device = 2, 'cpu'
    s.world = NS(n=2, device='cpu', dt=1/480, hands=hands)
    s.bindings, s.rows = [], {}
    for h in range(2):
        for bi, link in enumerate(links):
            digit = runtime.DISTAL_LINKS.index(link) if link in runtime.DISTAL_LINKS else None
            view = NS(filter_count=1, calls=0, data=[
                torch.full((8, 1), float('nan'), dtype=torch.float64),
                torch.full((8, 3), float('nan'), dtype=torch.float64),
                torch.full((8, 3), float('nan'), dtype=torch.float64),
                torch.full((8, 1), float('nan'), dtype=torch.float64),
                torch.zeros(2, 1, dtype=torch.long), torch.zeros(2, 1, dtype=torch.long)])
            def read(dt, view=view):
                view.calls += 1
                return tuple(view.data)
            view.get_contact_data = read
            s.bindings.append((h, bi, digit, NS(contact_physx_view=view)))
            if digit is not None:
                s.rows[h, digit] = dict(center_local_m=tensor([0, -.01, .04]),
                    tangent_u=tensor([1, 0, 0]), tangent_v=tensor([0, 0, 1]),
                    half_widths_m=tensor([.004, .004]), planes_local=tensor([[0, -1, 0, -.01]]),
                    surface_tolerance_m=.001, normal_cosine_min=.8)
    return s


def binding(s, hand=0, digit=1):
    return next(row for row in s.bindings if row[0] == hand and row[2] == digit)


def put(s, env=0, hand=0, digit=1, slot=0, force=(1.483,),
        points=((.0042, -.01, .04),), normals=None):
    _, bi, _, sensor = binding(s, hand, digit)
    buf = sensor.contact_physx_view.data; k = len(force)
    pose = s.world.hands[hand].data.body_link_pose_w[env, bi]
    r = runtime._rotation(pose)
    buf[0][slot:slot+k, 0] = tensor(force)
    buf[1][slot:slot+k] = tensor(points) @ r.T+pose[:3]
    buf[2][slot:slot+k] = tensor(normals or [[0, 1, 0]]*k) @ r.T
    buf[3][slot:slot+k, 0] = -.0001
    buf[4][env, 0] = k; buf[5][env, 0] = slot
    return buf


def inputs(s):
    velocity = torch.zeros(2, 2, 5, 3, dtype=torch.float64)
    inward = torch.zeros_like(velocity); inward[..., 1] = -1.
    object_pose = torch.zeros(2, 7, dtype=torch.float64); object_pose[:, 3] = 1.
    return velocity, inward, object_pose, torch.zeros(2, 6, dtype=torch.float64)


def no_contacts(s):
    for *_, sensor in s.bindings:
        sensor.contact_physx_view.data[4].zero_()


def test_outside_ellipse_recovers_geometry_without_promoting_valid_load():
    s = sampler(); put(s)
    before = s.sample()
    assert before['effective_pad_loads_n'][0, 0, 1] == 0
    assert before['rejected_contact_loads_n'][0, 0] == pytest.approx(1.483)
    v, inward, pose, speed = inputs(s); v[0, 0, 1, 1] = -.001
    helper = module().PadCentering(s)
    adjusted, diag = helper.adjust(v, inward, pose, speed, 1/60)
    assert diag['geometry_valid'][0, 0, 1]
    assert diag['ellipse'][0, 0, 1] == pytest.approx(1.1025)
    assert adjusted[0, 0, 1, 0] > 0
    assert (adjusted[0, 0, 1]*inward[0, 0, 1]).sum() == pytest.approx(0.)
    assert diag['positive_normal_suppressed'][0, 0, 1]
    after = s.sample()
    for key in before:
        torch.testing.assert_close(after[key], before[key], atol=0, rtol=0, equal_nan=True)


def test_original_rows_native_buffers_velocity_and_signed_force_are_not_modified():
    s = sampler(); put(s)
    s.sample(); signed = s.last_signed_force_w.clone()
    rows = copy.deepcopy(s.rows)
    raw = [[v.clone() for v in sensor.contact_physx_view.data] for *_, sensor in s.bindings]
    args = inputs(s); original_velocity = args[0].clone()
    module().PadCentering(s).adjust(*args, 1/60)
    for key, row in s.rows.items():
        for field in row:
            if torch.is_tensor(row[field]):
                torch.testing.assert_close(row[field], rows[key][field], atol=0, rtol=0)
            else: assert row[field] == rows[key][field]
    for (*_, sensor), snapshot in zip(s.bindings, raw):
        for value, before in zip(sensor.contact_physx_view.data, snapshot):
            torch.testing.assert_close(value, before, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(args[0], original_velocity, atol=0, rtol=0)
    torch.testing.assert_close(s.last_signed_force_w, signed, atol=0, rtol=0)


def test_only_ten_distal_sensors_are_read():
    s = sampler()
    module().PadCentering(s).adjust(*inputs(s), 1/60)
    for _, _, digit, sensor in s.bindings:
        assert sensor.contact_physx_view.calls == (1 if digit is not None else 0)


@pytest.mark.parametrize('radius,weight', [(.45, 0.), (.65, .5), (.85, 1.)])
def test_weight_ramp_and_normal_component_inside_original_mask(radius, weight):
    s = sampler(); put(s, points=((radius*.004, -.01, .04),))
    args = inputs(s); args[0][0, 0, 1] = tensor([-.002, -.0012, .0004])
    old = args[0].clone()
    new, diag = module().PadCentering(s).adjust(*args, 1/60)
    assert diag['weight'][0, 0, 1] == pytest.approx(weight)
    torch.testing.assert_close((new*args[1]).sum(-1), (old*args[1]).sum(-1), atol=1e-12, rtol=0)
    assert not diag['positive_normal_suppressed'].any()
    if not weight: torch.testing.assert_close(new, old, atol=1e-12, rtol=0)


def test_outside_patch_still_preserves_unloading_normal_velocity():
    s = sampler(); put(s)
    args = inputs(s); args[0][0, 0, 1, 1] = .001
    new, diag = module().PadCentering(s).adjust(*args, 1/60)
    torch.testing.assert_close((new*args[1]).sum(-1), (args[0]*args[1]).sum(-1), atol=1e-12, rtol=0)
    assert not diag['positive_normal_suppressed'].any()


def test_oblique_force_axis_is_preserved_even_when_pad_tangent_is_not_orthogonal_to_it():
    s = sampler(); put(s, points=((.003, -.01, .04),))
    args = inputs(s)
    args[1][0, 0, 1] = tensor([.6, -.8, 0])
    args[0][0, 0, 1] = tensor([.002, -.001, .0005])
    args[3][0, :3] = tensor([.002, .004, .006])
    new, diag = module().PadCentering(s).adjust(*args, 1/60)
    torch.testing.assert_close((new*args[1]).sum(-1), (args[0]*args[1]).sum(-1), atol=1e-12, rtol=0)
    assert diag['recovery_tangent_velocity'].norm(dim=-1).max() <= .003+1e-12
    assert abs((diag['recovery_tangent_velocity'][0, 0, 1]*args[1][0, 0, 1]).sum()) < 1e-12


def test_near_field_ellipse_two_point_two_five_is_a_hard_recovery_boundary():
    s = sampler()
    put(s, env=0, slot=3, points=((.006, -.01, .04),))
    put(s, env=1, slot=0, points=((.00601, -.01, .04),))
    args = inputs(s); args[0][:, 0, 1, 1] = -.001
    new, diag = module().PadCentering(s).adjust(*args, 1/60)
    assert diag['geometry_valid'][0, 0, 1] and not diag['geometry_valid'][1, 0, 1]
    assert diag['ellipse'][0, 0, 1] == pytest.approx(2.25)
    assert new[0, 0, 1, 1] == pytest.approx(0.)
    torch.testing.assert_close(new[1], args[0][1], atol=0, rtol=0)


def test_weighted_local_native_point_uses_forces_not_contact_count():
    s = sampler(); put(s, force=(1., 3.), points=((.002, -.01, .04), (.004, -.01, .04)))
    new, diag = module().PadCentering(s).adjust(*inputs(s), 1/60)
    torch.testing.assert_close(diag['contact_local_m'][0, 0, 1], tensor([.0035, -.01, .04]))
    assert diag['ellipse'][0, 0, 1] == pytest.approx(.875**2)
    assert new[0, 0, 1, 0] > 0


def test_rotation_frame_moves_pad_center_towards_native_contact_not_away():
    s = sampler(); _, bi, _, _ = binding(s)
    body = s.world.hands[0].data.body_link_pose_w[0, bi]
    body[:3] = tensor([.3, -.2, .4]); body[3:] = tensor([2**-.5, 0, 0, 2**-.5])
    put(s, points=((.003, -.01, .04),))
    args = inputs(s); args[1][0, 0, 1] = tensor([1, 0, 0]); args[0][0, 0, 1, 0] = -.001
    new, diag = module().PadCentering(s).adjust(*args, 1/60)
    torch.testing.assert_close(diag['tangent_error_w'][0, 0, 1], tensor([0, .003, 0]), atol=1e-12, rtol=0)
    assert new[0, 0, 1, 1] == pytest.approx(.00225)
    assert new[0, 0, 1, 0] == pytest.approx(-.001)


def test_object_point_velocity_includes_angular_cross_lever_and_is_tangent_limited():
    s = sampler(); put(s, points=((.0035, -.01, .04),))
    args = inputs(s); args[3][0] = tensor([0, 0, .001, 0, 0, 1.])
    new, diag = module().PadCentering(s, kp=0.).adjust(*args, 1/60)
    lever = diag['contact_point_w'][0, 0, 1]-args[2][0, :3]
    expected = args[3][0, :3]+torch.linalg.cross(args[3][0, 3:], lever)
    nn = args[1][0, 0, 1]; expected -= (expected*nn).sum()*nn
    expected *= min(1., .003/float(expected.norm()))
    torch.testing.assert_close(new[0, 0, 1], expected, atol=1e-12, rtol=0)


def test_cache_uses_current_link_pose_and_expires_after_point_one_seconds():
    s = sampler(); put(s, points=((.0035, -.01, .04),))
    h = module().PadCentering(s); args = inputs(s)
    h.adjust(*args, 1/60); no_contacts(s)
    _, bi, _, _ = binding(s)
    body = s.world.hands[0].data.body_link_pose_w[0, bi]
    body[:3] += tensor([0, 0, .1]); body[3:] = tensor([2**-.5, 0, 0, 2**-.5])
    args[1][0, 0, 1] = tensor([1, 0, 0])
    new, diag = h.adjust(*args, .1)
    assert diag['cached'][0, 0, 1] and diag['geometry_valid'][0, 0, 1]
    expected = runtime._rotation(body) @ tensor([.0035, -.01, .04])+body[:3]
    torch.testing.assert_close(diag['contact_point_w'][0, 0, 1], expected)
    assert new[0, 0, 1, 1] > 0
    new, diag = h.adjust(*args, .0001)
    assert not diag['geometry_valid'][0, 0, 1]
    torch.testing.assert_close(new, args[0], atol=0, rtol=0)


def test_reset_clears_only_requested_environment_cache():
    s = sampler(); put(s); put(s, env=1, slot=3)
    h = module().PadCentering(s); args = inputs(s)
    h.adjust(*args, 1/60); no_contacts(s); h.reset(torch.tensor([0]))
    new, diag = h.adjust(*args, 1/60)
    assert not diag['geometry_valid'][0].any() and diag['geometry_valid'][1, 0, 1]
    torch.testing.assert_close(new[0], args[0][0], atol=0, rtol=0)


@pytest.mark.parametrize('kind', ['far_hinge', 'wrong_face', 'opposite_normal', 'nonunit_normal',
                                 'negative_force', 'zero_force', 'nan_force', 'overlapping_owners'])
def test_unknown_or_invalid_contact_does_not_drive_recovery(kind):
    s = sampler()
    buf = put(s)
    if kind == 'far_hinge': buf = put(s, points=((0, -.01, .028),))
    elif kind == 'wrong_face': buf = put(s, points=((.003, .01, .04),))
    elif kind == 'opposite_normal': buf = put(s, normals=((0, -1, 0),))
    elif kind == 'nonunit_normal': buf = put(s, normals=((0, 2, 0),))
    elif kind == 'negative_force': buf[0][0, 0] = -.5
    elif kind == 'zero_force': buf[0][0, 0] = 0.
    elif kind == 'nan_force': buf[0][0, 0] = float('nan')
    elif kind == 'overlapping_owners': buf[4][1, 0] = 1; buf[5][1, 0] = 0
    args = inputs(s); args[0][0, 0, 1, 1] = -.001
    new, diag = module().PadCentering(s).adjust(*args, 1/60)
    assert not diag['geometry_valid'].any()
    torch.testing.assert_close(new, args[0], atol=0, rtol=0)


def test_bad_owner_read_clears_a_previously_valid_cache():
    s = sampler(); buf = put(s)
    helper = module().PadCentering(s); args = inputs(s)
    helper.adjust(*args, 1/60)
    buf[5][0, 0] = 100
    new, diag = helper.adjust(*args, 1/60)
    assert diag['bad_env'][0] and not diag['geometry_valid'][0].any()
    torch.testing.assert_close(new[0], args[0][0], atol=0, rtol=0)


def test_no_contacts_is_an_exact_passthrough_and_default_does_not_activate_elsewhere():
    s = sampler(); args = inputs(s); args[0][:] = .0007
    new, diag = module().PadCentering(s).adjust(*args, 1/60)
    assert not diag['geometry_valid'].any() and not diag['active'].any()
    torch.testing.assert_close(new, args[0], atol=0, rtol=0)


def test_explicit_lift_transport_survives_centering_that_would_track_a_stationary_rod():
    s=sampler();put(s,points=((0.,-.01,.0365),))
    args=inputs(s);transport=tensor([[0.,0.,.003],[0.,0.,.003]])
    args[0][...,2]=.003
    new,diag=module().PadCentering(s).adjust(*args,1/60,transport_velocity_w=transport)
    assert diag['applied'][0,0,1]
    assert new[0,0,1,2]>=.003-1e-12


def test_recovery_outside_pad_suppresses_extra_press_not_the_transport_normal_component():
    s=sampler();put(s,points=((.0042,-.01,.04),))
    args=inputs(s);transport=tensor([[0.,-.003,0.],[0.,-.003,0.]])
    args[0][...,1]=-.004
    new,diag=module().PadCentering(s).adjust(*args,1/60,transport_velocity_w=transport)
    assert diag['positive_normal_suppressed'][0,0,1]
    assert new[0,0,1,1]==pytest.approx(-.003)
