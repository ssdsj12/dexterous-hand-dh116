"""Pure reference geometry; no simulation, contact or grasp qualification."""
import importlib
import importlib.util
import math

import pytest
import torch


def module():
    name = 'go2_pvcnn.control.dh116.tabletop_pinch_reference'
    assert importlib.util.find_spec(name) is not None, 'pinch reference helpers are missing'
    return importlib.import_module(name)


def cloud(dtype=torch.float64):
    return torch.tensor([[[0., 0., 0.], [1., 0., 0.],
                          [0., 2., 0.], [0., 0., 3.]]], dtype=dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('shape', [(2, 3), (2, 4, 3), (2, 2, 4, 3)])
def test_reflect_y_uses_each_object_center_preserves_xz_and_is_an_involution(dtype, shape):
    points = torch.arange(torch.tensor(shape).prod().item(), dtype=dtype).reshape(shape) / 10
    center = torch.tensor([[3., .7, 9.], [-4., -.3, 2.]], dtype=dtype)
    original, original_center = points.clone(), center.clone()
    reflected = module().reflect_y(points, center)
    assert reflected.shape == points.shape and reflected.dtype == dtype
    assert reflected.device == points.device
    assert torch.equal(reflected[..., [0, 2]], points[..., [0, 2]])
    cy = center[:, 1].reshape((2,) + (1,) * (points.ndim - 2))
    torch.testing.assert_close(reflected[..., 1] - cy, -(points[..., 1] - cy))
    torch.testing.assert_close(module().reflect_y(reflected, center), points)
    assert torch.equal(points, original) and torch.equal(center, original_center)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_rigid_fit_recovers_batched_rotations_and_translations_with_row_vector_convention(dtype):
    source = cloud(dtype).repeat(2, 1, 1)
    source[1] = source[1] * .003 + torch.tensor([.2, -.3, .4], dtype=dtype)
    expected_r = torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
                               [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]], dtype=dtype)
    expected_t = torch.tensor([[.3, -.5, .1], [2., 3., -4.]], dtype=dtype)
    target = source @ expected_r.transpose(-1, -2) + expected_t[:, None, :]
    before_source, before_target = source.clone(), target.clone()
    r, t = module().proper_rigid_fit(source, target)
    assert r.shape == (2, 3, 3) and t.shape == (2, 3)
    assert r.dtype == dtype and t.dtype == dtype
    assert r.device == source.device and t.device == source.device
    torch.testing.assert_close(source @ r.transpose(-1, -2) + t[:, None, :], target)
    torch.testing.assert_close(r, expected_r, atol=2e-4 if dtype == torch.float32 else 1e-12, rtol=0)
    torch.testing.assert_close(t, expected_t)
    assert torch.equal(source, before_source) and torch.equal(target, before_target)


def test_rigid_fit_accepts_three_noncollinear_planar_points():
    source = cloud()[:, :3]
    target = source[..., [2, 0, 1]] + torch.tensor([.1, -.2, .3])
    r, t = module().proper_rigid_fit(source, target)
    torch.testing.assert_close(source @ r.transpose(-1, -2) + t[:, None, :], target)
    torch.testing.assert_close(torch.linalg.det(r), torch.ones(1, dtype=r.dtype))


def test_reflected_volumetric_cloud_gets_best_proper_fit_and_cannot_be_fit_exactly():
    source = cloud()
    target = module().reflect_y(source, torch.tensor([[.3, .5, -.2]], dtype=source.dtype))
    r, t = module().proper_rigid_fit(source, target)
    torch.testing.assert_close(torch.linalg.det(r), torch.ones(1, dtype=r.dtype))
    torch.testing.assert_close(r @ r.transpose(-1, -2), torch.eye(3, dtype=r.dtype)[None])
    residual = (source @ r.transpose(-1, -2) + t[:, None, :] - target).square().sum()
    centered = source[0] - source[0].mean(0)
    best_proper_error = 4 * torch.linalg.eigvalsh(centered.T @ centered)[0]
    assert residual > .1
    torch.testing.assert_close(residual, best_proper_error)


@pytest.mark.parametrize('bad_side', ['source', 'target'])
@pytest.mark.parametrize('degeneracy', ['coincident', 'collinear'])
def test_fit_rejects_a_degenerate_slot_even_in_an_otherwise_valid_batch(bad_side, degeneracy):
    source, target = cloud().repeat(2, 1, 1), cloud().repeat(2, 1, 1)
    bad = source if bad_side == 'source' else target
    bad[1] = torch.tensor([2., 3., 4.], dtype=bad.dtype)
    if degeneracy == 'collinear':
        bad[1] += torch.arange(4, dtype=bad.dtype)[:, None] * torch.tensor([1., 2., -3.])
    with pytest.raises(ValueError, match='rank|collinear'):
        module().proper_rigid_fit(source, target)


@pytest.mark.parametrize('side', ['points', 'center'])
@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_reflect_rejects_nonfinite_inputs(side, value):
    points, center = cloud(), torch.zeros(1, 3, dtype=torch.float64)
    (points if side == 'points' else center).reshape(-1)[0] = value
    with pytest.raises(ValueError, match='finite'):
        module().reflect_y(points, center)


@pytest.mark.parametrize('side', ['source', 'target'])
@pytest.mark.parametrize('value', [float('nan'), -float('inf')])
def test_fit_rejects_nonfinite_inputs(side, value):
    source, target = cloud(), cloud()
    (source if side == 'source' else target).reshape(-1)[0] = value
    with pytest.raises(ValueError, match='finite'):
        module().proper_rigid_fit(source, target)


@pytest.mark.parametrize('points,center', [
    (torch.zeros(4, 3), torch.zeros(3)),
    (torch.zeros(1, 4, 2), torch.zeros(1, 3)),
    (torch.zeros(0, 4, 3), torch.zeros(0, 3)),
    (torch.zeros(1, 0, 3), torch.zeros(1, 3)),
    (torch.zeros(1, 4, 3), torch.zeros(2, 3)),
    (torch.zeros(1, 4, 3), torch.zeros(1, 1, 3)),
    (torch.zeros(1, 4, 3, dtype=torch.int64), torch.zeros(1, 3)),
    (torch.zeros(1, 4, 3), torch.zeros(1, 3, dtype=torch.float64)),
    (torch.zeros(1, 4, 3), [[0., 0., 0.]]),
])
def test_reflect_rejects_malformed_or_mismatched_inputs(points, center):
    with pytest.raises(ValueError):
        module().reflect_y(points, center)


@pytest.mark.parametrize('source,target', [
    (torch.zeros(4, 3), torch.zeros(4, 3)),
    (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)),
    (torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)),
    (torch.zeros(0, 4, 3), torch.zeros(0, 4, 3)),
    (cloud(), cloud().repeat(2, 1, 1)),
    (cloud(), cloud()[:, :3]),
    (cloud(), cloud().float()),
    (cloud().long(), cloud().long()),
    (cloud(), cloud().to(torch.complex128)),
    (cloud(), [[[0., 0., 0.]]]),
])
def test_fit_rejects_malformed_or_mismatched_inputs(source, target):
    with pytest.raises(ValueError):
        module().proper_rigid_fit(source, target)


@pytest.mark.parametrize('operation', ['reflect', 'fit'])
def test_device_mismatch_is_rejected_before_tensor_operations(operation):
    if operation == 'reflect':
        args = cloud(), torch.empty(1, 3, device='meta', dtype=torch.float64)
        function = module().reflect_y
    else:
        args = cloud(), torch.empty(1, 4, 3, device='meta', dtype=torch.float64)
        function = module().proper_rigid_fit
    with pytest.raises(ValueError, match='device'):
        function(*args)


def frame(position=(0., 0., 0.), tilt=0., dtype=torch.float64):
    from go2_pvcnn.control.dh116.tabletop_surface_kinematics import pose_matrix
    pose = torch.tensor([[*position, math.cos(tilt / 2), math.sin(tilt / 2), 0., 0.]],
                        dtype=dtype)
    return pose_matrix(pose)


def test_reframing_tilted_object_lifts_negative_end_finger_instead_of_only_shifting_center():
    source = frame((.1, .2, .35), tilt=.052)
    target = frame((.4, -.3, .308))
    local_finger = frame((0., -.25, .008), tilt=.2)
    actual_finger = source @ local_finger
    reframed = module().reframe_transforms(actual_finger, source, target)
    torch.testing.assert_close(reframed, target @ local_finger)
    center_only = actual_finger.clone()
    center_only[:, :3, 3] += target[:, :3, 3] - source[:, :3, 3]
    lift_correction = (reframed[0, 2, 3] - center_only[0, 2, 3]).item()
    assert .0129 < lift_correction < .0131
    torch.testing.assert_close(reframed[0, 2, 3], torch.tensor(.316, dtype=reframed.dtype))
    assert not torch.allclose(reframed[:, :3, :3], center_only[:, :3, :3])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('extra_shape', [(), (3,), (2, 3)])
def test_reframe_broadcast_preserves_relative_geometry_and_roundtrips(dtype, extra_shape):
    source = torch.cat((frame((.1, .2, .4), .15, dtype),
                        frame((-.2, .3, .5), -.3, dtype)))
    target = torch.cat((frame((.4, -.1, .3), -.1, dtype),
                        frame((.6, .2, .3), .25, dtype)))
    count = math.prod(extra_shape)
    local = frame((.01, -.2, .008), .4, dtype).repeat(2 * count, 1, 1)
    local[:, 0, 3] += torch.arange(2 * count, dtype=dtype) * .002
    local = local.reshape((2,) + extra_shape + (4, 4))
    broadcast = (2,) + (1,) * len(extra_shape) + (4, 4)
    transforms = source.reshape(broadcast) @ local
    before = [x.clone() for x in (transforms, source, target)]
    result = module().reframe_transforms(transforms, source, target)
    assert result.shape == transforms.shape and result.dtype == dtype
    assert result.device == transforms.device
    torch.testing.assert_close(result, target.reshape(broadcast) @ local)
    torch.testing.assert_close(torch.linalg.inv(target).reshape(broadcast) @ result, local)
    torch.testing.assert_close(module().reframe_transforms(result, target, source), transforms)
    for value, original in zip((transforms, source, target), before):
        assert torch.equal(value, original)


@pytest.mark.parametrize('slot', ['transforms', 'source', 'target'])
@pytest.mark.parametrize('fault', ['nan', 'inf', 'reflection', 'scale', 'shear', 'last_row'])
def test_reframe_rejects_nonfinite_and_improper_rigid_matrices_in_each_input(slot, fault):
    values = dict(transforms=frame().repeat(2, 2, 1, 1),
                  source=frame().repeat(2, 1, 1), target=frame().repeat(2, 1, 1))
    matrix = values[slot].reshape(-1, 4, 4)[-1]
    if fault == 'nan':
        matrix[0, 3] = float('nan')
    elif fault == 'inf':
        matrix[0, 0] = float('inf')
    elif fault == 'reflection':
        matrix[1, 1] = -1.
    elif fault == 'scale':
        matrix[0, 0] = 1.01
    elif fault == 'shear':
        matrix[0, 1] = .01
    else:
        matrix[3, 0] = .01
    with pytest.raises(ValueError):
        module().reframe_transforms(values['transforms'], values['source'], values['target'])


@pytest.mark.parametrize('fault', ['unbatched', 'empty', 'wrong_tail', 'source_batch',
                                  'source_extra_axis', 'target_batch', 'integer',
                                  'dtype', 'device', 'not_tensor'])
def test_reframe_rejects_bad_shapes_dtypes_and_devices(fault):
    transforms, source, target = frame().repeat(2, 3, 1, 1), frame().repeat(2, 1, 1), frame().repeat(2, 1, 1)
    if fault == 'unbatched':
        transforms = transforms[0, 0]
    elif fault == 'empty':
        transforms, source, target = transforms[:0], source[:0], target[:0]
    elif fault == 'wrong_tail':
        transforms = transforms[..., :3, :]
    elif fault == 'source_batch':
        source = source[:1]
    elif fault == 'source_extra_axis':
        source = source[:, None]
    elif fault == 'target_batch':
        target = target[:1]
    elif fault == 'integer':
        transforms = transforms.long()
    elif fault == 'dtype':
        target = target.float()
    elif fault == 'device':
        source = torch.empty(2, 4, 4, device='meta', dtype=source.dtype)
    else:
        target = target.tolist()
    with pytest.raises(ValueError):
        module().reframe_transforms(transforms, source, target)


@pytest.mark.parametrize('roll', [-1.3, .7, math.pi])
def test_gravity_cylinder_frame_discards_arbitrary_roll_about_cylinder_y(roll):
    from go2_pvcnn.control.dh116.tabletop_surface_kinematics import pose_matrix
    center = torch.tensor([[.1, -.2, .308]], dtype=torch.float64)
    pose = torch.cat((center, center.new_tensor([[math.cos(roll / 2), 0., math.sin(roll / 2), 0.]])), -1)
    actual = pose_matrix(pose)
    result = module().gravity_cylinder_frame(center, actual[:, :3, 1])
    torch.testing.assert_close(result, frame(center[0].tolist()))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_gravity_cylinder_frame_tilt_is_proper_and_reframing_preserves_relative_geometry(dtype):
    source_center = torch.tensor([[.1, .2, .35]], dtype=dtype)
    expected = frame(source_center[0].tolist(), .052, dtype)
    axis = expected[:, :3, 1] * 7.
    before_center, before_axis = source_center.clone(), axis.clone()
    source = module().gravity_cylinder_frame(source_center, axis)
    target = module().gravity_cylinder_frame(source_center.new_tensor([[0., 0., .308]]),
                                            axis.new_tensor([[0., 1., 0.]]))
    assert source.shape == (1, 4, 4) and source.dtype == dtype and source.device == axis.device
    torch.testing.assert_close(source, expected)
    torch.testing.assert_close(torch.linalg.det(source[:, :3, :3]), torch.ones(1, dtype=dtype))
    local = frame((0., -.25, .008), .2, dtype)
    moved = module().reframe_transforms(source @ local, source, target)
    torch.testing.assert_close(moved, target @ local)
    assert torch.equal(source_center, before_center) and torch.equal(axis, before_axis)


def test_gravity_cylinder_frame_near_vertical_uses_finite_proper_fallback():
    axes = torch.tensor([[0., 0., 1.], [0., 0., -1.], [.001, .001, 1.],
                         [0., 1e-300, 0.], [0., 1e300, 0.]], dtype=torch.float64)
    result = module().gravity_cylinder_frame(torch.zeros_like(axes), axes)
    r = result[:, :3, :3]
    assert torch.isfinite(result).all()
    torch.testing.assert_close(r.transpose(-1, -2) @ r, torch.eye(3, dtype=r.dtype).expand_as(r))
    torch.testing.assert_close(torch.linalg.det(r), torch.ones(len(axes), dtype=r.dtype))
    scaled = axes / axes.abs().amax(-1, keepdim=True)
    torch.testing.assert_close(r[:, :, 1], scaled / scaled.norm(dim=-1, keepdim=True))
    torch.testing.assert_close(r[:2, :, 0], torch.tensor([[0., 1., 0.], [0., -1., 0.]], dtype=r.dtype))


@pytest.mark.parametrize('bad_axis', [[0., 0., 0.], [0., float('nan'), 0.], [0., float('inf'), 0.]])
def test_gravity_cylinder_frame_rejects_bad_axis_slots(bad_axis):
    center = torch.zeros(2, 3, dtype=torch.float64)
    axis = center.new_tensor([[0., 1., 0.], bad_axis])
    with pytest.raises(ValueError):
        module().gravity_cylinder_frame(center, axis)


def test_gravity_cylinder_frame_rejects_bad_center_shape_dtype_and_device():
    center = torch.zeros(1, 3, dtype=torch.float64)
    axis = center.new_tensor([[0., 1., 0.]])
    bad_inputs = [(center[:, :2], axis), (center, axis.repeat(2, 1)),
                  (center[:, None], axis), (center[:0], axis[:0]),
                  (center.float(), axis), (center.long(), axis),
                  (center, torch.empty_like(axis, device='meta')),
                  (center + float('nan'), axis)]
    for bad_center, bad_axis in bad_inputs:
        with pytest.raises(ValueError):
            module().gravity_cylinder_frame(bad_center, bad_axis)


def seek_packet(dtype=torch.float64):
    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    lower, upper = (torch.as_tensor(a, dtype=dtype).repeat(2) for a in active_limits())
    seed = ((lower + upper) / 2).repeat(2, 1)
    return seed.clone(), seed, torch.zeros(2, 2, 2, dtype=dtype), torch.zeros(2, 2, 2, dtype=dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_contact_seek_digits_independently_close_hold_and_back_off_at_strict_thresholds(dtype):
    command, seed, loads, depth = seek_packet(dtype)
    loads[:] = loads.new_tensor([[[0., .1], [100., 0.]], [[0., 100.], [.2, 0.]]])
    depth[:] = depth.new_tensor([[[0., .0009], [.0008, .0008]],
                                 [[.0009, 0.], [0., 0.]]])
    originals = [v.clone() for v in (command, seed, loads, depth)]
    result = module().contact_seek(command, seed, loads, depth, .1)
    expected = command.clone()
    expected[:, [1, 2, 7, 8]] += command.new_tensor([[.002, -.002, 0., 0.], [-.002, 0., 0., .002]])
    torch.testing.assert_close(result, expected)
    assert result.dtype == dtype and result.device == command.device
    for value, original in zip((command, seed, loads, depth), originals):
        assert torch.equal(value, original)


def test_contact_seek_high_force_is_not_a_release_trigger_but_depth_always_is():
    command, seed, loads, depth = seek_packet()
    loads.fill_(100.)
    depth[1] = .000801
    result = module().contact_seek(command, seed, loads, depth, .1)
    torch.testing.assert_close(result[0], command[0])
    torch.testing.assert_close(result[1, [1, 2, 7, 8]], command[1, [1, 2, 7, 8]] - .002)


def test_contact_seek_never_changes_ab_or_other_fingers_and_does_not_drift_when_holding():
    command, seed, loads, depth = seek_packet()
    untouched = [0, 3, 4, 5, 6, 9, 10, 11]
    command[:, untouched] += .2
    original = command.clone()
    for _ in range(100):
        command = module().contact_seek(command, seed, loads, depth, .1)
    assert torch.equal(command[:, untouched], original[:, untouched])
    torch.testing.assert_close(command[:, [1, 2, 7, 8]], seed[:, [1, 2, 7, 8]] + .12)
    loads.fill_(100.)
    assert torch.equal(module().contact_seek(command, seed, loads, depth, .1), command)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_contact_seek_respects_search_hardware_and_per_call_rate_bounds(dtype):
    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    command, seed, loads, depth = seek_packet(dtype)
    lower, upper = (seed.new_tensor(a).repeat(2) for a in active_limits())
    selected = [1, 2, 7, 8]
    seed[0, selected] = upper[selected] - .0001
    seed[1, selected] = lower[selected] + .0001
    command = seed.clone()
    small_step = module().contact_seek(command, seed, loads, depth, .01)
    assert ((small_step - command).abs() <= .0002 + 1e-7).all()
    closed = module().contact_seek(command, seed, loads, depth, 1., rate_rad_s=.3)
    expected_upper = (seed + .12).minimum(upper)
    torch.testing.assert_close(closed[:, selected], expected_upper[:, selected])
    depth.fill_(.001)
    opened = module().contact_seek(closed, seed, loads, depth, 1., rate_rad_s=.3)
    expected_lower = (seed - .03).maximum(lower)
    torch.testing.assert_close(opened[:, selected], expected_lower[:, selected])
    assert ((opened - closed).abs() <= .3 + 1e-7).all()
    assert ((opened >= lower) & (closed <= upper)).all()


@pytest.mark.parametrize('fault', ['command_hardware', 'seed_hardware', 'command_envelope'])
def test_contact_seek_rejects_infeasible_joint_targets(fault):
    command, seed, loads, depth = seek_packet()
    if fault == 'command_hardware':
        command[1, 1] = -1.
    elif fault == 'seed_hardware':
        seed[1, 0] = 5.
    else:
        command[1, 1] = seed[1, 1] + .121
    with pytest.raises(ValueError):
        module().contact_seek(command, seed, loads, depth, .1)


def test_contact_seek_rejects_bad_shapes_dtypes_devices_and_nonfinite_joint_targets():
    packet = seek_packet()
    bad = [(0, packet[0][:, :6]), (1, packet[1][:1]), (2, packet[2][:, 0]),
           (3, packet[3][:, :, :1]), (0, packet[0].long()), (2, packet[2].float()),
           (1, torch.empty_like(packet[1], device='meta')),
           (0, packet[0] + float('nan')), (1, packet[1] + float('inf'))]
    for slot, replacement in bad:
        args = list(packet)
        args[slot] = replacement
        with pytest.raises(ValueError):
            module().contact_seek(*args, .1)
    with pytest.raises(ValueError):
        module().contact_seek(*(v[:0] for v in packet), .1)


@pytest.mark.parametrize('fault', ['negative_load', 'negative_depth', 'nan_load', 'inf_depth'])
def test_contact_seek_rejects_invalid_feedback_per_slot(fault):
    command, seed, loads, depth = seek_packet()
    if fault == 'negative_load':
        loads[1, 1, 1] = -.001
    elif fault == 'negative_depth':
        depth[1, 1, 1] = -.001
    elif fault == 'nan_load':
        loads[1, 1, 1] = float('nan')
    else:
        depth[1, 1, 1] = float('inf')
    with pytest.raises(ValueError):
        module().contact_seek(command, seed, loads, depth, .1)


def test_contact_seek_rejects_invalid_scalar_controls():
    bad = [dict(dt=0.), dict(dt=-.1), dict(dt=float('inf')), dict(dt=True),
           dict(rate_rad_s=0.), dict(rate_rad_s=.3001), dict(rate_rad_s=float('nan')),
           dict(min_contact_n=0.), dict(min_contact_n=-1.), dict(min_contact_n=float('inf')),
           dict(max_depth_m=0.), dict(max_depth_m=-1.), dict(max_depth_m=float('nan'))]
    for overrides in bad:
        kwargs = dict(dt=.1)
        kwargs.update(overrides)
        with pytest.raises(ValueError):
            module().contact_seek(*seek_packet(), **kwargs)


def test_manifold_depth_uses_only_each_environments_valid_contacts():
    separation = torch.tensor([float('nan'), .0003, -.0002, -.0005, float('nan')], dtype=torch.float64)
    count = torch.tensor([2, 1, 0], dtype=torch.int32)
    start = torch.tensor([1, 3, 0], dtype=torch.int32)
    depth = module().manifold_depth(separation, count, start)
    torch.testing.assert_close(depth, separation.new_tensor([.0002, .0005, 0.]))
    assert torch.isnan(separation[0])


def test_manifold_depth_positive_gap_is_not_penetration_and_empty_buffer_is_valid():
    value = torch.tensor([.0002], dtype=torch.float32)
    one, zero = torch.tensor([1]), torch.tensor([0])
    assert torch.equal(module().manifold_depth(value, one, zero), value.new_zeros(1))
    assert torch.equal(module().manifold_depth(value[:0], zero, zero), value.new_zeros(1))


@pytest.mark.parametrize('fault', ['negative_count', 'negative_start', 'out_of_bounds',
                                  'nonfinite_active', 'float_count', 'wrong_shape', 'device'])
def test_manifold_depth_rejects_bad_contact_slices(fault):
    value = torch.tensor([-.0002], dtype=torch.float64)
    count, start = torch.tensor([1]), torch.tensor([0])
    if fault == 'negative_count': count[0] = -1
    elif fault == 'negative_start': start[0] = -1
    elif fault == 'out_of_bounds': count[0] = 2
    elif fault == 'nonfinite_active': value[0] = float('nan')
    elif fault == 'float_count': count = count.float()
    elif fault == 'wrong_shape': value = value[:, None]
    else: start = start.to('meta')
    with pytest.raises(ValueError):
        module().manifold_depth(value, count, start)
