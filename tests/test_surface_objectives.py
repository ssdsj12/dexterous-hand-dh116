"""CPU geometry contracts; proximity is never measured physical contact."""
import importlib
import importlib.util
import math

import pytest
import torch


MODULE = 'go2_pvcnn.control.dh116.tabletop_surface_objectives'
DTYPE = torch.float64


def objectives():
    assert importlib.util.find_spec(MODULE) is not None, 'surface objectives module is missing'
    return importlib.import_module(MODULE)


def tensor(value):
    return torch.tensor(value, dtype=DTYPE)


def scene(n=1, p=4):
    center = tensor([[0., 0., .5]]).repeat(n, 1)
    quaternion = tensor([[1., 0., 0., 0.]]).repeat(n, 1)
    patches = tensor([.008, 0., .5]).expand(n, 2, 2, p, 3).clone()
    collisions = tensor([[[.010, 0., .5], [-.010, 0., .5]]]).repeat(n, 1, 1)
    return dict(patches=patches, collision_points=collisions,
                center=center, quaternion=quaternion)


def rotation(axis, angle):
    """Independent Rodrigues reference; vectors rotate local to world."""
    axis = tensor(axis)
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis.tolist()
    skew = tensor([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    matrix = (math.cos(angle) * torch.eye(3, dtype=DTYPE)
              + (1. - math.cos(angle)) * torch.outer(axis, axis)
              + math.sin(angle) * skew)
    quaternion = torch.cat((tensor([math.cos(angle / 2.)]), axis * math.sin(angle / 2.)))
    return matrix, quaternion


def test_signed_distance_distinguishes_side_caps_interior_and_exterior_corner():
    points = tensor([[[.016, 0., 0.], [0., .258, 0.], [0., -.258, 0.],
                      [0., 0., 0.], [.004, 0., 0.], [.008, 0., 0.],
                      [0., .25, 0.], [.014, .258, 0.], [.007, .2495, 0.]]])
    result = objectives().finite_cylinder_sdf(points, tensor([[0., 0., 0.]]),
                                              tensor([[1., 0., 0., 0.]]))
    torch.testing.assert_close(result, tensor([[.008, .008, .008, -.008, -.004,
                                               0., 0., .010, -.0005]]))


@pytest.mark.parametrize('tail', [(), (9,), (3, 3), (1, 3, 3)])
def test_arbitrary_translated_rotated_batch_and_point_dimensions(tail):
    local = tensor([[.016, 0., 0.], [0., .258, 0.], [.014, -.258, 0.],
                    [.004, .02, 0.], [.008, -.05, 0.], [0., -.258, 0.],
                    [0., 0., 0.], [0., .25, 0.], [0., 0., .020]])
    expected = tensor([.008, .008, .010, -.004, 0., .008, -.008, 0., .012])
    if not tail:
        local, expected = local[0], expected[0]
    else:
        local, expected = local.reshape(*tail, 3), expected.reshape(tail)
    centers = tensor([[.3, -.4, .6], [-1.2, .3, 1.7], [.01, -2., -.8]])
    rotations = [rotation([1., 2., -3.], .731), rotation([.5, -3., 2.], -1.21),
                 rotation([-1., 0., 2.], 2.03)]
    points = torch.stack([local @ r.T + c for (r, _), c in zip(rotations, centers)])
    q = torch.stack([q for _, q in rotations]) * tensor([[2.], [-3.], [7.]])
    result = objectives().finite_cylinder_sdf(points, centers, q)
    assert result.shape == (3, *tail)
    torch.testing.assert_close(result, expected.expand(3, *tail), atol=1e-14, rtol=1e-10)


@pytest.mark.parametrize('dtype,scale', [(torch.float32, 1e-38), (torch.float32, 1e38),
                                       (torch.float64, 1e-300), (torch.float64, 1e300)])
def test_quaternion_normalization_handles_extreme_finite_scales(dtype, scale):
    points = tensor([[[.016, .03, .01], [0., .26, 0.]]]).to(dtype)
    c = torch.zeros((1, 3), dtype=dtype)
    q = tensor([[.25, .5, .75, 1.]]).to(dtype)
    sdf = objectives().finite_cylinder_sdf
    expected = sdf(points, c, q)
    torch.testing.assert_close(sdf(points, c, q * scale), expected)
    torch.testing.assert_close(sdf(points, c, -q * scale), expected)


def test_sdf_gradients_point_outward_on_both_sides_caps_and_corner():
    points = tensor([[[.012, 0., 0.], [-.012, 0., 0.], [0., .26, 0.],
                      [0., -.26, 0.], [.014, .258, 0.], [.006, 0., 0.]]]).requires_grad_()
    sdf = objectives().finite_cylinder_sdf(points, tensor([[0., 0., 0.]]),
                                         tensor([[1., 0., 0., 0.]]))
    gradient, = torch.autograd.grad(sdf.sum(), points)
    torch.testing.assert_close(gradient, tensor([[[1., 0., 0.], [-1., 0., 0.],
                                                 [0., 1., 0.], [0., -1., 0.],
                                                 [.6, .8, 0.], [1., 0., 0.]]]))


def test_sdf_gradcheck_for_points_center_and_nonunit_quaternion():
    points = tensor([[[.031, .08, .017], [.05, .31, -.02]]]).requires_grad_()
    center = tensor([[.001, -.003, .002]]).requires_grad_()
    q = tensor([[1.7, .11, -.23, .31]]).requires_grad_()
    assert torch.autograd.gradcheck(objectives().finite_cylinder_sdf, (points, center, q))


def test_terms_have_only_geometry_outputs_per_row_and_do_not_mutate_inputs():
    data = scene(n=2)
    before = {k: v.clone() for k, v in data.items()}
    terms = objectives().surface_terms(**data)
    assert set(terms) == {'reach', 'penetration', 'table', 'normal_alignment',
                          'self', 'object_in_hand', 'patch_distance_m'}
    for name, value in terms.items():
        assert value.shape == ((2, 2, 2) if name == 'patch_distance_m' else (2,))
        assert value.dtype == DTYPE and value.device == data['patches'].device
        torch.testing.assert_close(value, torch.zeros_like(value))
    for name in data:
        torch.testing.assert_close(data[name], before[name])


@pytest.mark.parametrize('p,count,expected', [(4, 3, .003), (2, 3, .0015),
                                           (4, 1, .001), (4, 99, .00725)])
def test_reach_averages_nearest_absolute_sdf_samples_per_patch(p, count, expected):
    data = scene(p=p)
    # Signed distances: -1, +2, +6, +20 mm. Signed averaging is incorrect.
    data['patches'][..., 0] = tensor([.007, .010, .014, .028])[:p]
    result = objectives().surface_terms(**data, contact_points_per_patch=count)
    torch.testing.assert_close(result['patch_distance_m'], torch.full((1, 2, 2), expected, dtype=DTYPE))
    torch.testing.assert_close(result['reach'], tensor([expected / .008]))


def test_every_patch_matters_even_when_the_other_three_are_on_surface():
    data = scene(n=5)
    for i in range(4):
        data['patches'][i + 1, i // 2, i % 2, :, 0] += .008
    result = objectives().surface_terms(**data)
    torch.testing.assert_close(result['reach'], tensor([0., .25, .25, .25, .25]))
    assert (result['patch_distance_m'][1:].reshape(4, 4).diag() > .0079).all()


def test_inside_and_outside_reach_equal_but_collision_penetration_is_distinct_maximum():
    data = scene(n=2)
    data['patches'][0, ..., 0] = .006
    data['patches'][1, ..., 0] = .010
    data['collision_points'][0, :, 0] = tensor([.007, .005])
    result = objectives().surface_terms(**data)
    torch.testing.assert_close(result['reach'], tensor([.25, .25]))
    torch.testing.assert_close(result['penetration'], tensor([3., 0.]))


def test_custom_radius_and_length_keep_fixed_eight_mm_and_one_mm_scales():
    data = scene()
    data['patches'][..., 0] = .024
    data['collision_points'][0] = tensor([[.015, 0., .5], [.020, 0., .5]])
    result = objectives().surface_terms(**data, radius_m=.016, length_m=2.)
    torch.testing.assert_close(result['reach'], tensor([1.]))
    torch.testing.assert_close(result['penetration'], tensor([1.]))
    sdf = objectives().finite_cylinder_sdf(tensor([[[0., 1.02, 0.]]]),
                                         tensor([[0., 0., 0.]]), tensor([[1., 0., 0., 0.]]), .016, 2.)
    torch.testing.assert_close(sdf, tensor([[.02]]))


def test_table_uses_lowest_collision_sample_with_point_three_mm_clearance():
    data = scene(n=3)
    data['collision_points'][:, :, 2] = tensor([[.3003, .4], [.3000, .2993], [.4, .5]])
    result = objectives().surface_terms(**data)
    torch.testing.assert_close(result['table'], tensor([0., 1., 0.]), atol=1e-12, rtol=0.)
    custom = objectives().surface_terms(**data, table_z=.31)
    torch.testing.assert_close(custom['table'], tensor([10., 11., 0.]), atol=1e-12, rtol=0.)


def test_normal_alignment_cost_opposes_hand_and_object_outward_normals():
    data = scene(n=3)
    normals = torch.zeros_like(data['patches'])
    normals[0, ..., 0], normals[1, ..., 0], normals[2, ..., 2] = -7., 3., 2.
    result = objectives().surface_terms(**data, patch_normals=normals)
    torch.testing.assert_close(result['normal_alignment'], tensor([0., 2., 1.]))


def test_normal_alignment_uses_nearest_samples_only():
    data = scene(p=4)
    data['patches'][..., 0] = tensor([.008, .009, .010, .1])
    normals = torch.zeros_like(data['patches'])
    normals[..., 0] = tensor([-1., -1., -1., 1.])
    result = objectives().surface_terms(**data, patch_normals=normals)
    torch.testing.assert_close(result['normal_alignment'], tensor([0.]))


def test_normal_alignment_handles_local_y_caps_and_exterior_corner():
    data = scene(n=3)
    data['patches'][0] = tensor([0., .26, .5])
    data['patches'][1] = tensor([0., -.26, .5])
    data['patches'][2] = tensor([.014, .258, .5])
    normals = torch.empty_like(data['patches'])
    normals[0], normals[1], normals[2] = tensor([0., -1., 0.]), tensor([0., 1., 0.]), tensor([-.6, -.8, 0.])
    result = objectives().surface_terms(**data, patch_normals=normals)
    torch.testing.assert_close(result['normal_alignment'], torch.zeros(3, dtype=DTYPE), atol=1e-14, rtol=0.)


def test_surface_and_normal_terms_are_invariant_under_rigid_batch_transforms():
    data = scene(n=2)
    data['patches'][0] = tensor([.014, .258, .5])
    data['patches'][1] = tensor([.006, .01, .501])
    normals = tensor([-.5, -.7, .2]).expand_as(data['patches']).clone()
    expected = objectives().surface_terms(**data, patch_normals=normals)
    rotations = [rotation([1., -2., 3.], 1.23), rotation([-.5, 1., 2.], -.71)]
    for i, (r, q) in enumerate(rotations):
        offset = tensor([.3, -.4, 2.])
        for name in ('patches', 'collision_points', 'center'):
            data[name][i] = data[name][i] @ r.T + offset
        data['quaternion'][i] = -3. * q
        normals[i] = normals[i] @ r.T
    actual = objectives().surface_terms(**data, patch_normals=normals)
    for name in expected:
        if name != 'table':
            torch.testing.assert_close(actual[name], expected[name], atol=1e-12, rtol=1e-10)


def test_optional_penetration_terms_are_independent_nonnegative_depths_in_meters():
    data = scene(n=2)
    result = objectives().surface_terms(**data, self_penetration_m=tensor([.002, 0.]),
                                       object_hand_penetration_m=tensor([0., .003]))
    torch.testing.assert_close(result['self'], tensor([2., 0.]))
    torch.testing.assert_close(result['object_in_hand'], tensor([0., 3.]))
    torch.testing.assert_close(result['penetration'], tensor([0., 0.]))


def test_reach_gradient_moves_each_patch_toward_surface_and_ignores_distant_samples():
    data = scene(p=4)
    data['patches'][..., 0] = tensor([.010, .011, .012, .020])
    data['patches'].requires_grad_()
    result = objectives().surface_terms(**data)
    gradient, = torch.autograd.grad(result['reach'].sum(), data['patches'])
    torch.testing.assert_close(gradient[..., :3, 0], torch.full((1, 2, 2, 3), 1. / (.008 * 12), dtype=DTYPE))
    assert not gradient[..., 3, :].any()
    assert not gradient[..., 1:].any()


def test_penetration_table_and_supplemental_gradients_have_physical_signs():
    data = scene()
    data['collision_points'][0] = tensor([[.006, 0., .5], [.010, 0., .2993]])
    data['collision_points'].requires_grad_()
    self_depth = tensor([.002]).requires_grad_()
    hand_depth = tensor([.003]).requires_grad_()
    result = objectives().surface_terms(**data, self_penetration_m=self_depth,
                                       object_hand_penetration_m=hand_depth)
    gradients = torch.autograd.grad(result['penetration'].sum() + result['table'].sum()
                                    + result['self'].sum() + result['object_in_hand'].sum(),
                                    (data['collision_points'], self_depth, hand_depth))
    torch.testing.assert_close(gradients[0][0, 0], tensor([-1000., 0., 0.]))
    torch.testing.assert_close(gradients[0][0, 1], tensor([0., 0., -1000.]))
    torch.testing.assert_close(gradients[1], tensor([1000.]))
    torch.testing.assert_close(gradients[2], tensor([1000.]))


def test_normal_term_gradcheck_includes_geometry_normals_center_and_quaternion():
    data = scene(p=1)
    data['patches'][:] = tensor([.017, .258, .506])
    normals = tensor([-.4, -.8, .2]).expand_as(data['patches']).clone().requires_grad_()
    points = data['patches'].requires_grad_()
    center = data['center'].requires_grad_()
    q = tensor([[1.7, .002, -.003, .004]]).requires_grad_()

    def alignment(p, n, c, quat):
        return objectives().surface_terms(p, data['collision_points'], c, quat,
                                           patch_normals=n)['normal_alignment']

    assert torch.autograd.gradcheck(alignment, (points, normals, center, q), atol=2e-5)


@pytest.mark.parametrize('position', [[0., 0., .5], [.008, 0., .5],
                                     [0., .25, .5], [.008, .25, .5]])
def test_axis_surface_and_rim_conventions_have_finite_backward(position):
    data = scene(p=1)
    data['patches'][:] = tensor(position)
    data['patches'].requires_grad_()
    normals = tensor([-1., -.5, .2]).expand_as(data['patches']).clone().requires_grad_()
    result = objectives().surface_terms(**data, patch_normals=normals)
    gradients = torch.autograd.grad(result['reach'].sum() + result['normal_alignment'].sum(),
                                    (data['patches'], normals))
    assert all(torch.isfinite(g).all() for g in gradients)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_surface_terms_preserve_floating_dtype(dtype):
    data = {k: v.to(dtype) for k, v in scene().items()}
    result = objectives().surface_terms(**data)
    assert all(v.dtype == dtype for v in result.values())


@pytest.mark.parametrize('name,bad', [
    ('points', torch.zeros(3)), ('points', torch.zeros(1, 4)),
    ('points', torch.zeros(0, 3)), ('points', torch.zeros(1, 0, 3)),
    ('points', torch.zeros(1, 1, 3, dtype=torch.int64)), ('points', []),
    ('center', torch.zeros(3)), ('center', torch.zeros(2, 3)),
    ('quaternion', torch.ones(1, 3)), ('quaternion', torch.ones(2, 4)),
    ('quaternion', torch.zeros(1, 4)),
    ('center', torch.zeros(1, 3, device='meta')),
    ('quaternion', torch.ones(1, 4, device='meta')),
])
def test_sdf_rejects_invalid_shapes_types_devices_and_zero_quaternion(name, bad):
    args = dict(points=tensor([[[.01, 0., .5]]]), center=tensor([[0., 0., .5]]),
                quaternion=tensor([[1., 0., 0., 0.]]))
    args[name] = bad.to(DTYPE) if isinstance(bad, torch.Tensor) and bad.is_floating_point() else bad
    with pytest.raises(ValueError):
        objectives().finite_cylinder_sdf(**args)


@pytest.mark.parametrize('name', ['points', 'center', 'quaternion'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_sdf_rejects_nonfinite_values(name, bad):
    args = dict(points=tensor([[[.01, 0., .5]]]), center=tensor([[0., 0., .5]]),
                quaternion=tensor([[1., 0., 0., 0.]]))
    args[name].reshape(-1)[0] = bad
    with pytest.raises(ValueError):
        objectives().finite_cylinder_sdf(**args)


@pytest.mark.parametrize('name', ['radius_m', 'length_m'])
@pytest.mark.parametrize('bad', [0., -1., float('nan'), float('inf'), True])
def test_both_apis_reject_nonpositive_or_nonfinite_dimensions(name, bad):
    data = scene()
    with pytest.raises(ValueError):
        objectives().finite_cylinder_sdf(data['patches'], data['center'], data['quaternion'], **{name: bad})
    with pytest.raises(ValueError):
        objectives().surface_terms(**data, **{name: bad})


@pytest.mark.parametrize('name,bad', [
    ('patches', torch.zeros(1, 4, 3)), ('patches', torch.zeros(1, 2, 3, 4, 3)),
    ('patches', torch.zeros(1, 2, 2, 0, 3)), ('patches', torch.zeros(0, 2, 2, 4, 3)),
    ('collision_points', torch.zeros(1, 3)), ('collision_points', torch.zeros(1, 0, 3)),
    ('collision_points', torch.zeros(2, 2, 3)), ('center', torch.zeros(2, 3)),
    ('quaternion', torch.zeros(1, 4)), ('quaternion', torch.ones(1, 3)),
    ('patch_normals', torch.ones(1, 2, 2, 3)),
    ('patch_normals', torch.zeros(1, 2, 2, 4, 3)),
    ('self_penetration_m', torch.zeros(1, 1)),
    ('self_penetration_m', tensor([-.001])),
    ('object_hand_penetration_m', tensor([-.001])),
    ('object_hand_penetration_m', tensor([.001, .002])),
    ('collision_points', torch.zeros(1, 2, 3, device='meta')),
    ('patch_normals', torch.ones(1, 2, 2, 4, 3, device='meta')),
    ('self_penetration_m', torch.zeros(1, device='meta')),
    ('object_hand_penetration_m', torch.zeros(1, device='meta')),
])
def test_terms_reject_invalid_shapes_devices_normals_and_negative_depths(name, bad):
    data = scene()
    data[name] = bad.to(DTYPE)
    with pytest.raises(ValueError):
        objectives().surface_terms(**data)


@pytest.mark.parametrize('name', ['patches', 'collision_points', 'center', 'quaternion',
                                  'patch_normals', 'self_penetration_m', 'object_hand_penetration_m'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_terms_reject_nonfinite_inputs(name, bad):
    data = scene()
    data.update(patch_normals=torch.ones_like(data['patches']),
                self_penetration_m=tensor([0.]), object_hand_penetration_m=tensor([0.]))
    data[name].reshape(-1)[0] = bad
    with pytest.raises(ValueError):
        objectives().surface_terms(**data)


@pytest.mark.parametrize('count', [0, -1, 1.5, True])
def test_terms_reject_invalid_contact_sample_count(count):
    with pytest.raises(ValueError):
        objectives().surface_terms(**scene(), contact_points_per_patch=count)


@pytest.mark.parametrize('height', [float('nan'), float('inf')])
def test_terms_reject_nonfinite_table_height(height):
    with pytest.raises(ValueError):
        objectives().surface_terms(**scene(), table_z=height)
