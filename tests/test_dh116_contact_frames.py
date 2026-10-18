import numpy as np
import pytest

from go2_pvcnn.control.dh116 import contact_frames as cf
from go2_pvcnn.control.dh116.linear_coupling import JOINTS, expand


TRIANGLE = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])


def test_surface_reference_and_winding():
    point, normal = cf.surface_reference(TRIANGLE, [.5, .25, .25])
    np.testing.assert_allclose(point, [.25, .25, 0.])
    np.testing.assert_allclose(normal, [0., 0., 1.])
    _, reversed_normal = cf.surface_reference(TRIANGLE[::-1], [.25, .25, .5])
    np.testing.assert_allclose(reversed_normal, -normal)


@pytest.mark.parametrize('triangle,weights', [
    (np.zeros((3, 3)), [.5, .25, .25]),
    (np.zeros((2, 3)), [.5, .25, .25]),
    (np.full((3, 3), np.nan), [.5, .25, .25]),
    (TRIANGLE, [-.1, .5, .6]),
    (TRIANGLE, [.5, .25, .3]),
    (TRIANGLE, [.5, .5]),
    (TRIANGLE, [np.inf, 0., 0.]),
])
def test_surface_rejects_invalid_geometry(triangle, weights):
    with pytest.raises(ValueError):
        cf.surface_reference(triangle, weights)


def test_palm_frame_axes_and_round_trip():
    transform = cf.palm_frame([.1, -.2, .3], [1., 0., 0.], [1., 0., 2.])
    rotation = transform[:3, :3]
    np.testing.assert_allclose(rotation[:, 2], [1., 0., 0.])
    np.testing.assert_allclose(rotation[:, 1], [0., 0., 1.])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.)
    point, normal = cf.transform_reference(transform, [.02, .03, .04], [0., 0., 1.])
    np.testing.assert_allclose(point, rotation @ [.02, .03, .04] + [.1, -.2, .3])
    back_point, back_normal = cf.transform_reference(np.linalg.inv(transform), point, normal)
    np.testing.assert_allclose(back_point, [.02, .03, .04], atol=1e-15)
    np.testing.assert_allclose(back_normal, [0., 0., 1.])


@pytest.mark.parametrize('origin,outward,direction', [
    ([0., 0., 0.], [0., 0., 0.], [0., 1., 0.]),
    ([0., 0., 0.], [0., 0., 1.], [0., 0., 2.]),
    ([0., 0., 0.], [0., 0., 1.], [0., 0., 0.]),
    ([np.nan, 0., 0.], [0., 0., 1.], [0., 1., 0.]),
    ([0., 0.], [0., 0., 1.], [0., 1., 0.]),
])
def test_palm_frame_rejects_invalid_inputs(origin, outward, direction):
    with pytest.raises(ValueError):
        cf.palm_frame(origin, outward, direction)


@pytest.mark.parametrize('transform', [
    np.diag([-1., 1., 1., 1.]),
    np.diag([2., 1., 1., 1.]),
    np.diag([1., 1., 1., 2.]),
    np.full((4, 4), np.nan),
    np.eye(3),
])
def test_reference_rejects_improper_transform(transform):
    with pytest.raises(ValueError):
        cf.transform_reference(transform, [0., 0., 0.], [0., 0., 1.])


@pytest.mark.parametrize('point,normal', [
    ([0., 0., np.inf], [0., 0., 1.]),
    ([0., 0., 0.], [0., 0., 0.]),
    ([0., 0., 0.], [0., 0., 2.]),
    ([0., 0., 0.], [0., np.nan, 1.]),
])
def test_reference_rejects_bad_point_or_normal(point, normal):
    with pytest.raises(ValueError):
        cf.transform_reference(np.eye(4), point, normal)


def names_for(side):
    return tuple(f'{side}_panda_joint{i}' for i in range(1, 8)) + JOINTS


@pytest.mark.parametrize('side', ['left', 'right'])
def test_jacobian_shift_and_reordered_columns(side):
    names = names_for(side)
    spatial = np.zeros((6, 18))
    spatial[0, 0] = 2.
    spatial[5, 0] = 1.
    spatial[5, names.index('finger21')] = 1.
    spatial[5, names.index('finger22')] = 2.
    result = cf.point_jacobian(spatial, [.1, 0., 0.], names, side)
    assert result.shape == (3, 13)
    np.testing.assert_allclose(result[:, 0], [2., .1, 0.])
    np.testing.assert_allclose(result[:, 9], [0., .1 * (1. + 2. * .9206), 0.])
    np.testing.assert_allclose(result[:, [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]], 0.)
    order = np.random.default_rng(42).permutation(18)
    reordered = cf.point_jacobian(spatial[:, order], [.1, 0., 0.],
                                  tuple(names[i] for i in order), side)
    np.testing.assert_allclose(reordered, result)


@pytest.mark.parametrize('step', [1e-4, 1e-5, 1e-6])
def test_jacobian_against_coupled_rotating_point_difference(step):
    names = names_for('left')
    q = np.full(6, .3)
    def point(active):
        full = dict(zip(JOINTS, expand(active)))
        angle = full['finger21'] + full['finger22']
        return np.array([.1 * np.cos(angle), .1 * np.sin(angle), 0.])
    spatial = np.zeros((6, 18))
    spatial[5, names.index('finger21')] = 1.
    spatial[5, names.index('finger22')] = 1.
    analytic = cf.point_jacobian(spatial, point(q), names, 'left')[:, 7:]
    finite_difference = np.column_stack([
        (point(q + np.eye(6)[i] * step) - point(q - np.eye(6)[i] * step)) / (2. * step)
        for i in range(6)
    ])
    np.testing.assert_allclose(analytic, finite_difference, rtol=0., atol=1e-8)


@pytest.mark.parametrize('names,side', [
    (names_for('left')[:-1], 'left'),
    (names_for('left')[:-1] + ('finger11',), 'left'),
    (names_for('left')[:-1] + ('unknown',), 'left'),
    (names_for('right'), 'left'),
    (names_for('left'), 'both'),
])
def test_point_jacobian_rejects_wrong_identity(names, side):
    with pytest.raises(ValueError):
        cf.point_jacobian(np.zeros((6, 18)), np.zeros(3), names, side)


@pytest.mark.parametrize('spatial,offset', [
    (np.zeros((3, 18)), np.zeros(3)),
    (np.zeros((6, 17)), np.zeros(3)),
    (np.full((6, 18), np.nan), np.zeros(3)),
    (np.zeros((6, 18)), np.array([0., np.inf, 0.])),
    (np.zeros((6, 18)), np.zeros((1, 3))),
    (np.full((6, 18), 1e308), np.array([1e308, 0., 0.])),
])
def test_point_jacobian_rejects_bad_numbers(spatial, offset):
    with pytest.raises(ValueError):
        cf.point_jacobian(spatial, offset, names_for('left'), 'left')
