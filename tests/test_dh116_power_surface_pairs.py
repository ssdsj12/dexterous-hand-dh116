import numpy as np
from scipy.spatial.transform import Rotation

from scripts.search_dh116_power_surface_pairs import anchored_translation, plane_ray_candidates


def test_anchor_keeps_the_owned_node_under_a_rotated_pad_surface():
    rotation = Rotation.from_euler("Z", 90, degrees=True).as_matrix()
    pad = np.array([.02, -.01, .003])
    node = np.array([.42, .17, .34])
    origin = anchored_translation(rotation, pad, node)
    np.testing.assert_allclose(rotation @ pad + origin, node, atol=1e-14)


def test_surface_ray_is_placed_at_the_locked_negative_sdf_gap():
    step = .001
    origin = np.array([-.015, -.015, -.015])
    coords = origin[0] + step * np.arange(31)
    # A locally planar interpolation in the +X direction.
    values = np.broadcast_to((coords - .001)[:, None, None], (31, 31, 31))
    grid = dict(origin=origin, spacing=step, values=values)
    cooked = np.array([[.001, 0., 0.]])
    rays = np.array([[1., 0., 0.]])
    points, normals = plane_ray_candidates(cooked + [.00005, 0., 0.], cooked, rays, grid, radius=.005)
    np.testing.assert_allclose(points, [[.00595, 0., 0.]], atol=1e-10)
    np.testing.assert_allclose(normals, rays, atol=1e-12)
    none, _ = plane_ray_candidates(cooked + [.00020, 0., 0.], cooked, rays, grid, radius=.005)
    assert not len(none)


def test_axial_error_uses_world_contact_normal_and_material_tangent():
    from scripts.search_dh116_power_surface_pairs import axial_contact_error

    n = np.array([[1., 0., 0.], [0., 1., 0.]])
    t = np.array([[0., 1., 0.], [0., -1., 0.]])
    np.testing.assert_allclose(axial_contact_error(n, t, radius=.005), [0., -.005])
