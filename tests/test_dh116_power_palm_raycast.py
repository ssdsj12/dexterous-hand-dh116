import numpy as np
import pytest

from scripts.audit_dh116_power_palm_raycast import first_triangle_hit


def test_first_ray_hit_is_nearest_and_respects_finite_range():
    triangles = np.array([[[0., 0., z], [1., 0., z], [0., 1., z]] for z in (0., .5)])
    assert first_triangle_hit(triangles, [.25, .25, 1.], [0., 0., -1.], 2.) == pytest.approx(.5)
    assert first_triangle_hit(triangles, [.25, .25, 1.], [0., 0., -1.], .49) is None
    assert first_triangle_hit(triangles, [.75, .75, 1.], [0., 0., -1.], 2.) is None


def test_ray_parallel_to_original_face_is_not_a_hit():
    triangle = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    assert first_triangle_hit(triangle, [.25, .25, 0.], [1., 0., 0.], 2.) is None
