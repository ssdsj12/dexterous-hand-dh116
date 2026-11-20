import numpy as np
import pytest

from scripts.audit_dh116_power_palm_self_intersection import audit_intersections


@pytest.mark.parametrize(("other", "expected"), [
    ((0, 1, 3), []),       # The triangles share an edge and are otherwise disjoint.
    ((0, 1, 4), [(0, 1)]), # Coplanar overlap beyond the shared edge.
    ((0, 2, 1), [(0, 1)]), # A duplicated, reverse-wound face.
    ((5, 6, 7), [(0, 1)]), # A noncoplanar crossing without shared vertices.
    ((8, 9, 10), []),      # Parallel planes separated along Z.
    ((0, 11, 12), []),     # A boundary intersection only at the shared vertex.
])
def test_triangle_overlap_is_distinguished_from_expected_shared_boundary(other, expected):
    vertices = np.array([
        [0., 0., 0.], [2., 0., 0.], [0., 2., 0.],
        [0., -1., 0.], [.5, .5, 0.],
        [.25, .25, -1.], [.25, .25, 1.], [.5, .5, 0.],
        [0., 0., 1.], [2., 0., 1.], [0., 2., 1.],
        [-1., 0., 1.], [0., -1., 1.],
    ])
    result = audit_intersections(vertices, np.array([(0, 1, 2), other]))
    assert result["face_pairs"] == 1
    assert result["crossings"] == expected
