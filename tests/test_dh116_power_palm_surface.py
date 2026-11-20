import numpy as np
import pytest

from scripts.audit_dh116_power_palm_surface import audit_surface


VERTICES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float) * .03
FACES = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])


def test_original_surface_uses_triangle_normals_not_reported_plane_coefficients():
    report = audit_surface(dict(vertices=VERTICES, faces=FACES, polygons=[{"plane": [1, 0, 0, 100]}]))
    assert report["matches_convex_boundary"]
    assert report["winding_consistent"]
    assert report["worst_support_violation_m"] <= report["arithmetic_tolerance_m"]


def test_closed_dented_surface_cannot_be_relabelled_convex():
    vertices = np.vstack((VERTICES, [.007, .007, .002]))
    faces = np.vstack(([[0, 2, 4], [2, 1, 4], [1, 0, 4]], FACES[1:]))
    report = audit_surface(dict(vertices=vertices, faces=faces))
    assert report["winding_consistent"] and report["watertight"]
    assert not report["matches_convex_boundary"]
    assert report["worst_support_violation_m"] > .001


def test_reversed_triangle_fails_directed_surface():
    faces = FACES.copy()
    faces[0] = faces[0, ::-1]
    result = audit_surface(dict(vertices=VERTICES, faces=faces))
    assert result["watertight"]
    assert not result["winding_consistent"]
    assert not result["matches_convex_boundary"]


def test_wrong_index_rejected():
    with pytest.raises(ValueError, match="triangle indices"):
        audit_surface(dict(vertices=VERTICES, faces=np.array([[4, 1, 2]])))
