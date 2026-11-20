import numpy as np
import pytest
import trimesh

from scripts.bake_dh116_power_right_palm_sdf import audit_grid, reference_distances


def test_oriented_box_reference_distances():
    mesh = trimesh.creation.box(extents=[.02] * 3)
    points = np.array([[.009, 0., 0.], [.01, 0., 0.], [.011, 0., 0.]])
    np.testing.assert_allclose(reference_distances(mesh, points), [-.001, 0., .001], atol=1e-12)


def test_sample_audit_detects_sign_inversion_near_original_faces():
    mesh = trimesh.creation.box(extents=[.02] * 3)

    def inverted(points):
        return -reference_distances(mesh, points), np.ones(len(points), dtype=bool)

    result = audit_grid(mesh, inverted, spacing_m=.001)
    assert result["groups"]["face_0.1_voxel"]["opposite_sign_count"] == 24
    assert result["groups"]["surface"]["signed_samples"] == 0
    assert result["worst"]["absolute_error_m"] > .001


def test_audit_rejects_samples_outside_the_lattice():
    mesh = trimesh.creation.box(extents=[.02] * 3)

    def empty(points):
        return np.zeros(len(points)), np.zeros(len(points), dtype=bool)

    with pytest.raises(ValueError, match="lattice"):
        audit_grid(mesh, empty, spacing_m=.001)
