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


def test_proximal_ray_witness_requires_the_exact_body_and_owned_collider(monkeypatch):
    from scripts import audit_dh116_power_palm_raycast as audit

    vertices = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    mesh = {"vertices": vertices, "faces": [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]}
    monkeypatch.setattr(audit, "_pad_hull", lambda record: (mesh, 0.))
    monkeypatch.setattr(audit, "compare_hull_topology", lambda *a: {"parity_passed": True})
    src = dict(body_path="/World/left/finger52_link", geometry_sha256="same",
               mesh_path="/World/left/finger52_link/collisions/mesh",
               collision_owner="/World/left/finger52_link/collisions")
    row = dict(side="left", link="finger52_link", source=src, clock={"sdk": 2},
               sdk_body={"position_m": [0., 0., 0.], "quaternion_xyzw": [0., 0., 0., 1.]})

    def cast(body):
        return lambda start, direction, limit: dict(hit=True, rigidBody=body, collision=src["mesh_path"],
                                                    position=start + direction * .004, normal=-direction)

    result = audit.ray_audit(row, row, cast(src["body_path"]), expected_link="finger52_link")
    assert result["owned_rays"] == result["total_rays"] == 4
    rejected = audit.ray_audit(row, row, cast("/World/right/finger52_link"), expected_link="finger52_link")
    assert rejected["owned_rays"] == 0
    with pytest.raises(ValueError, match="finger52_link"):
        audit.ray_audit(row, dict(row, link="finger53_link"), cast(src["body_path"]), expected_link="finger52_link")


def test_proximal_tensor_pose_requires_one_exact_body_and_a_finite_unit_quaternion():
    from types import SimpleNamespace
    from scripts.audit_dh116_power_proximal_raycast import owned_tensor_pose

    path = "/World/left/finger52_link"
    valid = np.array([[1., 2., 3., 0., 0., 0., 1.]])
    def view(p, values):
        return SimpleNamespace(count=1, prim_paths=[p], get_transforms=lambda: values)
    assert owned_tensor_pose(view(path, valid), path) == {
        "position_m": [1., 2., 3.], "quaternion_xyzw": [0., 0., 0., 1.]}
    with pytest.raises(ValueError, match="exact"):
        owned_tensor_pose(view(path.replace("left", "right"), valid), path)
    with pytest.raises(ValueError, match="unit"):
        owned_tensor_pose(view(path, np.zeros((1, 7))), path)
