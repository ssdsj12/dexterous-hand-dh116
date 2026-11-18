import numpy as np
import pytest

from scripts.audit_dh116_cooked_convex_hulls import (
    compare_surfaces, decode_cooked_hull, geometric_hull_estimate, query_physx_hulls,
)


_TETRA = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
_FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])


def _convex(indices=None):
    return {"vertices": _TETRA.tolist(),
            "indices": _FACES.reshape(-1).tolist() if indices is None else indices,
            "polygons": [{"index_base": 3 * i, "num_vertices": 3, "plane": [0, 0, 1, 0]}
                         for i in range(4)]}


def test_decode_rejects_bad_native_indices():
    with pytest.raises(ValueError, match="indices"):
        decode_cooked_hull(_convex([4] * 12))
    decoded = decode_cooked_hull(_convex())
    assert decoded["faces"] == _FACES.tolist()


def test_queries_keep_live_and_requested_geometry_separate():
    class Cooking:
        def get_nb_convex_mesh_data(self, path):
            return int(path == "/owner")

        def get_convex_mesh_data(self, path, index):
            assert (path, index) == ("/owner", 0)
            return _convex()

        def request_convex_collision_representation(self, *, stage_id, collision_prim_id,
                                                    run_asynchronously, on_result):
            assert stage_id == 13 and not run_asynchronously
            on_result("ok", [_convex()] if collision_prim_id == 2 else [])

    result = query_physx_hulls(Cooking(), ("/owner", "/mesh"), stage_id=13,
                               encode_path={"/owner": 1, "/mesh": 2}.__getitem__, valid_result="ok",
                               simulation_running=True)
    assert len(result["/owner"]["live_simulation_cooked"]["hulls"]) == 1
    assert not result["/owner"]["requested_cooking"]["hulls"]
    assert result["/owner"]["live_simulation_cooked"]["coordinate_frame"] == "unconfirmed"
    assert len(result["/mesh"]["requested_cooking"]["hulls"]) == 1


def test_geometric_estimate_and_surface_comparison_have_explicit_provenance():
    vertices, faces, estimate = geometric_hull_estimate(np.vstack((_TETRA, [0.1, 0.1, 0.1])))
    assert len(vertices) == 4
    assert estimate["source"] == "scipy_qhull_geometric_estimate"
    assert not estimate["physx_cooking_applied"]
    metrics = compare_surfaces(_TETRA, _FACES, vertices, faces, samples=16)
    assert metrics["sampled_symmetric_hausdorff_m"] == pytest.approx(0, abs=1e-12)
    assert metrics["volume_delta_m3"] == pytest.approx(0)


def test_native_query_preserves_zero_and_error_independently():
    class Cooking:
        def get_nb_convex_mesh_data(self, path):
            if path == "/unsupported":
                raise RuntimeError("path does not map to a physics shape")
            return 0

        def request_convex_collision_representation(self, *, stage_id, collision_prim_id,
                                                    run_asynchronously, on_result):
            if collision_prim_id == 1:
                raise RuntimeError("request could not parse proxy owner")
            on_result("not_ready", [])

    result = query_physx_hulls(Cooking(), ("/unsupported", "/zero"), stage_id=7,
                               encode_path={"/unsupported": 1, "/zero": 2}.__getitem__,
                               valid_result="ok", simulation_running=True)
    assert result["/unsupported"]["live_simulation_cooked"]["hulls"] is None
    assert "does not map" in result["/unsupported"]["live_simulation_cooked"]["error"]
    assert "parse proxy" in result["/unsupported"]["requested_cooking"]["error"]
    assert result["/zero"]["live_simulation_cooked"]["hulls"] == []
    assert result["/zero"]["requested_cooking"]["callback_status"] == "not_ready"


def test_asymmetric_frame_calibration_fails_closed_on_ambiguous_coordinates():
    from scripts.verify_dh116_cooked_convex_hulls_native import classify_reference_frame

    link = _TETRA @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) + [0.5, 0.2, 0.3]
    world = link + [5, 6, 7]
    targets = {"mesh_local": _TETRA, "link_local": link, "world": world}
    for frame, points in targets.items():
        result = classify_reference_frame(points[::-1], targets)
        assert result["confirmed_frame"] == frame
    assert classify_reference_frame(link, {"mesh_local": link, "link_local": link})["confirmed_frame"] is None
    assert classify_reference_frame(world + 1, targets)["confirmed_frame"] is None
