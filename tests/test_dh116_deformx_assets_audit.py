import hashlib
import json

import numpy as np
import pytest

from scripts.audit_dh116_deformx_assets import audit_trajectory, audit_usd, verify_files


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verify_files_keeps_pinned_document_drift_separate(tmp_path):
    (tmp_path / "README.md").write_text("current card")
    (tmp_path / "usd").mkdir()
    asset = tmp_path / "usd" / "mesh.usdc"
    asset.write_bytes(b"official bytes")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [
        {"path": "README.md", "bytes": 8, "sha256": "a" * 64},
        {"path": "usd/mesh.usdc", "bytes": 14, "sha256": _sha(asset)},
    ]}))
    lock = {"manifest_sha256": _sha(manifest), "include_prefixes": ["usd/"],
            "include_files": ["README.md"], "metadata_sha256": {
                "README.md": _sha(tmp_path / "README.md"),
            }}
    report = verify_files(tmp_path, lock)
    assert report["verified_count"] == 3
    assert report["metadata_manifest_drift"] == ["README.md"]
    asset.write_bytes(b"tampered bytes")
    with pytest.raises(ValueError, match="digest mismatch.*mesh"):
        verify_files(tmp_path, lock)


def test_verify_files_rejects_bundle_escape(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [
        {"path": "usd/../../escape", "bytes": 1, "sha256": "a" * 64},
    ]}))
    lock = {"manifest_sha256": _sha(manifest), "include_prefixes": ["usd/"],
            "include_files": [], "metadata_sha256": {}}
    with pytest.raises(ValueError, match="invalid relative path"):
        verify_files(tmp_path, lock)


def test_usd_audit_follows_local_reference_and_separates_missing_texture(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    child = Usd.Stage.CreateNew(str(tmp_path / "wire.usda"))
    mesh = UsdGeom.Mesh.Define(child, "/Wire")
    mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    child.SetDefaultPrim(mesh.GetPrim())
    child.GetRootLayer().Save()
    main = Usd.Stage.CreateNew(str(tmp_path / "scene.usda"))
    instance = main.OverridePrim("/Instance")
    instance.GetReferences().AddReference("./wire.usda")
    instance.CreateAttribute("texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./missing.exr"))
    UsdLux.DomeLight.Define(main, "/Dome").CreateTextureFileAttr(Sdf.AssetPath("./missing.hdr"))
    main.GetRootLayer().Save()
    report = audit_usd(tmp_path, "scene.usda", expect_mesh=False)
    assert "wire.usda" in report["local_dependencies"]
    assert report["meshes"][0]["vertices"] == 3
    assert report["unresolved_dependencies"] == ["missing.exr", "missing.hdr"]
    assert report["unresolved_dome_dependencies"] == ["missing.hdr"]


def test_trajectory_audit_verifies_node_and_director_alignment(tmp_path):
    path = tmp_path / "source.npz"
    fields = {
        "time_arr": np.array([0., 0.1, 0.2]),
        "pos_arr": np.zeros((3, 2, 3, 4)),
        "dir_arr": np.broadcast_to(np.eye(3)[None, None, :, :, None], (3, 2, 3, 3, 3)),
        "seed_arr": np.arange(3),
        "radius_arr": np.full((3, 2), 0.005),
    }
    np.savez(path, **fields)
    report = audit_trajectory(path, expected_rods=2)
    assert report["record_count"] == 3
    assert report["nodes_per_rod"] == 4
    assert report["time_vector_samples"] == 3
    fields["pos_arr"] = np.zeros((3, 2, 3, 5))
    np.savez(path, **fields)
    with pytest.raises(ValueError, match="director layout"):
        audit_trajectory(path, expected_rods=2)


def test_hang_records_keep_seed_count_distinct_from_frame_count(tmp_path):
    path = tmp_path / "hang.npz"
    np.savez(path, time_arr=np.linspace(0, 2, 5),
             pos_arr=np.zeros((4, 2, 3, 4)),
             dir_arr=np.zeros((4, 2, 3, 3, 3)),
             seed_arr=np.array([1, 2]), radius_arr=np.full((4, 2), 0.005),
             hanging_count_arr=np.ones(4, dtype=int),
             sweep_count_arr=np.ones(4, dtype=int))
    report = audit_trajectory(path, expected_rods=2)
    assert report["record_count"] == 4
    assert report["seed_count"] == 2
    assert report["time_vector_samples"] == 5
