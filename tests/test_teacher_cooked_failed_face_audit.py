"""Offline evidence for the unsatisfied cooked polygon plane checks."""
import copy

import pytest

from test_teacher_cooked_representation_audit import row_fixture
from scripts import audit_dh116_teacher_cooked_readback as audit


def test_export_reports_polygon_geometry_without_changing_qualification():
    row = row_fixture()
    raw = row["cooked_hulls"][0]["raw"]
    raw["polygons"][0]["plane"][3] += 0.0002
    packet = {"capture": {"complete": False, "colliders": [row]}}
    before = copy.deepcopy(packet)
    result = audit.analyze_capture(packet, export_failed_faces=True)
    report = result["failed_face_audit"]
    face, = report["faces"]
    assert face["polygon_index"] == 0
    assert face["cooked_vertex_indices"] == [0, 3, 2, 1]
    assert face["cooked_vertices_mesh_m"] == [raw["vertices"][i] for i in [0, 3, 2, 1]]
    assert face["source_exact_vertex_matches"][0] == {"first_index": 0, "count": 2}
    assert face["stored_normal"] == [0, 0, -1]
    assert face["fan_area_m2"] == pytest.approx(4e-6)
    assert face["best_fit_max_residual_m"] == pytest.approx(0)
    assert face["max_face_residual_m"] == pytest.approx(2e-4)
    assert report["hulls_unchanged_under_reindex"] == 1
    assert result["hulls_passing_existing_validator"] == 0
    assert result["native_qualification"] is False
    assert packet == before


def test_reindex_preserves_directed_rings_and_detects_nonplanar_face():
    row = row_fixture()
    vertices = row["cooked_hulls"][0]["raw"]["vertices"]
    x, y, z = vertices[0]
    vertices[0] = [x, y, z + 0.0002]
    packet = {"capture": {"complete": False, "colliders": [row]}}
    report = audit.analyze_capture(packet, export_failed_faces=True)["failed_face_audit"]
    face = next(f for f in report["faces"] if f["polygon_index"] == 0)
    assert face["best_fit_max_residual_m"] > 0
    assert face["source_exact_vertex_matches"][0] is None
    assert report["hulls_unchanged_under_reindex"] == 1


def test_malformed_hull_is_explicitly_unavailable():
    row = row_fixture()
    row["cooked_hulls"][0]["raw"]["indices"][0] = 999
    packet = {"capture": {"complete": False, "colliders": [row]}}
    result = audit.analyze_capture(packet, export_failed_faces=True)
    report = result["failed_face_audit"]
    assert report["hulls_examined"] == 1
    assert report["hulls_unchanged_under_reindex"] == 0
    assert len(report["errors"]) == 1
    assert report["faces"] == []


def test_cli_writes_opt_in_evidence_and_preserves_default(tmp_path):
    import json

    row = row_fixture()
    row["cooked_hulls"][0]["raw"]["polygons"][0]["plane"][3] += 0.0002
    packet = {"capture": {"complete": False, "colliders": [row]}}
    source, output, default = (tmp_path / name for name in ("source.json", "output.json", "default.json"))
    source.write_text(json.dumps(packet))
    args = ["--capture", str(source)]
    assert audit.main(args + ["--output", str(default)]) == 0
    assert "failed_face_audit" not in json.loads(default.read_text())
    assert audit.main(args + ["--output", str(output), "--export-failed-faces"]) == 0
    assert len(json.loads(output.read_text())["failed_face_audit"]["faces"]) == 1
