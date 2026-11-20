import numpy as np
import pytest

from scripts import audit_dh116_power_palm_parity as palm


def test_original_left_and_right_palm_reads_are_enabled_and_distinct():
    from pxr import Usd

    records = []
    for source in palm.probe._sources(palm.ROOT).values():
        stage = Usd.Stage.Open(source["path"])
        parent = stage.GetDefaultPrim().GetPath()
        vertices, faces, row = palm.read_palm_source(stage, str(parent.AppendChild("base_link")))
        assert np.isfinite(vertices).all() and faces.shape[1] == 3
        assert row["collision_approximation"] == "convexHull"
        assert row["frame"] == "PhysX_link_origin"
        assert row["mesh_path"].startswith(row["body_path"] + "/collisions/")
        with pytest.raises(ValueError, match="base_link"):
            palm.read_palm_source(stage, str(parent.AppendChild("finger14_link")))
        records.append(row["geometry_sha256"])
    assert len(set(records)) == 2


def test_successful_receipt_rejects_unsealed_or_failed_exit(tmp_path):
    native = tmp_path / "native"
    native.mkdir()
    path = native / "result.json"
    path.write_text("{}")
    outer = tmp_path / "outer.json"
    report = dict(exit_code=0, forced_exit=False, device="cpu", native=str(native),
                  files={"result.json": palm.file_sha(path)})
    palm.write_json(outer, report)
    assert palm.checked_capture(tmp_path, "cpu") == native
    with pytest.raises(ValueError):
        palm.checked_capture(tmp_path, "cuda:0")
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError):
        palm.checked_capture(tmp_path, "cpu")
