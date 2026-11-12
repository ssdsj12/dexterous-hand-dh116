import json
from pathlib import Path

import numpy as np
import pytest

from go2_pvcnn.control.dh116.deformx_dh116_colliders import collider_plan, read_pad_triangles


def test_single_pad_and_bilateral_share_one_rod():
    one = collider_plan("left_thumb", fixture_asset=None)
    assert one["rods_per_env"] == 1
    assert one["specs"] == [{"asset": "left_hand", "bodies": ["finger14_link"], "dynamic": True}]
    both = collider_plan("tripod", fixture_asset="fixture")
    assert both["rods_per_env"] == 1
    assert both["specs"][0]["bodies"] == ["finger14_link", "finger23_link", "finger33_link"]
    assert both["specs"][1]["asset"] == "right_hand"
    assert both["specs"][2] == {"asset": "fixture", "bodies": None, "dynamic": False}
    with pytest.raises(ValueError, match="fixture"):
        collider_plan("tripod", fixture_asset="right_hand")


def test_original_instance_proxy_mesh_is_in_link_metres():
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "assets/fixed_dual_panda_dh116_v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    per_side = []
    for side in ("left", "right"):
        source = next(x for x in manifest["sources"] if x["kind"] == "dh116" and x["side"] == side)
        usd = manifest_path.parent / "dependencies" / Path(source["path"]).name
        stage = Usd.Stage.Open(str(usd))
        path = str(stage.GetDefaultPrim().GetPath().AppendChild("finger14_link"))
        vertices, faces, record = read_pad_triangles(stage, path)
        assert vertices.shape[1] == faces.shape[1] == 3
        assert record["instance_proxy"]
        assert record["collision_owner"] != record["mesh_path"]
        assert record["collision_approximation"] == "convexHull"
        assert abs(vertices).max() < 0.1
        per_side.append(record["geometry_sha256"])
        composed = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(composed, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(composed, 1.0)
        world = UsdGeom.Xform.Define(composed, "/World")
        world.AddTranslateOp().Set(Gf.Vec3d(2.0, -3.0, 1.0))
        world.AddRotateXYZOp().Set(Gf.Vec3f(13.0, -40.0, 21.0))
        hand = composed.OverridePrim("/World/Hand")
        hand.GetReferences().AddReference(str(usd))
        composed_v, composed_f, _ = read_pad_triangles(composed, "/World/Hand/finger14_link")
        assert np.allclose(vertices, composed_v, atol=1e-8, rtol=0)
        assert np.array_equal(faces, composed_f)
        with pytest.raises(ValueError, match="distal"):
            read_pad_triangles(stage, path.replace("finger14_link", "finger13_link"))
        stage.SetEditTarget(stage.GetSessionLayer())
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        with pytest.raises(ValueError, match="metre"):
            read_pad_triangles(stage, path)
    assert per_side[0] != per_side[1]
