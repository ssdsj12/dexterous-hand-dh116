import hashlib
import json

import pytest

from scripts.prepare_dh116_deformx_render_scene import prepare_render_scene


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def snapshot(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    root = tmp_path / "original"
    (root / "background").mkdir(parents=True)
    (root / "usd").mkdir()
    hdri = root / "background/boiler_room_4k.hdr"
    hdri.write_bytes(b"fixture")
    template = root / "usd/template.usdc"
    stage = Usd.Stage.CreateNew(str(template))
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    light = UsdLux.DomeLight.Define(stage, "/World/Dome")
    light.CreateTextureFileAttr(Sdf.AssetPath("./textures/missing.exr"))
    light.CreateIntensityAttr(2.0)
    UsdGeom.Camera.Define(stage, "/World/Camera")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()
    lock = {
        "repository": "fixture",
        "revision": "fixture",
        "include_files": ["background/boiler_room_4k.hdr"],
        "include_prefixes": ["usd/"],
        "metadata_sha256": {},
        "scene_usds": ["usd/template.usdc"],
        "expected_template_dome_references": {
            "usd/template.usdc": ["usd/textures/missing.exr"],
        },
        "wire_usdc_count": 0,
    }

    def reseal():
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"files": [
            {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
             "sha256": _sha(path)}
            for path in (hdri, template)
        ]}))
        lock["manifest_sha256"] = _sha(manifest)

    reseal()
    return root, lock, stage, reseal


def test_dome_override_resolves_without_touching_geometry_or_sources(tmp_path, snapshot):
    from pxr import Usd, UsdGeom, UsdLux

    root, lock, _, _ = snapshot
    output = tmp_path / "derived"
    source_hash = _sha(root / "usd/template.usdc")
    report = prepare_render_scene(root, lock, output)
    assert report["derived_authored_usd_dependencies_closed"]
    assert not report["original_authored_usd_dependencies_closed"]
    assert not report["native_render_verified"]
    assert _sha(root / "usd/template.usdc") == source_hash
    row = report["overrides"]["usd/template.usdc"]
    assert row["original_sha256"] == source_hash
    assert row["original_asset_path"] == "./textures/missing.exr"
    assert row["property"] == "/World/Dome.inputs:texture:file"
    assert _sha(output / row["original_copy"]) == source_hash
    derived = output / "assets/usd/template.usdc"
    assert _sha(derived) == row["derived_sha256"] != source_hash
    stage = Usd.Stage.Open(str(derived))
    dome = UsdLux.DomeLight.Get(stage, "/World/Dome")
    texture = dome.GetTextureFileAttr().Get()
    assert texture.resolvedPath == str(output / "assets/background/boiler_room_4k.hdr")
    assert dome.GetIntensityAttr().Get() == 2.0
    assert UsdGeom.Mesh.Get(stage, "/World/Mesh").GetFaceVertexIndicesAttr().Get() == [0, 1, 2]
    assert UsdGeom.Camera.Get(stage, "/World/Camera")
    assert (output / "render_scene_manifest.json").is_file()


def test_unlisted_missing_dependencies_and_existing_outputs_are_rejected(tmp_path, snapshot):
    from pxr import Sdf

    root, lock, stage, reseal = snapshot
    stage.GetPrimAtPath("/World").CreateAttribute("extra", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./unapproved.exr"))
    stage.GetRootLayer().Save()
    reseal()
    with pytest.raises(ValueError, match="unexpected USD dependencies"):
        prepare_render_scene(root, lock, tmp_path / "derived")
    assert not (tmp_path / "derived").exists()
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError):
        prepare_render_scene(root, lock, occupied)
