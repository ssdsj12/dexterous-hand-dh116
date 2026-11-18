#!/usr/bin/env python3
"""Materialize independent DeformX render assets with audited DomeLight overrides."""

import argparse
import json
import os
from pathlib import Path
import shutil

if __package__:
    from .audit_dh116_deformx_assets import _sha256, audit_usd, verify_files
else:
    from audit_dh116_deformx_assets import _sha256, audit_usd, verify_files


BACKGROUND = "background/boiler_room_4k.hdr"
_STRUCTURE_KEYS = ("up_axis", "meters_per_unit", "default_prim", "meshes", "skeletons")


def _preflight(root, lock, files):
    scenes = lock["scene_usds"]
    expected = lock["expected_template_dome_references"]
    if set(scenes) != set(expected) or any(len(paths) != 1 for paths in expected.values()):
        raise ValueError("expected one explicitly locked DomeLight dependency per template")
    wires = sorted(name for name in files
                   if name.startswith("usd/wire_usdc/") and name.endswith(".usdc"))
    if len(wires) != lock["wire_usdc_count"] or not set(scenes).issubset(files):
        raise ValueError("pinned USD inventory differs")
    audit = {}
    for name in (*scenes, *wires):
        report = audit_usd(root, name, expect_mesh=name in wires)
        allowed = expected.get(name, [])
        if report["external_dependencies"] or report["unresolved_dependencies"] != allowed:
            raise ValueError(f"unexpected USD dependencies: {name}")
        if report["unresolved_dome_dependencies"] != allowed:
            raise ValueError(f"expected texture is not an unresolved DomeLight dependency: {name}")
        audit[name] = report
    return audit, set(wires)


def _override_dome(asset_root, name, missing):
    from pxr import Sdf, Usd, UsdGeom, UsdLux

    scene = asset_root / name
    background = asset_root / BACKGROUND
    stage = Usd.Stage.Open(str(scene))
    if not stage:
        raise ValueError(f"unreadable copied template: {name}")
    layer = stage.GetRootLayer()
    cameras = []
    matches = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Camera):
            cameras.append(str(prim.GetPath()))
        if prim.IsA(UsdLux.DomeLight):
            attr = UsdLux.DomeLight(prim).GetTextureFileAttr()
            value = attr.Get()
            if value and not value.resolvedPath and (scene.parent / value.path).resolve() == asset_root / missing:
                matches.append((attr, value))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one matching DomeLight property: {name}")
    attr, original = matches[0]
    stack = attr.GetPropertyStack(Usd.TimeCode.Default())
    if len(stack) != 1 or stack[0].layer != layer or attr.GetTimeSamples():
        raise ValueError(f"DomeLight texture is not a root-layer default: {name}")
    prop = stack[0].path
    original_text = layer.ExportToString()
    replacement = Sdf.AssetPath(os.path.relpath(background, scene.parent))
    stack[0].default = replacement

    if not layer.Save():
        raise ValueError(f"failed to save copied template: {name}")
    saved = Sdf.Layer.OpenAsAnonymous(str(scene))
    restored = Sdf.Layer.CreateAnonymous("restore_check.usda")
    restored.TransferContent(saved)
    restored.GetAttributeAtPath(prop).default = original
    if restored.ExportToString() != original_text:
        raise ValueError(f"unexpected authored USD changes: {name}")
    return {
        "property": str(prop),
        "original_asset_path": original.path,
        "replacement_asset_path": replacement.path,
        "background": BACKGROUND,
        "camera_paths": sorted(cameras),
        "only_locked_default_changed": True,
    }


def prepare_render_scene(source, lock, output):
    source = Path(source).resolve(strict=True)
    output = Path(output)
    if output.is_symlink():
        raise FileExistsError(output)
    output = output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("output must be separate from the read-only asset root")
    if output.exists():
        raise FileExistsError(output)

    inventory = verify_files(source, lock)
    files = inventory["files"]
    if BACKGROUND not in files:
        raise ValueError("replacement HDRI is not pinned in the source inventory")
    original_audit, wires = _preflight(source, lock, files)
    output.mkdir(parents=True)
    assets = output / "assets"
    originals = output / "original-scenes"
    for name in files:
        copied = assets / name
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, copied)
        if _sha256(copied) != files[name]["sha256"]:
            raise ValueError(f"copied source SHA256 mismatch: {name}")

    overrides = {}
    for name in lock["scene_usds"]:
        original = originals / name
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(assets / name, original)
        if _sha256(original) != files[name]["sha256"]:
            raise ValueError(f"archived source SHA256 mismatch: {name}")
        missing = lock["expected_template_dome_references"][name][0]
        overrides[name] = {
            **_override_dome(assets, name, missing),
            "missing_dependency": missing,
            "original_copy": original.relative_to(output).as_posix(),
            "original_sha256": files[name]["sha256"],
            "derived_sha256": _sha256(assets / name),
        }

    derived_audit = {}
    for name, prior in original_audit.items():
        report = audit_usd(assets, name, expect_mesh=name in wires)
        if any(report[key] for key in ("unresolved_dependencies", "unresolved_dome_dependencies",
                                       "external_dependencies")):
            raise ValueError(f"derived USD dependencies are not closed: {name}")
        local = set(prior["local_dependencies"])
        if name in overrides:
            local.add(BACKGROUND)
        if set(report["local_dependencies"]) != local or any(
            report[key] != prior[key] for key in _STRUCTURE_KEYS
        ):
            raise ValueError(f"derived USD changed audited structure or other dependencies: {name}")
        derived_audit[name] = report

    if verify_files(source, lock) != inventory:
        raise ValueError("pinned source inventory changed during preparation")
    report = {
        "repository": lock["repository"],
        "revision": lock["revision"],
        "scope": "deformx_render_preparation_cpu",
        "original_inventory": inventory,
        "derived_asset_root": "assets",
        "original_authored_usd_dependencies_closed": False,
        "derived_authored_usd_dependencies_closed": True,
        "native_render_verified": False,
        "isaac_launched": False,
        "overrides": overrides,
        "original_usd": original_audit,
        "derived_usd": derived_audit,
    }
    manifest = output / "render_scene_manifest.json"
    temporary = output / ".render_scene_manifest.tmp"
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    prepare_render_scene(args.asset_root, lock, args.output)
    print(args.output.resolve() / "render_scene_manifest.json")


if __name__ == "__main__":
    main()
