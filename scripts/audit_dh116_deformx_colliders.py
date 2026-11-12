#!/usr/bin/env python3
"""Audit real DH116 pad meshes and ColliderSpec inputs without Isaac startup."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from xml.etree import ElementTree

from go2_pvcnn.control.dh116.deformx_dh116_colliders import (
    DISTAL_PAD_LINKS, HAND_SCENE_KEYS, collider_plan, read_pad_triangles,
)
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, MIMIC


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _urdf_contract(path):
    joints = {joint.attrib["name"]: joint for joint in ElementTree.parse(path).findall("joint")}
    actual = tuple((name, value.attrib["joint"], float(value.attrib["multiplier"]),
                    float(value.attrib["offset"]))
                   for name, joint in joints.items() if (value := joint.find("mimic")) is not None)
    if set(actual) != set(MIMIC):
        raise ValueError(f"unexpected passive coupling in {path.name}")
    if any(joints[name].attrib["type"] != "revolute" for name in ACTIVE):
        raise ValueError(f"unexpected active joint type in {path.name}")
    pad_joints = [joint for joint in joints.values()
                  if joint.find("child").attrib["link"] in DISTAL_PAD_LINKS]
    if len(pad_joints) != len(DISTAL_PAD_LINKS) or any(
            joint.attrib["type"] != "fixed" for joint in pad_joints):
        raise ValueError(f"unexpected distal pad topology in {path.name}")
    return {"source_sha256": _digest(path), "active": list(ACTIVE), "passive_mimic": list(actual)}


def audit(project, bridge, sdf):
    from pxr import Usd

    contract_path = project / "docs/contracts/stereo_action_contract.json"
    action = json.loads(contract_path.read_text())["action_contract"]
    expected_order = [f"{side}.{name}" for side in HAND_SCENE_KEYS for name in ACTIVE]
    if (action["size"] != 24 or action["joint_order"] != expected_order
            or action["force_action_dimensions"] != 0):
        raise ValueError("canonical DH116 action contract differs")
    manifest_path = project / "assets/fixed_dual_panda_dh116_v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hands = {}
    for side, tag in (("left", "L"), ("right", "R")):
        row = next(x for x in manifest["sources"] if x["kind"] == "dh116" and x["side"] == side)
        source = manifest_path.parent / "dependencies" / Path(row["path"]).name
        if _digest(source) != row["sha256"]:
            raise ValueError(f"unexpected original {side} USD digest")
        urdf = (project / "resources/manufacturer/DH116V2.0_URDF_MJCF_Files"
                / f"DH116-V2.0-{tag}/urdf/DH116-V2.0-{tag}.urdf")
        stage = Usd.Stage.Open(str(source))
        root = stage.GetDefaultPrim()
        pads = {}
        for link in DISTAL_PAD_LINKS:
            path = str(root.GetPath().AppendChild(link))
            vertices, triangles, record = read_pad_triangles(stage, path)
            vertices, triangles = bridge.weld(vertices, triangles)
            if not sdf.is_closed(triangles):
                raise ValueError(f"unexpected open source pad: {path}")
            try:
                direct_vertices, direct_faces = bridge.body_geometry(stage, path)
            except ValueError as exc:
                if "no collision geometry under " not in str(exc):
                    raise
                direct = {"success": False, "reason": str(exc)}
            else:
                direct = {"success": True, "vertices": len(direct_vertices), "triangles": len(direct_faces)}
            pads[link] = {
                **record, "welded_vertices": len(vertices), "triangles": len(triangles),
                "bounds_link_m": [vertices.min(0).tolist(), vertices.max(0).tolist()],
                "closed_manifold": True, "upstream_direct_geometry": direct,
            }
        hands[side] = {"usd_sha256": row["sha256"], "urdf": _urdf_contract(urdf), "pads": pads}
    plans = {mode: collider_plan(mode, fixture_asset=None if mode.endswith("_thumb") else "fixture")
             for mode in ("left_thumb", "right_thumb", "pinch", "tripod", "power")}
    for plan in plans.values():
        for spec in plan["specs"]:
            bridge.ColliderSpec(**spec)
    return {
        "task_kind": "deformable_rod", "scope": "dh116_original_pad_source_triangles_cpu",
        "hands": hands, "plans": plans, "action_sha256": _digest(contract_path),
        "bridge_sha256": _digest(Path(bridge.__file__)),
        "geometry_is_physx_cooked_mesh": False,
        "sdf_baked": False, "isaac_launched": False, "native_contact_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ropesim-package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.ropesim_package_root.resolve(strict=True)))
    from ropesim import isaac_bridge, sdf

    result = audit(Path(__file__).resolve().parents[1], isaac_bridge, sdf)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
