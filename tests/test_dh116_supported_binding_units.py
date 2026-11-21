"""Verified collider readers must not lose their explicit frame/unit at registration."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import numpy as np

from scripts import verify_dh116_power_shared_rod as shared


def test_registered_palm_and_proximal_bindings_preserve_verified_frame_unit_and_grid_path(monkeypatch):
    from scripts import audit_dh116_power_palm_parity as audit
    monkeypatch.setitem(sys.modules, "ropesim", NS(sdf=NS(SdfGrid=lambda *a, **kw: object())))
    matrix = np.eye(4).tolist()
    live = dict(geometry_sha256="a"*64, mesh_to_body=matrix, frame="PhysX_link_origin", unit="m")
    hands, original = {}, NS(bindings={})
    for side in shared.SIDES:
        root = "/World/" + side + "Hand"
        paths = [root+"/base_link", root+"/finger52_link"]
        hands[side] = NS(mount=NS(base_id=0), hand=NS(body_names=["base_link", "finger52_link"],
            root_physx_view=NS(link_paths=[paths])))
        for distal in ("finger14_link", "finger53_link"):
            original.bindings[side+"."+distal] = dict(prim_path=root+"/"+distal,
                source_usd_sha256=("a" if side == "left" else "b")*64)
    args = NS(rays=Path("/cpu/rays.json"), left_bake=Path("/cpu/left"),
              right_bake=Path("/cpu/right"), proximal_capture=Path("/cpu/capture"),
              proximal_bake=Path("/cpu/proximal"))
    archive = dict(source=dict(live))
    monkeypatch.setattr(shared.sources, "checked_json",
        lambda path, digest: {"palms": {"left": {}, "right": {}}} if path == args.rays else archive)
    monkeypatch.setattr(shared.sources, "fixture", lambda *a: (
        (np.ones((2,2,2), np.float32), np.zeros(3), .001), None,
        dict(gpu_row_sha256="b"*64, sdf_file_sha256="c"*64), None))
    monkeypatch.setattr(shared, "archive_geometry_digest", lambda row: "d"*64)
    monkeypatch.setattr(audit, "read_palm_source", lambda *a: (None, None, dict(live)))
    palms, _ = shared.bind_palm_sources(NS(stage=object()), hands, original, args=args)
    records = {}
    for side in shared.SIDES:
        root = "/World/" + side + "Hand/finger52_link"
        source = dict(live, body_path=root, mesh_path=root+"/collision", collision_owner=root)
        records[side+".finger52_link"] = (
            dict(side=side, source=source), dict(values=np.ones((2,2,2),np.float32),
                 origin=np.zeros(3), spacing=.0005),
            dict(native_row_sha256="e"*64, grid_sha256="f"*64))
    monkeypatch.setattr(shared, "proximal_registration_inputs", lambda *a: (records, []))
    monkeypatch.setattr(audit, "read_collision_source", lambda stage, path, **kw: (
        None, None, dict(live, body_path=path, mesh_path=path+"/collision", collision_owner=path)))
    proximal, _ = shared.bind_proximal_sources(NS(stage=object()), hands, original, args=args)
    for key, row in {**palms, **proximal}.items():
        assert row["frame"] == "PhysX_link_origin" and row["unit"] == "m"
        if row["body_name"] == "base_link":
            expected = getattr(args, row["side"]+"_bake") / (key+".npz")
        else:
            expected = args.proximal_bake / (key+".npz")
        assert row["sdf_path"] == str(expected)
