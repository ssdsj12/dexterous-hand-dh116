"""Offline identity cell for a future positive static-table full report.

No subscription, setting mutation, SDK step, native launcher, or qualification.
Reuse the existing copied callback/poll arrays; do not infer absence from zero.
The static actor path is retained as reported, not guessed from collider path.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import numpy as np

from . import tabletop_stereo_contact_calibration as calibration
from . import tabletop_bimanual_table_contacts_v2 as table_source

VERSION = "dh116-static-table-full-report-cell-cpu-v1"
SDK = Path("/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extsPhysics/omni.physx/omni/physx/bindings/_physx.pyi")
PINS = {
    SDK: "b75b94176c7e6b766bf6f36058d2115016eaabc0d45d8b36a28c64cbf29f4dd9",
    Path(calibration.__file__): "8a5e6a9bd5569e0453741eb083748976e529b8d82f2e67554dfaca200c92a1b0",
    Path(table_source.__file__): "c8420d3a738b7f0268ed56060747f9e5f9fbfb5871a49e081baa7c3e13712628",
}


def source_identity(stage, *, table_root, moving_actor, moving_collider, stage_id):
    """Read real USD paths for one dynamic reporter and the static Cuboid.

    Stage ID is an explicit caller input, not a fabricated backend measurement.
    Original table dimensions/material must additionally pass scene preparation.
    """
    actor = collider = ancestor = root = None
    try:
        from pxr import UsdGeom, UsdPhysics, PhysxSchema
        pins = {}
        for path, expected in PINS.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("contact source implementation changed: "+str(path))
            pins[str(path)] = expected
        if type(stage_id) is not int or stage_id < 0:
            raise ValueError("explicit nonnegative stage identity required")
        if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage),1.,rel_tol=0,abs_tol=1e-12):
            raise ValueError("meter USD stage required")
        table = table_source.resolve_static_table_colliders(stage,table_root)
        if table["collider_paths"] != [table_root+"/geometry/mesh"]:
            raise ValueError("original Cuboid table collision hierarchy required")
        root = stage.GetPrimAtPath(table_root)
        collider = stage.GetPrimAtPath(table["collider_paths"][0])
        if root.GetTypeName() != "Xform" or collider.GetTypeName() != "Cube":
            raise ValueError("original Xform/Cube table hierarchy required")
        actor = stage.GetPrimAtPath(moving_actor)
        collider = stage.GetPrimAtPath(moving_collider)
        if (not actor or not actor.IsActive() or not actor.HasAPI(UsdPhysics.RigidBodyAPI)
                or not UsdPhysics.RigidBodyAPI(actor).GetRigidBodyEnabledAttr().Get()
                or UsdPhysics.RigidBodyAPI(actor).GetKinematicEnabledAttr().Get()
                or not actor.HasAPI(PhysxSchema.PhysxContactReportAPI)
                or PhysxSchema.PhysxContactReportAPI(actor).GetThresholdAttr().Get() != 0
                or not collider or not collider.IsActive() or not collider.HasAPI(UsdPhysics.CollisionAPI)
                or not UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get()):
            raise ValueError("enabled dynamic moving body/collider and threshold-zero reporter required")
        ancestor = collider
        while ancestor and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            ancestor = ancestor.GetParent()
        if not ancestor or str(ancestor.GetPath()) != moving_actor:
            raise ValueError("actual collider rigid ancestor differs")
        return dict(version=VERSION,table=table,moving_actor=moving_actor,
            moving_collider=moving_collider,stage_id=stage_id,stage_id_provenance="caller_supplied",
            source_sha256=pins,table_actor_identity_verified=None,
            source_scope="USD identity only; dimensions/material remain scene-preparation responsibility",
            native_cell_qualified=False,clock_qualified=False)
    finally:
        stage = actor = collider = ancestor = root = None


def classify_full_report(raw_report, identity):
    """Count exact table collider samples; preserve unverified table actor identity.

    The result concerns the supplied report only. It does not establish native
    provenance, stream completeness, force sign, no contact, or same-tick capture.
    """
    if identity.get("version") != VERSION:
        raise ValueError("explicit static table source identity required")
    if not isinstance(raw_report,dict) or raw_report.get("friction_source_available") is not True:
        raise ValueError("copied full report with separate friction channel required")
    report = deepcopy(raw_report)
    pairs = calibration._report_pairs(dict(full_report=report))
    samples, loaded, unmatched, matched = 0, 0, 0, []
    for pair in pairs:
        if type(pair["stage_id"]) is not int or pair["stage_id"] != identity["stage_id"]:
            raise ValueError("report stage identity differs")
        event = pair.get("type")
        if type(event) is not int or event not in (0,1,2):
            raise ValueError("source-pinned contact event type required")
        if event == 1 and pair["contacts"]:
            raise ValueError("lost event cannot supply positive contact data")
        # Check raw vector fields even on unrelated headers; no NaN is accepted.
        for contact in pair["contacts"]:
            for field in ("position","impulse","normal"):
                calibration._world_vector(contact[field])
            if (not math.isfinite(contact["separation"])
                    or not math.isclose(float(np.linalg.norm(contact["normal"])),1.,rel_tol=0,abs_tol=1e-4)):
                raise ValueError("finite contact separation and unit normal required")
        ci = (pair["collider0"],pair["collider1"])
        ai = (pair["actor0"],pair["actor1"])
        sides = [i for i in (0,1) if ci[i] in identity["table"]["collider_paths"]
                 and ci[1-i] == identity["moving_collider"] and ai[1-i] == identity["moving_actor"]]
        if len(sides) != 1:
            unmatched += 1
            continue
        side = sides[0]
        count = len(pair["contacts"])
        samples += count
        nonzero = sum(float(np.linalg.norm(c["impulse"]))>0 for c in pair["contacts"])
        loaded += nonzero
        matched.append(dict(table_collider=ci[side],reported_table_actor=ai[side],
            moving_actor=ai[1-side],moving_collider=ci[1-side],table_side=side,
            contact_samples=count,nonzero_contact_impulse_samples=nonzero,
            raw_contact_impulse_world_n_s=pair["impulse"],
            raw_friction_impulses_world_n_s=pair["friction_impulses"]))
    # Friction positions are not consumed by the existing force reducer.
    for anchor in report["friction_anchors"]:
        calibration._world_vector(anchor["position"])
    return dict(version=VERSION,exact_table_contact_samples=samples,
        nonzero_contact_impulse_samples=loaded,unmatched_headers=unmatched,pairs=matched,
        table_actor_identity_verified=None,force_on_moving_body_world_n=None,
        absence_verified=False,native_cell_qualified=False,clock_qualified=False,
        force_sign_frame_qualified=False,teacher_admission=False)
