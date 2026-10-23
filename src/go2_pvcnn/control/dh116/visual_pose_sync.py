"""Static-preview synchronization for imported USD articulation links."""

from __future__ import annotations

import os


def pose_quaternion_wxyz(pose):
    """Return IsaacLab's body-pose quaternion in USD's (w,x,y,z) order."""
    return tuple(float(value) for value in pose[3:7])


def _find_body_prims(stage, root_path, body_names):
    from pxr import Usd

    root = stage.GetPrimAtPath(root_path)
    found = {}
    for prim in Usd.PrimRange(root):
        name = prim.GetName()
        if name in body_names and name not in found:
            found[name] = prim
    return found


def apply_static_visual_poses(stage, arms, root_paths=None, pose_overrides=None):
    """Author the measured PhysX link poses into the preview USD layer.

    The imported asset uses Fabric-backed articulation transforms.  A paused
    static preview can retain the authored default link transforms even after
    PhysX has accepted the requested joints.  This helper is intentionally
    restricted to the replay script's static pose mode; the real runner never
    calls it, so physics and contact calculations continue to use articulation
    joints and their actual readback.
    """
    from pxr import Gf, UsdGeom

    updated = 0
    diagnostics = []
    for side, arm in arms.items():
        root_path = (root_paths or {}).get(side, f"/World/Rig/{side}_arm")
        names = list(arm.body_names)
        prims = _find_body_prims(stage, root_path, set(names))
        poses = arm.data.body_link_pose_w[0].detach().cpu().numpy()
        by_name = dict(zip(names, poses))
        if pose_overrides and side in pose_overrides:
            by_name.update(pose_overrides[side])
        # Parent links must be authored before children so local transforms are
        # computed relative to the already-updated parent world pose.
        ordered = sorted(
            ((prims[n], n) for n in prims if n in by_name),
            key=lambda item: len(str(item[0].GetPath()).split('/')),
        )
        for prim, body_name in ordered:
            pose = by_name[body_name]
            world = Gf.Matrix4d(1.0)
            world.SetRotate(Gf.Quatd(*pose_quaternion_wxyz(pose)))
            world.SetTranslateOnly(Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2])))
            parent = prim.GetParent()
            parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(parent)
            # USD/Gf uses row-vector transforms: child_world = local *
            # parent_world.  Inverting in the opposite order produces large
            # bogus translations and makes the imported DH116 disappear from
            # the viewport even though the physics readback is correct.
            local = world * parent_world.GetInverse()
            xform = UsdGeom.Xformable(prim)
            ops = xform.GetOrderedXformOps()
            if not ops:
                xform.AddTransformOp().Set(local)
            else:
                for op in ops:
                    op_name = str(op.GetOpName())
                    if op_name.endswith('translate'):
                        op.Set(local.ExtractTranslation())
                    elif op_name.endswith('orient'):
                        rotation = local.ExtractRotation().GetQuat()
                        # Native Panda links commonly author orient ops as
                        # Quatf while imported DH116 links may use Quatd.
                        # USD rejects a precision-mismatched value, so retain
                        # the existing op's declared type.
                        if "quatf" in str(op.GetAttr().GetTypeName()).lower():
                            op.Set(Gf.Quatf(
                                float(rotation.GetReal()),
                                Gf.Vec3f(*[float(v) for v in rotation.GetImaginary()]),
                            ))
                        else:
                            op.Set(Gf.Quatd(
                                float(rotation.GetReal()),
                                Gf.Vec3d(*[float(v) for v in rotation.GetImaginary()]),
                            ))
            updated += 1
            if os.environ.get("DH116_DEBUG_VISUAL_SYNC") and len(diagnostics) < 8:
                diagnostics.append({
                    "side": side,
                    "body": body_name,
                    "prim": str(prim.GetPath()),
                    "target_world": [float(pose[i]) for i in range(3)],
                    "local": [float(v) for v in local.ExtractTranslation()],
                })
    if diagnostics:
        print("DH116_VISUAL_SYNC", {"updated": updated, "samples": diagnostics}, flush=True)
    return updated
