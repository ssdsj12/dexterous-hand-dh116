"""Read-only DH116 distal-pad geometry for the optional DeformX backend."""

import numpy as np


DISTAL_PAD_LINKS = (
    "finger14_link", "finger23_link", "finger33_link", "finger43_link", "finger53_link",
)
HAND_SCENE_KEYS = {"left": "left_hand", "right": "right_hand"}
_GRASP_LINK_COUNTS = {"pinch": 2, "tripod": 3, "power": 5}


def collider_plan(mode, *, fixture_asset=None):
    """ColliderSpec identities for one continuous rod per environment."""
    if mode in ("left_thumb", "right_thumb"):
        sides = (mode.split("_")[0],)
        bodies = DISTAL_PAD_LINKS[:1]
    elif mode in _GRASP_LINK_COUNTS:
        sides = tuple(HAND_SCENE_KEYS)
        bodies = DISTAL_PAD_LINKS[:_GRASP_LINK_COUNTS[mode]]
    else:
        raise ValueError(f"unknown DH116 collider plan: {mode}")
    specs = [{"asset": HAND_SCENE_KEYS[side], "bodies": list(bodies), "dynamic": True}
             for side in sides]
    if fixture_asset is not None:
        if not isinstance(fixture_asset, str) or not fixture_asset or fixture_asset in HAND_SCENE_KEYS.values():
            raise ValueError("fixture must name a separate rigid asset")
        specs.append({"asset": fixture_asset, "bodies": None, "dynamic": False})
    return {"task_kind": "deformable_rod", "rods_per_env": 1, "specs": specs}


def read_pad_triangles(stage, body_path):
    """Read an enabled original pad mesh, including proxy-owned collision Xforms."""
    from pxr import Sdf, UsdGeom, UsdPhysics

    from .usd_surface import read_surface

    path = Sdf.Path(body_path)
    if not path.IsAbsolutePath() or not path.IsPrimPath() or path.name not in DISTAL_PAD_LINKS:
        raise ValueError("expected an absolute DH116 distal pad body path")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError("expected Z-up DH116 stage")
    source = read_surface(stage, body_path, branch="collisions")
    mesh = stage.GetPrimAtPath(source["mesh_path"])
    ancestor = mesh
    while ancestor.GetPath() != path:
        if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("collision shape belongs to a nested rigid body")
        ancestor = ancestor.GetParent()
    vertices = np.asarray(source["points_body"], dtype=np.float32)
    triangles = np.asarray(source["triangles"], dtype=np.int64)
    if source["orientation"] == "leftHanded":
        triangles = triangles[:, [0, 2, 1]]
    record = {key: source[key] for key in (
        "body_path", "mesh_path", "collision_owner", "collision_approximation",
        "orientation", "mesh_to_body", "geometry_sha256",
    )}
    record.update(instance_proxy=mesh.IsInstanceProxy(), frame="PhysX_link_origin", unit="m")
    return vertices, triangles, record
