"""Independent supported Power scene. The guarded pilot owns no policy."""

import numpy as np

from .linear_coupling import expand
from .tabletop_dual_hand_reset import prepare_reset_state
from .tabletop_power_shared_rod import END_SUPPORT_NAMES, SIDES


def prepare_initial_state(config, table):
    """Bind an independent pose to the free rod, support boxes and original Reset."""
    nodes = np.asarray(config["rod_nodes_w_m"], dtype=np.float32)
    if (config["scope"] != "power_end_supported_contact_diagnostic"
            or config["physics_qualified"] is not False
            or config["fixed_rod_nodes"] != []
            or config["gravity_w_m_s2"] != [0., 0., -9.81]
            or nodes.shape != (41, 3) or not np.isfinite(nodes).all()):
        raise ValueError("explicit unqualified free Power rod and real gravity required")
    support = config["support_scene"]
    boxes = support["supports"]
    if [box["name"] for box in boxes] != ["negative_y", "positive_y"]:
        raise ValueError("two separately identified supports required")
    centers = np.asarray([box["center_m"] for box in boxes], dtype=np.float64)
    sizes = np.asarray([box["size_m"] for box in boxes], dtype=np.float64)
    if centers.shape != (2, 3) or sizes.shape != (2, 3) or not np.isfinite([centers, sizes]).all() or np.any(sizes <= 0):
        raise ValueError("two finite positive world-coordinate support boxes required")
    tops = centers[:, 2] + sizes[:, 2] / 2
    table_top = float(table["center_m"][2] + table["size_m"][2] / 2)
    radius = support["wire_radius_m"]
    if (radius != .005 or support["wire_length_m"] != .5
            or not np.allclose(centers[:, 2] - sizes[:, 2] / 2, table_top, atol=1e-7, rtol=0)
            or not np.allclose(tops, support["lift_reference_bottom_m"], atol=1e-7, rtol=0)
            or not np.allclose(nodes[:, 2] - radius, tops[0], atol=1e-7, rtol=0)
            or not np.allclose(np.linalg.norm(np.diff(nodes, axis=0), axis=-1), .5 / 40, atol=1e-7, rtol=0)):
        raise ValueError("both supports must sit on the original table under the unloaded free rod")
    for center, size in zip(centers, sizes):
        horizontal = nodes[:, :2] - center[:2]
        if not np.any(np.all(np.abs(horizontal) <= size[:2] / 2, axis=-1)):
            raise ValueError("each support must lie under measured material stations")
    if abs(centers[1, 1] - centers[0, 1]) <= (sizes[0, 1] + sizes[1, 1]) / 2:
        raise ValueError("initial support boxes overlap")
    hands = config["hands"]
    if set(hands) != set(SIDES):
        raise ValueError("independent candidate states required for both original hands")
    if (len(config["offline_candidate_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in config["offline_candidate_sha256"])):
        raise ValueError("offline candidate source SHA-256 required")
    roots = {side: hands[side]["pose_wxyz"] for side in SIDES}
    active = np.asarray([hands[side]["q_rad"] for side in SIDES], dtype=np.float64)
    seed = prepare_reset_state(hand_roots=roots, origin=np.zeros(3), root_xy=np.zeros((2, 2)),
                               initial_offset_m=0., joint_positions=np.stack([expand(q) for q in active]),
                               joint_velocities=np.zeros((2, 11)),
                               source_description=config["offline_candidate_sha256"])
    poses_xyzw = {name: (center, np.array([0., 0., 0., 1.]))
                  for name, center in zip(END_SUPPORT_NAMES, centers)}
    return seed, active, nodes, poses_xyzw


def guard_hand_supports(world_vertices, support_scene):
    """Measure finite-box separation using actual body poses and known envelopes."""
    from scripts.audit_dh116_pinch_initial_pose import box_relation

    supports = support_scene["supports"]
    if not world_vertices or len(supports) != len(END_SUPPORT_NAMES):
        raise ValueError("hand envelopes and two supports required")
    rows = {}
    for shape, vertices in world_vertices.items():
        rows[shape] = {}
        for slot, support in zip(END_SUPPORT_NAMES, supports):
            row = box_relation(vertices, support["center_m"], support["size_m"])
            rows[shape][slot] = row
            if row["intersects"]:
                raise ValueError(f"{shape} intersects {slot}: {row['max_separating_axis_gap_m']:.9g} m")
    return {"minimum_gap_m": min(row["max_separating_axis_gap_m"]
                                 for slots in rows.values() for row in slots.values()),
            "per_shape": rows}


def assemble_supported_scene(sim, project, prepared, seed, active, support):
    """Author both support collisions in the owned scene before counted setup."""
    from . import tabletop_dual_hand_scene_preparation as scene_api
    from .tabletop_deformx_end_supported_scene import _bind_support_material
    from .tabletop_end_supported_scene import author_end_supports
    from .tabletop_vec_isaac import spawn_scene

    args, layout, usd = scene_api.spawn_parameters(project, prepared)
    prefix = prepared["prefix"]
    if sim.stage.GetPrimAtPath(prefix):
        raise ValueError("isolated empty supported Power scene required")
    for i, side in enumerate(SIDES):
        layout["hand_roots"][side] = np.asarray(seed["authored_root_seed_w"][i]).copy()
        layout["hand_roots"][side][2] -= .04  # Cancel spawn_scene's fixed authoring offset.
        layout["active_joints"][side] = active[i].copy()
    scene = spawn_scene(sim, args, layout, usd, prefix, np.zeros(3), external_wire=True)
    scene_api._bind_materials(sim.stage, prepared)
    scene_api.inspect_source_stage(sim.stage, prepared)
    from pxr import Usd, UsdGeom, UsdPhysics

    material = prepared["candidate_physics"]["table"]
    paths = author_end_supports(sim.stage, prefix, support)
    binding = _bind_support_material(sim.stage, prefix, paths, material["mu_s"], material["mu_k"])
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for path, expected in zip(paths, support["supports"]):
        prim = sim.stage.GetPrimAtPath(path)
        world = np.asarray(cache.GetLocalToWorldTransform(prim))
        if (not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                or prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or not np.allclose(world[3, :3], expected["center_m"], atol=1e-8, rtol=0)
                or not np.allclose(world[:3, :3], np.diag(expected["size_m"]), atol=1e-8, rtol=0)):
            raise ValueError(f"static support stage binding differs from frozen world box: {path}")
    return scene, paths, binding
