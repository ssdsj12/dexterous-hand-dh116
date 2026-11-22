"""Prepare independent, fixed-wrist resets for unseen wire specimens."""
from copy import deepcopy

import numpy as np

from .tabletop_dual_hand_reset import prepare_reset_state
from .tabletop_power_shared_rod import END_SUPPORT_NAMES, SIDES
from .tabletop_power_supported_task import PINCH_SCOPE, POWER_SCOPE, TRIPOD_SCOPE, prepare_initial_state
from .tabletop_supported_wire_specs import build_variant, catalog_wire_specs, digest


def place_fixed_wrist_variant(variant, scene, table, placement):
    """Place a fresh variant's wire and supports without moving either hand."""
    shift_yaw = np.asarray(placement, dtype=float)
    if shift_yaw.shape != (3,) or not np.isfinite(shift_yaw).all():
        raise ValueError("finite world XY shift and yaw required")
    shift_yaw = np.where(shift_yaw == 0., 0., shift_yaw)
    layout = variant["scene"]
    layout["hands"] = deepcopy(scene["hands"])

    nodes = np.asarray(layout["rod_nodes_w_m"], dtype=float)
    origin = nodes[:, :2].mean(axis=0)
    cosine, sine = np.cos(shift_yaw[2]), np.sin(shift_yaw[2])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    nodes[:, :2] = (nodes[:, :2] - origin) @ rotation.T + origin + shift_yaw[:2]
    layout["rod_nodes_w_m"] = nodes.tolist()

    table_center = np.asarray(table["center_m"], dtype=float)
    table_size = np.asarray(table["size_m"], dtype=float)
    lower, upper = table_center - table_size / 2, table_center + table_size / 2
    radius = variant["rod"]["radius_m"]
    if (np.any(nodes[:, :2] - radius < lower[:2]) or np.any(nodes[:, :2] + radius > upper[:2])
            or np.any(nodes[:, 2] - radius <= upper[2])):
        raise ValueError("wire must rest above and within the original table")
    midpoint = len(nodes) // 2
    boxes = []
    support_halves = set()
    for support in layout["support_scene"]["supports"]:
        center = np.asarray(support["center_m"], dtype=float)
        center[:2] = rotation @ (center[:2] - origin) + origin + shift_yaw[:2]
        size = np.asarray(support["size_m"], dtype=float)
        bottom, top = center - size / 2, center + size / 2
        if (np.any(bottom[:2] < lower[:2]) or np.any(top[:2] > upper[:2])
                or not np.isclose(bottom[2], upper[2], atol=1e-7, rtol=0.)):
            raise ValueError("each axis-aligned support must fit on the original table")
        on_support = np.flatnonzero((np.abs(nodes[:, :2] - center[:2]) <= size[:2] / 2).all(axis=1))
        negative_half = np.all(on_support < midpoint)
        positive_half = np.all(on_support > midpoint)
        if (not on_support.size or not (negative_half or positive_half)
                or not np.allclose(nodes[on_support, 2] - radius, top[2], atol=1e-7, rtol=0.)):
            raise ValueError("each support must lie under one half of the resting wire")
        support_halves.add(bool(positive_half))
        support["center_m"] = center.tolist()
        boxes.append((bottom, top))
    if len(support_halves) != 2 or np.all(np.maximum(boxes[0][0], boxes[1][0]) <= np.minimum(boxes[0][1], boxes[1][1])):
        raise ValueError("two disjoint supports on opposite wire halves required")

    return nodes


def prepare_holdout_variant(scene, table, baseline, *, catalog, variant_id, stations, placement):
    """Return an evaluation-only CPU scene and seed.

    Placement is (world X metres, world Y metres, yaw radians). The wire and
    support centres share this transform. Support boxes remain axis-aligned
    in both backends. The nominal hands are identical across all placements
    and specimens for a grasp. The returned metadata is for the evaluator.
    """
    cases = [case for case in catalog_wire_specs(catalog, baseline)["test"] if case["id"] == variant_id]
    if len(cases) != 1:
        raise ValueError("one explicit held-out wire specimen required")
    case = cases[0]
    baseline_seed, active, _, _ = prepare_initial_state(scene, table, expected_scope=scene["scope"])
    variant = build_variant(scene, baseline, radius_m=case["rod"]["radius_m"], length_m=case["length_m"],
                            elastic_modulus_scale=case.get("elastic_modulus_scale", 1.),
                            grasp_material_ids=stations)
    if variant["spec_sha256"] != case["spec_sha256"]:
        raise ValueError("held-out solver specification differs from catalog")
    nodes = place_fixed_wrist_variant(variant, scene, table, placement)
    layout = variant["scene"]
    table_center = np.asarray(table["center_m"], dtype=float)
    table_size = np.asarray(table["size_m"], dtype=float)
    physical_sha256 = digest(dict(scope=layout["scope"], hands=layout["hands"], spec_sha256=variant["spec_sha256"],
                                  nodes=layout["rod_nodes_w_m"], supports=layout["support_scene"]["supports"],
                                  table_center_m=table_center.tolist(), table_size_m=table_size.tolist()))
    seed = prepare_reset_state(
        hand_roots={side: layout["hands"][side]["pose_wxyz"] for side in SIDES},
        **{key: baseline_seed[key] for key in ("origin", "root_xy", "initial_offset_m",
                                               "joint_positions", "joint_velocities")},
        source_description="wire-holdout:" + physical_sha256)
    support_poses = {name: (np.asarray(box["center_m"]), np.array([0., 0., 0., 1.]))
                     for name, box in zip(END_SUPPORT_NAMES, layout["support_scene"]["supports"], strict=True)}
    return dict(variant, physical_initial_condition_sha256=physical_sha256, reset_seed=seed,
                active=active, nodes=np.asarray(nodes, dtype=np.float32), support_poses=support_poses)


def fixed_pregrasp_reset(packet, table):
    """Rebuild the authored Reset for a sealed training-only pregrasp candidate."""
    scopes = {PINCH_SCOPE: "pinch", TRIPOD_SCOPE: "tripod", POWER_SCOPE: "power"}
    scene = packet.get("config")
    if (packet.get("version") != "dh116-wire-common-pregrasp-v1" or packet.get("status") != "cpu_reset_candidate"
            or not isinstance(scene, dict) or packet.get("scope") not in scopes
            or any(packet.get(key) is not False for key in
                   ("native_reset_verified", "approach_verified", "grasp_qualified"))):
        raise ValueError("sealed unqualified fixed pregrasp candidate required")
    scope = packet["scope"]
    seed, active, _, _ = prepare_initial_state(scene, table, expected_scope=scope)
    if seed["sha256"] != packet.get("reset_seed_sha256"):
        raise ValueError("fixed pregrasp CPU reset seed differs")
    return scopes[scope], scene, seed, active


def prepare_fixed_pregrasp_variant(packet, table, baseline, *, radius_m, length_m, stations,
                                   variant_id, elastic_modulus_scale=1.):
    """Resize only the wire and supports while retaining the exact authored hand seed."""
    grasp, scene, seed, active = fixed_pregrasp_reset(packet, table)
    variant = build_variant(scene, baseline, radius_m=radius_m, length_m=length_m,
                            elastic_modulus_scale=elastic_modulus_scale, grasp_material_ids=stations)
    if packet["training_spec_sha256"].get(variant_id) != variant["spec_sha256"]:
        raise ValueError("fixed pregrasp requires the same named training wire specification")
    nodes = place_fixed_wrist_variant(variant, scene, table, (0., 0., 0.))
    support_poses = {
        name: (np.asarray(box["center_m"]), np.array([0., 0., 0., 1.]))
        for name, box in zip(END_SUPPORT_NAMES, variant["scene"]["support_scene"]["supports"], strict=True)
    }
    return seed, active, np.asarray(nodes, dtype=np.float32), support_poses, dict(variant, grasp=grasp)
