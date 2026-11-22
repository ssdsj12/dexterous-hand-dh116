"""Build a clear Tripod Reset candidate from an actual native opposed contact."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, expand
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned, _digest
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import checked_file
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import measure_supported_distal_contact
from go2_pvcnn.control.dh116.tabletop_power_supported_task import guard_hand_supports
from go2_pvcnn.control.dh116.tabletop_supported_wire_specs import build_variant
from go2_pvcnn.control.dh116.tabletop_wire_live_palm_approach import SIDES, wire_tube
from scripts.audit_dh116_pinch_initial_pose import load_geometry, world_vertices
from scripts.screen_dh116_tripod_supported_reset import separation


ANCHORS = {"left": 35, "right": 3}
PADS = ("finger14_link", "finger23_link", "finger33_link")
MIN_RESET_ROD_GAP_M = .0012
MIN_TABLE_GAP_M = .0021
MAX_RETREAT_M = .022
ROOT = Path(__file__).resolve().parents[1]


def align_palm(palm, xyzw, source_point, tangent, target_point, target_tangent):
    """Apply one rigid correspondence of the same rod point and directed tangent."""
    a, b = np.asarray(tangent, dtype=float), np.asarray(target_tangent, dtype=float)
    if (a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all()
            or min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-9 or float(a @ b) <= 0.):
        raise ValueError("finite consistently oriented material tangent required")
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    q = np.r_[np.cross(a, b), 1. + a @ b]
    align = Rotation.from_quat(q)
    world = align * Rotation.from_quat(xyzw)
    return np.r_[np.asarray(target_point)+align.apply(np.asarray(palm)-source_point),
                 world.as_quat()[[3, 0, 1, 2]]]


def read_receipt(run, receipt_sha256):
    receipt = json.loads(checked_file(run/"outer.json", receipt_sha256))
    if (receipt["exit_code"] != 0 or receipt["debugger_inferior_exit_code"] != 0
            or receipt["debugger_status_verified"] is not True or receipt["forced_exit"]
            or receipt["exceeded_budget"] or receipt["debugger_inferior_signal"] is not None):
        raise ValueError("closed native interval with verified outer and debugger receipts required")
    return receipt


def measured_contact(run, *, receipt_sha256, sdk):
    """Consume an actual bilateral tick and match the material nodes per distinct link."""
    receipt = read_receipt(run, receipt_sha256)
    artifact = f"tripod_control_{sdk}.json"
    artifact_sha256 = receipt["artifact_sha256"][artifact]
    row = decode_owned(json.loads(checked_file(run/"native"/artifact, artifact_sha256)))
    rod = row["rod"]
    posture = row["sdk_hand_geometry"]
    endpoint = row["action_readback"]["post"]["identity"][0]
    epoch = ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid", "physics_step", "rod_tick")
    if (any(endpoint[key] != rod[key] for key in epoch) or row["physics_step"] != sdk or posture["physics_step"] != sdk or rod["physics_step"] != sdk
            or rod["rod_tick"] != 12*(sdk-2) or row["guard_failure"] is not None):
        raise ValueError("same completed, guarded SDK tick required")
    contact = measure_supported_distal_contact(rod, grasp="tripod")
    if not contact["bilateral"] or contact["wrong_side_contacts"] or contact["unexpected_hand_contacts"]:
        raise ValueError("source must have actual bilateral opposition and loads for all solver substeps")
    hands = {}
    for side, node in ANCHORS.items():
        hand = row["action_readback"]["post"]["hands"][side]
        indices = hand["active_indices"]
        if tuple(hand["joint_names"][i] for i in indices) != ACTIVE:
            raise ValueError("measured ordered DH116 active joints required")
        q = np.asarray(hand["q"], dtype=float)[0, indices]
        expand(q)
        pairs = [p for p in rod["rod_mesh_contact_pairs"]["pairs"]
                 if p["mesh_name"] in {f"{side}.{name}" for name in PADS}]
        if node not in {p["node_index"] for p in pairs}:
            raise ValueError("chosen material anchor must be a measured contact on its own side")
        xyz, quat = posture["poses"][f"{side}.base_link"]
        hands[side] = dict(xyz=xyz, quat=quat, q=q)
    provenance = dict(sdk=sdk, receipt_sha256=receipt_sha256, raw_row_sha256=artifact_sha256,
                      source_head=receipt["head"], stable_reference=False)
    return rod["node_positions"][0], hands, provenance


def parked_geometry(run, receipt_sha256):
    """Read the real unchanged 8-step wire motion for the same resized free-rod spec."""
    receipt = read_receipt(run, receipt_sha256)
    inventory = receipt["artifact_sha256"]
    def read(name):
        return decode_owned(json.loads(checked_file(run/"native"/name, inventory[name])))

    plan = read("frozen_plan.json")
    settled = read("tripod_shared_result.json")
    if (plan["version"] != "dh116-wire-variant-isaac-settle-v1"
            or settled["sdk_steps"] != 10 or settled["measured_exchanges"] != 8
            or settled["solver_substeps"] != 96
            or settled["peak_support_force_n"]["table"] > 1e-6
            or any(count for link, count in settled["pair_counts"].items()
                   if link not in ("table", "support_0", "support_1"))):
        raise ValueError("original unchanged hands-clear parked wire envelope required")
    first = read("wire_variant_reset_getters.json")["clock"]
    keys = ("run_uuid", "episode_uid", "reset_id", "env_id", "rod_uid")
    samples = []
    for sdk in range(3, 11):
        row = read(f"tripod_exchange_{sdk}.json")["rod"]
        if (row["physics_step"] != sdk or row["rod_tick"] != 12*(sdk-2)
                or any(row[key] != first[key] for key in keys)
                or {pair["mesh_name"] for pair in row["rod_mesh_contact_pairs"]["pairs"]}
                   != {"support_0", "support_1"}):
            raise ValueError("ordered source samples must retain both individually bound support loads")
        samples.append(row["node_positions"][0])
    return np.asarray(samples), plan["spec_sha256"]


def reset_tube(nodes, settled_frames, *, radius_m):
    """Enclose both the authored wire and the measured early sag."""
    nodes = np.asarray(nodes)
    frames = np.asarray(settled_frames)
    if (nodes.shape != (41, 3) or frames.ndim != 3 or frames.shape[1:] != nodes.shape
            or not len(frames)):
        raise ValueError("complete measured parked material frames required")
    return wire_tube(np.concatenate((nodes[None], frames), axis=0), radius_m=radius_m)


def screen_hand(shapes, support_scene, table_top, tube):
    """Find the smallest safe XZ retreat from a candidate contact placement."""
    names = list(shapes)
    normals, upper = tube
    projection = np.stack([(shapes[name] @ normals.T).min(axis=0) for name in names])
    z0 = min(float(points[:, 2].min()) for points in shapes.values())
    angles = np.linspace(0., 2*np.pi, 16, endpoint=False)
    rays = np.c_[np.cos(angles), np.zeros(len(angles)), np.sin(angles)]
    for radius in np.arange(1., int(round(MAX_RETREAT_M/.0005))+1)*.0005:
        feasible = []
        for shift in rays*radius:
            table_gap = float(z0+shift[2]-table_top)
            if table_gap < MIN_TABLE_GAP_M:
                continue
            moved = projection + shift @ normals.T
            gaps = (moved[:, None]-upper[None]).max(axis=-1).min(axis=-1)
            if float(gaps.min()) <= MIN_RESET_ROD_GAP_M:
                continue
            shifted = {name: shapes[name]+shift for name in names}
            try:
                support = guard_hand_supports(shifted, support_scene)
            except ValueError:
                continue
            pad_gaps = {name: separation(shifted["cooked."+name], *tube) for name in PADS}
            feasible.append(dict(offset_world_m=shift.tolist(), retreat_m=float(radius),
                minimum_rod_gap_m=float(gaps.min()), minimum_table_gap_m=table_gap,
                minimum_support_gap_m=support["minimum_gap_m"], required_pad_rod_gaps_m=pad_gaps))
        if feasible:
            return min(feasible, key=lambda row: max(row["required_pad_rod_gaps_m"].values()))
    raise ValueError("no nearby retreat passes original hand envelopes and both supports")


def derive(base, rod_config, measured, parked, *, radius_m, length_m, table_top):
    old_nodes, hands, provenance = measured
    candidate = deepcopy(base)
    straight = np.asarray(base["rod_nodes_w_m"], dtype=float)
    for side, index in ANCHORS.items():
        hand = hands[side]
        align = align_palm(hand["xyz"], hand["quat"], old_nodes[index],
                           old_nodes[index+1]-old_nodes[index-1],
                           straight[index], straight[index+1]-straight[index-1])
        candidate["hands"][side] = dict(pose_wxyz=align.tolist(), q_rad=hand["q"].tolist())
    resized = build_variant(candidate, rod_config, radius_m=radius_m, length_m=length_m,
                            grasp_material_ids=ANCHORS)
    scene = resized["scene"]
    frames, parked_spec = parked
    if parked_spec != resized["spec_sha256"]:
        raise ValueError("parked archive must match the actual screened wire specification")
    tube = reset_tube(scene["rod_nodes_w_m"], frames, radius_m=radius_m)
    metrics, shifted_y = {}, {}
    for side in SIDES:
        geo = load_geometry(ROOT, side=side)
        hand = scene["hands"][side]
        shapes = {}
        for cooked, prefix in ((False, "source"), (True, "cooked")):
            for name, points in world_vertices(geo, hand["pose_wxyz"], np.asarray(hand["q_rad"]),
                                               cooked=cooked).items():
                shapes[f"{prefix}.{name}"] = points
        if len(shapes) != 22:
            raise ValueError("17 source envelopes and 5 cooked pad hulls required per hand")
        result = screen_hand(shapes, scene["support_scene"], table_top, tube)
        metrics[side] = result
        shift = np.asarray(result["offset_world_m"])
        seed = candidate["hands"][side]["pose_wxyz"]
        seed[:3] = (np.asarray(seed[:3])+shift).tolist()
        shifted_y[side] = (min(float(p[:, 1].min()) for p in shapes.values()),
                           max(float(p[:, 1].max()) for p in shapes.values()))
    if shifted_y["left"][0] <= shifted_y["right"][1]:
        raise ValueError("original two hand sides must retain disjoint envelopes")
    report = dict(grasp="tripod", origin=provenance, anchors=ANCHORS, candidate_wire={
        "radius_m": radius_m, "length_m": length_m, "spec_sha256": resized["spec_sha256"]},
        geometry_screen=metrics, original_active_joint_counts=dict(left=6, right=6),
        hull_count=44, measured_support_samples=len(frames),
        scope="CPU geometry proposal for new native Reset", teacher_admission=False)
    candidate["offline_candidate_sha256"] = _digest(report)
    return candidate, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "nominal-scene", "rod-config", "parked-run", "output", "report"):
        parser.add_argument("--"+name, required=True, type=Path)
    for name in ("receipt-sha256", "scene-sha256", "rod-config-sha256", "parked-receipt-sha256"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--sdk", required=True, type=int)
    parser.add_argument("--radius-m", required=True, type=float)
    parser.add_argument("--length-m", required=True, type=float)
    args = parser.parse_args()
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import prepare_scene
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import TRIPOD_SCOPE, prepare_initial_state

    base = json.loads(checked_file(args.nominal_scene, args.scene_sha256))
    rod = json.loads(checked_file(args.rod_config, args.rod_config_sha256))["rod"]
    table = prepare_scene(ROOT)["table"]
    prepare_initial_state(base, table, expected_scope=TRIPOD_SCOPE)
    data = measured_contact(args.run, receipt_sha256=args.receipt_sha256, sdk=args.sdk)
    parked = parked_geometry(args.parked_run, args.parked_receipt_sha256)
    data[2]["parked_receipt_sha256"] = args.parked_receipt_sha256
    scene, report = derive(base, rod, data, parked, radius_m=args.radius_m, length_m=args.length_m,
                           table_top=table["center_m"][2]+table["size_m"][2]/2)
    for path, value in ((args.report, report), (args.output, scene)):
        with path.open("x") as output:
            json.dump(value, output, indent=2, allow_nan=False)
            output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
