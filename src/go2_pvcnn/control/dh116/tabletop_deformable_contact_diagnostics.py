"""Reduce validated node/mesh pairs without claiming physical grasp qualification."""
import math

import numpy as np

from .deformx_dh116_colliders import collider_plan
from .tabletop_deformx_pair_capture import validate_pair_capture
from .tabletop_dual_hand_rod_runtime import MESH_NAMES, PADS, SIDES
from .tabletop_deformable_grasp_runtime import DT


SCOPE = "deformable_pair_contact_diagnostics_only"
MIN_NORMAL_FORCE_N = 1e-5


def summarize_contact_pairs(row, *, grasp):
    """Count simultaneous loads on named pads, preserving material-node identities.

    A loaded node's rest station is a discrete solver label. It is not a
    sub-node anchor measurement or a bound on physical slip.
    """
    grasp_specs = collider_plan(grasp)["specs"]
    required = dict(zip(SIDES, (tuple(item["bodies"]) for item in grasp_specs)))
    result = dict(scope=SCOPE, grasp=grasp, hands=None, table_sdf_loaded=None,
                  simultaneous_required_pad_substeps=None, native_qualified=False,
                  effective_opposed_contact=None, material_anchor_slip=None,
                  external_support=None, training_reward=None, r2_qualified=False)
    packet = row.get("rod_mesh_contact_pairs")
    if packet is None:
        return result
    if (packet.get("scope") != "rod_node_to_sdf_mesh_substep_force"
            or packet.get("physics_step") != row["physics_step"]
            or packet.get("rod_tick") != row["rod_tick"] or row.get("clock_valid") is not True
            or tuple(row["mesh_names"]) != MESH_NAMES):
        raise ValueError("contact packet clock or mesh identities differ")
    step_dt = packet["substep_dt_s"]
    if type(step_dt) not in (float, int) or not math.isfinite(step_dt) or not 0 < step_dt <= DT:
        raise ValueError("finite positive solver substep duration required")
    substeps = round(DT / step_dt)
    if not 1 <= substeps <= 64 or not math.isclose(substeps * step_dt, DT, rel_tol=0, abs_tol=1e-12):
        raise ValueError("pair interval does not cover the full physics step")
    pairs = packet["pairs"]
    if not isinstance(pairs, list):
        raise ValueError("explicit pair rows required")
    ids, vectors = [], []
    for p in pairs:
        mesh = p["mesh_index"]
        if (type(mesh) is not int or not 0 <= mesh < len(MESH_NAMES)
                or MESH_NAMES[mesh] != p["mesh_name"]
                or any(type(p[key]) is not int for key in ("substep", "node_index"))):
            raise ValueError("pair identity differs from its actual slot")
        ids.append((p["substep"], p["node_index"], mesh))
        vectors.append([*p["contact_point_world_m"], *p["outward_normal_world"],
                        *p["force_on_rod_world_n"], *p["torque_on_mesh_about_origin_world_nm"]])
    node_forces = np.asarray(row["distributed_contact"])[0]
    rest = np.asarray(row["rest_lengths_m"])
    if rest.shape != (len(node_forces) - 1,) or not np.isfinite(rest).all() or np.any(rest <= 0):
        raise ValueError("matching positive rod rest segments required")
    # Reopen the individual raw quantities and reconcile with independently
    # retained solver aggregates before deriving any count or mean.
    clean = validate_pair_capture(dict(substeps=substeps, mesh_count=len(MESH_NAMES),
        capacity=max(1, len(pairs)), attempted=len(pairs), overflow=False,
        indices=np.asarray(ids, dtype=np.int64).reshape((-1, 3)),
        vectors=np.asarray(vectors, dtype=float).reshape((-1, 12))), dt_s=DT,
        node_forces=node_forces, mesh_wrenches=[row["named_wrenches"][name] for name in MESH_NAMES])
    stations = np.concatenate(([0.], np.cumsum(rest)))
    table_loaded = False
    load_by_slot = {name: np.zeros(substeps) for name in MESH_NAMES}
    impulse_by_slot = {name: np.zeros(3) for name in MESH_NAMES}
    node_by_slot = {name: set() for name in MESH_NAMES}
    for pair in clean["pairs"]:
        name = MESH_NAMES[pair["mesh_index"]]
        normal = pair["normal_force_on_rod_n"]
        if normal <= MIN_NORMAL_FORCE_N:
            continue
        load_by_slot[name][pair["substep"]] += normal
        impulse_by_slot[name] += pair["impulse_on_rod_ns"]
        node_by_slot[name].add(pair["node_index"])
        table_loaded |= name == "table"
    hands, masks = {}, []
    for side in SIDES:
        per_pad = {}
        for pad in PADS:
            key = f"{side}.{pad}"
            nodes = sorted(node_by_slot[key])
            per_pad[pad] = dict(mean_normal_force_n=float(load_by_slot[key].mean()),
                loaded_substeps=int(np.count_nonzero(load_by_slot[key])),
                loaded_node_indices=nodes, loaded_node_rest_stations_m=stations[nodes].tolist(),
                impulse_on_rod_world_ns=impulse_by_slot[key].tolist())
        all_loaded = np.logical_and.reduce([load_by_slot[f"{side}.{pad}"] > 0 for pad in required[side]])
        masks.append(all_loaded)
        hands[side] = dict(per_pad=per_pad, required_pads=list(required[side]),
            all_required_loaded_substeps=int(np.count_nonzero(all_loaded)))
    result.update(hands=hands, table_sdf_loaded=table_loaded,
        simultaneous_required_pad_substeps=int(np.count_nonzero(masks[0] & masks[1])),
        physics_step=row["physics_step"], rod_tick=row["rod_tick"], substeps=substeps,
        min_normal_force_n=MIN_NORMAL_FORCE_N)
    return result
