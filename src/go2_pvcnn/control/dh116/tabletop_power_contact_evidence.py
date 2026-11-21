"""Reduce same-boundary contacts from the original five pads and palm per hand."""

from collections import defaultdict
import math
import re

import numpy as np

from .tabletop_deformx_pair_capture import validate_pair_capture
from .tabletop_power_shared_rod import (
    END_SUPPORT_NAMES, PALM_NAMES, PADS, POWER_MESH_NAMES, SUPPORTED_POWER_MESH_NAMES,
    SUPPORTED_LAYOUTS, PROXIMAL_LINK, PROXIMAL_NAMES, PROXIMAL_POWER_MESH_NAMES, _bindings,
)
from .tabletop_teacher_force_closure import _opposition, _pad_summary, _summarize_distal_pairs


DT = 1 / 480
SUBSTEPS = 12
NODE_COUNT = 41
MIN_LINK_LOAD_N = .05
MIN_HAND_LOAD_N = .3
MAX_OPPOSITION_COS = -.25
TABLE_LOAD_LIMIT_N = 1e-6
BUFFER_FIELDS = ("substeps", "mesh_count", "capacity", "attempted", "overflow")
JOINT_CONTACT_LINKS = (*PADS, "base_link")
IDENTITY_FIELDS = ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid", "physics_step", "rod_tick")


def _power_pairs(rod):
    names = tuple(rod["mesh_names"])
    if names not in (POWER_MESH_NAMES, *SUPPORTED_LAYOUTS):
        raise ValueError("the named Power layout with palms at slots 11 and 12 is required")
    if (rod["pair_capture_status"] != "validated_solver_samples"
            or rod["source"] != "RopeBatch.positions/velocities/contact_forces/mesh_wrenches"
            or rod["clock_valid"] is not True or rod["contact_valid"] is not True
            or any(not isinstance(rod[key], str) or not rod[key] for key in ("run_uuid", "episode_uid", "rod_uid"))
            or any(type(rod[key]) is not int or rod[key] < 0
                   for key in ("env_id", "reset_id", "physics_step", "rod_tick"))
            or not math.isclose(rod["timestamp_s"], rod["physics_step"] * DT, abs_tol=2e-7, rel_tol=0.)
            or not rod["physics_step"] or not rod["rod_tick"] or rod["rod_tick"] % SUBSTEPS):
        raise ValueError("an actual completed one-rod exchange with valid contacts is required")
    bindings = _bindings(rod["mesh_bindings"])
    if set(bindings) != set(names) - {"table", *END_SUPPORT_NAMES}:
        raise ValueError("the complete physical binding set is required")
    source_pairs = (PALM_NAMES, PROXIMAL_NAMES) if names == PROXIMAL_POWER_MESH_NAMES else (PALM_NAMES,)
    for left, right in source_pairs:
        for key in ("capture_sha256", "cooked_geometry_sha256", "sdf_sha256"):
            a, b = bindings[left].get(key), bindings[right].get(key)
            if (not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                        and value != "0"*64 for value in (a,b)) or a == b):
                raise ValueError("left and right Power contacts must retain separate geometry sources")

    capture = rod["rod_mesh_contact_pairs"]
    if (capture["scope"] != "rod_node_to_sdf_mesh_substep_force"
            or (capture["physics_step"], capture["rod_tick"]) != (rod["physics_step"], rod["rod_tick"])
            or not math.isclose(capture["substep_dt_s"], DT / SUBSTEPS, abs_tol=1e-12, rel_tol=0.)):
        raise ValueError("contact pairs need the same physics step, rod tick and substep period")
    buffer = {key: capture["solver_buffer"][key] for key in BUFFER_FIELDS}
    pairs = capture["pairs"]
    if (any(type(buffer[key]) is not int for key in BUFFER_FIELDS[:-1])
            or buffer["mesh_count"] != len(names) or buffer["substeps"] != SUBSTEPS
            or buffer["capacity"] < NODE_COUNT * len(names) * SUBSTEPS
            or buffer["attempted"] != len(pairs)):
        raise ValueError("capacity and count must describe the original complete Power solver buffer")
    ids, vectors = [], []
    for pair in pairs:
        indices = pair["substep"], pair["node_index"], pair["mesh_index"]
        if (any(type(index) is not int for index in indices)
                or not 0 <= indices[2] < len(names) or names[indices[2]] != pair["mesh_name"]):
            raise ValueError("a pair's ordered solver mesh index and link name must agree")
        ids.append(indices)
        vectors.append([*pair["contact_point_world_m"], *pair["outward_normal_world"],
                        *pair["force_on_rod_world_n"], *pair["torque_on_mesh_about_origin_world_nm"]])
    node_forces = np.asarray(rod["distributed_contact"])
    wrench_by_name = rod["named_wrenches"]
    if (node_forces.shape != (1, NODE_COUNT, 3) or set(wrench_by_name) != set(names)
            or not np.array_equal(rod["table_wrench_world"], wrench_by_name["table"])):
        raise ValueError("same-boundary one-rod force array and complete named wrenches required")
    if names in SUPPORTED_LAYOUTS:
        static = rod["external_support_wrenches_world"]
        if (type(static) is not dict or set(static) != {"table", *END_SUPPORT_NAMES}
                or any(not np.array_equal(wrench, wrench_by_name[name]) for name,wrench in static.items())
                or buffer["capacity"] != NODE_COUNT * len(names) * SUBSTEPS):
            raise ValueError("static support wrenches must come from the same ordered solver capture")
    return validate_pair_capture(
        dict(buffer, indices=np.asarray(ids, dtype=int).reshape(-1, 3),
             vectors=np.asarray(vectors, dtype=float).reshape(-1, 12)),
        dt_s=DT, node_forces=node_forces[0], mesh_wrenches=[wrench_by_name[name] for name in names],
    )["pairs"]


def measure_power_contact(rod):
    """Measure per-substep Power reactions without assigning grasp success."""
    try:
        pairs = _power_pairs(rod)
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("missing or malformed physical Power contact sources") from exc

    group = defaultdict(list)
    for pair in pairs:
        group[pair["substep"], pair["mesh_index"]].append(pair)

    hands = {}
    for side in ("left", "right"):
        evidence = []
        slots = {name: POWER_MESH_NAMES.index(side + "." + name) for name in JOINT_CONTACT_LINKS}
        for step in range(SUBSTEPS):
            links = {name: _pad_summary(group[step, slot], 1, slot) for name, slot in slots.items()}
            fingers = {name: links[name] for name in PADS}
            palm = links["base_link"]
            finger_load = sum(link["normal_load_n"] for link in fingers.values())
            palm_normal = palm["normal_on_rod_world"]
            opposing = {}
            for name, link in fingers.items():
                if (palm_normal is not None and link["normal_on_rod_world"] is not None
                        and link["normal_load_n"] > MIN_LINK_LOAD_N):
                    opposing[name] = float(np.clip(np.dot(palm_normal, link["normal_on_rod_world"]), -1., 1.))
            palm_cos = min(opposing.values(), default=None)
            pad_opposition = _opposition(fingers, "Power")
            passed = (
                all(link["normal_load_n"] > MIN_LINK_LOAD_N for link in links.values())
                and finger_load + palm["normal_load_n"] > MIN_HAND_LOAD_N
                and pad_opposition["effective"] and palm_cos is not None and palm_cos <= MAX_OPPOSITION_COS
            )
            evidence.append(dict(substep=step, links=links, palm_normal_load_n=palm["normal_load_n"],
                                 five_pad_normal_load_n=finger_load, palm_pad_cosines=opposing,
                                 pad_opposition=pad_opposition, loaded_opposed=bool(passed)))
        hands[side] = dict(per_substep=evidence,
                           loaded_opposed_substeps=[item["substep"] for item in evidence if item["loaded_opposed"]])

    table_slot = POWER_MESH_NAMES.index("table")
    table_load = [_pad_summary(group[step, table_slot], 1, table_slot)["normal_load_n"]
                  for step in range(SUBSTEPS)]
    result = dict(identity={key: rod[key] for key in IDENTITY_FIELDS},
                evidence_scope="same_boundary_simulated_distal_and_palm_contact",
                hands=hands,
                all_substeps_power_contacts=all(len(hand["loaded_opposed_substeps"]) == SUBSTEPS
                                                for hand in hands.values()),
                table_normal_load_n=table_load, zero_table_load=max(table_load) <= TABLE_LOAD_LIMIT_N,
                proximal_sdf_registered=False, native_qualification=False,
                grasp_qualified=False, teacher_admission=False)
    names = tuple(rod["mesh_names"])
    if names == PROXIMAL_POWER_MESH_NAMES:
        result["registered_proximal_links"] = [PROXIMAL_LINK]
        result["proximal_contacts"] = {}
        for side in ("left", "right"):
            slot = names.index(f"{side}.{PROXIMAL_LINK}")
            result["proximal_contacts"][side] = [_pad_summary(group[step,slot], 1, slot)
                                                 for step in range(SUBSTEPS)]
    if names in SUPPORTED_LAYOUTS:
        loads = {}
        for name in ("table", *END_SUPPORT_NAMES):
            slot = SUPPORTED_POWER_MESH_NAMES.index(name)
            loads[name] = [sum(float(np.linalg.norm(pair["force_on_rod_world_n"])) for pair in group[step,slot])
                           for step in range(SUBSTEPS)]
        result["external_support_force_norm_n"] = loads
        result["zero_external_support_load"] = all(max(per_step) <= TABLE_LOAD_LIMIT_N for per_step in loads.values())
    return result


def measure_supported_distal_contact(rod, *, grasp):
    """Assess Pinch or Tripod loads on the intact 17-mesh supported-rod capture."""
    if grasp not in ("pinch", "tripod"):
        raise ValueError("supported distal contact expects pinch or tripod")
    try:
        names = tuple(rod["mesh_names"])
        if names != PROXIMAL_POWER_MESH_NAMES:
            raise ValueError("the original ordered 17-slot rod capture is required")
        pairs = _power_pairs(rod)
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError("missing or malformed physical supported rod contact sources") from exc
    support_names = ("table", *END_SUPPORT_NAMES)
    evidence = _summarize_distal_pairs(
        pairs, substeps=SUBSTEPS, mesh_count=len(names), grasp=grasp.capitalize(),
        support_slots={name: names.index(name) for name in support_names},
    )
    nodes = np.asarray(rod.get("node_positions"), dtype=float)
    if nodes.shape != (1, NODE_COUNT, 3) or not np.isfinite(nodes).all():
        raise ValueError("the same-boundary 41 current world-space nodes are required")
    y = nodes[0, :, 1]
    if y[0] >= y[-1]:
        raise ValueError("supported rod material orientation must run from world negative Y to positive Y")
    center_y = (y[0] + y[-1]) / 2
    center_station = NODE_COUNT // 2
    target_sides = {slot: side for side, hand in evidence["per_hand"].items()
                    for slot in hand["required_mesh_ids"]}
    external = {name: [0.] * SUBSTEPS for name in support_names}
    unexpected = set()
    wrong_side = []
    for pair in pairs:
        name, step = names[pair["mesh_index"]], pair["substep"]
        if name in external:
            external[name][step] += float(np.linalg.norm(pair["force_on_rod_world_n"]))
        elif pair["mesh_index"] not in target_sides:
            unexpected.add(name)
        else:
            index = pair["node_index"]
            side = target_sides[pair["mesh_index"]]
            sign = 1 if side == "left" else -1
            if sign * (index - center_station) <= 0 or sign * (y[index] - center_y) <= 0:
                wrong_side.append(dict(link=name, substep=step, node_index=index, node_world_y_m=float(y[index])))
                row = evidence["per_hand"][side]["per_substep"][step]
                row["loaded"] = False
                reason = "required_pad_on_wrong_wire_side"
                if reason not in row["reasons"]:
                    row["reasons"].append(reason)
    for hand in evidence["per_hand"].values():
        hand["loaded_substeps"] = [row["substep"] for row in hand["per_substep"] if row["loaded"]]
        hand["loaded"] = len(hand["loaded_substeps"]) == SUBSTEPS
        hand["reasons"] = sorted({reason for row in hand["per_substep"] for reason in row["reasons"]})
    bilateral = all(hand["loaded"] for hand in evidence["per_hand"].values())
    evidence.update(
        bilateral=bilateral, all_solver_substeps_loaded=bilateral,
        wrong_side_contacts=wrong_side, material_side_split_world_y_m=float(center_y),
        identity={key: rod[key] for key in IDENTITY_FIELDS},
        external_support_force_norm_n=external,
        support_free=all(max(loads) <= TABLE_LOAD_LIMIT_N for loads in external.values()),
        unexpected_hand_contacts=sorted(unexpected),
        diagnostic_only=True, grasp_qualified=False, teacher_admission=False,
    )
    return evidence
