
"""Validate bilateral local Power opposition and its applied next-tick reaction."""

import numpy as np

from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts import IDENTITY, _clock
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import (
    MIN_LINK_LOAD_N, SUBSTEPS, measure_power_contact,
)
from go2_pvcnn.control.dh116.tabletop_power_shared_rod import PADS, PROXIMAL_POWER_MESH_NAMES
from scripts.screen_dh116_power_measured_reseat import support_adjacent_nodes
from scripts.verify_dh116_supported_distal_transfer import check_reaction_delivery, pairs_outside_station

SIDES = ("left", "right")
THUMB = PADS[0]


def _selected_links(stations, node_count):
    sides = {}
    center = node_count // 2
    for side in SIDES:
        side_links = {name.split(".",1)[1]: nodes for name, nodes in stations.items()
                      if name.startswith(side+".")}
        others = set(side_links) - {THUMB, "base_link"}
        if len(side_links) != 3 or not {THUMB, "base_link"} <= side_links.keys() or not others <= set(PADS[1:]):
            raise ValueError("each hand requires its own palm, thumb and one selected distal pad")
        if any(not nodes or any(type(i) is not int or
                   not (center < i < node_count-1 if side=="left" else 0 < i < center)
                   for i in nodes)
               for nodes in side_links.values()):
            raise ValueError("selected contact must be inside the assigned wire half")
        finger, = others
        if any(abs(a-b) > 1 for a in side_links[THUMB] for b in side_links[finger]):
            raise ValueError("local opposition requires the same or adjacent distal material nodes")
        sides[side] = (THUMB, finger, "base_link")
    if len(stations) != 6:
        raise ValueError("six original hand link identities required")
    return sides


def _opposed_substeps(steps, links):
    thumb, finger, palm = links
    evidence = []
    for step in steps:
        forces = {name: step["links"][name]["normal_load_n"] for name in links}
        left, right = (step["links"][name]["normal_on_rod_world"] for name in (thumb, finger))
        cosine = float(np.clip(np.dot(left, right), -1., 1.)) if left is not None and right is not None else None
        evidence.append(dict(substep=step["substep"], normal_load_by_link_n=forces,
                             thumb_partner_cos=cosine, palm_positive=forces[palm] > 0.,
                             opposed=bool(cosine is not None and cosine <= -.25
                                          and min(forces.values()) > MIN_LINK_LOAD_N)))
    return evidence


def _each_body_received(route, bindings, side, links):
    ids = route["body_ids"]
    forces = np.asarray(route["forces"], dtype=float)
    if len(ids) != len(set(ids)) or forces.shape != (1, len(ids), 3) or not np.isfinite(forces).all():
        raise ValueError("owned finite world force rows must match the issued body order")
    delivery = {}
    for name in links:
        body = bindings[f"{side}.{name}"]["body_index"]
        if body not in ids:
            raise ValueError("selected reaction body is absent from the PhysX composer")
        delivery[name] = float(np.linalg.norm(forces[0, ids.index(body)]))
    return delivery


def confirm_local_opposition(previous, current, stations, *, scene):
    """Check one complete local load window and the exact following applied world wrench."""
    rod = previous["rod"]
    next_rod = current["rod"]
    first, second = _clock(lambda: rod), _clock(lambda: next_rod)
    if (any(first[key] != second[key] for key in IDENTITY[:5])
            or previous["physics_step"] != first["physics_step"]
            or current["physics_step"] != second["physics_step"]
            or second["physics_step"] != first["physics_step"]+1
            or second["rod_tick"] != first["rod_tick"]+SUBSTEPS
            or tuple(rod["mesh_names"]) != PROXIMAL_POWER_MESH_NAMES
            or tuple(next_rod["mesh_names"]) != PROXIMAL_POWER_MESH_NAMES
            or rod["mesh_bindings"] != next_rod["mesh_bindings"]):
        raise ValueError("same consecutive Power exchange and owner bindings required")

    local = _selected_links(stations, node_count=len(rod["node_positions"][0]))
    first_contact, second_contact = measure_power_contact(rod), measure_power_contact(next_rod)
    transferred = check_reaction_delivery(previous, current)
    unexpected = set()
    near_support = set()
    for received in (rod, next_rod):
        pairs = received["rod_mesh_contact_pairs"]["pairs"]
        unexpected.update(pairs_outside_station(pairs, stations))
        flagged = support_adjacent_nodes(received, received["node_positions"][0], scene)
        near_support.update(f"{pair['mesh_name']}:{pair['node_index']}" for pair in pairs
                            if pair["mesh_name"] in stations and flagged[pair["node_index"]])
    hands = {}
    for side, links in local.items():
        first_steps = _opposed_substeps(first_contact["hands"][side]["per_substep"], links)
        second_steps = _opposed_substeps(second_contact["hands"][side]["per_substep"], links)
        delivery = _each_body_received(current["issued_composer_wrench"][side], rod["mesh_bindings"], side, links)
        hands[side] = dict(required_links=list(links), substeps=first_steps, following_substeps=second_steps,
                           all_substeps_opposed=all(row["opposed"] for row in (*first_steps,*second_steps)),
                           delivered_body_force_norm_n=delivery,
                           all_selected_bodies_received_force=all(force > 0. for force in delivery.values()))
    accepted = bool(transferred and not unexpected and not near_support and all(
        hand["all_substeps_opposed"] and hand["all_selected_bodies_received_force"] for hand in hands.values()))
    return dict(scope="power_partial_local_opposition_stage", contact_sdk_step=first["physics_step"],
                reaction_sdk_step=second["physics_step"], hands=hands, local_opposition_with_reaction=accepted,
                unexpected_pairs=sorted(unexpected), support_adjacent_pairs=sorted(near_support),
                all_substeps_power_contacts=first_contact["all_substeps_power_contacts"],
                partial_contact_only=True, grasp_qualified=False, teacher_admission=False)
