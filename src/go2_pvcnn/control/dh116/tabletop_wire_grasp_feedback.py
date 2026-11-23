"""Policy-facing geometry and reaction from a completed supported-wire interval."""

import math

import numpy as np
from scipy.spatial.transform import Rotation

from .tabletop_bimanual_table_contacts import _clock
from .tabletop_deformable_solver_clock import solver_clock_matches
from .tabletop_power_contact_evidence import SUBSTEPS, measure_power_contact, measure_supported_distal_contact
from .tabletop_power_shared_rod import PADS, PROXIMAL_POWER_MESH_NAMES, SIDES, SUPPORTED_LAYOUTS
from .tabletop_supported_wire_specs import MATERIAL_NODES, _validate_stations
from .tabletop_teacher_lift_progress import _palm_poses
from .tabletop_training_reward import WINDOW_SUBSTEPS

PAD_NAMES = ("thumb", "index", "middle", "ring", "little", "palm")
PAD_LINKS = (*PADS, "base_link")
PAD_INDEX = {name: index for names in (PAD_NAMES, PAD_LINKS)
             for index, name in enumerate(names)}


def pack_wire_grasp_feedback(observation, decision_clock):
    """Pack proposal inputs from the current pre-action boundary without specimen identity.

    Per side, positions are metres and tangents are unit vectors in palm coordinates.
    The six force rows contain minimum/mean normal load and XYZ hand reaction,
    in newtons. Unavailable rows have an explicit false mask and use zero only
    as the masked representation.
    """
    clock = _clock(lambda: observation["clock"])
    if (clock != _clock(lambda: decision_clock) or clock["reset_id"] == 0
            or clock["rod_tick"] < SUBSTEPS or clock["rod_tick"] % SUBSTEPS):
        raise ValueError("same completed pre-action SDK/rod boundary required")
    positions, tangents = [], []
    forces = np.zeros((len(SIDES), len(PAD_NAMES), 5))
    valid = np.zeros(forces.shape[:2], dtype=bool)
    for side_index, side in enumerate(SIDES):
        hand = observation["measured_hands"][side]
        position = np.asarray(hand["xyz_in_palm_m"], dtype=float)
        tangent = np.asarray(hand["tangent_in_palm"], dtype=float)
        if (position.shape != (3,) or tangent.shape != (3,) or not np.isfinite(position).all()
                or not math.isclose(np.linalg.norm(tangent), 1., abs_tol=1e-5, rel_tol=0.)):
            raise ValueError("finite measured palm-frame position and unit tangent required")
        positions.append(position.tolist())
        tangents.append(tangent.tolist())
        if observation["reaction_valid"] is not True:
            continue
        if not hand["reaction"]:
            raise ValueError("missing measured reaction for the hand")
        for name, pad in hand["reaction"].items():
            index = PAD_INDEX.get(name)
            if index is None or valid[side_index, index]:
                raise ValueError("unique canonical distal pad or palm required")
            row = np.asarray([pad["minimum_normal_n"], pad["mean_normal_n"],
                              *pad["mean_force_on_hand_palm_n"]], dtype=float)
            if row.shape != (5,) or not np.isfinite(row).all() or not 0. <= row[0] <= row[1]:
                raise ValueError("finite measured palm-frame force and ordered normal loads required")
            forces[side_index, index] = row
            valid[side_index, index] = True
    return dict(position_in_palm_m=positions, tangent_in_palm=tangents,
                reaction_force_n=forces.tolist(), reaction_valid=valid.tolist())


def pack_wire_control_feedback(exchanges, decision_clock, *, grasp, grasp_material_ids):
    """Reduce the eight SDK steps ending at this decision in the final palm frame.

    Raw solver pairs are validated each step. An unavailable measurement anywhere
    in the window leaves forces masked while retaining the measured geometry.
    """
    if grasp not in ("pinch", "tripod", "power"):
        raise ValueError("supported feedback requires Pinch, Tripod or Power")
    _validate_stations(grasp_material_ids)
    end = _clock(lambda: decision_clock)
    if len(exchanges) != WINDOW_SUBSTEPS:
        raise ValueError("complete previous control window required")
    clocks = [_clock(lambda: exchange["rod"]) for exchange in exchanges]
    if end != clocks[-1]:
        raise ValueError("contact window must end at the current decision boundary")
    epoch_keys = ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid")
    epoch = tuple(end[key] for key in epoch_keys)
    first = exchanges[0]["policy_step"]
    if type(first) is not int or first < 0:
        raise ValueError("nonnegative original control policy index required")
    next_index = decision_clock.get("policy_step")
    if type(next_index) is not int or next_index != first+1:
        raise ValueError("next decision policy index required at the contact window endpoint")
    names = tuple(exchanges[-1]["rod"]["mesh_names"])
    layouts = SUPPORTED_LAYOUTS if grasp == "power" else (PROXIMAL_POWER_MESH_NAMES,)
    if names not in layouts:
        raise ValueError("ordered supported contact layout required")
    for substep, (exchange, clock) in enumerate(zip(exchanges, clocks, strict=True)):
        remaining = WINDOW_SUBSTEPS - 1 - substep
        if (tuple(clock[key] for key in epoch_keys) != epoch
                or clock["physics_step"] != end["physics_step"] - remaining
                or clock["rod_tick"] != end["rod_tick"] - remaining*SUBSTEPS
                or clock["rod_tick"] < SUBSTEPS
                or exchange["physics_step"] != clock["physics_step"]
                or type(exchange["substep"]) is not int or exchange["substep"] != substep
                or type(exchange["policy_step"]) is not int or exchange["policy_step"] != first
                or tuple(exchange["rod"]["mesh_names"]) != names):
            raise ValueError("consecutive SDK/rod steps from one control interval and owner required")
    if any(not solver_clock_matches(a["solver_time_s"], b["solver_time_s"], steps=1)
           for a, b in zip(clocks, clocks[1:])):
        raise ValueError("consecutive native solver times required")

    nodes = np.asarray(exchanges[-1]["rod"]["node_positions"], dtype=float)
    if nodes.shape != (1, MATERIAL_NODES, 3) or not np.isfinite(nodes).all():
        raise ValueError("finite current 41-node wire geometry required")
    palms = _palm_poses(exchanges[-1], end["physics_step"])
    hands, rotations = {}, {}
    for side, (position, rotation) in zip(SIDES, palms, strict=True):
        if (np.shape(position) != (3,) or not np.isfinite(position).all()
                or np.shape(rotation) != (4,) or not np.isfinite(rotation).all()
                or not math.isclose(np.linalg.norm(rotation), 1., abs_tol=1e-5, rel_tol=0.)):
            raise ValueError("finite current palm position and unit XYZW quaternion required")
        inverse = rotations[side] = Rotation.from_quat(rotation).inv()
        index = grasp_material_ids[side]
        tangent = nodes[0, min(index+1, MATERIAL_NODES-1)] - nodes[0, max(index-1, 0)]
        length = np.linalg.norm(tangent)
        if not length > 0.:
            raise ValueError("current material station has no tangent")
        hands[side] = dict(xyz_in_palm_m=inverse.apply(nodes[0, index]-position).tolist(),
                           tangent_in_palm=inverse.apply(tangent/length).tolist(), reaction={})

    observation = dict(clock=end, measured_hands=hands, reaction_valid=False)
    if any(exchange["rod"]["contact_valid"] is not True for exchange in exchanges):
        return pack_wire_grasp_feedback(observation, end)

    power = grasp == "power"
    link_field = "links" if power else "per_pad"
    measured = []
    for exchange in exchanges:
        rod = exchange["rod"]
        sample = measure_power_contact(rod) if power else measure_supported_distal_contact(rod, grasp=grasp)
        if sample["wrong_side_contacts"] or sample.get("unexpected_hand_contacts"):
            return pack_wire_grasp_feedback(observation, end)
        measured.append(sample["hands" if power else "per_hand"])

    for side in SIDES:
        substeps = [row[link_field] for frame in measured for row in frame[side]["per_substep"]]
        if len(substeps) != WINDOW_SUBSTEPS*SUBSTEPS or any(set(row) != set(substeps[0]) for row in substeps):
            raise ValueError("complete consistent per-pad solver samples required")
        for name in substeps[0]:
            loads = [row[name]["normal_load_n"] for row in substeps]
            force_on_rod = np.mean([row[name]["force_on_rod_world_n"] for row in substeps], axis=0)
            hands[side]["reaction"][name] = dict(minimum_normal_n=min(loads), mean_normal_n=np.mean(loads),
                mean_force_on_hand_palm_n=rotations[side].apply(-force_on_rod).tolist())
    observation["reaction_valid"] = True
    return pack_wire_grasp_feedback(observation, end)
