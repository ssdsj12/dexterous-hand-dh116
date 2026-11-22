"""Policy-facing geometry and reaction from a completed supported-wire interval."""

import math

import numpy as np

from .tabletop_bimanual_table_contacts import _clock
from .tabletop_power_contact_evidence import SUBSTEPS
from .tabletop_power_shared_rod import PADS, SIDES

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
