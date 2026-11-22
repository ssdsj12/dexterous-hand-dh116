"""One bounded pose-feedback pulse before any valid hand contact."""
import math

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.screen_dh116_tripod_supported_reset import archived_tube
from .tabletop_surface_contract import ACTION_CONTRACT
from .tabletop_training_reward import CONTROL_HZ


SIDES = ("left", "right")
PULSE_LIMIT_M = 0.00005
REFERENCE_RADIUS_M = 0.005


def wire_tube(nodes, *, radius_m):
    """Retain the measured segment envelope and change only its radial support."""
    if (isinstance(radius_m, (bool, np.bool_)) or not math.isfinite(radius_m) or radius_m <= 0):
        raise ValueError("measured positive wire radius required")
    normals, reference_upper = archived_tube(nodes)
    upper = np.nextafter(reference_upper + (radius_m - REFERENCE_RADIUS_M), np.inf)
    return normals, upper


def choose_pulse(observation, geometry):
    """Aim in World XY transverse to the measured wire, with wrist Z held.

    Measured hand force ends the approach pulse. Closure and lift are owned
    by subsequent contact phases.
    """
    sdk = observation["clock"]["physics_step"]
    if type(sdk) is not int or geometry["physics_step"] != sdk:
        raise ValueError("same SDK tick required for palm pose and wire reaction")
    action = np.zeros(ACTION_CONTRACT["size"])
    if observation["reaction_valid"] is not True:
        return action
    hands = observation["measured_hands"]
    for side in SIDES:
        loads = [pad["mean_normal_n"] for pad in hands[side]["reaction"].values()]
        if not loads or not np.isfinite(loads).all() or min(loads) < 0:
            raise ValueError("finite measured hand reactions required")
        if max(loads) > 0:
            return action
    start, _ = ACTION_CONTRACT["slices"]["world_translation_velocity"]
    distance_per_control = ACTION_CONTRACT["translation_rate_m_s"] / CONTROL_HZ
    for index, side in enumerate(SIDES):
        hand = hands[side]
        _, quat = geometry["poses"][f"{side}.base_link"]
        wire = np.asarray(hand["xyz_in_palm_m"], dtype=float)
        axis = np.asarray(hand["tangent_in_palm"], dtype=float)
        if (wire.shape != (3,) or axis.shape != (3,) or np.shape(quat) != (4,)
                or not np.isfinite(wire).all() or not np.isfinite(axis).all()
                or not np.isfinite(quat).all()
                or not math.isclose(np.linalg.norm(quat), 1., rel_tol=0., abs_tol=1e-5)
                or not math.isclose(np.linalg.norm(axis), 1., rel_tol=0., abs_tol=1e-5)):
            raise ValueError("finite palm-local wire pose, tangent and unit XYZW required")
        world = Rotation.from_quat(quat)
        lateral, tangent = world.apply([wire, axis])[:, :2]
        length = np.linalg.norm(tangent)
        if length > 1e-8:
            tangent /= length
            lateral -= (lateral @ tangent) * tangent
        distance = np.linalg.norm(lateral)
        if distance > PULSE_LIMIT_M:
            lateral *= PULSE_LIMIT_M / distance
        action[start+3*index:start+3*index+2] = lateral / distance_per_control
    return action
