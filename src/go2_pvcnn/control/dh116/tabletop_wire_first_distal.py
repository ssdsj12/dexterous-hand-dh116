"""Hold the original targets as soon as an intended Tripod contact is measured."""
import numpy as np

from scripts.verify_dh116_supported_distal_transfer import check_reaction_delivery, pairs_outside_station
from .tabletop_power_shared_rod import END_SUPPORT_NAMES
from .tabletop_surface_contract import ACTION_CONTRACT
from .tabletop_training_reward import CONTROL_HZ


LEFT_TRIPOD_STATIONS = {
    "left.finger14_link": (34, 35, 36),
    "left.finger23_link": (34, 35, 36),
    "left.finger33_link": (32, 33, 34),
}
APPROACH_TRAVEL_M = .0033
APPROACH_WINDOWS = 20
HOLD_WINDOWS = 1


class FirstDistalContact:
    """Bind first intended force to its immediately following applied SDK reaction."""

    def __init__(self, previous):
        self.previous = previous
        self.first_contact_sdk = None
        self.first_transfer_sdk = None
        self.first_pairs = []
        self.violations = []
        self.contact_samples = 0
        self._right_target = None

    def select(self, action, clock):
        import torch

        if clock["physics_step"] != self.previous["physics_step"]:
            raise ValueError("first distal selector needs the previous physical endpoint")
        allowed = torch.zeros_like(action)
        allowed[0, 14] = action[0, 14]
        if not torch.equal(allowed, action) or not 0. <= float(action[0, 14]) <= 1.:
            raise ValueError("only the original unilateral upward approach slot may change")
        paused = self.first_contact_sdk is not None
        return (torch.zeros_like(action) if paused else action,
                dict(paused=paused, reason="first_distal_contact_hold" if paused else "bounded_left_approach"))

    def observe(self, row):
        sdk = row["physics_step"]
        rod = row["rod"]
        if rod["physics_step"] != sdk or rod["rod_tick"] != (sdk-2)*12:
            raise ValueError("same SDK and solver endpoint required for first distal forces")
        transferred = check_reaction_delivery(self.previous, row)
        hand = row["action_readback"]
        pre, post = (hand[key]["hands"]["right"]["backend_targets"] for key in ("pre", "post"))
        if self._right_target is None:
            self._right_target = np.array(pre, copy=True)
        if not np.array_equal(pre, self._right_target) or not np.array_equal(post, self._right_target):
            return "right_target_changed"
        pairs = rod["rod_mesh_contact_pairs"]["pairs"]
        self.violations = pairs_outside_station(pairs, LEFT_TRIPOD_STATIONS)
        if self.violations:
            return "unexpected_distal_station"
        wrench = rod["external_support_wrenches_world"]
        if (np.linalg.norm(wrench["table"][:3]) > 1e-6
                or any(wrench[name][2] >= 0. for name in END_SUPPORT_NAMES)):
            return "unexpected_support_transition"
        contacts = [pair for pair in pairs if pair["mesh_name"] in LEFT_TRIPOD_STATIONS
                    and pair["normal_force_on_rod_n"] > 0.]
        hand_wrenches = rod["hand_wrenches_com_world"]
        if any(np.any(hand_wrenches["right"][key]) for key in ("forces", "torques")):
            return "right_unexpected_wrench"
        if contacts:
            if not any(np.any(hand_wrenches["left"][key]) for key in ("forces", "torques")):
                return "distal_pair_without_reaction"
            self.contact_samples += 1
            if self.first_contact_sdk is None:
                self.first_contact_sdk = sdk
                self.first_pairs = contacts
        if self.first_contact_sdk is not None and sdk == self.first_contact_sdk+1:
            if not transferred:
                return "first_distal_reaction_not_applied"
            self.first_transfer_sdk = sdk
        self.previous = row
        return None


def right_relief_action(direction, *, travel_m, rotation_rad, windows):
    """Bind the screened World XYZ and original right intrinsic-Z channels."""
    direction = np.asarray(direction, dtype=float)
    if (direction.shape != (3,) or not np.isfinite(direction).all()
            or not np.isclose(np.linalg.norm(direction), 1., rtol=0., atol=1e-12)
            or type(windows) is not int or windows < 1
            or not 0. < travel_m <= .003 or not 0. < rotation_rad <= .02):
        raise ValueError("finite bounded right wrist relief required")
    rate = travel_m*CONTROL_HZ/(windows*ACTION_CONTRACT["translation_rate_m_s"])
    spin = rotation_rad*CONTROL_HZ/(windows*ACTION_CONTRACT["rotation_rate_rad_s"])
    if rate > 1.+1e-12 or spin > 1.:
        raise ValueError("right relief exceeds the original wrist rate")
    result = np.zeros(ACTION_CONTRACT["size"])
    result[15:18] = direction*min(1., rate)
    result[23] = spin
    return result
