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
SETTLED_THUMB_STATIONS = {"left.finger14_link": (34, 35, 36)}


class FirstDistalContact:
    """Bind first intended force to its immediately following applied SDK reaction."""

    def __init__(self, previous, *, approach=None):
        self.previous = previous
        self.approach = None if approach is None else np.asarray(approach, dtype=float).copy()
        self.stations = LEFT_TRIPOD_STATIONS if approach is None else SETTLED_THUMB_STATIONS
        self.first_contact_sdk = None
        self.first_transfer_sdk = None
        self.first_pairs = []
        self.violations = []
        self.contact_samples = 0

    def select(self, action, clock):
        import torch

        if clock["physics_step"] != self.previous["physics_step"]:
            raise ValueError("first distal selector needs the previous physical endpoint")
        if self.approach is None:
            allowed = torch.zeros_like(action)
            allowed[0, 14] = action[0, 14]
            if not torch.equal(allowed, action) or not 0. <= float(action[0, 14]) <= 1.:
                raise ValueError("only the original unilateral upward approach slot may change")
        elif action.any() and not torch.equal(action, action.new_tensor(self.approach)[None]):
            raise ValueError("only the screened left translation or zero may be proposed")
        paused = self.first_contact_sdk is not None
        return (torch.zeros_like(action) if paused else action,
                dict(paused=paused, reason="first_distal_contact_hold" if paused else "bounded_left_approach"))

    def observe(self, row):
        sdk = row["physics_step"]
        rod = row["rod"]
        if rod["physics_step"] != sdk or rod["rod_tick"] != (sdk-2)*12:
            raise ValueError("same SDK and solver endpoint required for first distal forces")
        transferred = check_reaction_delivery(self.previous, row)
        pairs = rod["rod_mesh_contact_pairs"]["pairs"]
        self.violations = pairs_outside_station(pairs, self.stations)
        if self.violations:
            return "unexpected_distal_station"
        wrench = rod["external_support_wrenches_world"]
        if (np.linalg.norm(wrench["table"][:3]) > 1e-6
                or any(wrench[name][2] >= 0. for name in END_SUPPORT_NAMES)):
            return "unexpected_support_transition"
        contacts = [pair for pair in pairs if pair["mesh_name"] in self.stations
                    and pair["normal_force_on_rod_n"] > 0.]
        hand_wrenches = rod["hand_wrenches_com_world"]
        if any(np.any(hand_wrenches["right"][key]) for key in ("forces", "torques")):
            return "right_unexpected_wrench"
        left_loaded = any(np.any(hand_wrenches["left"][key]) for key in ("forces", "torques"))
        if left_loaded and not contacts:
            return "left_wrench_without_positive_normal_pair"
        if contacts:
            if not left_loaded:
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


class HeldPalmTarget:
    """Check a fixed world target and the original mount conversion independently."""

    def __init__(self, side, reset_pose):
        import torch

        from .tabletop_free_wrist_grasp import compose_rotation

        self.side = side
        self.index = ("left", "right").index(side)
        seed = torch.as_tensor(reset_pose, dtype=torch.float64).reshape(1, 7)
        rotation = compose_rotation(seed[:, 3:], seed.new_zeros((1, 3)))
        self.pose = torch.cat((seed[:, :3], rotation), dim=1).numpy()
        self._previous = self._anchor = None

    def observe(self, readback):
        import torch

        from .tabletop_palm_mount import JOINT_NAMES, local_translation

        index = self.index
        slots = [*range(6*index, 6*index+6), *range(12+3*index, 15+3*index),
                 *range(18+3*index, 21+3*index)]
        submitted = readback["submission"]
        if (np.any(np.asarray(readback["request"]["action24"])[:, slots])
                or not np.array_equal(np.asarray(submitted["projected_root_poses"])[:, index], self.pose)
                or np.any(np.asarray(submitted["projected_angles"])[:, index])):
            return False
        pre, post = (readback[key]["hands"][self.side] for key in ("pre", "post"))
        anchor = pre["mount_anchor_pose_w"]
        if not np.array_equal(anchor, post["mount_anchor_pose_w"]):
            return False
        before, after = pre["backend_targets"], post["backend_targets"]
        translation = [pre["joint_names"].index(name) for name in JOINT_NAMES]
        if self._previous is None:
            if np.any(before[:, translation]):
                return False
        elif not np.array_equal(before, self._previous) or not np.array_equal(anchor, self._anchor):
            return False
        expected = before.copy()
        expected[:, translation] = local_translation(
            torch.from_numpy(self.pose), torch.from_numpy(anchor).double()).float().numpy()
        if not np.array_equal(after, expected):
            return False
        self._previous, self._anchor = after.copy(), anchor.copy()
        return True


def settled_left_action(world_translation_m, *, windows):
    """Normalize a frozen straight path without changing the 24D action contract."""
    from .tabletop_free_wrist_grasp import CONTRACT as WRIST

    delta = np.asarray(world_translation_m, dtype=float)
    if (delta.shape != (3,) or not np.isfinite(delta).all() or not np.any(delta)
            or type(windows) is not int or windows < 1
            or np.any(delta < WRIST["palm_lower_m"]) or np.any(delta > WRIST["palm_upper_m"])):
        raise ValueError("the screened translation must fit the original wrist workspace")
    rate = CONTROL_HZ * delta / (windows * ACTION_CONTRACT["translation_rate_m_s"])
    if np.linalg.norm(rate) > 1.:
        raise ValueError("the screened translation exceeds the original wrist rate")
    action = np.zeros(ACTION_CONTRACT["size"])
    action[12:15] = rate
    return action


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
