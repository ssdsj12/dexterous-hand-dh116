"""Versioned DH116 active-finger reference for a native approach diagnostic."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .linear_coupling import ACTIVE, active_limits


SIDES = ("left", "right")
ERROR_FIELDS = tuple("finger_goal_max_error_" + side for side in SIDES)
GOAL_SCHEMA = "dh116-teacher-finger-goal-v1"


@dataclass(frozen=True)
class FingerGoal:
    grasp: str
    reference_sha256: str
    target_rad: tuple
    participating: tuple
    tolerance_rad: float = 0.05
    scale_rad: float = 0.25
    arrival_steps: int = 16
    arrival_reward: float = 2.0

    def max_errors_rad(self, measured):
        q = np.asarray(measured, dtype=float)
        if q.shape != (2, len(ACTIVE)) or not np.isfinite(q).all():
            raise ValueError("finite two-hand active radians required")
        error = np.abs(q - self.target_rad)
        return tuple(float(value) for value in error[:, self.participating].max(axis=1))

    def potential(self, errors):
        return math.fsum(
            math.exp(-max(0., error - self.tolerance_rad) / self.scale_rad)
            for error in errors
        ) / len(SIDES)


def load_finger_goal(spec):
    if (type(spec) is not dict or set(spec) != {"schema", "grasp", "reference_sha256"}
            or spec["schema"] != GOAL_SCHEMA
            or spec["grasp"] not in ("pinch", "tripod", "power")):
        raise ValueError("versioned native finger goal required")
    grasp = spec["grasp"]
    root = Path(__file__).resolve().parents[4]
    configs = root / "configs/dh116_training"
    references = {
        "pinch": configs / "pose_reference.json",
        "tripod": configs / "gpu_teacher_style_seeds_v1.json",
        "power": root / "configs/dh116_tasks/flexible_power_near_open_settle_v1.json",
    }
    reference = references[grasp]
    raw = reference.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if spec["reference_sha256"] != actual_sha:
        raise ValueError("finger goal reference SHA differs")
    data = json.loads(raw)
    if grasp == "pinch":
        targets = tuple(tuple(data["pose_contract"][side]["hand_active_q"]) for side in SIDES)
    elif grasp == "power":
        targets = tuple(tuple(data["candidate"][side]["target_active_q_rad"]) for side in SIDES)
    else:
        targets = tuple(tuple(data["styles"][grasp][side]["active_q_rad"]) for side in SIDES)
    lower, upper = active_limits()
    q = np.asarray(targets, dtype=float)
    if (q.shape != (2, len(ACTIVE)) or not np.isfinite(q).all()
            or np.any(q < lower) or np.any(q > upper)):
        raise ValueError("finger goal requires two finite active states inside named DH116 joint limits")
    counts = {"pinch": 3, "tripod": 4, "power": 6}
    return FingerGoal(grasp, actual_sha, targets, tuple(range(counts[grasp])))


def measured_active_angles(hands):
    """Extract active joint positions from single-environment SDK measurements."""
    angles = []
    for side in SIDES:
        sample = hands[side]
        names = tuple(sample["joint_names"])
        q = np.asarray(sample["q"])
        if (q.shape != (1, len(names)) or len(set(names)) != len(names)
                or not set(ACTIVE).issubset(names) or not np.isfinite(q).all()):
            raise ValueError("named finite SDK joint readback required")
        angles.append(tuple(float(q[0, names.index(name)]) for name in ACTIVE))
    return tuple(angles)
