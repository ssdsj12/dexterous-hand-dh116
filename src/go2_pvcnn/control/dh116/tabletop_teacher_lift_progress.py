"""Measured Close, PreliftHold and Lift progression over supported-rod exchanges."""
from dataclasses import dataclass
import math

import numpy as np

from .tabletop_native_phase_contract_v2 import PhaseOwnerV2, PhaseReceiptV2, PhaseSampleV2
from .tabletop_supported_lift_observation import (
    FixedLiftReference, SupportedLiftObservation, observe_supported_lift,
)
from .tabletop_training_reward import PHYSICS_HZ, SUCCESS, WINDOW_SUBSTEPS, control_total


@dataclass(frozen=True, slots=True)
class LiftProgress:
    phase: PhaseReceiptV2
    observation: SupportedLiftObservation
    palm_speed_m_s: float | None
    relative_node_speed_m_s: float | None
    lift_potential: float


def _local_points(nodes, position, quaternion):
    vector = nodes - position
    # Rotate World vectors by the inverse of the measured XYZW quaternion.
    axis = -quaternion[:3]
    cross = 2 * np.cross(axis, vector)
    return vector + quaternion[3] * cross + np.cross(axis, cross)


def _palm_poses(exchange, tick):
    if "closure_geometry" in exchange:
        packet = exchange["closure_geometry"]
        raw = packet["raw"]
        rod = exchange["rod"]
        if (packet["source_complete"] is not True
                or any(raw["identity"].get(key) != rod[key] for key in (
                    "run_uuid", "episode_uid", "env_id", "reset_id", "rod_uid",
                    "physics_step", "rod_tick", "solver_time_s", "timestamp_s"))
                or not np.array_equal(raw["rod_nodes_m"], np.asarray(rod["node_positions"])[0])):
            raise ValueError("same physical rod and native palm geometry required")
        poses = []
        for side in ("left", "right"):
            hand = raw["hands"][side]
            names = hand["body_names"]
            i = names.index("base_link")
            if names.count("base_link") != 1:
                raise ValueError("unique measured palm link required")
            pose = np.asarray(hand["link_poses_xyzw"], dtype=float)
            if pose.shape != (1, len(names), 7):
                raise ValueError("complete measured hand pose array required")
            poses.append((pose[0, i, :3], pose[0, i, 3:]))
        return poses

    geometry = exchange["sdk_hand_geometry"]
    if type(geometry["physics_step"]) is not int or geometry["physics_step"] != tick:
        raise ValueError("same physical step required for the measured palm poses")
    return [geometry["poses"][f"{side}.base_link"] for side in ("left", "right")]


class SupportedLiftProgress:
    """One fixed Reset and consecutive measured physical endpoints.

    The velocity limits are from the existing force diagnostic. Palm-local
    motion includes palm rotation. This candidate phase validates prelift
    motion only. Final Hold requires independent fixed-anchor slip evidence.
    """

    def __init__(self, reference, *, grasp, prelift_stable_s):
        if type(reference) is not FixedLiftReference:
            raise ValueError("fixed measured Reset required")
        self._reference = reference
        self._grasp = grasp
        self._phase = PhaseOwnerV2(prelift_stable_s=prelift_stable_s)
        self._phase.reset(episode_uid=reference.epoch[2], physics_tick=reference.physics_step)
        self._previous = None

    def observe(self, exchange):
        """Use the rod and actual palm poses of the same completed SDK step."""
        obs = observe_supported_lift(exchange["rod"], self._reference, grasp=self._grasp)
        tick = obs.clock[5]
        palms = _palm_poses(exchange, tick)
        nodes = np.asarray(obs.node_positions_world_m)
        positions, relative = [], []
        for (position, quaternion), half in zip(palms, (nodes[21:], nodes[:20])):
            position = np.asarray(position, dtype=float)
            quaternion = np.asarray(quaternion, dtype=float)
            if (position.shape != (3,) or quaternion.shape != (4,)
                    or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
                    or not math.isclose(float(np.linalg.norm(quaternion)), 1., abs_tol=1e-5, rel_tol=0.)):
                raise ValueError("finite measured palm position and unit XYZW quaternion required")
            positions.append(position.copy())
            relative.append(_local_points(half, position, quaternion))

        positions = np.asarray(positions)
        relative = np.asarray(relative)
        palm_speed = relative_speed = None
        if self._previous is not None:
            old_palms, old_relative = self._previous
            palm_speed = float(np.linalg.norm(positions - old_palms, axis=-1).max() * PHYSICS_HZ)
            relative_speed = float(np.linalg.norm(relative - old_relative, axis=-1).max() * PHYSICS_HZ)
        stable = palm_speed is not None and palm_speed <= .001 and relative_speed <= .002
        receipt = self._phase.observe(PhaseSampleV2(obs.clock[2], tick, obs.bilateral, stable, False))
        self._previous = (positions, relative)
        height = obs.off_support_lift_m
        potential = 0.
        if receipt.phase == "Lift" and height is not None:
            potential = 4. * min(1., max(0., height / SUCCESS["net_lift_m"]))
        return LiftProgress(receipt, obs, palm_speed, relative_speed, potential)


def lift_reward_delta(pre, post, *, gamma, terminated):
    """Lift contribution at the original 60 Hz PPO settlement boundary."""
    if (type(pre) is not LiftProgress or type(post) is not LiftProgress
            or pre.observation.clock[:5] != post.observation.clock[:5]
            or pre.observation.reset_rod_sha256 != post.observation.reset_rod_sha256
            or pre.observation.grasp != post.observation.grasp
            or not 1 <= post.phase.physics_tick - pre.phase.physics_tick <= WINDOW_SUBSTEPS
            or type(terminated) is not bool):
        raise ValueError("same fixed Reset and one actual control interval required")
    return control_total(pre.lift_potential, post.lift_potential, (0.,), gamma, 0., terminated)
