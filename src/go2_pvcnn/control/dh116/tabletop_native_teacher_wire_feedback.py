"""Observe complete wire feedback beside the original Teacher policy."""
from copy import deepcopy

import numpy as np

from .tabletop_deformable_r0_replay import _digest
from .tabletop_teacher_lift_progress import _palm_poses
from .tabletop_training_reward import WINDOW_SUBSTEPS
from .tabletop_wire_grasp_feedback import _pack_measurement, _palm_frame_geometry, pack_wire_control_feedback


def _pre_action_sources(pre):
    common = pre["common"]
    clock = common["identity"]
    source = {packet["side"]: packet for packet in common["source_packets"]}
    if (len(common["source_packets"]) != 2 or set(source) != {"left", "right"}
            or any(row["identity"] != clock for row in source.values())):
        raise ValueError("same-clock mounted palm source for each hand required")
    palms = []
    for side in ("left", "right"):
        pose = np.asarray(source[side]["body_link_pose_w"], dtype=float)
        if pose.shape != (7,):
            raise ValueError("raw world position and WXYZ quaternion required")
        palms.append((pose[:3], pose[[4,5,6,3]]))
    snapshot = pre["rod_boundary"]["rod_snapshot"]
    nodes = snapshot["node_positions"]
    if hasattr(nodes, "detach"):
        nodes = nodes.detach().cpu()
    nodes = np.asarray(nodes, dtype=float)
    if nodes.shape != (1,41,3) or not np.isfinite(nodes).all():
        raise ValueError("finite current rod snapshot required")
    return clock, nodes, palms


def _stations(nodes, palms):
    """Pick initial named material samples from measured geometry alone."""
    return {
        side: int(indices[np.argmin(np.linalg.norm(nodes[0,indices] - pose[0], axis=1))])
        for side,indices,pose in zip(("left","right"), (np.arange(21,41),np.arange(20)), palms, strict=True)
    }


class WireFeedbackTrace:
    """One episode's read-only control history and independent diagnostic sink."""

    def __init__(self, grasp, sink):
        self._grasp = grasp
        self._sink = sink
        self._stations = None
        self.clear_window()

    def clear_window(self):
        self._rows = []
        self._bootstrap = None

    def record(self, exchange):
        if len(self._rows) >= WINDOW_SUBSTEPS:
            raise ValueError("wire trace received more than one control interval")
        # Own only what feedback reads. Never retain a view or articulation.
        keys = ("physics_step", "policy_step", "substep", "rod", "closure_geometry", "sdk_hand_geometry")
        self._rows.append(deepcopy({key: exchange[key] for key in keys if key in exchange}))

    def emit(self, pre, *, pre_action_sha256, purpose, collection_token):
        clock = pre["common"]["identity"]
        observation = None
        reason = None
        window = None
        try:
            clock, nodes, palms = _pre_action_sources(pre)
            if self._stations is None:
                self._stations = _stations(nodes, palms)
            if not self._rows and purpose == "decision" and clock["policy_step"] == 0:
                hands, _ = _palm_frame_geometry(nodes, palms, self._stations)
                observation = _pack_measurement(dict(measured_hands=hands, reaction_valid=False))
                status = "no_previous_control_window"
            else:
                if len(self._rows) != WINDOW_SUBSTEPS:
                    raise ValueError("previous control interval is incomplete")
                last = self._rows[-1]
                if not np.array_equal(nodes, last["rod"]["node_positions"]):
                    raise ValueError("pre-action rod snapshot differs from the final exchange")
                measured = _palm_poses(last, clock["physics_step"])
                for (pos,quat), (last_pos,last_quat) in zip(palms, measured, strict=True):
                    if (not np.array_equal(pos, last_pos)
                            or not (np.array_equal(quat, last_quat) or np.array_equal(quat, -np.asarray(last_quat)))):
                        raise ValueError("pre-action palm pose differs from the final exchange")
                observation = pack_wire_control_feedback(self._rows, clock, grasp=self._grasp,
                                                         grasp_material_ids=self._stations)
                window = dict(first_sdk_step=self._rows[0]["physics_step"], end_sdk_step=last["physics_step"],
                              end_rod_tick=last["rod"]["rod_tick"], control_index=last["policy_step"])
                status = "measured" if np.asarray(observation["reaction_valid"]).any() else "reaction_unavailable"
                if purpose == "decision" and self._bootstrap != (pre_action_sha256, _digest(observation)):
                    raise ValueError("wire feedback or owned pre-action state changed since bootstrap")
        except (KeyError, TypeError, ValueError) as error:
            # Reject only this diagnostic sample. The original Actor stays authoritative.
            observation, status, reason = None, "rejected", str(error)

        observation_sha256 = _digest(observation) if observation is not None else None
        if purpose == "bootstrap":
            self._bootstrap = pre_action_sha256, observation_sha256
        self._sink("wire_control_feedback", dict(purpose=purpose, clock=deepcopy(clock),
            collection_token=collection_token, pre_action_sha256=pre_action_sha256,
            source_window=window, status=status, reason=reason, observation=observation,
            observation_sha256=observation_sha256, actor_used=False, teacher_admission=False))
        if purpose != "bootstrap":
            self.clear_window()
