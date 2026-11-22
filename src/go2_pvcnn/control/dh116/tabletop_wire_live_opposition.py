"""Bounded left Tripod opposition diagnostic on the already-owned live wire."""

import math

import numpy as np

from .tabletop_batch_geometry import rotated_points
from .tabletop_deformable_wrist_projection import CLEARANCE_M, DT
from .tabletop_surface_contract import ACTION_CONTRACT
from .tabletop_training_reward import CONTROL_HZ
from .tabletop_wire_continuous_clearance import WireClearance

MODE = "quiet_wire_then_left_opposition_transfer"
VERSION = "dh116-wire-settled-left-opposition-v1"
HOLD_WINDOWS = 2
THUMB = "left.finger14_link"
INDEX = "left.finger23_link"


def ready_after_thumb(transfer_sdk, row):
    return (type(transfer_sdk) is int and row["physics_step"] >= transfer_sdk
            and row["substep"] == 7 and row["guard_failure"] is None)


def index_closure(increment):
    if not math.isfinite(increment) or not 0. < increment <= .12:
        raise ValueError("screened positive left index increment required")
    rate = ACTION_CONTRACT["joint_rate_rad_s"]
    windows = math.ceil(increment * CONTROL_HZ / rate)
    action = np.zeros(ACTION_CONTRACT["size"])
    action[2] = increment * CONTROL_HZ / (rate * windows)
    return action, windows


def left_loads(substeps):
    """Separate same-substep opposing reactions from the original force thresholds."""
    if len(substeps) != 12:
        raise ValueError("complete left-hand solver substeps required")
    counts = dict(thumb_loaded_substeps=0, index_loaded_substeps=0,
                  opposed_substeps=0, loaded_opposed_substeps=0)
    thumb_loads, index_loads, cosines = [], [], []
    for sample in substeps:
        thumb, index = (sample["per_pad"][name] for name in ("thumb", "index"))
        a, b = thumb["normal_load_n"], index["normal_load_n"]
        if not all(math.isfinite(x) and x >= 0. for x in (a, b)):
            raise ValueError("finite nonnegative actual pad loads required")
        thumb_loads.append(a)
        index_loads.append(b)
        counts["thumb_loaded_substeps"] += int(a > 0.)
        counts["index_loaded_substeps"] += int(b > 0.)
        n, m = (pad["normal_on_rod_world"] for pad in (thumb, index))
        if a > 0. and b > 0. and n is not None and m is not None:
            dot = float(np.dot(n, m))
            if not math.isfinite(dot):
                raise ValueError("finite measured pad normals required")
            cosines.append(dot)
            counts["opposed_substeps"] += int(dot <= -.25)
            counts["loaded_opposed_substeps"] += int(dot <= -.25 and min(a, b) > .05 and a+b > .3)
    return dict(**counts, thumb_peak_n=max(thumb_loads), index_peak_n=max(index_loads),
                opposition_cos_min=min(cosines, default=None))


class IndexReaction:
    """Bind the first index load to its own immediately following world force."""

    def __init__(self, *, body_id):
        self.body_id = body_id
        self.first_contact_sdk = self.first_transfer_sdk = self.transfer_force_n = None

    def deliver_after_stop(self, phase):
        """The reserved hold window may deliver a late index without moving targets."""
        return phase == "hold" and self.first_contact_sdk is not None and self.first_transfer_sdk is None

    def observe(self, sdk, evidence, issued):
        # The owner checks exact equality against the preceding solver wrench first.
        if self.first_contact_sdk is not None and self.first_transfer_sdk is None:
            if sdk != self.first_contact_sdk + 1 or self.body_id not in issued["body_ids"]:
                raise ValueError("next SDK must apply the original index reaction route")
            slot = issued["body_ids"].index(self.body_id)
            force = float(np.linalg.norm(issued["forces"][0, slot]))
            if not math.isfinite(force) or force <= 0.:
                raise ValueError("expected positive force magnitude on the index reaction route")
            self.transfer_force_n = force
            self.first_transfer_sdk = sdk
        if self.first_contact_sdk is None and evidence["index_loaded_substeps"]:
            self.first_contact_sdk = sdk


_PIVOT_FRACTIONS = np.linspace(0., 1., 5)


def close_action_joint_names(issued, *, allow_thumb_reseat):
    """Return the targets a recorded close increment was allowed to update."""
    from .tabletop_palm_mount import JOINT_NAMES

    action = np.asarray(issued)
    if action[2]:
        return "finger21", "finger22"
    if allow_thumb_reseat and np.any(action[12:15]):
        return JOINT_NAMES
    return ()


def choose_thumb_reseat(gap_after_translation, preview):
    """Select a bounded wrist-only increment from the current thumb SDF."""
    result = dict(action=np.zeros(ACTION_CONTRACT["size"]), reason="thumb_reseat_hold")
    zero = np.zeros(3)
    gap = gap_after_translation(zero)
    if gap is None or not math.isfinite(gap) or gap <= -2e-6:
        return result
    epsilon = 1e-6
    basis = np.eye(3) * epsilon
    neighbors = [gap_after_translation(direction * step) for direction in basis for step in (-1., 1.)]
    if any(value is None or not math.isfinite(value) for value in neighbors):
        return result
    gradient = np.diff(np.reshape(neighbors, (3, 2)), axis=1).ravel() / (2 * epsilon)
    magnitude = np.linalg.norm(gradient)
    if not math.isfinite(magnitude) or magnitude <= 1e-6:
        return result
    travel = min(10e-6, max(2e-6, (gap + 2e-6) / magnitude))
    meter_per_sdk_step = ACTION_CONTRACT["translation_rate_m_s"] * DT
    for length in (travel, travel / 2):
        delta = -gradient * (length / magnitude)
        endpoint = gap_after_translation(delta)
        if endpoint is None or not math.isfinite(endpoint) or not -.00015 <= endpoint < gap:
            continue
        action = np.zeros(ACTION_CONTRACT["size"])
        action[12:15] = delta / meter_per_sdk_step
        if np.linalg.norm(action[12:15]) <= 1. and preview(action)["passed"]:
            return dict(action=action, reason="thumb_wrist_reseat", gap_m=float(gap),
                        predicted_gap_m=float(endpoint))
    return result


def compose_projected_pivot(action, *, root_seed_w, measured_base_w, clearance_m, palm_offsets_m, angles_rad,
                            allow_translation_only=False):
    """Return world rotation increments and translations if the original projector keeps the move."""
    from scipy.spatial.transform import Rotation
    import torch

    from .tabletop_command_kernels import wrist_requested_targets
    from .tabletop_deformable_wrist_projection import project_policy_wrist_targets
    from .tabletop_free_wrist_grasp import compose_rotation

    action = np.asarray(action, dtype=float)
    if (action.shape != (24,) or not np.isfinite(action).all()
            or np.any(action[np.r_[:12, 15:18, 21:24]])
            or not (np.any(action[18:21]) or allow_translation_only and np.any(action[12:15]))):
        return None
    command = torch.as_tensor(action)[None]
    root, base, clearance, palms, angles = (
        torch.as_tensor(v, dtype=torch.float64) for v in
        (root_seed_w, measured_base_w, clearance_m, palm_offsets_m, angles_rad))
    rates = dict(wrist_rate_m_s=ACTION_CONTRACT["translation_rate_m_s"],
                 rotation_rate_rad_s=ACTION_CONTRACT["rotation_rate_rad_s"])
    requested_p, requested_a = wrist_requested_targets(palms, angles, command, rates, DT)
    projected = project_policy_wrist_targets(
        root_seed_w=root, measured_base_w=base, common_lift_offset_m=root.new_zeros(1),
        measured_clearance_m=clearance, palm_offsets_m=palms, requested_palm_offsets_m=requested_p,
        angles_rad=angles, requested_angles_rad=requested_a,
        allowed=torch.ones(1, dtype=torch.bool), live=torch.ones(1, dtype=torch.bool))
    delta = command[:, 12:18].reshape(1, 2, 3) * (DT * rates["wrist_rate_m_s"])
    angular_delta = command[:, 18:24].reshape(1, 2, 3) * (DT * rates["rotation_rate_rad_s"])
    if (projected["projection_reasons"].any()
            or not torch.allclose(projected["palm_offsets_m"] - palms, delta, rtol=0., atol=1e-12)
            or not torch.allclose(projected["projected_angles"] - angles, angular_delta, rtol=0., atol=1e-12)):
        return None
    fractions = torch.tensor(_PIVOT_FRACTIONS, dtype=torch.float64)[:, None]
    # C(t) C(0)^-1 acts on the measured world hull about the measured palm.
    # A fixed local servo residual cancels in this world increment.
    yaw = angles[0, 0] + fractions * angular_delta[0, 0]
    quaternions = compose_rotation(root[0, 0, 3:].expand(len(fractions), 4), yaw).numpy()
    rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])
    return rotations * rotations[0].inv(), fractions.numpy() * delta[0, 0].numpy()


def guard_prediction(*, action, frames, before_frames, nodes, before_nodes,
                     shapes, selected, left_palm, finger_travel, boxes, table_top, radius,
                     composed_pivot=None):
    """Check a one-step proposed-motion envelope using current and last readbacks."""
    from scripts.certify_dh116_tripod_distal_route import support_clearance_lower

    reserve = float(np.linalg.norm(nodes-before_nodes, axis=-1).max())
    if not math.isfinite(reserve):
        raise ValueError("finite measured rod motion required")
    envelope = WireClearance(nodes, radius_m=radius)
    translation = float(np.linalg.norm(action[12:15])) * ACTION_CONTRACT["translation_rate_m_s"] * DT
    rotation = float(np.abs(action[18:21]).sum()) * ACTION_CONTRACT["rotation_rate_rad_s"] * DT
    minimum = np.inf
    composed_count = 0
    sweep = None
    for i, (side, body, vertices) in enumerate(shapes):
        live = rotated_points(frames[side, body], vertices)
        older = rotated_points(before_frames[side, body], vertices)
        observed = float(np.linalg.norm(live-older, axis=1).max())
        requested = 0.
        if side == 0:
            lever = float(np.linalg.norm(live - left_palm, axis=1).max())
            requested = translation + rotation*lever + finger_travel.get(i, 0.)
        # This is an empirical preview allowance. Live post-step guards remain active.
        hand_motion = observed + requested
        is_selected = (side, body) in selected
        proof = envelope.certify(live, selected=is_selected, reserve_m=reserve+hand_motion)
        margin = proof["lower_m"] - proof["threshold_m"] - reserve - hand_motion
        table = float(live[:, 2].min() - table_top - CLEARANCE_M)
        support = min(support_clearance_lower(live, lo, hi) for lo, hi in boxes)
        hard_space = table > hand_motion and support > hand_motion
        passed = proof["passed"]
        if (not passed and hard_space and side == 0 and is_selected and composed_pivot is not None
                and not finger_travel and not np.any(action[np.r_[:12, 15:18, 21:24]])):
            if sweep is None:
                sweep = composed_pivot()
            if sweep is not None:
                rotations, shifts = sweep
                # The closest of five equal-spaced samples is at most 1/8 step away.
                sample_reserve = reserve + observed + requested / 8.
                checks = [envelope.certify(left_palm + shift + rot.apply(live-left_palm),
                                            selected=True, reserve_m=sample_reserve)
                          for rot, shift in zip(rotations, shifts, strict=True)]
                margin = min(q["lower_m"] - q["threshold_m"] - sample_reserve for q in checks)
                passed = all(q["passed"] for q in checks)
                composed_count += int(passed)
        minimum = min(minimum, margin)
        if not passed or not hard_space:
            return dict(passed=False, shape_index=i, side=side, body=body, rod_motion_m=reserve,
                        observed_hand_motion_m=observed, requested_hand_motion_m=requested,
                        minimum_rod_margin_m=minimum, table_margin_m=table, support_margin_m=support)
    result = dict(passed=True, rod_motion_m=reserve, minimum_rod_margin_m=float(minimum))
    if composed_count:
        result["composed_hull_count"] = composed_count
    return result
