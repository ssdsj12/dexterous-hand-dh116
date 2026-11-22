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


def guard_prediction(*, action, frames, before_frames, nodes, before_nodes,
                     shapes, selected, left_palm, finger_travel, boxes, table_top, radius):
    """Check a one-step proposed-motion envelope using current and last readbacks."""
    from scripts.certify_dh116_tripod_distal_route import support_clearance_lower

    reserve = float(np.linalg.norm(nodes-before_nodes, axis=-1).max())
    if not math.isfinite(reserve):
        raise ValueError("finite measured rod motion required")
    envelope = WireClearance(nodes, radius_m=radius)
    translation = float(np.linalg.norm(action[12:15])) * ACTION_CONTRACT["translation_rate_m_s"] * DT
    rotation = float(np.abs(action[18:21]).sum()) * ACTION_CONTRACT["rotation_rate_rad_s"] * DT
    minimum = np.inf
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
        proof = envelope.certify(live, selected=(side, body) in selected, reserve_m=reserve+hand_motion)
        minimum = min(minimum, proof["lower_m"] - proof["threshold_m"] - reserve - hand_motion)
        table = float(live[:, 2].min() - table_top - CLEARANCE_M)
        support = min(support_clearance_lower(live, lo, hi) for lo, hi in boxes)
        if not proof["passed"] or table <= hand_motion or support <= hand_motion:
            return dict(passed=False, shape_index=i, side=side, body=body, rod_motion_m=reserve,
                        observed_hand_motion_m=observed, requested_hand_motion_m=requested,
                        minimum_rod_margin_m=minimum, table_margin_m=table, support_margin_m=support)
    return dict(passed=True, rod_motion_m=reserve, minimum_rod_margin_m=float(minimum))
