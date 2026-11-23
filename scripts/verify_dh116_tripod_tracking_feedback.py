"""Tripod fixed-wrist force corrections bounded by measured actuator tracking."""

import numpy as np

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import VERSION as READBACK_VERSION
from go2_pvcnn.control.dh116.tabletop_joint_target_readback import BACKEND_SOURCE
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import IDENTITY_FIELDS, measure_supported_distal_contact
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from scripts.verify_dh116_supported_force_seating import BILATERAL_PADS, BilateralLoadFeedback


WINDOW_RATE_RAD = ACTION_CONTRACT["joint_rate_rad_s"] / 60
MEASURED_SOURCE = "physx_backend_get_dof_positions/get_dof_velocities"


def _target_leads(physical, sdk):
    """Read signed backend target minus measured q for each driven pad slot."""
    try:
        rod = physical["rod"]
        readback = physical["action_readback"]
        post = readback["post"]
        (clock,) = post["identity"]
        if (physical["guard_failure"] is not None or physical["physics_step"] != sdk
                or readback["version"] != READBACK_VERSION or readback["env_ids"] != [clock["env_id"]]
                or sdk != clock["physics_step"] or physical["substep"] != 7
                or physical["substep"] != clock["substep"] or physical["policy_step"] != clock["policy_step"]
                or clock["clock_valid"] is not True or rod["clock_valid"] is not True
                or any(clock[key] != rod[key] for key in (*IDENTITY_FIELDS, "timestamp_s", "solver_time_s"))):
            raise ValueError("tracking requires the completed same-clock control boundary")

        leads = {}
        for side in ("left", "right"):
            hand = post["hands"][side]
            (path,) = hand["prim_paths"]
            names = hand["joint_names"]
            active = [names.index(name) for name in ACTIVE]
            q, target = (np.asarray(hand[key]) for key in ("q", "backend_targets"))
            if (f"{side}Hand" not in path.split("/") or len(set(names)) != len(names)
                    or hand["active_indices"] != active or any(hand["units_by_name"][name] != "rad" for name in ACTIVE)
                    or hand["measured_source"] != MEASURED_SOURCE or hand["backend_source"] != BACKEND_SOURCE
                    or any(a.shape != (1, len(names)) or a.dtype.kind != "f" or not np.isfinite(a).all()
                           for a in (q, target))):
                raise ValueError("tracking requires actual named radian PhysX q and backend targets")
            for slot, (hand_side, _) in BILATERAL_PADS.items():
                if side == hand_side:
                    i = active[slot % len(ACTIVE)]
                    leads[slot] = float(target[0, i]) - float(q[0, i])
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ValueError("missing actual tracking source") from exc
    return leads


class BilateralTrackingLoadFeedback(BilateralLoadFeedback):
    """Use the original force and travel guards with one nominal dose of lead."""

    def decide(self, physical, *, sdk):
        tracking = _target_leads(physical, sdk)
        evidence = measure_supported_distal_contact(physical["rod"], grasp="tripod")
        issued, report = self._decide(evidence, sdk=sdk, tracking=tracking)
        report["target_lead_limit_rad"] = self.action_step * WINDOW_RATE_RAD
        for side, hand in report["hands"].items():
            hand["signed_backend_target_lead_rad"] = {
                pad: tracking[slot] for slot, (hand_side, pad) in BILATERAL_PADS.items() if side == hand_side
            }
        return issued, report

    def _select_unloaded(self, side, minimums, minimum_total, *, tracking):
        deficient = [slot for slot, (hand_side, pad) in BILATERAL_PADS.items()
                     if hand_side == side and minimums[pad] < self.pad_target_n]
        if not deficient:
            # Keep the original choice of opposing side when only total load is low.
            slot, _, reason = super()._select_unloaded(side, minimums, minimum_total, tracking=tracking)
            if slot is None:
                return None, 0., reason
            deficient = [slot]
        limit = self.action_step * WINDOW_RATE_RAD
        by_weakness = sorted(deficient, key=lambda slot: minimums[BILATERAL_PADS[slot][1]])
        for slot in by_weakness:
            travel = min(limit, self._limits[slot] - self.travel_rad[slot], limit - tracking[slot])
            if travel > 1e-12:
                return slot, travel / WINDOW_RATE_RAD, "tracking_eligible_acquire"
        return None, 0., "deficient_pad_tracking_or_travel_exhausted"
