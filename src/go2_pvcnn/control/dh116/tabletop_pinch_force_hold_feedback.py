"""Bound fixed-wrist Pinch preload using complete measured reaction windows."""

import math

SDK_STEPS = 8
SOLVER_SUBSTEPS = 12
SIDES = ("left", "right")
IDENTITY = ("run_uuid", "episode_uid", "reset_id", "rod_uid")
MAX_INDEX_PRELOAD_RAD = .0001
INDEX_INCREMENT_RAD = MAX_INDEX_PRELOAD_RAD / 2


class FixedWristPinchForceHold:
    """Permit at most two certified increments of left finger21; no lift output."""

    def __init__(self, first_boundary):
        if type(first_boundary) is not int or first_boundary < SDK_STEPS+2 or (first_boundary-2) % SDK_STEPS:
            raise ValueError("an explicit SDK control boundary is required for force feedback")
        self._next_boundary = first_boundary
        self._identity = None
        self._increments = 0

    def decide(self, rows, *, certify):
        end = self._next_boundary
        if len(rows) != SDK_STEPS or not callable(certify):
            raise ValueError("a complete preceding SDK window and explicit geometry proof are required")
        short_left = insufficient_right = 0
        has_opposition = has_pad_load = True
        previous_identity = self._identity
        for i, row in enumerate(rows):
            step = end - SDK_STEPS + i + 1
            rod, data = row["rod"], row["stage_reaction"]
            identity = tuple(rod[key] for key in IDENTITY)
            if (type(identity[2]) is not int or identity[2] < 1
                    or any(not isinstance(identity[j], str) or not identity[j] for j in (0, 1, 3))
                    or (previous_identity is not None and identity != previous_identity)
                    or (row["physics_step"], row["policy_step"], row["substep"]) != (step, (end-3)//SDK_STEPS, i)
                    or row["guard_failure"] is not None or rod["contact_valid"] is not True
                    or (rod["physics_step"], rod["rod_tick"]) != (step, (step-2)*SOLVER_SUBSTEPS)
                    or (data["physics_step"], data["rod_tick"]) != (step, (step-2)*SOLVER_SUBSTEPS)
                    or data["evidence_scope"] != "measured_simulated_reaction"
                    or len(data["per_substep"]) != SOLVER_SUBSTEPS):
                raise ValueError("stale, incomplete or foreign reaction at the fixed-wrist boundary")
            previous_identity = identity
            for j, sample in enumerate(data["per_substep"]):
                if sample["substep"] != j:
                    raise ValueError("ordered same-boundary solver substeps required")
                totals = []
                for side in SIDES:
                    hold = sample[side]
                    loads = (hold["thumb_load_n"], hold["index_load_n"])
                    if any(not math.isfinite(value) or value < 0. for value in loads):
                        raise ValueError("finite physical pad loads required")
                    if hold["signed_opposition"] not in (True, False):
                        raise ValueError("explicit signed opposition measurement required")
                    has_pad_load &= min(loads) > .05
                    cosine = hold["opposition_cos"]
                    if cosine is not None and (not math.isfinite(cosine) or not -1. <= cosine <= 1.):
                        raise ValueError("invalid measured contact opposition")
                    has_opposition &= hold["signed_opposition"] is True and cosine is not None and cosine <= -.25
                    totals.append(sum(loads))
                short_left += int(totals[0] <= .3)
                insufficient_right += int(totals[1] <= .3)
        self._identity = previous_identity
        self._next_boundary += SDK_STEPS
        window_size = SDK_STEPS*SOLVER_SUBSTEPS
        delta = 0.
        if not has_opposition:
            reason = "opposing_contact_not_continuous"
        elif not has_pad_load:
            reason = "required_pad_load_missing"
        elif insufficient_right >= window_size // 2:
            reason = "right_required_load_missing"
        elif short_left <= window_size // 2:
            reason = "left_load_adequate"
        elif self._increments >= 2:
            reason = "preload_limit_reached"
        elif certify(INDEX_INCREMENT_RAD, rows[-1]) is not True:
            reason = "collision_proof_rejected"
        else:
            delta = INDEX_INCREMENT_RAD
            self._increments += 1
            reason = "left_required_load_deficit"
        return dict(observed_sdk_step=end, proposed_control_step=(end-2)//SDK_STEPS,
                    identity=dict(zip(IDENTITY,self._identity)), solver_substeps=window_size,
                    left_insufficient_substeps=short_left, right_insufficient_substeps=insufficient_right,
                    reasons=[reason], left_index_delta_rad=delta,
                    total_index_delta_rad=self._increments*INDEX_INCREMENT_RAD)
