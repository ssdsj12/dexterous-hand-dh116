"""Small, measured seating diagnostics on existing supported-contact replays."""

import json
from pathlib import Path

import numpy as np

from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from go2_pvcnn.control.dh116.tabletop_power_contact_evidence import MIN_HAND_LOAD_N
from scripts import screen_dh116_tripod_measured_precontact as measured

TRIPOD_REQUEST = measured.ROOT / "configs/dh116_tasks/flexible_tripod_right_force_seating_v1.json"
TRIPOD_CONTRACT = measured.ROOT / "docs/contracts/dh116_teacher_tripod_measured_left_retarget_native_v2.json"
TRIPOD_SDK = 1296
RIGHT_PADS = {8: "index", 9: "middle"}
LEFT_PADS = {2: "index"}
PADS_BY_HAND = {"left": LEFT_PADS, "right": RIGHT_PADS}
DOSE_THRESHOLD_N = .085
TRIPOD_TRACKING_FEEDBACK_VARIANT = "bilateral_tracking_load_feedback_v1"


def validate_seating_schedule(schedule, pad_by_slot, *, max_travel_rad):
    schedule = np.asarray(schedule, dtype=float)
    if (not pad_by_slot or not any(pad_by_slot.items() <= expected.items() for expected in PADS_BY_HAND.values())
            or schedule.ndim != 2 or schedule.shape[1] != 24 or not len(schedule)
            or not np.isfinite(schedule).all() or (schedule < 0.).any() or (schedule > 1.).any()
            or any(np.any(schedule[:, slot]) for slot in range(24) if slot not in pad_by_slot)):
        raise ValueError("screened positive finger-only seating commands are required")
    travel = schedule.sum(axis=0) * (ACTION_CONTRACT["joint_rate_rad_s"] / 60)
    if not 0. < max_travel_rad <= .003 or np.any(travel > max_travel_rad + 1e-12):
        raise ValueError("seating exceeds the screened finger travel budget")


def select_seating_command(planned, evidence, pad_by_slot, *, side="right"):
    if (side not in PADS_BY_HAND or not pad_by_slot
            or not pad_by_slot.items() <= PADS_BY_HAND[side].items()):
        raise ValueError("a seating action must use the measured hand identity")
    hand = evidence["per_hand"][side]
    rows = hand["per_substep"]
    if len(rows) != 12:
        raise ValueError("complete 12-substep current solver load required")
    combined = np.asarray([row["total_normal_load_n"] for row in rows], dtype=float)
    if not np.isfinite(combined).all() or (combined < 0.).any():
        raise ValueError("seating requires a valid total normal force")
    total_ready = bool((combined > MIN_HAND_LOAD_N).all())
    issued = np.asarray(planned, dtype=float).copy()
    minimums = {}
    for slot, pad in pad_by_slot.items():
        forces = np.asarray([row["per_pad"][pad]["normal_load_n"] for row in rows], dtype=float)
        if not np.isfinite(forces).all() or np.any(forces < 0.):
            raise ValueError("seating requires measured per-pad normal force")
        minimums[pad] = float(forces.min())
        if hand["loaded"] or (total_ready and minimums[pad] >= DOSE_THRESHOLD_N):
            issued[slot] = 0.
    return issued, minimums


def load_tripod_seating(receipt, actions, *, start_sdk):
    expected = json.loads(TRIPOD_CONTRACT.read_bytes())
    if expected["force_followup_boundary"]["sdk"] != TRIPOD_SDK:
        raise ValueError("the measured Tripod seating checkpoint is required")
    rows, _, paths = measured.archive(receipt, contract=TRIPOD_CONTRACT, sdk=expected["actual"]["end_sdk"])
    if rows[-1]["physics_step"] <= TRIPOD_SDK:
        raise ValueError("completed retarget must cover the seating checkpoint")
    outer = measured.sources.checked_json(receipt, expected["actual"]["outer_sha256"])
    name = f"tripod_control_{TRIPOD_SDK}.json"
    checkpoint = receipt.parent / "native" / name
    row = measured.sources.checked_json(checkpoint, outer["artifact_sha256"][name])
    if (row["physics_step"] != TRIPOD_SDK or row["sdk_hand_geometry"]["physics_step"] != TRIPOD_SDK
            or row["rod"]["physics_step"] != TRIPOD_SDK or row["rod"]["rod_tick"] != (TRIPOD_SDK - 2) * 12
            or row["guard_failure"] is not None):
        raise ValueError("seating anchor must bind actual fingertip and rod data to the complete SDK step")

    request = json.loads(TRIPOD_REQUEST.read_bytes())
    if (request["version"] != "dh116-tripod-right-force-seating-v1" or request["control_boundary_sdk"] != TRIPOD_SDK
            or request["dose_windows"] != 10 or request["settle_windows"] != 18):
        raise ValueError("expected the frozen Tripod force seating contract")
    command = np.zeros(24)
    command[[8, 9]] = [request["planned_action_by_joint"][name] for name in ("finger21", "finger31")]
    appended = np.vstack((np.repeat(command[None], request["dose_windows"], axis=0),
                          np.zeros((request["settle_windows"], 24))))
    validate_seating_schedule(appended, RIGHT_PADS, max_travel_rad=request["maximum_finger_travel_rad"])
    count = (TRIPOD_SDK - start_sdk) // 8
    if not 0 < count < len(actions) or start_sdk + 8 * count != TRIPOD_SDK:
        raise ValueError("seating requires the intact original replay through SDK 1296")
    return np.vstack((actions[:count], appended)), dict(sdk_step=TRIPOD_SDK, archive=row,
        pad_by_slot={str(slot): pad for slot, pad in RIGHT_PADS.items()}), (
        *paths, Path(__file__), TRIPOD_REQUEST, checkpoint)


def load_pinch_seating(receipt):
    contract = measured.ROOT / "docs/contracts/dh116_teacher_pinch_right_first_contact_native_v1.json"
    request_path = measured.ROOT / "configs/dh116_tasks/flexible_pinch_right_force_seating_v1.json"
    request = json.loads(request_path.read_bytes())
    sdk = 656
    if (request["version"] != "dh116-pinch-right-force-seating-v1" or request["control_boundary_sdk"] != sdk
            or request["dose_windows"] != 8 or request["settle_windows"] != 16
            or not 0. < request["maximum_finger_travel_rad"] <= .0012):
        raise ValueError("expected frozen right Pinch index seating contract")
    rows, _, paths = measured.archive(receipt, contract=contract, sdk=sdk, task="pinch")
    command = np.zeros(24)
    command[8] = request["planned_index_action"]
    pads = {8: "index"}
    appended = np.vstack((np.repeat(command[None], request["dose_windows"], axis=0),
                          np.zeros((request["settle_windows"], 24))))
    validate_seating_schedule(appended, pads, max_travel_rad=request["maximum_finger_travel_rad"])
    return appended, dict(sdk_step=sdk, archive=rows[-1], pad_by_slot={"8": "index"}), (
        *paths, Path(__file__), request_path)


def load_tripod_left_stabilization(receipt, actions, *, start_sdk):
    request_path = measured.ROOT / "configs/dh116_tasks/flexible_tripod_left_index_stabilize_v1.json"
    contract = measured.ROOT / "docs/contracts/dh116_teacher_tripod_force_guided_right_seating_native_v1.json"
    req = json.loads(request_path.read_bytes())
    sdk = 1480
    if (req["version"] != "dh116-tripod-left-index-stabilize-v1" or req["source_sdk"] != sdk
            or req["dose_windows"] != 4 or req["hold_windows"] != 24
            or not 0. < req["max_travel_rad"] <= .0006):
        raise ValueError("the measured left index stabilization recipe is required")
    info = json.loads(contract.read_bytes())["actual"]
    _, _, bound = measured.archive(receipt, contract=contract, sdk=info["end_sdk"])
    outer = measured.sources.checked_json(receipt, info["outer_sha256"])
    path = receipt.parent / "native" / f"tripod_control_{sdk}.json"
    row = measured.sources.checked_json(path, outer["artifact_sha256"][path.name])
    if (row["physics_step"] != sdk or row["rod"]["physics_step"] != sdk
            or row["sdk_hand_geometry"]["physics_step"] != sdk or row["rod"]["rod_tick"] != (sdk-2)*12
            or row["guard_failure"] is not None):
        raise ValueError("left stabilization must start at the complete same-clock measured boundary")
    count = (sdk - start_sdk) // 8
    if not 0 < count < len(actions) or start_sdk + 8 * count != sdk:
        raise ValueError("left stabilization requires replay through all prior right doses")
    planned = np.zeros((req["dose_windows"] + req["hold_windows"], 24))
    planned[:req["dose_windows"], 2] = req["normalized_index_action"]
    pads = LEFT_PADS
    validate_seating_schedule(planned, pads, max_travel_rad=req["max_travel_rad"])
    return np.vstack((actions[:count], planned)), dict(sdk_step=sdk, archive=row, side="left",
        pad_by_slot={"2":"index"}), (*bound, path, request_path, Path(__file__))


BILATERAL_PADS = {
    1: ("left", "thumb"), 2: ("left", "index"), 3: ("left", "middle"),
    7: ("right", "thumb"), 8: ("right", "index"), 9: ("right", "middle"),
}


class BilateralLoadFeedback:
    """Opt-in fixed-wrist acquisition candidate; never a grasp-success classifier.

    Each 60 Hz decision uses the previous completed SDK sample's 12 solver
    substeps. At most one deficient finger per hand moves. Travel is the
    cumulative requested target increment, not measured articulation travel.
    """

    def __init__(self, *, start_sdk, windows, action_step, pad_target_n, max_travel_rad):
        if (not isinstance(max_travel_rad, dict)
                or any(str(key) not in {str(slot) for slot in BILATERAL_PADS} for key in max_travel_rad)):
            raise ValueError("canonical bilateral finger slots required")
        normalized_limits = {int(slot): value for slot, value in max_travel_rad.items()}
        if len(normalized_limits) != len(max_travel_rad):
            raise ValueError("duplicate bilateral finger slots")
        max_travel_rad = normalized_limits
        if (type(start_sdk) is not int or start_sdk < 2 or type(windows) is not int
                or not 1 <= windows <= 64 or type(action_step) not in (int, float)
                or type(pad_target_n) not in (int, float) or not np.isfinite(action_step)
                or not 0. < action_step <= .03 or not np.isfinite(pad_target_n)
                or not .05 < pad_target_n <= DOSE_THRESHOLD_N
                or set(max_travel_rad) != set(BILATERAL_PADS)):
            raise ValueError("explicit finite bilateral feedback configuration required")
        limits = np.zeros(24)
        for slot, value in max_travel_rad.items():
            if type(value) not in (int, float) or not np.isfinite(value) or not 0. <= value <= .003:
                raise ValueError("bilateral per-finger travel must remain bounded")
            limits[slot] = value
        self._limits = limits
        self.travel_rad = np.zeros(24)
        self.start_sdk = self.next_sdk = start_sdk
        self.stop_sdk = start_sdk + 8 * windows
        self.action_step = float(action_step)
        self.pad_target_n = float(pad_target_n)
        self._identity = None

    def decide(self, evidence, *, sdk):
        return self._decide(evidence, sdk=sdk)

    def _decide(self, evidence, *, sdk, tracking=None):
        if type(sdk) is not int or sdk != self.next_sdk or sdk >= self.stop_sdk:
            raise ValueError("bilateral feedback requires the next unused control window")
        identity = evidence.get("identity", {})
        if (evidence.get("grasp") != "Tripod" or evidence.get("solver_substeps") != 12
                or identity.get("physics_step") != sdk or identity.get("rod_tick") != (sdk-2)*12
                or evidence.get("wrong_side_contacts") != [] or evidence.get("unexpected_hand_contacts") != []):
            raise ValueError("bilateral feedback requires complete same-clock correct-side Tripod contact")
        keys = ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid")
        bound = {key: identity.get(key) for key in keys}
        if (any(not isinstance(bound[key], str) or not bound[key] for key in ("run_uuid", "episode_uid", "rod_uid"))
                or any(type(bound[key]) is not int or bound[key] < 0 for key in ("env_id", "reset_id"))
                or self._identity is not None and bound != self._identity):
            raise ValueError("bilateral feedback cannot cross run, episode or Reset identities")

        issued = np.zeros(24)
        reports = {}
        # Validate both hands before changing state or issuing any command.
        for side in ("left", "right"):
            try:
                hand = evidence["per_hand"][side]
                rows = hand["per_substep"]
                pads = ("thumb", "index", "middle")
                if (hand["required_pads"] != list(pads) or len(rows) != 12
                        or [row["substep"] for row in rows] != list(range(12))
                        or type(hand["loaded"]) is not bool
                        or any(type(row["loaded"]) is not bool for row in rows)
                        or hand["loaded"] != all(row["loaded"] for row in rows)):
                    raise ValueError("ordered complete bilateral per-pad solver evidence required")
                forces = np.array([[row["per_pad"][pad]["normal_load_n"] for pad in pads] for row in rows], dtype=float)
                total = np.array([row["total_normal_load_n"] for row in rows], dtype=float)
                if (not np.isfinite(forces).all() or (forces < 0.).any()
                        or not np.isfinite(total).all() or (total < 0.).any()
                        or not np.allclose(total, forces.sum(axis=1), atol=1e-10, rtol=1e-9)):
                    raise ValueError("finite actual per-pad and total force required")
                for i, row in enumerate(rows):
                    if row["loaded"] and (not (forces[i] > .05).all() or total[i] <= MIN_HAND_LOAD_N
                            or row.get("effective") is not True
                            or row.get("all_required_pads_on_opposing_faces") is not True
                            or row.get("opposition_cos") is None
                            or not np.isfinite(row["opposition_cos"]) or not -1. <= row["opposition_cos"] <= -.25):
                        raise ValueError("loaded flag contradicts measured pad loads or opposition")
            except (KeyError, TypeError, IndexError) as exc:
                raise ValueError("missing bilateral solver force source") from exc
            minimums = dict(zip(pads, forces.min(axis=0).tolist(), strict=True))
            report = dict(minimum_pad_force_n=minimums, minimum_total_load_n=float(total.min()),
                          loaded=hand["loaded"], selected_pad=None)
            if hand["loaded"]:
                report["reason"] = "loaded_hold"
            else:
                slot, action, reason = self._select_unloaded(
                    side, minimums, float(total.min()), tracking=tracking)
                report["reason"] = reason
                if slot is not None:
                    issued[slot] = action
                    report.update(selected_pad=BILATERAL_PADS[slot][1], selected_slot=slot)
            reports[side] = report
        self.travel_rad += np.abs(issued) * (ACTION_CONTRACT["joint_rate_rad_s"] / 60)
        self.next_sdk += 8
        self._identity = bound
        return issued, dict(source_sdk_step=sdk, identity=bound, hands=reports,
                            requested_target_travel_rad=self.travel_rad.tolist(),
                            diagnostic_only=True, grasp_qualified=False, teacher_admission=False)

    def _select_unloaded(self, side, minimums, minimum_total, *, tracking):
        pads = ("thumb", "index", "middle")
        deficits = {pad: self.pad_target_n - minimums[pad] for pad in pads}
        if max(deficits.values()) > 0.:
            weakest = max(pads, key=lambda pad: deficits[pad])
        elif minimum_total <= MIN_HAND_LOAD_N:
            # Magnitude-based diagnostic heuristic, not force-vector equilibrium proof.
            weakest = ("thumb" if minimums["thumb"] < minimums["index"] + minimums["middle"]
                       else min(("index", "middle"), key=lambda pad: minimums[pad]))
        else:
            return None, 0., "opposition_or_geometry_requires_retarget"
        tied = [slot for slot, (hand_side, pad) in BILATERAL_PADS.items()
                if hand_side == side and (pad == "thumb") == (weakest == "thumb")
                and abs(minimums[pad] - minimums[weakest]) <= 1e-12]
        slot = next((slot for slot in tied if self._limits[slot] - self.travel_rad[slot] > 1e-12), tied[0])
        remaining = max(0., self._limits[slot] - self.travel_rad[slot])
        action = min(self.action_step, remaining * 60 / ACTION_CONTRACT["joint_rate_rad_s"])
        if action <= 1e-12:
            return slot, 0., "weakest_pad_travel_exhausted"
        return slot, action, "weakest_pad_acquire"


def load_tripod_bilateral_feedback(receipt, actions, *, start_sdk, request_path):
    """Keep the measured replay, replacing serial doses with live bilateral load feedback."""
    request = json.loads(Path(request_path).read_bytes())
    fields = {"version", "control_boundary_sdk", "control_windows", "normalized_action_step", "pad_target_n",
              "maximum_target_travel_rad", "scope", "thumb_dose_status"}
    version = request.get("version")
    if (set(request) != fields or version not in ("dh116-tripod-bilateral-load-feedback-v1",
                                                  "dh116-tripod-bilateral-tracking-load-feedback-v1")
            or request.get("control_boundary_sdk") != TRIPOD_SDK):
        raise ValueError("an independent bilateral force feedback request at SDK 1296 is required")
    options = dict(start_sdk=TRIPOD_SDK, windows=request["control_windows"],
                   action_step=request["normalized_action_step"], pad_target_n=request["pad_target_n"],
                   max_travel_rad=dict(request["maximum_target_travel_rad"]))
    variant = "bilateral_load_feedback_v1"
    closure = ()
    if version == "dh116-tripod-bilateral-tracking-load-feedback-v1":
        from scripts import verify_dh116_tripod_tracking_feedback as tracking

        variant = TRIPOD_TRACKING_FEEDBACK_VARIANT
        closure = (Path(tracking.__file__),)
        tracking.BilateralTrackingLoadFeedback(**options)
    else:
        BilateralLoadFeedback(**options)
    prefix, stage, files = load_tripod_seating(receipt, actions, start_sdk=start_sdk)
    count = (TRIPOD_SDK-start_sdk)//8
    schedule = np.vstack((prefix[:count], np.zeros((options["windows"], 24))))
    stage.pop("pad_by_slot", None)
    stage.update(variant=variant, feedback=options,
                 stop_sdk=TRIPOD_SDK+8*options["windows"])
    return schedule, stage, (*files, *closure, Path(request_path))
