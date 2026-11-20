"""Run the frozen right Tripod sequence inside an independently owned Isaac lifecycle."""
from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from scripts.audit_dh116_pinch_initial_pose import checked_json
from scripts.audit_dh116_rod_quiet_window import IDENTITY, extract, finite
from scripts.prepare_dh116_tripod_right_tracking import ENDPOINT_SHA, REFERENCE_SHA


BUDGET = dict(sdk_steps=178, setup=2, quiet=8, motion=160, quiet_after=8,
              ppo_updates=0, instances=1)
VERSION = "dh116-right-tripod-noncontact-tracking-plan-v1"
MOVING_CHANNELS = (7, 8, 9, 17)


def _validate_action(action):
    action = finite(action, (24,))
    fixed = np.delete(action, MOVING_CHANNELS)
    if np.any(fixed) or np.any(np.abs(action) > 1.):
        raise ValueError("action outside the right-only Tripod tracking channels")
    return action


def validate_schedule(plan):
    if (plan["version"] != VERSION or plan["budget"] != BUDGET
            or plan["original_clearance_floor_m"] != .0021):
        raise ValueError("right Tripod tracking budget or original clearance differs")
    actions = finite(plan["actions24"], (22, 24))
    for action in actions:
        _validate_action(action)
    if np.any(actions[0]) or np.any(actions[-1]) or not np.all(np.any(actions[1:-1], axis=1)):
        raise ValueError("expected quiet, motion and quiet control windows")
    return actions.copy()


def validate_tracking_sample(row, expected_action, *, ordinal):
    if (not 1 <= ordinal <= 176 or row["physics_step"] != ordinal + 2
            or row["rod_tick"] != ordinal * 12 or row["policy_step"] != (ordinal - 1) // 8
            or row["substep"] != (ordinal - 1) % 8
            or not math.isclose(row["timestamp_s"], (ordinal + 2) / 480, abs_tol=2e-7, rel_tol=0.)):
        raise ValueError("right Tripod sample clock differs")
    if not np.array_equal(_validate_action(row["action24"]), _validate_action(expected_action)):
        raise ValueError("measured action differs from the right Tripod plan")
    if row["guard_failure"] is not None:
        raise ValueError("original clearance or joint guard: " + row["guard_failure"])
    if row["hand_loaded_substeps"]:
        raise ValueError("unexpected hand-rod contact in right Tripod tracking")


class _TrackingRecorder:
    def __init__(self, actions, record, quiet):
        self._actions = actions
        self._record = record
        self._samples = 0
        self._identity = None
        self._quiet = quiet
        self._failed = False

    def __call__(self, payload):
        if self._failed:
            raise RuntimeError("right Tripod tracking stopped")
        self._record(payload)
        if payload["kind"] != "physical":
            return
        # A failed boundary remains in the existing journal, even if the control
        # window cannot be settled. Its owner closes and fsyncs the prefix.
        self._failed = True
        exchange = payload["exchange"]
        if exchange["guard_failure"] is not None:
            raise ValueError("original clearance or joint guard: " + exchange["guard_failure"])
        row = extract(payload)
        ordinal = self._samples + 1
        validate_tracking_sample(row, self._actions[(ordinal - 1) // 8], ordinal=ordinal)
        identity = tuple(row[key] for key in IDENTITY)
        if self._identity is not None and identity != self._identity:
            raise ValueError("right Tripod tracking physical identity changed")
        self._identity = identity
        for side in ("left", "right"):
            hand = exchange["action_readback"]["post"]["hands"][side]
            if len(set(hand["joint_names"])) != 17 or hand["mounted"] is not True:
                raise ValueError("all 17 original mounted joint identities required per side")
            finite(hand["backend_targets"], (1, 17))
            finite(hand["native_anchor_pose_w"], (1, 7))
        if self._quiet is not None and ordinal <= 8:
            quiet = self._quiet.observe_measurement(row)
            if quiet["status"] == "failed":
                raise ValueError("right Tripod entry quiet failed: " + ", ".join(quiet["failure_reasons"]))
        self._samples = ordinal
        self._failed = False

    def finish(self):
        if self._failed or self._samples != 176 or self._quiet is None:
            raise ValueError("incomplete right Tripod tracking")
        self._quiet.require_ready()
        return self._samples


def record_tracking_window(plan, record, *, quiet=None):
    return _TrackingRecorder(validate_schedule(plan), record, quiet)


def run_owned_tracking(sim, owners, trace, *, project, plan_path, plan_sha256, output,
                       run_uuid, episode_uid):
    """Borrow a fresh context under the existing lock/normal-close supervisor."""
    from functools import partial

    import torch

    from go2_pvcnn.control.dh116 import tabletop_deformable_dual_hand_episode as episode_api
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    from go2_pvcnn.control.dh116.tabletop_deformable_clearance_guard import ClearanceGuardedExchange
    from go2_pvcnn.control.dh116.tabletop_deformable_dual_evidence import DualHandEvidenceJournal
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile, profile_reader_scope
    from go2_pvcnn.control.dh116.tabletop_tripod_quiet_gate import TripodQuietGate
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

    episode = rod = journal = recorder = None
    try:
        plan = checked_json(plan_path, plan_sha256)
        actions = validate_schedule(plan)
        if (plan["endpoint_sha256"] != ENDPOINT_SHA or plan["reference_sha256"] != REFERENCE_SHA
                or plan["proved_right_sweep"]["proved"] is not True
                or plan["native_qualification"] is not False):
            raise ValueError("frozen right Tripod static provenance required")
        nodes = finite(plan["rod_nodes_m"], (41, 3))
        packet = deepcopy(builder.load_pair_proposal())
        packet["reset_seed"] = plan["reset_seed"]
        packet["rod_seed"]["proposed_nodes_m"] = nodes.tolist()
        profile = BackendProfile.load(project)
        prepared = scene.prepare_scene(project)
        journal = DualHandEvidenceJournal(Path(output) / "physical.jsonl",
                                          physics_budget_end=BUDGET["sdk_steps"], solver_substeps=12)
        quiet = TripodQuietGate(reset_nodes_m=nodes.astype(np.float32))
        recorder = record_tracking_window(plan, journal.record, quiet=quiet)

        def absent(name, context):
            return dict(owner=name, status="not_present", episode_uid=context["episode_uid"],
                        source="right-only tracking has no policy, task filter, prior contacts or camera history")

        owner_resets = {name: partial(absent, name)
                        for name in ("controller", "filter", "contact", "camera_history")}
        with profile_reader_scope(profile), bindings(
                (builder, "load_pair_proposal", lambda: packet),
                (builder, "SEED_SHA", packet["reset_seed"]["sha256"]),
                (builder, "BACKEND", Path(profile.binary_path)),
                (builder, "BACKEND_SHA", profile.binary_sha256),
                (episode_api, "FlexiblePhysicalExchange", ClearanceGuardedExchange)):
            builder.build_reset_owned(sim, owners, project=project, prepared=prepared,
                ownership=trace, record=recorder, owner_resets=owner_resets,
                run_uuid=run_uuid, episode_uid=episode_uid, physics_budget_end=BUDGET["sdk_steps"])
            episode = owners.resources["dual_episode"]
            rod = owners.resources["runtime"]
            for index, values in enumerate(actions):
                if index:
                    quiet.require_ready()
                action = torch.from_numpy(values[None, :])
                journal.begin(episode.pre_action_state(), action)
                result = episode.control(action)
                journal.end(result)
                if result["termination_reason"] is not None:
                    raise ValueError("right Tripod terminated: " + result["termination_reason"])
            if rod.clock()["physics_step"] != BUDGET["sdk_steps"]:
                raise ValueError("right Tripod final physical clock differs")
            return dict(physics_rows=recorder.finish(), control_steps=len(actions),
                        source_scope="current_instance_right_tripod_noncontact_tracking",
                        native_qualification=False, grasp_qualified=False, teacher_admission=False,
                        ppo_updates=0)
    finally:
        if journal is not None:
            journal.close()
        sim = owners = trace = episode = rod = journal = recorder = None
