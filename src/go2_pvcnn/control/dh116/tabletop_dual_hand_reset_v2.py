"""Independent zero-step dual-hand Reset v2; frozen physical APIs unchanged."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from .tabletop_dual_hand_reset import (_seed, _hands, _targets, _verify_after,
    _no_step, RESET_OWNERS, ACTIVE, JOINTS, WRIST_LINEAR, WRIST_ANGULAR,
    write_contact_effort, _bindings, DT, PADS, DualHandRodRuntime)
from .tabletop_native_training_clock import (ClockPolicyV2, read_clock_v2,
    validate_sdk_clock, _validate_rod, VERSION as CLOCK_VERSION)

VERSION = "dh116-dual-hand-reset-transaction-v2"
FROZEN_V1_SOURCES = {'tabletop_dual_hand_reset.py': 'da35b878d6ef4d5236649e2cadd97ae1a66972ce3f0646419d69c0a6d0c6bce2', 'tabletop_mesh_reset_order.py': '94394cd6a10c528422bda28f7ee8813587ba1ca970dff54d9eabcc41ab19b50a', 'tabletop_mesh_reset_readback.py': '53c362e365df241280caa8d89ebd821a8a541d7c62c845d4956a6c12530ca0cc', 'tabletop_mesh_reset_numeric_state.py': 'e02853b0e7be2d38f9887d35549f335430fc92ae7380857be931d07b7dda0346'}
CLOCK_SOURCE_SHA256 = "90bc0e7bab5117d5724d35ea5faaee25cb378981b1f3d58815d295c34439a32b"

def verify_frozen_source(name, expected):
    path = Path(__file__).with_name(name)
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("frozen reset source changed: " + name)

def verify_source_lineage():
    for name, expected in FROZEN_V1_SOURCES.items():
        verify_frozen_source(name, expected)
    verify_frozen_source("tabletop_native_training_clock.py", CLOCK_SOURCE_SHA256)
    return dict(FROZEN_V1_SOURCES, clock_source_sha256=CLOCK_SOURCE_SHA256)

def _validate_clock_packet(packet, policy):
    if type(packet) is not dict or set(packet) != {
            "schema", "sdk_physics_step", "sdk_timestamp_s", "sdk_dt_s",
            "sdk_tolerance_s", "clock_policy_sha256", "rod"}:
        raise ValueError("exact v2 clock fields required")
    checked = validate_sdk_clock(step=packet["sdk_physics_step"],
        timestamp_s=packet["sdk_timestamp_s"],dt_s=packet["sdk_dt_s"],policy=policy)
    if (packet["schema"] != CLOCK_VERSION
            or packet["clock_policy_sha256"] != checked.policy_sha256
            or type(packet["sdk_tolerance_s"]) is not float
            or packet["sdk_tolerance_s"] != checked.tolerance_s):
        raise ValueError("v2 clock policy/envelope identity differs")
    _validate_rod(deepcopy(packet["rod"]), checked, policy)
    return packet

def failure_clock_start(sim, rod):
    """Raw diagnostic readback only, never a validated or repaired clock."""
    result=dict(sdk_step=None,sdk_timestamp_s=None,solver_time_s=None,read_errors=[])
    try:
        for name,getter in (
                ("sdk_step",lambda:sim.current_time_step_index),
                ("sdk_timestamp_s",lambda:sim.current_time),
                ("solver_time_s",lambda:rod.rope.time)):
            try:
                value=getter()
                if name=="sdk_step":
                    if type(value) is not int or value<0:raise ValueError("actual SDK step unavailable")
                elif type(value) not in (int,float) or not math.isfinite(value) or value<0:
                    raise ValueError("actual diagnostic time unavailable")
                result[name]=value
            except BaseException as error:
                result["read_errors"].append(name+": "+type(error).__name__+": "+str(error))
        return result
    finally:
        sim=rod=getter=None


def failure_clock_progress(before, sim, rod):
    after=failure_clock_start(sim,rod)
    initial=before["sdk_step"] if before is not None else None
    final=after["sdk_step"]
    delta=final-initial if type(initial) is int and type(final) is int and final>=initial else None
    return dict(sdk_step_before=initial,sdk_step_after=final,observed_sdk_steps=delta,
        before=deepcopy(before),after=after,
        observed_solver_calls=None,solver_step_count_status="unknown_no_actual_call_counter",
        solver_step_count_inferred=False,clock_qualified=False)


class DualHandResetTransactionV2:
    """Single-use physical transaction with explicit immutable clock policy."""
    def __init__(self, seed, *, episode_uid, clock_policy):
        self.source_lineage = verify_source_lineage()
        if type(clock_policy) is not ClockPolicyV2:
            raise ValueError("exact v2 clock policy required")
        self.clock_policy = deepcopy(clock_policy)
        self.clock_policy_sha256 = self.clock_policy.sha256
        self.seed = _seed(seed)
        if type(episode_uid) is not str or not episode_uid.strip():
            raise ValueError("explicit fresh globally unique episode UID required")
        self.episode_uid = episode_uid
        self.status = "new"
        self.failure = None
        self.cleanup_failures = []

    def _verify(self):
        if self.clock_policy.sha256 != self.clock_policy_sha256:
            raise ValueError("reset clock policy changed")
        if verify_source_lineage() != self.source_lineage:
            raise ValueError("reset source lineage changed")

    def run(self, sim, rod, bindings, *, owner_resets):
        try:
            self._verify()
            if self.status != "new":
                raise RuntimeError("reset transaction cannot be reused")
            seed = _seed(self.seed)
            if not isinstance(rod, DualHandRodRuntime) or rod.rope.num_envs != 1:
                raise ValueError("existing single-environment DualHandRodRuntime required")
            _bindings(bindings, (rod.env_id,))
            if set(owner_resets) != set(RESET_OWNERS) or not all(callable(v) for v in owner_resets.values()):
                raise ValueError("explicit controller/filter/contact/camera_history reset owners required")
            if self.episode_uid in rod._seen_episodes:
                raise ValueError("fresh episode UID required")
            if not callable(getattr(sim.physics_sim_view, "update_articulations_kinematic", None)):
                raise ValueError("actual no-step kinematics refresh API required")
            before_clock = read_clock_v2(sim, rod, policy=self.clock_policy)
            before = _hands(bindings, rod, finite=False)
            _no_step(before_clock, read_clock_v2(sim, rod, policy=self.clock_policy))
            effort_limit = float(rod._base.cfg["hand"]["active_effort_limit_nm"])
            if not math.isfinite(effort_limit) or effort_limit <= 0:
                raise ValueError("explicit candidate active effort limit required")
            reset_id = before_clock["rod"]["reset_id"] + 1
            rope_identity = id(rod.rope)
            self.status = "mutating"
            phase = "hands"
            try:
                for hi, binding in enumerate(bindings):
                    hand, mount = binding.hand, binding.mount
                    names = before[binding.side]["joint_names"]
                    active = [names.index(n) for n in ACTIVE]
                    q, qd = _targets(seed, hi, names)
                    tensor = lambda v: torch.as_tensor(v, dtype=torch.float32, device=hand.device)
                    ids = torch.tensor([0], dtype=torch.long, device=hand.device)
                    pose = tensor(np.asarray(seed["reset_root_pose_w"])[hi:hi+1])
                    hand.reset(ids)
                    hand.write_joint_state_to_sim(tensor(q), tensor(qd), env_ids=ids)
                    mount.reset(pose, ids)
                    hand.write_root_pose_to_sim(pose, env_ids=ids)
                    hand.write_root_velocity_to_sim(tensor(np.zeros((1, 6))), env_ids=ids)
                    hand.set_joint_position_target(tensor(q[:, active]), joint_ids=active, env_ids=ids)
                    hand.write_joint_stiffness_to_sim(tensor(np.full((1, 6), 10.)), joint_ids=active, env_ids=ids)
                    hand.actuators["active"].stiffness[ids] = 10.
                    hand.set_joint_velocity_target(tensor(np.zeros((1, 17))), env_ids=ids)
                    hand.set_joint_effort_target(tensor(np.zeros((1, 17))), env_ids=ids)
                    write_contact_effort(hand, active, tensor(np.zeros((1, 6))), effort_limit, env_ids=ids)
                    permanent, instantaneous = hand.permanent_wrench_composer, hand.instantaneous_wrench_composer
                    permanent.reset()
                    instantaneous.reset()
                    zeros = tensor(np.zeros((1, len(hand.body_names), 3)))
                    permanent.set_forces_and_torques(forces=zeros, torques=zeros, is_global=False)
                    hand.write_data_to_sim()
                    permanent.reset()
                    instantaneous.reset()
                phase = "kinematics"
                sim.physics_sim_view.update_articulations_kinematic()
                for binding in bindings:
                    binding.hand.update(DT)  # ArticulationData cache timestamp only; no SDK tick.
                _no_step(before_clock, read_clock_v2(sim, rod, policy=self.clock_policy))
                updated = _hands(bindings, rod, finite=True)
                pad_poses = {}
                for binding in bindings:
                    row = updated[binding.side]
                    for name in PADS:
                        pose = row["body_poses_xyzw"][0, row["body_names"].index(name)]
                        pad_poses[binding.side+"."+name] = (pose[:3].copy(), pose[3:].copy())
                phase = "rod"
                raw_reset = rod.reset(physics_step=before_clock["sdk_physics_step"],
                    episode_uid=self.episode_uid, reset_id=reset_id, pad_poses=pad_poses)
                phase = "owners"
                context = dict(version=VERSION, **rod.clock(),
                    authored_root_seed_w=deepcopy(seed["authored_root_seed_w"]),
                    reset_root_pose_w=deepcopy(seed["reset_root_pose_w"]),
                    source_seed_sha256=seed["sha256"])
                owner_results = {}
                for name in RESET_OWNERS:
                    packet = owner_resets[name](deepcopy(context))
                    # No retained native handles, arbitrary object encoders, NaN sensor defaults.
                    packet = json.loads(json.dumps(packet, allow_nan=False))
                    if (not isinstance(packet, dict) or packet.get("owner") != name
                            or packet.get("status") not in ("cleared", "not_present", "unavailable")
                            or packet.get("episode_uid") != self.episode_uid
                            or not isinstance(packet.get("source"), str) or not packet["source"]):
                        raise ValueError("explicit reset owner identity/status/source required")
                    owner_results[name] = packet
                phase = "readback"
                after = _hands(bindings, rod, finite=True)
                after_clock = read_clock_v2(sim, rod, policy=self.clock_policy)
                _no_step(before_clock, after_clock)
                row = after_clock["rod"]
                if (id(rod.rope) != rope_identity or row["episode_uid"] != self.episode_uid
                        or row["reset_id"] != reset_id or row["rod_tick"] != 0
                        or row["physics_step"] != before_clock["sdk_physics_step"]
                        or row["clock_valid"] is not True):
                    raise ValueError("same rod fresh reset identity/readback differs")
                _verify_after(seed, before, after, effort_limit)
                self.status = "complete"
                return dict(version=VERSION, status="complete", seed=deepcopy(seed),
                clock_policy_sha256=self.clock_policy_sha256, reset_epoch=reset_id,
                source_lineage=deepcopy(self.source_lineage),
                    before_clock=before_clock, after_clock=deepcopy(row), after_actual_clocks=after_clock,
                    before_hands=before, hands=after, pad_poses_xyzw=deepcopy(pad_poses),
                    rod_reset=deepcopy(raw_reset), owner_resets=owner_results,
                    new_sdk_steps=0, new_solver_steps=0,
                    external_body_force_native_readback=dict(status="unavailable_no_getter", values=None,
                        zero_write_source="Articulation.write_data_to_sim -> apply_forces_and_torques_at_position",
                        zero_write_observed_at_backend=False),
                    initial_pose_qualified=False, native_qualification=False, teacher_admission=False,
                    r0_reset_complete=False, accepted_controller_snapshot=False)
            except BaseException as error:
                self.status = "failed"
                self.failure = dict(phase=phase, error_type=type(error).__name__, error=str(error),
                                    rollback_performed=False)
                rod._needs_reset = True
                rod._base.last = None
                raise
        finally:
            sim=rod=bindings=binding=owner_resets=hand=mount=permanent=instantaneous=tensor=ids=pose=zeros=None
