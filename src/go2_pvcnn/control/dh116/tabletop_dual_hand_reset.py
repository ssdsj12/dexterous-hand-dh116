"""Single-environment reset transaction using the original mounted-hand APIs.

No app, prim, stepping, safety projection or R0 reward owner is created here.
Inputs are authored reset values, not a collision certificate. Both hands and
the existing rod are reset in place. A partial mutation invalidates the rod;
there is no rollback or retry of this transaction.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import PurePosixPath

import numpy as np
import torch

from .linear_coupling import ACTIVE, JOINTS, active_limits, _SOURCE_LOWER, _SOURCE_UPPER
from .tabletop_deformable_bimanual_step import DT, SIDES, _bindings, _owned
from .tabletop_dual_hand_rod_runtime import DualHandRodRuntime, PADS
from .tabletop_joint_target_readback import read_joint_target_evidence, WRIST_LINEAR, WRIST_ANGULAR
from .tabletop_tensor_world import write_contact_effort

VERSION = "dh116-dual-hand-reset-transaction-v1"
RESET_OWNERS = ("controller", "filter", "contact", "camera_history")
SEED_VERSION = "dh116-authored-reset-state-v1"


def _array(value, shape, label):
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"finite {label} {shape} required")
    return array


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def prepare_reset_state(*, hand_roots, origin, root_xy, initial_offset_m,
                        joint_positions, joint_velocities, source_description):
    """Apply TensorWorld's authored seed arithmetic; all physical inputs explicit.

    root_seed excludes initial_offset. reset_root_pose includes it. The caller's
    provenance label is retained verbatim but is not verified source approval.
    Preserve all eleven supplied q/qd; do not infer follower measurements.
    """
    if set(hand_roots) != set(SIDES) or not isinstance(source_description, str) or not source_description:
        raise ValueError("both authored hand roots and explicit source description required")
    roots = _array([hand_roots[s] for s in SIDES], (2, 7), "authored wxyz roots")
    if not np.allclose(np.linalg.norm(roots[:, 3:], axis=1), 1., rtol=0, atol=1e-6):
        raise ValueError("normalized authored root quaternions required")
    origin = _array(origin, (3,), "environment origin")
    xy = _array(root_xy, (2, 2), "authored root_xy")
    if type(initial_offset_m) not in (float, int) or not math.isfinite(initial_offset_m):
        raise ValueError("explicit finite initial root offset required")
    q = _array(joint_positions, (2, 11), "canonical joint positions")
    qd = _array(joint_velocities, (2, 11), "canonical joint velocities")
    lo, hi = active_limits()
    active = [JOINTS.index(n) for n in ACTIVE]
    if np.any(q[:, active] < lo) or np.any(q[:, active] > hi):
        raise ValueError("authored active joints exceed original coupled limits")
    if np.any(q < np.asarray(_SOURCE_LOWER)) or np.any(q > np.asarray(_SOURCE_UPPER)):
        raise ValueError("authored full joints exceed original source limits")
    root_seed = roots.copy()
    root_seed[:, :3] += origin
    root_seed[:, :2] += xy
    reset_pose = root_seed.copy()
    reset_pose[:, 2] += float(initial_offset_m)
    payload = dict(version=SEED_VERSION, sides=list(SIDES), joint_names=list(JOINTS),
        hand_roots=roots.tolist(), origin=origin.tolist(), root_xy=xy.tolist(),
        initial_offset_m=float(initial_offset_m), joint_positions=q.tolist(),
        joint_velocities=qd.tolist(), authored_root_seed_w=root_seed.tolist(),
        reset_root_pose_w=reset_pose.tolist(), source_description=source_description,
        source_qualification=False, initial_pose_qualified=False)
    return dict(payload, sha256=_digest(payload))


def _seed(value):
    packet = deepcopy(value)
    expected = packet.pop("sha256", None)
    if expected != _digest(packet):
        raise ValueError("reset seed content fingerprint differs")
    rebuilt = prepare_reset_state(hand_roots=dict(zip(SIDES, packet["hand_roots"])),
        origin=packet["origin"], root_xy=packet["root_xy"],
        initial_offset_m=packet["initial_offset_m"], joint_positions=packet["joint_positions"],
        joint_velocities=packet["joint_velocities"], source_description=packet["source_description"])
    if rebuilt != value:
        raise ValueError("authored reset arithmetic/schema differs")
    return rebuilt


def _clock(sim, rod):
    step, stamp, dt = sim.current_time_step_index, sim.current_time, sim.get_physics_dt()
    if type(step) is not int or step < 0 or not math.isclose(float(dt), DT, rel_tol=0, abs_tol=1e-12):
        raise ValueError("actual nonnegative 480Hz SDK clock required")
    if not math.isfinite(float(stamp)) or not math.isclose(float(stamp), step*DT, rel_tol=0, abs_tol=1e-9):
        raise ValueError("actual SDK timestamp differs from its step counter")
    row = deepcopy(rod.clock())
    if not math.isfinite(row["solver_time_s"]):
        raise ValueError("finite actual solver clock required")
    return dict(sdk_physics_step=step, sdk_timestamp_s=float(stamp), sdk_dt_s=float(dt),
                rod=row)


def _no_step(before, after):
    if (any(before[k] != after[k] for k in ("sdk_physics_step", "sdk_timestamp_s", "sdk_dt_s"))
            or before["rod"]["solver_time_s"] != after["rod"]["solver_time_s"]):
        raise ValueError("reset/refresh/owner unexpectedly advanced SDK or solver time")
    for key in ("run_uuid", "env_id", "rod_uid"):
        if before["rod"][key] != after["rod"][key]:
            raise ValueError("run/environment/rod identity changed during reset")


def _composer(composer, body_count):
    force = _owned(composer.composed_force_as_torch)
    torque = _owned(composer.composed_torque_as_torch)
    if force.shape != (1, body_count, 3) or torque.shape != force.shape:
        raise ValueError("actual full-body composer scratch shape required")
    return dict(force=force, torque=torque, active=bool(composer.active),
                source="WrenchComposer.composed_force_as_torch/composed_torque_as_torch")


def _hands(bindings, rod, *, finite):
    """Own actual backend buffers; cached command targets never replace them."""
    result = {}
    for binding in bindings:
        hand = binding.hand
        view = hand.root_physx_view
        evidence = read_joint_target_evidence(hand)
        names, bodies = list(hand.joint_names), list(hand.body_names)
        if (not evidence["joint_names_valid"] or len(names) != 17
                or set(names) != set(JOINTS + WRIST_LINEAR + WRIST_ANGULAR)
                or len(bodies) != 23 or len(set(bodies)) != 23):
            raise ValueError("actual named 17DOF/23-body mounted original hand required")
        if not evidence["backend"]["available"] or not evidence["dof_types"]["available"]:
            raise ValueError("actual backend targets and DOF types required")
        types = evidence["dof_types"]["values"]
        if types.shape != (1, 17) or any(types[0, i] != (1 if n in WRIST_LINEAR else 0)
                                          for i, n in enumerate(names)):
            raise ValueError("actual native DOF types differ")
        mount = binding.mount
        if (mount is None or mount.ids != [names.index(n) for n in WRIST_LINEAR]
                or mount.angular_ids != [names.index(n) for n in WRIST_ANGULAR]
                or mount.base_id != bodies.index("base_link")):
            raise ValueError("complete original MountView required")
        paths = deepcopy(list(view.link_paths))
        if len(paths) != 1 or len(paths[0]) != len(bodies):
            raise ValueError("actual backend link paths required")
        container = str(PurePosixPath(rod.bindings[binding.side+"."+PADS[0]]["prim_path"]).parent)
        if list(view.prim_paths) not in ([container], [container+"/wrist_fixed"]):
            raise ValueError("actual articulation root differs from pad container")
        if paths[0] != [container+"/"+name for name in bodies]:
            raise ValueError("actual body names/paths differ from rod hand container")
        for name in PADS:
            row = rod.bindings[binding.side+"."+name]
            if row["body_index"] != bodies.index(name) or row["prim_path"] != paths[0][bodies.index(name)]:
                raise ValueError("rod pad binding differs from actual articulation")
        packet = dict(joint_names=names, body_names=bodies, link_paths=paths,
            articulation_paths=list(view.prim_paths), dof_types=_owned(types),
            backend_targets=_owned(evidence["backend"]["values"]),
            backend_source=evidence["backend"]["source"],
            command_cache_targets=(_owned(evidence["command_cache"]["values"])
                if evidence["command_cache"]["available"] else None))
        getters = dict(q=("get_dof_positions", (1, 17)),
            qd=("get_dof_velocities", (1, 17)),
            root_pose_xyzw=("get_root_transforms", (1, 7)),
            root_velocity=("get_root_velocities", (1, 6)),
            stiffness=("get_dof_stiffnesses", (1, 17)),
            effort_limits=("get_dof_max_forces", (1, 17)),
            explicit_actuation_force=("get_dof_actuation_forces", (1, 17)),
            velocity_targets=("get_dof_velocity_targets", (1, 17)),
            body_poses_xyzw=("get_link_transforms", (1, len(bodies), 7)))
        for key, (method, shape) in getters.items():
            packet[key] = _owned(getattr(view, method)())
            if packet[key].shape != shape or packet[key].dtype.kind != "f":
                raise ValueError(f"actual {method} shape/dtype differs")
        packet["root_pose_wxyz"] = packet["root_pose_xyzw"][:, [0, 1, 2, 6, 3, 4, 5]].copy()
        packet["active_actuator_stiffness"] = _owned(hand.actuators["active"].stiffness)
        packet["composers"] = dict(permanent=_composer(hand.permanent_wrench_composer, len(bodies)),
                                  instantaneous=_composer(hand.instantaneous_wrench_composer, len(bodies)))
        arrays = [v for v in packet.values() if isinstance(v, np.ndarray)]
        packet["all_finite"] = all(np.isfinite(v).all() for v in arrays)
        if finite and (not packet["all_finite"] or packet["backend_targets"].shape != (1, 17)
                or packet["command_cache_targets"] is None
                or packet["command_cache_targets"].shape != (1, 17)
                or packet["active_actuator_stiffness"].shape != (1, 6)):
            raise ValueError("complete finite actual hand reset readbacks required")
        if names != list(hand.joint_names) or paths != list(view.link_paths):
            raise ValueError("actual hand identity changed during readback")
        result[binding.side] = packet
    return result


def _targets(seed, hi, names):
    q, qd = np.zeros((1, 17)), np.zeros((1, 17))
    for j, name in enumerate(JOINTS):
        q[0, names.index(name)] = seed["joint_positions"][hi][j]
        qd[0, names.index(name)] = seed["joint_velocities"][hi][j]
    return q, qd


def _verify_after(seed, before, after, effort_limit):
    for hi, side in enumerate(SIDES):
        old, row = before[side], after[side]
        for key in ("joint_names", "body_names", "link_paths", "articulation_paths"):
            if old[key] != row[key]:
                raise ValueError("actual articulation identity changed during reset")
        names = row["joint_names"]
        q, qd = _targets(seed, hi, names)
        active = [names.index(n) for n in ACTIVE]
        controlled = active + [names.index(n) for n in WRIST_LINEAR+WRIST_ANGULAR]
        for value, expected, label in (
            (row["q"], q, "q"), (row["qd"], qd, "qd"),
            (row["root_pose_wxyz"], np.asarray(seed["reset_root_pose_w"])[hi:hi+1], "root"),
            (row["root_velocity"], 0., "root velocity"),
            (row["backend_targets"][:, controlled], q[:, controlled], "SDK targets"),
            (row["command_cache_targets"][:, controlled], q[:, controlled], "cache targets"),
            (row["velocity_targets"], 0., "velocity targets"),
            (row["explicit_actuation_force"], 0., "explicit DOF actuation"),
            (row["stiffness"][:, active], 10., "active stiffness"),
            (row["active_actuator_stiffness"], 10., "active actuator stiffness"),
            (row["effort_limits"][:, active], effort_limit, "active effort limit"),
        ):
            if not np.allclose(value, expected, rtol=0, atol=1e-6):
                raise ValueError(f"actual post-reset {side} {label} differs")
        passive = [i for i in range(17) if i not in active]
        if (not np.array_equal(row["stiffness"][:, passive], old["stiffness"][:, passive])
                or not np.array_equal(row["effort_limits"][:, passive], old["effort_limits"][:, passive])):
            raise ValueError("reset changed original passive/wrist gain or effort limit")
        for composer in row["composers"].values():
            if composer["active"] or np.any(composer["force"] != 0) or np.any(composer["torque"] != 0):
                raise ValueError("actual post-reset composer is not zero and inactive")


class DualHandResetTransaction:
    """Single-use physical reset; object retains only owned seed/scalar failure data."""

    def __init__(self, seed, *, episode_uid):
        self.seed = _seed(seed)
        if not isinstance(episode_uid, str) or not episode_uid:
            raise ValueError("explicit fresh globally unique episode UID required")
        self.episode_uid = episode_uid
        self.status = "new"
        self.failure = None

    def run(self, sim, rod, bindings, *, owner_resets):
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
        before_clock = _clock(sim, rod)
        before = _hands(bindings, rod, finite=False)
        _no_step(before_clock, _clock(sim, rod))
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
            _no_step(before_clock, _clock(sim, rod))
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
            after_clock = _clock(sim, rod)
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
