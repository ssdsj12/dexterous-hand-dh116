"""Independent measured mounted-wrist extension of the unchanged 371D state.

The original source gate is untouched. This version supplies at most 66/371
channels and never grants Teacher admission. Actual SDK buffers are compared
with the exact IsaacLab fields used by TensorWorld; no target/anchor fallback.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import tabletop_teacher_preaction_readback as base
from .linear_coupling import JOINTS
from .tabletop_joint_target_readback import WRIST_LINEAR, WRIST_ANGULAR
from .tabletop_palm_mount import MountView

VERSION = "dh116-teacher-mounted-preaction-source-v1"
SOURCE = "isaaclab_base_link_pose_and_body_com_velocity_verified_against_physx"
SEED_VERSION = "dh116-authored-reset-root-seed-v1"
SEED_SOURCE = "tensorworld_layout_hand_roots_plus_origins_plus_root_xy"
RESET_KEYS = ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid")
SIDES = ("left", "right")
VELOCITY_REFERENCE = "body_com_world_existing_body_lin_vel_w_body_ang_vel_w"


def _implementation():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
        Path(__file__).name, "tabletop_teacher_preaction_readback.py",
        "tabletop_tensor_world.py", "tabletop_free_wrist_grasp.py", "tabletop_palm_mount.py",
        "linear_coupling.py", "tabletop_joint_target_readback.py")}


def _clock(value):
    identity = base._identity(value)
    if (value.get("clock_valid") is not True or type(value.get("substep")) is not int
            or not 0 <= value["substep"] < 8
            or type(value.get("solver_time_s")) not in (int, float)
            or not math.isfinite(value["solver_time_s"]) or value["solver_time_s"] < 0):
        raise ValueError("valid shared SDK/rod clock and control substep required")
    identity.update(clock_valid=True, substep=value["substep"], solver_time_s=value["solver_time_s"])
    return identity


def _array(value, shape):
    result = base._owned(value)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"finite array {shape} required")
    return result


def _seed_checked(seed, current):
    if (not isinstance(seed, dict) or seed.get("version") != SEED_VERSION
            or seed.get("source") != SEED_SOURCE):
        raise ValueError("authored reset seed source required; no measured-palm fallback")
    identity = _clock(seed["identity"])
    if any(identity[key] != current[key] for key in RESET_KEYS):
        raise ValueError("reset seed identity differs from current episode/reset/rod")
    authored = _array(seed["authored_hand_roots_w"], (2, 7))
    origin = _array(seed["environment_origin_w"], (3,))
    xy = _array(seed["root_xy"], (2, 2))
    actual = _array(seed["root_seed_w"], (2, 7))
    expected = authored.copy()
    expected[:, :3] += origin
    expected[:, :2] += xy
    if not np.array_equal(actual, expected):
        raise ValueError("reset seed differs from authored layout+origin+root_xy; offset is excluded")
    return actual


def capture_authored_reset_seed(tensor_world, *, read_identity, data_env_index):
    """Copy and verify TensorWorld.root_seed, never initial measured palm pose.

    The reset owner supplies the identity. root_offset/initial_offset is
    deliberately excluded, exactly as free_wrist_grasp.observe subtracts it.
    """
    if type(data_env_index) is not int or data_env_index < 0:
        raise ValueError("explicit data row required")
    before = _clock(read_identity())
    seeds = base._owned(tensor_world.root_seed)
    origins = base._owned(tensor_world.origins)
    if (seeds.ndim != 3 or seeds.shape[1:] != (2, 7)
            or origins.shape != (len(seeds), 3) or data_env_index >= len(seeds)):
        raise ValueError("complete authored per-environment seed arrays required")
    result = dict(version=SEED_VERSION, source=SEED_SOURCE, identity=before,
        authored_hand_roots_w=base._owned([tensor_world.w.layout["hand_roots"][s] for s in SIDES]).tolist(),
        environment_origin_w=origins[data_env_index].tolist(),
        root_xy=base._owned(tensor_world.w.seed["root_xy"]).tolist(),
        root_seed_w=seeds[data_env_index].tolist())
    _seed_checked(result, before)
    if _clock(read_identity()) != before:
        raise ValueError("clock changed while copying authored reset seed")
    return result


def _packet(packet, expected):
    if (not isinstance(packet, dict) or packet.get("version") != VERSION
            or packet.get("source") != SOURCE or packet.get("source_valid") is not True
            or packet.get("velocity_reference") != VELOCITY_REFERENCE
            or packet.get("implementation_sha256") != _implementation()):
        raise ValueError("current measured mounted source and implementation binding required")
    if _clock(packet["identity"]) != expected:
        raise ValueError("mounted source identity differs")
    side = packet.get("side")
    names = packet.get("joint_names")
    if (side not in SIDES or not isinstance(names, list) or len(names) != 17
            or any(not isinstance(n, str) for n in names)
            or len(set(names)) != 17 or set(names) != set(JOINTS+WRIST_LINEAR+WRIST_ANGULAR)):
        raise ValueError("complete actual named 17DOF hand required")
    types = np.asarray(packet["dof_types"])
    if (types.shape != (17,) or types.dtype.kind not in "iu"
            or not np.array_equal(types, [1 if n in WRIST_LINEAR else 0 for n in names])):
        raise ValueError("actual angular/prismatic SDK DOF types required")
    q, qd = _array(packet["q"], (17,)), _array(packet["qd"], (17,))
    pose = _array(packet["body_link_pose_w"], (7,))
    velocity = np.r_[_array(packet["body_lin_vel_w"], (3,)), _array(packet["body_ang_vel_w"], (3,))]
    if not math.isclose(float(np.linalg.norm(pose[3:])), 1., abs_tol=1e-5, rel_tol=0):
        raise ValueError("raw measured WXYZ quaternion must be finite and unit")
    path = packet.get("hand_path")
    if (not isinstance(path, str) or not path.startswith("/") or side+"Hand" not in path.split("/")
            or packet.get("body_path") != path+"/base_link"):
        raise ValueError("exact actual mounted base_link path required")
    seed = _seed_checked(packet["reset_seed"], expected)
    return names, q, qd, pose, velocity, seed


def capture_mounted_hand(hand, mount, *, side, read_identity, data_env_index,
                         expected_hand_path, reset_seed):
    if side not in SIDES or type(data_env_index) is not int or data_env_index < 0:
        raise ValueError("explicit side and data row required")
    before = _clock(read_identity())
    _seed_checked(reset_seed, before)
    names, bodies = list(hand.joint_names), list(hand.body_names)
    if (len(names) != 17 or len(set(names)) != 17
            or set(names) != set(JOINTS+WRIST_LINEAR+WRIST_ANGULAR)
            or len(set(bodies)) != len(bodies) or bodies.count("base_link") != 1
            or not isinstance(mount, MountView) or mount.hand is not hand
            or mount.base_id != bodies.index("base_link")
            or mount.ids != [names.index(n) for n in WRIST_LINEAR]
            or mount.angular_ids != [names.index(n) for n in WRIST_ANGULAR]):
        raise ValueError("original complete named 17DOF MountView/base_link required")
    view, data = hand.root_physx_view, hand.data
    paths = list(view.prim_paths)
    links = deepcopy(view.link_paths)
    n, count = len(paths), len(bodies)
    if (data_env_index >= n or len(set(paths)) != n or paths[data_env_index] != expected_hand_path
            or len(links) != n or any(len(row) != count for row in links)
            or links[data_env_index][mount.base_id] != expected_hand_path+"/base_link"):
        raise ValueError("actual SDK environment/body paths differ")
    cached_q, cached_qd = _array(data.joint_pos, (n, 17)), _array(data.joint_vel, (n, 17))
    cached_pose = _array(data.body_link_pose_w, (n, count, 7))
    cached_linear = _array(data.body_lin_vel_w, (n, count, 3))
    cached_angular = _array(data.body_ang_vel_w, (n, count, 3))
    # These getters have the same source semantics as the exact TensorWorld
    # fields: link actor pose, but COM linear/angular velocity.
    q, qd = _array(view.get_dof_positions(), (n, 17)), _array(view.get_dof_velocities(), (n, 17))
    pose = _array(view.get_link_transforms(), (n, count, 7))[..., [0, 1, 2, 6, 3, 4, 5]]
    velocity = _array(view.get_link_velocities(), (n, count, 6))
    types = base._owned(view.get_dof_types(), dtype=None)
    if types.shape != (n, 17) or types.dtype.kind not in "iu":
        raise ValueError("complete integer SDK DOF readback required")
    for cached, native in ((cached_q, q), (cached_qd, qd), (cached_pose, pose),
                           (cached_linear, velocity[..., :3]), (cached_angular, velocity[..., 3:])):
        if not np.array_equal(cached[data_env_index], native[data_env_index]):
            raise ValueError("TensorWorld cache differs from actual SDK readback")
    idx, body = data_env_index, mount.base_id
    result = dict(version=VERSION, source=SOURCE, source_valid=True, side=side, identity=before,
        joint_names=names, q=q[idx].tolist(), qd=qd[idx].tolist(), dof_types=types[idx].tolist(),
        hand_path=paths[idx], body_path=links[idx][body], body_link_pose_w=cached_pose[idx, body].tolist(),
        body_lin_vel_w=cached_linear[idx, body].tolist(), body_ang_vel_w=cached_angular[idx, body].tolist(),
        velocity_reference=VELOCITY_REFERENCE, reset_seed=deepcopy(reset_seed),
        implementation_sha256=_implementation())
    _packet(result, before)
    if (list(hand.joint_names) != names or list(hand.body_names) != bodies
            or list(view.prim_paths) != paths or view.link_paths != links or _clock(read_identity()) != before):
        raise ValueError("clock or actual names/paths changed during capture")
    return result


def adapt_mounted_hands(packets, *, expected_identity):
    expected = _clock(expected_identity)
    packets = deepcopy(list(packets))
    checked, fingers = [], []
    seeds = []
    for packet in packets:
        names, q, qd, pose, velocity, seed = _packet(packet, expected)
        ids = [names.index(name) for name in JOINTS]
        fingers.append(dict(version=base.VERSION, source=base.SOURCE, source_valid=True,
            side=packet["side"], identity=expected, joint_names=list(JOINTS),
            q=q[ids].tolist(), qd=qd[ids].tolist(), dof_types=[0]*11,
            dof_type_source="physx_backend_get_dof_types", units={"q":"rad","qd":"rad/s"}))
        checked.append((packet["side"], pose, velocity, q[[names.index(n) for n in WRIST_ANGULAR]], seed))
        seeds.append(packet["reset_seed"])
    if seeds and any(seed != seeds[0] for seed in seeds[1:]):
        raise ValueError("both mounted hands must use the same authored reset seed packet")
    result = base.adapt_named_hands(fingers, expected_identity=expected)
    result.update(version=VERSION, identity=expected, source_packets=packets)
    for side, pose, velocity, angles, seed in checked:
        hi = SIDES.index(side)
        normalized_pose = pose.copy()
        normalized_pose[:3] = (pose[:3]-seed[hi, :3])/.05
        for start, values, field in (
                (153+7*hi, normalized_pose, "base_link_pose_wxyz_from_authored_seed"),
                (167+6*hi, velocity/np.array([.05,.05,.05,1.,1.,1.]), "body_com_velocity"),
                (179+3*hi, angles/(math.pi/2), "intrinsic_mount_joint_angles")):
            stop = start+len(values)
            result["values"][start:stop] = values
            result["valid"][start:stop] = True
            result["sources"][start:stop] = [SOURCE+":"+side+":"+field]*len(values)
    if not np.isfinite(result["values"][result["valid"]]).all():
        raise ValueError("mounted source normalization produced nonfinite values")
    return result


def validate_mounted_observation(observation):
    if not isinstance(observation, dict) or observation.get("version") != VERSION:
        raise ValueError("independent mounted source version required")
    expected = adapt_mounted_hands(observation["source_packets"], expected_identity=observation["identity"])
    for key in ("contract_sha256", "sources", "qualified_for_teacher_ppo", "new_physics_steps", "optimizer_updates"):
        if observation.get(key) != expected[key]:
            raise ValueError("mounted observation differs from source reconstruction")
    valid, values = observation.get("valid"), observation.get("values")
    if (not isinstance(valid, np.ndarray) or valid.dtype != np.bool_
            or not isinstance(values, np.ndarray) or values.shape != (371,)
            or not np.array_equal(valid, expected["valid"])
            or not np.array_equal(values, expected["values"], equal_nan=True)):
        raise ValueError("mounted observation mask/values differ from source reconstruction")
    return dict(valid_channels=int(valid.sum()), unknown_channels=int((~valid).sum()),
                unknown_indices=np.flatnonzero(~valid).tolist(), native_qualification=False)


def require_mounted_teacher_common(observation):
    status = validate_mounted_observation(observation)
    if status["unknown_channels"]:
        raise ValueError(f"Teacher blocked: {status['unknown_channels']} unknown source channels")
    return observation["values"].copy()
