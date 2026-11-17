"""CPU source adapter for the unchanged 371D Teacher pre-action contract.

This captures owned joint state, never targets. The caller must update the
Articulation from the SDK and provide its live shared clock before any target
write. Identity checks detect overlap/staleness; they do not prove that a caller
actually performed an SDK update. This module grants no native qualification.
Other common-state sources remain unknown until independently implemented.
"""
from copy import deepcopy
import math

import numpy as np

from .linear_coupling import ACTIVE, JOINTS, MIMIC, active_limits

VERSION = "dh116-teacher-named-preaction-source-v1"
SOURCE = "isaaclab_articulation_data_joint_pos_joint_vel"
CONTRACT_SHA256 = "9f145cf8e68b1dfe8e2c7464ec5fc0cb00e81a2707115c2fe148ab49d77bbb62"
COUPLING_SHA256 = "168e30634c2e240fbaecd225766b52a1f2a3c87afafc8c2a7b5a23dfcb1afe93"
PHASE = "pre_action_before_target_write"
IDENTITY_KEYS = ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid",
                 "physics_step", "rod_tick", "policy_step", "timestamp_s", "phase")
DT = 1 / 480


def _identity(value):
    if not isinstance(value, dict) or any(key not in value for key in IDENTITY_KEYS):
        raise ValueError("complete pre-action identity required")
    result = {key: deepcopy(value[key]) for key in IDENTITY_KEYS}
    for key in ("run_uuid", "episode_uid", "rod_uid"):
        if not isinstance(result[key], str) or not result[key]:
            raise ValueError(f"nonempty {key} required")
    for key in ("env_id", "reset_id", "physics_step", "rod_tick", "policy_step"):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f"nonnegative integer {key} required")
    stamp = result["timestamp_s"]
    if (type(stamp) not in (int, float) or not math.isfinite(stamp)
            or not math.isclose(stamp, result["physics_step"] * DT, rel_tol=0, abs_tol=1e-9)
            or result["phase"] != PHASE):
        raise ValueError("pre-action phase and exact 480Hz SDK clock required")
    return result


def _owned(value, dtype=np.float64):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=dtype, copy=True)


def _checked_packet(packet, expected):
    if not isinstance(packet, dict):
        raise ValueError("named readback packet required")
    if (packet.get("version") != VERSION or packet.get("source") != SOURCE
            or packet.get("source_valid") is not True
            or packet.get("units") != {"q": "rad", "qd": "rad/s"}
            or packet.get("side") not in ("left", "right")):
        raise ValueError("actual named measured joint position/velocity source required")
    if _identity(packet.get("identity")) != expected:
        raise ValueError("joint readback differs from expected pre-action identity")
    names = packet.get("joint_names")
    if (not isinstance(names, (list, tuple)) or len(names) != 11
            or any(not isinstance(n, str) for n in names)
            or len(set(names)) != 11 or set(names) != set(JOINTS)):
        raise ValueError("actual SDK joint names/order required; no guessed fallback")
    types = _owned(packet.get("dof_types"), dtype=None)
    if (packet.get("dof_type_source") != "physx_backend_get_dof_types"
            or types.shape != (11,) or types.dtype.kind not in "iu"
            or not np.all(types == 0)):
        raise ValueError("integer angular DOF type readback and source required")
    q, qd = _owned(packet.get("q")), _owned(packet.get("qd"))
    if q.shape != (11,) or qd.shape != (11,) or not np.isfinite([q, qd]).all():
        raise ValueError("finite measured 11-joint q and qd required")
    return list(names), q, qd


def capture_named_hand(hand, *, side, read_identity, data_env_index):
    """Copy measured SDK buffers between two matching shared-clock reads.

    read_identity is supplied by the live environment, not a target cache.
    data_env_index selects an articulation buffer row; env_id is the independent
    external identity. Neither is inferred from the other.
    """
    if type(data_env_index) is not int or data_env_index < 0:
        raise ValueError("explicit nonnegative articulation data row required")
    before = _identity(read_identity())
    names = list(hand.joint_names)
    data = hand.data
    q, qd = _owned(data.joint_pos), _owned(data.joint_vel)
    if q.ndim != 2 or qd.shape != q.shape or data_env_index >= len(q):
        raise ValueError("matching articulation joint buffers and valid data row required")
    try:
        dof_types = _owned(hand.root_physx_view.get_dof_types(), dtype=None)
    except (AttributeError, TypeError, RuntimeError) as error:
        raise ValueError("actual SDK angular DOF type readback required") from error
    if (dof_types.shape != q.shape or dof_types.dtype.kind not in "iu"
            or not np.all(dof_types[data_env_index] == 0)):
        raise ValueError("all named DH116 DOF types must be angular (SDK DofType 0)")
    after = _identity(read_identity())
    if after != before or list(hand.joint_names) != names:
        raise ValueError("shared clock or joint names changed during measured-state copy")
    result = dict(version=VERSION, source=SOURCE, side=side, identity=before,
                  joint_names=names, q=q[data_env_index].tolist(), qd=qd[data_env_index].tolist(),
                  units={"q": "rad", "qd": "rad/s"}, source_valid=True,
                  dof_types=dof_types[data_env_index].tolist(),
                  dof_type_source="physx_backend_get_dof_types")
    _checked_packet(result, before)
    return result


def adapt_named_hands(packets, *, expected_identity):
    """Map only measured q, qd and mimic residual; all other fields stay NaN."""
    expected = _identity(expected_identity)
    values = np.full(371, np.nan)
    valid = np.zeros(371, dtype=bool)
    sources = [None] * 371
    seen = set()
    lower, upper = active_limits()
    for packet in packets:
        names, q, qd = _checked_packet(packet, expected)
        side = packet["side"]
        if side in seen:
            raise ValueError("duplicate hand source")
        seen.add(side)
        active = [names.index(name) for name in ACTIVE]
        residual = [(q[names.index(f)] - ratio*q[names.index(m)] - offset)/.02
                    for f, m, ratio, offset in MIMIC]
        hand = 0 if side == "left" else 1
        groups = ((6*hand, 2*(q[active]-lower)/(upper-lower)-1, "joint_position"),
                  (12+6*hand, qd[active]/.3, "joint_velocity"),
                  (78+5*hand, residual, "passive_mimic_error"))
        for start, block, name in groups:
            end = start + len(block)
            values[start:end] = block
            valid[start:end] = True
            sources[start:end] = [f"{SOURCE}:{side}:{name}"] * len(block)
    return dict(version=VERSION, contract_sha256=CONTRACT_SHA256, identity=expected,
                values=values, valid=valid, sources=sources,
                qualified_for_teacher_ppo=False, new_physics_steps=0, optimizer_updates=0)


def require_teacher_common(observation):
    """Fail before an unmasked Teacher call; finite placeholders are insufficient.

    This adapter currently supplies at most 34 channels, so no output from it
    can pass this gate. Completing sources requires a separately reviewed
    extension, not a caller setting valid=True for finite placeholders.
    """
    if (observation.get("version") != VERSION
            or observation.get("contract_sha256") != CONTRACT_SHA256):
        raise ValueError("exact pre-action source contract required")
    _identity(observation.get("identity"))
    values = observation.get("values")
    valid = observation.get("valid")
    sources = observation.get("sources")
    if (not isinstance(values, np.ndarray) or values.shape != (371,)
            or not isinstance(valid, np.ndarray) or valid.shape != (371,)
            or valid.dtype != np.bool_ or not isinstance(sources, list) or len(sources) != 371):
        raise ValueError("371 values, boolean validity and per-field sources required")
    for i, name in enumerate(sources):
        if valid[i] and (not isinstance(name, str) or not name.startswith(SOURCE + ":")):
            raise ValueError("valid field has no admitted source")
    missing = np.flatnonzero(~valid).tolist()
    if missing:
        raise ValueError(f"Teacher blocked: {len(missing)} unknown source channels: {missing}")
    # Only these sources are implemented; a hand-written mask cannot extend it.
    implemented = set(range(24)) | set(range(78, 88))
    if any(valid[i] for i in range(371) if i not in implemented):
        raise ValueError("valid field has no implemented source adapter")
    if not np.isfinite(values).all():
        raise ValueError("Teacher common contains nonfinite values")
    return values.copy()


def audit_legacy_pinch(rows):
    """Inventory the locked single-left-hand ledger without retroactive names.

    Its row clock is post-step; pre_state was copied before that step. The
    derived previous SDK step is an audit observation, not a historical clock
    field or a complete policy observation. Rod and force arrays are post-step.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty historical ledger required")
    audits = []
    first = rows[0]
    for index, row in enumerate(rows, 1):
        for key in ("phase_step", "physics_step", "sdk_step", "rod_tick", "rod_substep",
                    "policy_step", "substep", "env_id", "reset_id"):
            if type(row.get(key)) is not int:
                raise ValueError(f"integer historical {key} required")
        if (row["phase_step"], row["sdk_step"], row["physics_step"], row["rod_tick"],
                row["rod_substep"], row["policy_step"], row["substep"]) != (
                index, index+2, index+2, index*12, index*12, (index-1)//8, (index-1)%8):
            raise ValueError("historical candidate SDK/rod/phase clock differs")
        for key, expected in (("timestamp_s", (index+2)*DT),
                              ("sdk_time_s", (index+2)*DT), ("dt_s", DT)):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or not math.isclose(value, expected, rel_tol=0, abs_tol=2e-7):
                raise ValueError("historical timestamp differs")
        if (row.get("side") != "left" or row.get("full_state371_bound") is not False
                or row["env_id"] != 0 or row["reset_id"] != 1):
            raise ValueError("only the original explicitly unbound left-hand source is supported")
        for key in ("episode_uid", "rod_uid"):
            if not isinstance(row.get(key), str) or not row[key] or row[key] != first[key]:
                raise ValueError("historical episode/rod identity differs")
        for phase in ("pre_state", "post_state"):
            state = row.get(phase, {})
            for key in ("q", "qd"):
                value = _owned(state.get(key))
                if value.shape != (11,) or not np.isfinite(value).all():
                    raise ValueError("historical measured q/qd unavailable or nonfinite")
        audits.append(dict(phase_step=index, pre_state_sdk_step_derived=index+1,
                           post_state_sdk_step=index+2, policy_boundary=row["substep"] == 0,
                           joint_names_recorded="joint_names" in row.get("pre_state", {}),
                           admitted_371d_channels=0, failure=row.get("failure")))
    return dict(version="dh116-teacher-legacy-source-inventory-v1", rows=len(rows),
                row_audits=audits, admitted_371d_channels=0, unknown_indices=list(range(371)),
                blockers=["missing_actual_sdk_joint_names", "missing_pre_action_shared_identity",
                          "missing_right_hand", "post_step_rod_not_pre_action",
                          "aggregate_forces_not_contact_normal_sources",
                          "missing_controller_reset_and_completed_interval_sources"],
                qualified_for_teacher_ppo=False, native_success=False,
                new_physics_steps=0, optimizer_updates=0)
