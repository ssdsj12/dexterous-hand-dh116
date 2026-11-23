"""Opt-in fixed finger targets for an authored native Pinch or Tripod phase."""

import numpy as np
import torch

from .linear_coupling import ACTIVE, JOINTS
from .tabletop_deformable_bimanual_step import SIDES, VERSION, _clock
from .tabletop_deformable_solver_clock import solver_clock_matches
from .tabletop_joint_target_readback import BACKEND_SOURCE, WRIST_ANGULAR, WRIST_LINEAR
from .tabletop_surface_contract import ACTION_CONTRACT, validate_actions


_PARTICIPATING = {
    "pinch": ("finger11", "finger12", "finger21"),
    "tripod": ("finger11", "finger12", "finger21", "finger31"),
    "power": ACTIVE,
}
_TARGET_IDENTITY = ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid")
_PHYSICAL_CLOCK = _TARGET_IDENTITY + ("physics_step", "rod_tick", "timestamp_s", "solver_time_s")
_TRANSACTION_IDENTITY = _TARGET_IDENTITY + ("policy_step", "substep")
_ROD_SUBSTEPS = 12


def _validate_action(action):
    if isinstance(action, torch.Tensor) and action.ndim in (1, 2) and action.numel():
        validate_actions(action.unsqueeze(0) if action.ndim == 1 else action)
        return
    if (isinstance(action, np.ndarray) and action.ndim in (1, 2) and action.shape[-1] == ACTION_CONTRACT["size"]
            and action.size and action.dtype.kind == "f" and np.isfinite(action).all()
            and (np.abs(action) <= 1.0).all()):
        return
    raise ValueError("finite bounded normalized [24] or [N,24] action required")


def _target_words(targets, shape, indices):
    if (not isinstance(targets, np.ndarray) or targets.dtype != np.dtype(np.float32)
            or targets.shape != shape or not np.isfinite(targets).all()):
        raise ValueError("finite binary32 target buffer with the named shape required")
    return targets[:, indices].tobytes()


def _physical_clock(rows):
    return tuple(tuple(row[key] for key in _PHYSICAL_CLOCK) for row in rows)


def _hand_sample(side, hand, env_count, held_local):
    names = tuple(hand["joint_names"])
    full_names = JOINTS + WRIST_LINEAR + WRIST_ANGULAR if hand["mounted"] else JOINTS
    if len(names) != len(full_names) or set(names) != set(full_names):
        raise ValueError(f"{side} requires the complete named DH116 backend joint buffer")
    active = tuple(hand["active_indices"])
    if any(type(i) is not int for i in active) or active != tuple(names.index(joint) for joint in ACTIVE):
        raise ValueError(f"{side} requires the exact canonical ACTIVE mapping")
    paths = tuple(hand["prim_paths"])
    if (len(paths) != env_count or len(set(paths)) != env_count
            or any(f"{side}Hand" not in path.split("/") for path in paths)
            or sorted(hand["row_indices"]) != list(range(env_count))
            or hand["backend_source"] != BACKEND_SOURCE):
        raise ValueError(f"{side} requires the owned backend target source and side binding")
    binding = (names, active, paths, tuple(hand["row_indices"]), hand["mounted"],
               hand["instance_object_id"], hand["view_object_id"])
    shape = (env_count, len(names))
    held_backend = [active[i] for i in held_local]
    return binding, _target_words(hand["backend_targets"], shape, held_backend)


class GraspFingerHold:
    """Project candidate actions and check a contiguous phase of native target readbacks."""

    def __init__(self, grasp):
        if grasp not in _PARTICIPATING:
            raise ValueError("known Pinch, Tripod or Power grasp required")
        self._held_local = tuple(i for i, joint in enumerate(ACTIVE) if joint not in _PARTICIPATING[grasp])
        self._held_action_indices = tuple(len(ACTIVE) * hand + i for hand in range(len(SIDES))
                                          for i in self._held_local)
        self.clear()

    @property
    def held_action_indices(self):
        return self._held_action_indices

    def clear(self):
        """Start a new phase explicitly, including after Reset."""
        self._phase = None

    def project_action(self, action):
        """Return the original 24D shape and zero only the held joint action channels."""
        _validate_action(action)
        if not self._held_action_indices:
            return action
        if isinstance(action, torch.Tensor):
            held = torch.tensor(self._held_action_indices, device=action.device)
            return action.index_fill(-1, held, 0)
        projected = action.copy()
        projected[..., self._held_action_indices] = 0
        return projected

    def verify_action(self, action):
        """Reject an authored action that would move a held joint target."""
        _validate_action(action)
        if self._held_action_indices and bool((action[..., self._held_action_indices] != 0).any()):
            raise ValueError("held action channels must be exactly zero")

    def observe(self, row):
        """Bind the first pre-action targets and check actual pre/post and submitted targets."""
        if not self._held_action_indices:
            return
        from .tabletop_deformable_r0_replay import decode_owned

        try:
            wrapped = "action_readback" in row
            readback = decode_owned(row["action_readback"] if wrapped else row)
            post_clock, mappings, held_targets = self._observe_readback(readback)
            if wrapped and (row.get("guard_failure") is not None
                            or row["physics_step"] != readback["post"]["identity"][0]["physics_step"]):
                raise ValueError("the wrapped native SDK clock or guard status differs")
        except (AttributeError, KeyError, IndexError, TypeError) as error:
            raise ValueError("complete owned bimanual target readback required") from error
        self._phase = post_clock, mappings, held_targets

    def _observe_readback(self, readback):
        if readback["version"] != VERSION or readback["request"]["action_contract"] != ACTION_CONTRACT:
            raise ValueError("original bimanual step and 24D action contract required")
        if readback["observed_sdk_step_delta"] != 1 or readback["observed_rod_tick_delta"] != _ROD_SUBSTEPS:
            raise ValueError("native readback must report one original SDK/rod clock transition")
        env_ids = tuple(readback["env_ids"])
        if not env_ids or any(type(i) is not int for i in env_ids) or len(set(env_ids)) != len(env_ids):
            raise ValueError("distinct explicit readback environment IDs required")
        n = len(env_ids)
        pre, post = readback["pre"], readback["post"]
        pre_clock, post_clock = (_clock(lambda: state["identity"], env_ids) for state in (pre, post))
        for before, after in zip(pre_clock, post_clock, strict=True):
            if (any(before[k] != after[k] for k in _TRANSACTION_IDENTITY)
                    or after["physics_step"] != before["physics_step"] + 1
                    or after["rod_tick"] != before["rod_tick"] + _ROD_SUBSTEPS
                    or not solver_clock_matches(before["solver_time_s"], after["solver_time_s"], steps=1)):
                raise ValueError("one same-Reset SDK and solver clock transition required")
        if self._phase is not None and _physical_clock(pre_clock) != self._phase[0]:
            raise ValueError("the phase readback skipped or changed its physical clock or Reset")

        action = readback["request"]["action24"]
        self.verify_action(action)
        if action.shape != (n, ACTION_CONTRACT["size"]):
            raise ValueError("readback actions must match the environment batch")
        submissions = readback["submission"]["expected_backend_targets"]
        if any(set(hands) != set(SIDES) for hands in (pre["hands"], post["hands"], submissions)):
            raise ValueError("exactly one left and one right hand are required")

        bindings, held = [], []
        for side in SIDES:
            before, before_words = _hand_sample(side, pre["hands"][side], n, self._held_local)
            after, after_words = _hand_sample(side, post["hands"][side], n, self._held_local)
            if before != after:
                raise ValueError(f"{side} backend joint binding changed during the SDK step")
            entry = submissions[side]
            indices = tuple(entry["joint_indices"])
            if indices[:len(ACTIVE)] != before[1]:
                raise ValueError(f"{side} submission differs from its canonical ACTIVE mapping")
            submitted = _target_words(entry["values"], (n, len(indices)), list(self._held_local))
            if not before_words == after_words == submitted:
                raise ValueError(f"{side} held target changed during the SDK step or its submission")
            bindings.append(before)
            held.append(before_words)

        bindings, held = tuple(bindings), tuple(held)
        if self._phase is not None:
            if bindings != self._phase[1]:
                raise ValueError("backend joint binding changed across the phase")
            if held != self._phase[2]:
                raise ValueError("held target differs from the first phase pre-action readback")
        return _physical_clock(post_clock), bindings, held
