"""Explicit single-owner exchange binding for pre-submit wrist experiments.

A caller must provide a same-boundary geometry checker and durable recorder.
No existing recipe is enabled by importing this module.
"""
from contextlib import contextmanager
from copy import deepcopy
from threading import Lock, get_ident

import numpy as np
import torch

from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
from . import tabletop_deformable_physical_exchange as exchange
from .tabletop_deformable_bimanual_step import BimanualStepConnection
from .tabletop_deformable_wrist_projection import CLEARANCE_M, project_policy_wrist_targets
from .tabletop_free_wrist_grasp import CONTRACT as WRIST, compose_rotation
from .tabletop_native_teacher_swept_projection import ANGULAR_STEP_RAD, LINEAR_STEP_M, _finite
from .tabletop_palm_mount import CONTRACT as MOUNT


_LOCK = Lock()
SIDES = ("left", "right")


def _selected_wrist(decisions, state):
    xyz = np.stack([_finite(row["xyz"], (3,)) for row in decisions])
    angles = np.stack([_finite(row["angles"], (3,)) for row in decisions])
    seed = state["root_seed_w"][0]
    previous_xyz = seed[:, :3].numpy() + state["palm_offsets_m"][0].numpy()
    offset = state["common_lift_offset_m"].item()
    previous_xyz[:, 2] += offset
    previous_angles = state["angles_rad"][0].numpy()
    measured = state["measured_base_w"][0, :, :3].numpy()
    palm = xyz - seed[:, :3].numpy()
    palm[:, 2] -= offset
    tol = 1e-12
    if (np.any(np.linalg.norm(xyz - previous_xyz, axis=-1) > LINEAR_STEP_M + tol)
            or np.any(np.linalg.norm(angles - previous_angles, axis=-1) > ANGULAR_STEP_RAD + tol)
            or np.any(np.linalg.norm(xyz - measured, axis=-1) > MOUNT["tracking_guard_m"])
            or np.any(abs(angles) > np.pi / 2 + tol)
            or np.any(palm < np.array(WRIST["palm_lower_m"]) - tol)
            or np.any(palm > np.array(WRIST["palm_upper_m"]) + tol)):
        raise ValueError("swept projection candidate exceeds original wrist contract")
    for decision in decisions:
        proof = decision["certificate"]
        if (proof["proved"] is not True
                or (not np.isfinite([proof[k] for k in ("start_m", "end_m", "margin_lower_m")]).all())
                or min(proof["start_m"], proof["end_m"]) < CLEARANCE_M
                or proof["margin_lower_m"] < 0):
            raise ValueError("all selected wrist paths must clear the original table floor")
    angle_tensor = seed.new_tensor(angles)
    roots = torch.cat((seed.new_tensor(xyz), compose_rotation(seed[:, 3:], angle_tensor)), dim=-1)
    return roots[None], angle_tensor[None], seed.new_tensor(palm)[None]


@contextmanager
def swept_projection_scope(choose, *, record):
    """Intercept one synchronous owner's step, before composer/SDK writes."""
    if not callable(choose) or not callable(record):
        raise TypeError("wrist checker and refusal evidence recorder required")
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("swept exchange binding cannot be nested or shared")
    captured = None
    owner = get_ident()
    try:
        if (exchange.BimanualStepConnection is not BimanualStepConnection
                or exchange.project_policy_wrist_targets is not project_policy_wrist_targets):
            raise RuntimeError("original exchange symbols required for the swept scope")

        class CapturedConnection(BimanualStepConnection):
            def begin(self, *args, **kwargs):
                nonlocal captured
                if get_ident() != owner or captured is not None:
                    raise RuntimeError("foreign or repeated step during swept projection")
                request = super().begin(*args, **kwargs)
                captured = deepcopy(self._pre), deepcopy(request)
                return request

        def project(**state):
            nonlocal captured
            if get_ident() != owner or captured is None:
                raise RuntimeError("same-owner SDK request required for swept projection")
            pre, request = captured
            captured = None
            nominal = project_policy_wrist_targets(**state)
            decisions = choose(pre, request, deepcopy(nominal), deepcopy(state))
            if (not isinstance(decisions, (tuple, list)) or len(decisions) != 2
                    or any(type(row.get("accepted")) is not bool for row in decisions)):
                raise ValueError("explicit decision required for each original hand")
            rejected = [side for side, decision in zip(SIDES, decisions) if not decision["accepted"]]
            item = dict(schema="dh116-swept-projection-step-v1", physics_step=pre["identity"][0]["physics_step"],
                        accepted=not rejected, rejected_sides=rejected, native_qualification=False)
            if rejected:
                record(dict(item, reasons={side: row["reason"] for side, row in zip(SIDES, decisions)
                                           if not row["accepted"]}))
                raise RuntimeError("wrist_sweep_rejected:" + ",".join(rejected))
            try:
                roots, angles, palms = _selected_wrist(decisions, state)
            except ValueError as error:
                record(dict(item, accepted=False, binding_failure=str(error)))
                raise
            record(dict(item, certified_target_w=roots.tolist(), certified_wrist_angles_rad=angles.tolist(),
                        actual_clearance_m=state["measured_clearance_m"].tolist(),
                        endpoint_bound_m=[row["certificate"]["end_m"] for row in decisions],
                        path_margin_lower_m=[row["certificate"]["margin_lower_m"] for row in decisions]))
            return dict(nominal, projected_root_poses=roots, projected_angles=angles, palm_offsets_m=palms)

        with bindings((exchange, "BimanualStepConnection", CapturedConnection),
                      (exchange, "project_policy_wrist_targets", project)):
            yield
    finally:
        captured = None
        _LOCK.release()
