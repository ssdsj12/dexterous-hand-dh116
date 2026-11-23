"""Explicit training-only hand-height derivation; no asset/material/rod changes."""
import math
import numpy as np

from .tabletop_dual_hand_reset import prepare_reset_state,_seed,SIDES
from .tabletop_native_teacher_profile import SHA


def derive_height_seed(seed,*,grasp,delta_z_m,evidence_sha256):
    original=_seed(seed)
    delta=np.asarray(delta_z_m,dtype=float)
    if (grasp not in ("pinch","tripod","power") or delta.shape!=(2,)
            or not np.isfinite(delta).all() or np.any(delta<0) or np.any(delta>.02)
            or not isinstance(evidence_sha256,str) or not SHA.fullmatch(evidence_sha256)):
        raise ValueError("grasp-specific independent height candidate and actual failure SHA required")
    roots=np.array(original["hand_roots"],dtype=float,copy=True)
    roots[:,2]+=delta
    description=("shared_training_height_v1;grasp="+grasp+";parent="+seed["sha256"]
        +";dz="+",".join(format(float(x),".17g") for x in delta)
        +";failure="+evidence_sha256)
    result=prepare_reset_state(hand_roots=dict(zip(SIDES,roots.tolist(),strict=True)),
        origin=original["origin"],root_xy=original["root_xy"],
        initial_offset_m=original["initial_offset_m"],joint_positions=original["joint_positions"],
        joint_velocities=original["joint_velocities"],source_description=description)
    return result,dict(schema="dh116-shared-training-height-v1",grasp=grasp,
        parent_seed_sha256=seed["sha256"],derived_seed_sha256=result["sha256"],
        delta_z_m=delta.tolist(),failure_evidence_sha256=evidence_sha256,
        rod_translation_m=[0.,0.,0.],joint_or_asset_or_physics_parameter_changed=False,
        native_reset_qualified=False,teacher_admission=False)


def initial_mesh_world_poses(solver, plan, measured):
    """Use measured SDK contact links while retaining all native static poses."""
    snapshot=solver.mesh_state_snapshot()
    xyz=np.array(snapshot["current_positions_world_m"],dtype=np.float32,copy=True)
    quat=np.array(snapshot["current_quaternions_xyzw"],dtype=np.float32,copy=True)
    expected=(len(measured),len(plan["entries"]))
    if xyz.shape!=(*expected,3) or quat.shape!=(*expected,4):
        raise ValueError("complete initial mesh world layout required")
    for lane,(poses,_) in enumerate(measured):
        seen=set()
        for slot,entry in enumerate(plan["entries"]):
            name=entry["name"]
            if name in poses:
                xyz[lane,slot],quat[lane,slot]=poses[name]
                seen.add(name)
        if seen!=set(poses):
            raise ValueError("all measured contact links must belong to native slots")
    if not np.isfinite(xyz).all() or not np.isfinite(quat).all():
        raise ValueError("finite initial mesh world state required")
    if not np.allclose(np.linalg.norm(quat,axis=-1),1.,atol=2e-6,rtol=0):
        raise ValueError("actual SDK unit quaternion required")
    return xyz,quat
