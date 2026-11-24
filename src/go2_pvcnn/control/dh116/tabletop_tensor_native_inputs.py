"""Batched tensor equivalent of the shared Teacher's original observation math.

This is a numerical adapter, not an SDK capture or admission certificate. The
caller must establish the original mounted base_link/COM-velocity readback,
authored root seed, shared material parameters and same-boundary context owner.
Every numerical input is already a Tensor on the same device. No lane is
copied to NumPy/CPU, and the 305 unavailable common fields remain NaN/false.
"""
from dataclasses import dataclass
import math

import torch

from .linear_coupling import ACTIVE, JOINTS, MIMIC, active_limits
from .tabletop_joint_target_readback import WRIST_LINEAR, WRIST_ANGULAR
from .tabletop_flexible_measured_common_teacher import common_selection
from .tabletop_deformable_privileged_teacher import _features
from .tabletop_native_teacher_context import ContextIdentity, validate_context_tensors
from .tabletop_shared_teacher_source import FAMILY, SharedSourceIdentity


VERSION = "dh116-shared-tensor-native-inputs-v1"


@dataclass(frozen=True, slots=True)
class TensorTeacherPacket:
    """Owned six-key input with the unchanged shared parent/row source identity."""

    inputs: dict
    source: SharedSourceIdentity
    grasp: str
    context_identities: tuple
    source_family: str = FAMILY

    @property
    def native_qualified(self):
        return False

    def validate_fresh(self, current_identity):
        if type(current_identity) is not SharedSourceIdentity or current_identity != self.source:
            raise ValueError("shared source parent/row/episode Reset epoch or clock changed")
        return self

    def for_teacher(self, model, *, current_identity):
        """New explicit hook; the old pack_shared class gate stays untouched."""
        from .tabletop_shared_native_teacher import SharedNativeTeacher
        self.validate_fresh(current_identity)
        from .tabletop_tensor_native_teacher import TensorNativeTeacher
        if type(model) not in (SharedNativeTeacher,TensorNativeTeacher):
            raise ValueError("explicit original or tensor-source shared Teacher required")
        identity = model.model_identity
        if (identity["source_family"] != self.source_family
                or identity["parent_num_envs"] != self.source.parent_num_envs
                or identity["grasp"] != self.grasp
                or identity["backend_profile_sha256"] != self.source.profile_sha256):
            raise ValueError("shared Teacher source/grasp/profile/environment contract differs")
        model._parts(self.inputs)
        return {key: value.detach().clone() for key, value in self.inputs.items()}


def _tensor(name, value, shape, device, *, dtype=None, finite=True):
    if (not isinstance(value, torch.Tensor) or value.shape != shape or value.device != device
            or (value.dtype != dtype if dtype is not None else value.dtype not in (torch.float32, torch.float64))):
        raise ValueError(f"{name} requires same-device {shape} tensor with original units/dtype")
    if finite and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} requires finite native tensor values")
    return value.detach()


def _source(source, n, contexts, profile_sha, grasp, scope):
    if (type(source) is not SharedSourceIdentity or source.parent_num_envs != n
            or source.env_rows != tuple(range(n)) or not 2 <= n <= 512
            or source.scope != scope or scope not in ("cpu_fixture", "owner_source_unqualified")
            or source.profile_sha256 != profile_sha or type(profile_sha) is not str or len(profile_sha) != 64
            or grasp not in ("pinch", "tripod", "power")
            or any(type(x) is not tuple or len(x) != n for x in
                (source.episode_uids,source.rod_uids,source.reset_ids,source.rod_ticks))
            or type(contexts) is not tuple or len(contexts) != n
            or any(type(c) is not ContextIdentity for c in contexts)):
        raise ValueError("explicit same shared parent rows/profile/grasp/source scope required")
    expected = tuple(ContextIdentity(source.run_uuid, i, source.episode_uids[i], source.reset_ids[i],
        source.rod_uids[i], source.sdk_step, source.rod_ticks[i], source.policy_time_s,
        source.solver_time_s) for i in range(n))
    if contexts != expected:
        raise ValueError("common/rod/context shared parent row/time/episode source differs")


@torch.no_grad()
def pack_shared_tensor_inputs(q, qd, *, joint_names, body_link_pose_w,
        body_com_velocity_w, root_seed_w, env_origins, rod_positions_w,
        rod_velocities_w, rest_lengths_m, task_context, task_context_valid,
        source_identity, context_identities, backend_profile_sha256, grasp, scope):
    """Pack [N,2,17] native q/qd into the original common66+rod41x13+context50.

    ``joint_names`` is a pair of complete, source-verified 17DOF name tuples,
    one per hand; q/qd angular units are rad and rad/s, prismatic units m/m/s.
    Pose is measured base_link WXYZ (no sign canonicalization); velocity is
    the original body COM world linear/angular velocity in m/s and rad/s.
    ``root_seed_w`` excludes reset root_offset, exactly as the original source.
    Context values/masks are copied from the existing owner, never synthesized.
    The caller rechecks its live shared boundary before consuming the packet.
    """
    if not isinstance(q, torch.Tensor) or q.ndim != 3 or q.shape[1:] != (2,17):
        raise ValueError("q requires complete native [N,2,17] tensor")
    n, device = len(q), q.device
    _source(source_identity, n, context_identities, backend_profile_sha256, grasp, scope)
    complete = set(JOINTS+WRIST_LINEAR+WRIST_ANGULAR)
    if (type(joint_names) not in (tuple,list) or len(joint_names) != 2
            or any(type(names) not in (tuple,list) or len(names) != 17
                or any(type(name) is not str for name in names)
                or len(set(names)) != 17 or set(names) != complete for names in joint_names)):
        raise ValueError("complete original named 17DOF joint source required for both hands")
    q = _tensor("q",q,(n,2,17),device).double()
    qd = _tensor("qd",qd,(n,2,17),device).double()
    pose = _tensor("base_link WXYZ pose",body_link_pose_w,(n,2,7),device).double()
    velocity = _tensor("body COM world velocity",body_com_velocity_w,(n,2,6),device).double()
    seed = _tensor("authored root seed",root_seed_w,(n,2,7),device).double()
    origins = _tensor("environment origins",env_origins,(n,3),device)
    if (not bool((origins[:,2] == 0).all())
            or not bool((abs(torch.linalg.vector_norm(pose[...,3:],dim=-1)-1.) <= 1e-5).all())):
        raise ValueError("horizontal origins and raw unit WXYZ quaternion required")
    context = _tensor("task context",task_context,(n,50),device,dtype=torch.float32,finite=False)
    context_valid = _tensor("task context validity",task_context_valid,(n,50),device,dtype=torch.bool)
    validate_context_tensors(context,context_valid,mode="approach_wiring_smoke_v1")

    # The legacy readback converts native samples to NumPy float64 before
    # normalization and casts the resulting ledger to float32. Promote first
    # to retain that mathematical order, including mimic offsets/limits.
    ledger = torch.full((n,371),torch.nan,dtype=torch.float32,device=device)
    valid = torch.zeros((n,371),dtype=torch.bool,device=device)
    lower, upper = active_limits()
    lower, upper = q.new_tensor(lower), q.new_tensor(upper)
    # This fixed two-hand metadata loop is independent of environment count.
    for hi, names in enumerate(joint_names):
        ids = [names.index(name) for name in ACTIVE]
        ledger[:,6*hi:6*(hi+1)] = (2*(q[:,hi,ids]-lower)/(upper-lower)-1).float()
        ledger[:,12+6*hi:18+6*hi] = (qd[:,hi,ids]/.3).float()
        follower = [names.index(f) for f, _, _, _ in MIMIC]
        master = [names.index(m) for _, m, _, _ in MIMIC]
        ratio = q.new_tensor([r for _, _, r, _ in MIMIC])
        offset = q.new_tensor([o for _, _, _, o in MIMIC])
        ledger[:,78+5*hi:83+5*hi] = ((q[:,hi,follower]-ratio*q[:,hi,master]-offset)/.02).float()
        normalized_pose = pose[:,hi].clone()
        normalized_pose[:,:3] = (pose[:,hi,:3]-seed[:,hi,:3])/.05
        ledger[:,153+7*hi:160+7*hi] = normalized_pose.float()
        ledger[:,167+6*hi:173+6*hi] = (velocity[:,hi]/q.new_tensor([.05,.05,.05,1.,1.,1.])).float()
        ledger[:,179+3*hi:182+3*hi] = (q[:,hi,[names.index(k) for k in WRIST_ANGULAR]]/(math.pi/2)).float()
    valid[:,common_selection()] = True
    if not bool(torch.isfinite(ledger[valid]).all()):
        raise ValueError("native common normalization produced nonfinite values")

    # Same float32 order as SharedTeacherSource.capture: world differences
    # precede subtracting origins; other vectors stay in the world frame.
    p = _tensor("rod world positions",rod_positions_w,(n,41,3),device).float()
    v = _tensor("rod world velocities",rod_velocities_w,(n,41,3),device).float()
    rest = _tensor("material rest lengths",rest_lengths_m,(n,40),device).float()
    if not bool((rest > 0).all()):
        raise ValueError("finite positive material rest lengths required")
    edges = p.diff(dim=1)
    lengths = torch.linalg.vector_norm(edges,dim=-1)
    if not bool((lengths > 0).all()):
        raise ValueError("degenerate shared material segment")
    direction = edges/lengths[...,None]
    tangent = torch.cat((direction[:,:1],direction[:,:-1]+direction[:,1:],direction[:,-1:]),1)
    tangent /= torch.linalg.vector_norm(tangent,dim=-1,keepdim=True)
    curvature = torch.zeros((n,41),dtype=p.dtype,device=device)
    curvature[:,1:-1] = torch.linalg.vector_norm(direction.diff(dim=1),dim=-1)/((lengths[:,:-1]+lengths[:,1:])/2)
    snapshot = dict(node_positions=p-origins.to(dtype=p.dtype)[:,None],velocities=v,
        tangent=tangent,curvature=curvature,rest_lengths_m=rest,deformation=lengths/rest-1)
    mask = torch.ones((n,41),dtype=torch.bool,device=device)
    nodes = _features(snapshot,ledger[:,common_selection()],mask)
    inputs = dict(common371=ledger,common_valid=valid,rod_nodes=nodes,rod_mask=mask,
        task_context=context.clone(),task_context_valid=context_valid.clone())
    return TensorTeacherPacket(inputs,source_identity,grasp,context_identities)
