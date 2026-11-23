"""Explicit shared parent/row source; old one-solver-per-lane contracts stay intact."""
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch

from .tabletop_native_teacher_profile import checked_file, digest, ROOT
from .tabletop_native_teacher_context import ContextIdentity, validate_context
from .tabletop_teacher_mounted_wrist_source import validate_mounted_observation
from .tabletop_deformx_parameter_snapshot import validate_snapshot
from .tabletop_deformable_privileged_teacher import _features
from .tabletop_shared_rod_owner import SharedRodOwner

LOCK = ROOT / "docs/contracts/dh116_shared_backend_source_v1.json"
FAMILY = "shared_batched_solver_v1"


@dataclass(frozen=True, slots=True)
class SharedBackendProfile:
    binary_path: str
    binary_sha256: str
    source_tree_sha256: str
    patch_sha256: str
    digest: str

    @classmethod
    def load(cls):
        import json
        lock = json.loads(LOCK.read_text())
        if (lock["schema"] != "dh116-shared-backend-source-v1" or lock["teacher_admission"] is not False
                or lock["source_file_count"] != 72
                or lock["upstream_commit"] != "96b8514c7c55c724f1abba1fbf6ed6642b7db72b"):
            raise ValueError("explicit source-locked shared backend required")
        for path, sha in lock["source_inventory"].items():
            checked_file(path, sha)
        checked_file(lock["binary_path"], lock["binary_sha256"])
        receipt = lock["native_mechanical_receipt"]
        outer = json.loads(checked_file(receipt["path"], receipt["sha256"]))
        if (outer["inferior_status"] != dict(exit_code=0, signal=None) or outer["forced_exit"]):
            raise ValueError("natural mechanical Reset receipt required")
        return cls(lock["binary_path"], lock["binary_sha256"], digest(lock["source_inventory"]),
                   lock["source_inventory"][str(ROOT/"scripts/patches/deformx_shared_mesh_reset.patch")],
                   digest(lock))

    def verify(self):
        if self != type(self).load():
            raise ValueError("shared backend source identity changed")
        return self


@dataclass(frozen=True, slots=True)
class SharedSourceIdentity:
    parent_solver_id: int
    parent_num_envs: int
    env_rows: tuple
    run_uuid: str
    episode_uids: tuple
    rod_uids: tuple
    reset_ids: tuple
    rod_ticks: tuple
    sdk_step: int
    sdk_time_s: float
    sdk_dt_s: float
    sdk_timer_dt_s: float
    policy_time_s: float
    solver_time_s: float
    profile_sha256: str
    scope: str


@dataclass(frozen=True, slots=True)
class SharedTeacherPacket:
    inputs: dict
    source: SharedSourceIdentity
    parent_parameters: dict
    grasp: str


class SharedTeacherSource:
    """Read each shared parent once, never relabel a sliced snapshot as N=1."""

    def __init__(self, runtime, sim, *, backend_profile, env_origins, scope, device, rod_uids=None):
        if (type(runtime) is not SharedRodOwner or type(backend_profile) is not SharedBackendProfile
                or scope not in ("cpu_fixture", "owner_source_unqualified")):
            raise ValueError("actual shared owner, profile and explicit source scope required")
        backend_profile.verify()
        self.runtime, self.sim, self.profile, self.scope = runtime, sim, backend_profile, scope
        self.device = torch.device(device)
        self.rod_uids = (tuple("rod"+str(i) for i in range(runtime.solver.num_envs))
                         if rod_uids is None else tuple(rod_uids))
        if (len(self.rod_uids) != runtime.solver.num_envs or len(set(self.rod_uids)) != len(self.rod_uids)
                or any(type(x) is not str or not x for x in self.rod_uids)):
            raise ValueError("one source-owned material rod identity per parent row required")
        self.origins = np.array(env_origins, dtype=np.float64, copy=True)
        if (self.origins.shape != (runtime.solver.num_envs, 3) or not np.isfinite(self.origins).all()
                or np.any(self.origins[:, 2] != 0)):
            raise ValueError("source-owned horizontal environment origins required")
        self.origins.setflags(write=False)
        if scope != "cpu_fixture":
            loaded = sys.modules.get("ropesim._ropesim")
            if (loaded is None or Path(loaded.__file__).resolve() != Path(backend_profile.binary_path)
                    or type(runtime.solver) is not loaded.RopeBatch):
                raise ValueError("loaded actual native shared solver required")
        self._boundary()

    def _boundary(self):
        r, solver = self.runtime, self.runtime.solver
        getter = getattr(self.sim,"get_physics_dt",None)
        dt = float(getter()) if callable(getter) else 1/480
        fp32_dt = float(np.float32(1/480))
        # Isaac 4.5 current_time accumulates the native callback step_size,
        # while get_physics_dt returns the configured nominal reciprocal rate.
        # Retain both, verify raw timer progression; never fit a free offset.
        timer_dt = float(np.float32(dt)) if callable(getter) else dt
        if (dt not in (1/480,fp32_dt) or not np.isfinite(float(self.sim.current_time))):
            raise ValueError("actual 480Hz SDK dt and finite native time required")
        if (r._failed or solver.last_error != 0 or float(solver.time) != r._time
                or self.sim.current_time_step_index != r.physics_step
                or abs(float(self.sim.current_time)-r.physics_step*timer_dt) > max(1e-9,r.physics_step*1e-14)):
            raise ValueError("shared SDK/solver source boundary changed or failed")
        return SharedSourceIdentity(id(solver), solver.num_envs, tuple(range(solver.num_envs)),
            r.run_uuid, r.episodes, self.rod_uids, tuple(r.resets), tuple(r.ticks), r.physics_step,
            float(self.sim.current_time), dt, timer_dt, r.physics_step/480, float(solver.time), self.profile.digest, self.scope)

    def validate_fresh(self, packet):
        if type(packet) is not SharedTeacherPacket or packet.source != self._boundary():
            raise ValueError("shared parent/row/episode Reset epoch or clock changed")
        return packet

    def _parameters(self, raw, boundary):
        import json
        checked = validate_snapshot(raw, expected_num_envs=boundary.parent_num_envs,
            expected_nodes=41, expected_time_s=boundary.solver_time_s)
        cfg = json.loads((ROOT/"configs/dh116_tasks/flexible_pinch_candidate_v1.json").read_text())["rod"]
        expected = dict(contact_radius_m=cfg["radius_m"], gravity_world_m_s2=cfg["gravity_world_m_s2"],
            contact_mu_static=cfg["mu_s"], contact_mu_kinetic=cfg["mu_k"],
            contact_v_slip_m_s=cfg["v_slip_m_s"], contact_kappa=cfg["kappa"],
            stretch_ea_n=cfg["axial_stiffness_n"], bend_ei_n_m2=cfg["bend_stiffness_n_m2"])
        if checked["fixed_node_indices"] or any(
                not np.allclose(raw[k], v, atol=1e-9, rtol=2e-6) for k, v in expected.items()):
            raise ValueError("native shared material/gravity/radius/free-root source differs")
        rest = np.asarray(raw["rest_lengths_m"], dtype=np.float64)
        mass = np.concatenate((rest[:, :1]/2, (rest[:, :-1]+rest[:, 1:])/2, rest[:, -1:]/2), 1)*cfg["linear_density_kg_m"]
        if not np.allclose(1/np.asarray(raw["inverse_masses_kg_inv"]), mass, rtol=1e-6, atol=1e-10):
            raise ValueError("actual shared device masses differ")
        return torch.as_tensor(rest, dtype=torch.float32, device=self.device).clone()

    def capture(self, common, contexts):
        boundary = self._boundary()
        n = boundary.parent_num_envs
        common, contexts = tuple(common), tuple(contexts)
        grasps = {c.metadata.grasp for c in contexts}
        if len(grasps) != 1:
            raise ValueError("one grasp-specific source batch required")
        if len(common) != n or len(contexts) != n:
            raise ValueError("one actual common/context per shared environment row required")
        values, valid, cv, cm = [], [], [], []
        for i, (c, context) in enumerate(zip(common, contexts, strict=True)):
            validate_mounted_observation(c)
            identity = ContextIdentity.from_clock(c["identity"])
            expected = ContextIdentity(boundary.run_uuid, i, boundary.episode_uids[i],
                boundary.reset_ids[i], boundary.rod_uids[i], boundary.sdk_step, boundary.rod_ticks[i],
                boundary.policy_time_s, boundary.solver_time_s)
            if identity != expected or context.metadata.scope != self.scope:
                raise ValueError("same shared parent row/time/episode and source scope required")
            validate_context(context, backend_profile=self.profile, identities=(expected,))
            if context.metadata.mode != "approach_wiring_smoke_v1":
                raise ValueError("shared initial implementation is explicit finger-goal smoke only")
            if any(not np.array_equal(p["reset_seed"]["environment_origin_w"], self.origins[i])
                   for p in c["source_packets"]):
                raise ValueError("mounted source belongs to a different environment origin")
            values.append(c["values"])
            valid.append(c["valid"])
            cv.append(context.values.to(self.device))
            cm.append(context.valid.to(self.device))
        ledger = torch.as_tensor(np.stack(values), dtype=torch.float32, device=self.device).clone()
        mask = torch.as_tensor(np.stack(valid), dtype=torch.bool, device=self.device).clone()
        # Read full native batch once; preserve N and row provenance in packet.
        solver = self.runtime.solver
        p = torch.as_tensor(solver.positions(), dtype=torch.float32, device=self.device).clone()
        v = torch.as_tensor(solver.velocities(), dtype=torch.float32, device=self.device).clone()
        raw = solver.parameter_snapshot()
        rest = self._parameters(raw, boundary)
        if p.shape != (n,41,3) or v.shape != p.shape or not torch.isfinite(p).all() or not torch.isfinite(v).all():
            raise ValueError("finite complete shared material41 position/velocity required")
        edges = p.diff(dim=1)
        lengths = torch.linalg.vector_norm(edges, dim=-1)
        if not bool((lengths > 0).all()):
            raise ValueError("degenerate shared material segment")
        direction = edges/lengths[...,None]
        tangent = torch.cat((direction[:,:1],direction[:,:-1]+direction[:,1:],direction[:,-1:]),1)
        tangent /= torch.linalg.vector_norm(tangent,dim=-1,keepdim=True)
        curvature = torch.zeros((n,41),dtype=p.dtype,device=p.device)
        curvature[:,1:-1] = torch.linalg.vector_norm(direction.diff(dim=1),dim=-1)/(
            (lengths[:,:-1]+lengths[:,1:])/2)
        snapshot = dict(node_positions=p-torch.as_tensor(self.origins.copy(),dtype=p.dtype,device=p.device)[:,None],
            velocities=v,tangent=tangent,curvature=curvature,rest_lengths_m=rest,deformation=lengths/rest-1)
        node_mask = torch.ones((n,41),dtype=torch.bool,device=self.device)
        from .tabletop_native_teacher_model import _common
        selected = _common(ledger,mask)
        features = _features(snapshot,selected,node_mask)
        packet = SharedTeacherPacket(dict(common371=ledger,common_valid=mask,rod_nodes=features,
            rod_mask=node_mask,task_context=torch.cat(cv),task_context_valid=torch.cat(cm)),boundary,deepcopy(raw),next(iter(grasps)))
        return self.validate_fresh(packet)
