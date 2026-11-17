"""Opt-in read-only endpoint directors from the already locked RopeBatch API.

This owner never steps, resets, gathers or changes physics. Call only within the
existing physical owner's exclusive read phase, after reset/step has returned.
Clock checks detect drift; they are not a native mutex or substep-history proof.
"""
from copy import deepcopy
import hashlib
import math
import weakref
from pathlib import Path
import numpy as np

from .tabletop_bimanual_table_contacts import _clock as shared_clock
from .tabletop_cosserat_material_marker_reference import source_contract
from .tabletop_dual_pinch_builder import BACKEND, BACKEND_SHA
from .tabletop_deformable_solver_clock import solver_clock_matches

VERSION="dh116-material-director-endpoint-readback-v1"
BUILD=BACKEND.parents[2]
SOURCE_SHA={
    "csrc/batch.cu":"899e95da5aab86a40442be8effbcadbd6d2923a27b9554475d38f6fef631d868",
    "csrc/batch.h":"7b057d1e2bbfaa812f4a8c0dd201229e54a24e7a2d6a2dbfb017126a7f1d7010",
}
PACKAGE_SHA="adc864e6bed274b88a0b15f44dcf8c2628cae5d6913518d147d5d8b4ac0d8419"
DT=1/480


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_basis():
    proof=source_contract(BUILD/"source")
    paths={BUILD/"source"/name:sha for name,sha in SOURCE_SHA.items()}
    paths[BUILD/"python/ropesim/__init__.py"]=PACKAGE_SHA
    paths[BACKEND]=BACKEND_SHA
    for path,expected in paths.items():
        if _sha(path)!=expected:raise ValueError("locked director API source/binary differs: "+str(path))
    return dict(proof,version=VERSION,additional_sources={str(p):s for p,s in paths.items()},
        backend_path=str(BACKEND),backend_sha256=BACKEND_SHA,
        quaternion_api="RopeBatch.quaternions",positions_api="RopeBatch.positions",
        quaternion_order="xyzw",native_rest_lengths_getter_available=False,
        native_qualification=False)


def _array(value,shape,label):
    array=np.array(value,copy=True)
    if array.shape!=shape or array.dtype.kind!="f" or not np.isfinite(array).all():
        raise ValueError("finite "+label+" shape required: "+str(shape))
    return array


def _copy_endpoint(rope,device):
    """Use only the installed CUDA-view API, with explicit completion ordering."""
    p=q=p_view=q_view=None
    try:
        import torch
        dev=torch.device(device)
        if dev.type!="cuda" or dev.index is None:
            raise ValueError("explicit CUDA device index required")
        if torch.cuda.current_stream(dev)!=torch.cuda.default_stream(dev):
            raise ValueError("locked RopeBatch requires the Torch default CUDA stream")
        # step_free checks solver errors before its final asynchronous gather.
        # Complete that gather without advancing either physics clock.
        torch.cuda.synchronize(dev)
        p_view=rope.positions()
        q_view=rope.quaternions()
        p=torch.as_tensor(p_view)
        q=torch.as_tensor(q_view)
        if any(t.device!=dev or t.dtype!=torch.float32 for t in (p,q)):
            raise ValueError("actual float32 CUDA views on the declared device required")
        nodes=p.detach().clone().cpu().numpy().copy()
        quaternions=q.detach().clone().cpu().numpy().copy()
        return nodes,quaternions
    finally:
        rope=p=q=p_view=q_view=None


class MaterialDirectorReadback:
    """One 41-node free rod, reset then every completed 12-substep SDK interval."""

    def __init__(self,constructor_nodes_m,*,device):
        self.sources=source_basis()
        self._nodes=_array(np.asarray(constructor_nodes_m,dtype=np.float32),(41,3),"constructor nodes")
        self._rest=np.linalg.norm(np.diff(self._nodes,axis=0),axis=1)
        if np.any(self._rest<=0):raise ValueError("positive constructor material segments required")
        if not isinstance(device,str) or not device.startswith("cuda:") or not device[5:].isdigit():
            raise ValueError("explicit CUDA device index required")
        self.device=device
        self._last=self._reset_identity=self._material_uid=self._rope_ref=None
        self._seen_episodes=set();self._seen_material=set()
        self._failed=True

    def _read(self,row,*,rope,backend,sim,read_clock):
        try:
            expected=shared_clock(lambda:row)
            if self._rope_ref is None:
                self._rope_ref=weakref.ref(rope)
            elif self._rope_ref() is not rope:
                raise ValueError("same actual RopeBatch owner instance required")
            nodes=_array(row["node_positions"],(1,41,3),"completed rod nodes")
            rest=_array(row["rest_lengths_m"],(40,),"constructor material rest")
            if not np.array_equal(rest,self._rest):
                raise ValueError("constructor material reference changed")
            if (Path(backend.__file__).resolve()!=BACKEND.resolve() or _sha(BACKEND)!=BACKEND_SHA
                    or type(rope) is not backend.RopeBatch or rope.num_envs!=1 or rope.verts_per_rope!=41):
                raise ValueError("exact locked RopeBatch binary/type/shape required")
            def boundary():
                clock=shared_clock(read_clock)
                if (clock!=expected or sim.current_time_step_index!=expected["physics_step"]
                        or not math.isclose(float(sim.current_time),expected["timestamp_s"],rel_tol=0,abs_tol=2e-7)
                        or float(rope.time)!=expected["solver_time_s"] or rope.last_error!=0):
                    raise ValueError("actual SDK/solver clock or error differs from completed row")
                return clock,float(sim.current_time),float(rope.time)
            before=boundary()
            actual,directors=_copy_endpoint(rope,self.device)
            after=boundary()
            if before!=after:raise ValueError("clock changed during director readback")
            actual=_array(actual,(1,41,3),"actual endpoint positions")
            directors=_array(directors,(1,41,4),"actual XYZW directors plus padding")
            if not np.array_equal(actual,nodes):
                raise ValueError("actual endpoint positions differ from same-clock rod snapshot")
            if not np.allclose(np.linalg.norm(directors[:,:40],axis=-1),1.,atol=1e-4,rtol=0):
                raise ValueError("real segment directors must be unit quaternions")
            return dict(version=VERSION,identity=expected,sources=deepcopy(self.sources),source_instance_id=id(rope),
                sdk_time_s=before[1],solver_time_readback_s=before[2],
                node_positions_m=actual,quaternions_xyzw_with_padding=directors,
                segment_quaternions_xyzw=directors[:,:40].copy(),padding_quaternion_xyzw=directors[:,40].copy(),
                real_segment_indices=list(range(40)),padding_index=40,
                constructor_rest_lengths_m=rest,native_rest_lengths_readback_m=None,
                rest_source="FreeRodRuntime constructor float32 node differences; no RopeBatch rest getter",
                boundary="completed_physics_endpoint",solver_substeps_since_reset=expected["rod_tick"],
                material_reference_uid=self._material_uid,
                synchronization="explicit torch.cuda.synchronize(device), default stream, owned blocking CPU copies",
                declared_device=self.device,exclusive_owner_read_phase_required=True,
                substep_director_history_available=False,contact_pair_substep_alignment_verified=False,
                continuous_contact_s_m=None,r0_tangential_slip_m=None,physical_error_bound_m=None,
                native_qualification=False,teacher_admission=False,new_sdk_steps=0,new_solver_steps=0)
        finally:
            self=row=rope=backend=sim=read_clock=None

    def capture_reset(self,row,*,material_reference_uid,rope,backend,sim,read_clock):
        """Read an already completed actual reset; this method does not reset."""
        self._failed=True
        try:
            identity=shared_clock(lambda:row)
            if (identity["rod_tick"]!=0 or row["contact_valid"] is not False
                    or not isinstance(material_reference_uid,str) or not material_reference_uid
                    or material_reference_uid in self._seen_material
                    or identity["episode_uid"] in self._seen_episodes):
                raise ValueError("fresh actual reset and material reference required")
            if self._reset_identity is not None:
                prior=self._reset_identity
                history=[prior]+([self._last] if self._last is not None else [])
                if (any(identity[k]!=prior[k] for k in ("run_uuid","env_id","rod_uid"))
                        or identity["reset_id"]!=prior["reset_id"]+1
                        or any(identity[key]<old[key] for old in history
                               for key in ("physics_step","solver_time_s"))):
                    raise ValueError("fresh monotonic reset on the same run/env/rod required")
            if identity["env_id"]!=0 or identity["reset_id"]<1:
                raise ValueError("single environment zero and actual positive reset ID required")
            self._seen_episodes.add(identity["episode_uid"]);self._seen_material.add(material_reference_uid)
            self._material_uid=material_reference_uid
            self._reset_identity=deepcopy(identity)  # Preserve an attempted real reset even if its read fails.
            packet=self._read(row,rope=rope,backend=backend,sim=sim,read_clock=read_clock)
            if not np.array_equal(packet["node_positions_m"][0],self._nodes):
                raise ValueError("actual reset nodes differ from frozen constructor nodes")
            self._reset_identity=deepcopy(identity);self._last=deepcopy(identity);self._failed=False
            packet["boundary"]="completed_reset_endpoint"
            return packet
        finally:
            self=row=rope=backend=sim=read_clock=None

    def capture(self,row,*,rope,backend,sim,read_clock):
        try:
            if self._failed or self._last is None:
                raise RuntimeError("fresh reset readback required before another endpoint")
            self._failed=True
            identity=shared_clock(lambda:row);prior=self._last
            if (any(identity[k]!=prior[k] for k in ("run_uuid","env_id","episode_uid","reset_id","rod_uid"))
                    or identity["physics_step"]!=prior["physics_step"]+1
                    or identity["rod_tick"]!=prior["rod_tick"]+12
                    or row["contact_valid"] is not True
                    or not solver_clock_matches(prior["solver_time_s"],identity["solver_time_s"],steps=1)):
                raise ValueError("one completed SDK/12-substep interval with unchanged identity required")
            packet=self._read(row,rope=rope,backend=backend,sim=sim,read_clock=read_clock)
            self._last=deepcopy(identity);self._failed=False
            return packet
        finally:
            self=row=rope=backend=sim=read_clock=None
