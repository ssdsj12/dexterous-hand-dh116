"""Small opt-in actual getter/Reset hooks; complete pair mathematics is post-exit."""
from copy import deepcopy
import json,math,os,sys
from pathlib import Path
import numpy as np
from . import tabletop_mesh_observation_source as source
from . import tabletop_deformx_pair_observations_v2 as pairs
from . import tabletop_mesh_reset_order as rebuild
from .tabletop_mesh_reset_readback import MeasuredMeshReset
from .tabletop_bimanual_table_contacts import IDENTITY,_clock
from .tabletop_deformable_r0_replay import encode_owned
from .tabletop_dual_hand_reset import DualHandResetTransaction

VERSION="dh116-mesh-observation-getter-evidence-v1"
RESET_FILE="mesh_reset_state.json"
REASON="mesh_pair_v2_source_invalid"

def _require(ok,message):
    if not ok:raise ValueError(message)

def _boundary(runtime,sim):
    try:
        return dict(sdk_step=sim.current_time_step_index,sdk_time_s=float(sim.current_time),
            solver_time_s=float(runtime.rope.time),runtime_clock=_clock(runtime.clock),
            source_instance_id=id(runtime.rope))
    finally:runtime=sim=None

def _same_loaded_owner(runtime):
    backend=package=None
    try:
        # Source bytes are verified by the outer source_scope and reset. Do not
        # hash 144 files per physical row or import an alternative extension.
        from . import tabletop_native_parameter_endpoints as reader
        backend=sys.modules.get("ropesim._ropesim");package=sys.modules.get("ropesim")
        _require(reader.BACKEND==source.BACKEND and reader.BACKEND_SHA==source.BACKEND_SHA
            and reader.VERSION==source.PARAMETER_VERSION
            and backend is not None and getattr(package,"_ropesim",None) is backend
            and Path(backend.__file__).resolve()==source.BACKEND
            and type(runtime.rope) is backend.RopeBatch,"active exact observation backend owner required")
    finally:runtime=backend=package=None

def _light_raw(raw,clock):
    _require(isinstance(raw,dict) and set(raw)==pairs.RAW_KEYS,"complete actual v2 raw fields required")
    _require(raw["schema"]=="dh116.sdf_pair_samples.v2"
        and type(raw["attempted"]) is int and 0<=raw["attempted"]<=5412
        and type(raw["capacity"]) is int and raw["capacity"]==5412
        and type(raw["substeps"]) is int and raw["substeps"]==12
        and type(raw["mesh_count"]) is int and raw["mesh_count"]==11
        and raw["overflow"] is False,"bounded actual v2 counts required")
    _require(raw["owner_time_s_before"]==raw["owner_time_s_after"]==clock["solver_time_s"]
        and type(raw["owner_time_s_before"]) in (int,float)
        and type(raw["owner_time_s_after"]) in (int,float)
        and type(raw["last_error"]) is int and raw["last_error"]==0,
        "actual v2 owner time/error differs")
    n=raw["attempted"]
    shapes={"indices":((n,3),np.int32),"vectors":((n,12),np.float32),
        "mesh_positions_world_m":((n,3),np.float32),"mesh_quaternions_xyzw":((n,4),np.float32),
        "surface_gap_m":((n,),np.float32),"normal_force_n":((n,),np.float32),
        "node_forces_on_rod_world_n":((41,3),np.float32),
        "mesh_wrenches_on_mesh_world":((11,6),np.float32)}
    for name,(shape,dtype) in shapes.items():
        a=np.asarray(raw[name])
        _require(a.shape==shape and a.dtype==dtype and np.isfinite(a).all(),
            "actual v2 shape/dtype/finite differs: "+name)

def validate_pair_boundary(packet,exchange):
    _require(isinstance(packet,dict) and set(packet)=={"version","expected_clock","boundary_before",
        "boundary_after","raw","capture_error","new_sdk_steps","new_solver_steps","native_qualification"},
        "exact pair getter envelope required")
    expected={k:exchange["rod"][k] for k in IDENTITY};_clock(lambda:expected)
    _require(packet["version"]==VERSION and packet["expected_clock"]==expected
        and type(packet["new_sdk_steps"]) is int and packet["new_sdk_steps"]==0
        and type(packet["new_solver_steps"]) is int and packet["new_solver_steps"]==0
        and packet["native_qualification"] is False,"same completed row and no added steps required")
    error=packet["capture_error"]
    _require(error is None or isinstance(error,str) and bool(error.strip()),"explicit v2 capture error required")
    if error is not None:return dict(source_complete=False,capture_error=error)
    b,a=packet["boundary_before"],packet["boundary_after"]
    _require(isinstance(b,dict) and b==a and set(b)=={
        "sdk_step","sdk_time_s","solver_time_s","runtime_clock","source_instance_id"},
        "identical actual getter boundaries required")
    _require(type(b["sdk_step"]) is int and b["sdk_step"]==expected["physics_step"]
        and type(b["sdk_time_s"]) in (int,float) and math.isfinite(b["sdk_time_s"])
        and math.isclose(b["sdk_time_s"],expected["timestamp_s"],abs_tol=2e-7,rel_tol=0)
        and b["solver_time_s"]==expected["solver_time_s"] and b["runtime_clock"]==expected
        and type(b["source_instance_id"]) is int and b["source_instance_id"]>0,
        "v2 actual SDK/solver/owner boundary differs")
    anchor=exchange["material_director"]
    _require(isinstance(anchor,dict) and exchange.get("director_capture_error") is None
        and anchor["version"]==source.PARAMETER_VERSION and anchor["identity"]==expected
        and anchor["source_instance_id"]==b["source_instance_id"]
        and anchor["sources"]["backend_sha256"]==source.BACKEND_SHA,"v2 and parameter owner/source differ")
    _light_raw(packet["raw"],expected)
    return dict(source_complete=True,capture_error=None)

def capture_pair_v2(runtime,*,sim,exchange):
    packet=dict(version=VERSION,expected_clock={k:exchange["rod"][k] for k in IDENTITY},
        boundary_before=None,boundary_after=None,raw=None,capture_error=None,
        new_sdk_steps=0,new_solver_steps=0,native_qualification=False)
    rope=None
    try:
        try:
            _same_loaded_owner(runtime)
            rope=runtime.rope
            packet["boundary_before"]=_boundary(runtime,sim)
            packet["raw"]=deepcopy(rope.contact_pair_samples_v2())  # One actual getter call.
            packet["boundary_after"]=_boundary(runtime,sim)
            _same_loaded_owner(runtime)
            _require(runtime.rope is rope,"v2 owner changed during capture")
            validate_pair_boundary(packet,exchange)
        except Exception as error:
            packet["capture_error"]=type(error).__name__+": "+str(error)
        return packet
    finally:runtime=sim=exchange=rope=None

class PairV2EvidenceMixin:
    def _step(self,action,substep):
        row=None
        try:
            row=super()._step(action,substep)
            row["mesh_observation_prior_guard_failure"]=row["guard_failure"]
            try:
                row["mesh_pair_v2"]=capture_pair_v2(self._rod,sim=self._sim,exchange=row)
                checked=validate_pair_boundary(row["mesh_pair_v2"],row)
            except Exception as error:
                row["mesh_pair_v2"]=dict(version=VERSION,
                    expected_clock={k:row["rod"][k] for k in IDENTITY},
                    boundary_before=None,boundary_after=None,raw=None,
                    capture_error=type(error).__name__+": "+str(error),
                    new_sdk_steps=0,new_solver_steps=0,native_qualification=False)
                checked=dict(source_complete=False)
            row["guard_failure"]=row["mesh_observation_prior_guard_failure"] or (
                None if checked["source_complete"] else REASON)
            return row
        finally:self=action=row=None

def _write(path,value):
    with Path(path).open("x") as stream:
        json.dump(encode_owned(value),stream,allow_nan=False,separators=(",",":"))
        stream.write("\n");stream.flush();os.fsync(stream.fileno())

def make_reset_type(project,prepared,output):
    """Factory closure retains source dictionaries/paths only, never actual owners."""
    project=Path(project);output=Path(output);prepared=deepcopy(prepared)
    class BoundMeasuredReset(DualHandResetTransaction):
        def run(self,sim,rod,bindings,*,owner_resets):
            plan=transaction=sdf=emit=None;written=False
            try:
                if self.status!="new":raise RuntimeError("one reset transaction only")
                self.status="mutating"
                sdf=sys.modules.get("ropesim.sdf")
                _require(sdf is not None,"actual loaded original SDF module required")
                plan=rebuild.prepare_mesh_reset_plan(project,prepared,rod,sdf_module=sdf)
                transaction=MeasuredMeshReset(self.seed,episode_uid=self.episode_uid,mesh_plan=plan)
                def emit(value):
                    nonlocal written
                    _write(output/RESET_FILE,dict(version=VERSION,mesh_plan=plan,readback=value,
                        source_profile=plan["source_basis"],native_qualification=False,teacher_admission=False))
                    written=True
                result=transaction.run(sim,rod,bindings,owner_resets=owner_resets,record_readback=emit)
                self.status="complete"
                return result
            except Exception as error:
                self.status="failed";rod._needs_reset=True;rod._base.last=None
                if not written and not (output/RESET_FILE).exists():
                    _write(output/RESET_FILE,dict(version=VERSION,mesh_plan=plan,readback=None,
                        source_profile=None if plan is None else plan["source_basis"],
                        capture_error=type(error).__name__+": "+str(error),
                        native_qualification=False,teacher_admission=False))
                raise
            finally:self=sim=rod=bindings=owner_resets=plan=transaction=sdf=emit=None
    return BoundMeasuredReset
