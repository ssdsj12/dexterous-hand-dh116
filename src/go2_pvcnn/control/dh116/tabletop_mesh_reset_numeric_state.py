"""Opt-in numeric reset state copies; frozen v1 guard and recipes stay unchanged."""
from contextlib import contextmanager
from copy import deepcopy
import math
import numpy as np
from . import tabletop_mesh_reset_order as order
from . import tabletop_mesh_observation_source as profile
from .tabletop_material_director_readback import _copy_endpoint
from .tabletop_deformx_parameter_snapshot import validate_snapshot
from .tabletop_deformable_r0_replay import decode_owned
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

VERSION="dh116-mesh-reset-numeric-state-v1"
ORIGINAL_STATE=order._state
STATE_FIELDS={"parameters","last_error","positions","velocities","quaternions"}
SHAPES={"positions":(1,41,3),"velocities":(1,41,3),"quaternions":(1,41,4)}
MAX_SAMPLES=2
MAX_LEAVES=64
MAX_ARRAY_ELEMENTS=1024

def _require(condition,message):
    if not condition:raise ValueError(message)

def _numeric(value,shape,name):
    array=np.array(value,copy=True)
    _require(array.shape==shape and array.dtype==np.float32 and np.isfinite(array).all(),
             "finite owned float32 "+name+" "+str(shape)+" required")
    return array

def _leaves(value,prefix=""):
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            _require(isinstance(key,str),"string numeric-state field name required")
            result.update(_leaves(item,prefix+"."+key if prefix else key))
        _require(len(result)<=MAX_LEAVES,"bounded numeric-state field count required")
        return result
    if isinstance(value,np.ndarray):
        _require(value.size<=MAX_ARRAY_ELEMENTS and value.dtype==np.float32 and np.isfinite(value).all(),
                 "bounded finite float32 state array required")
    else:
        _require(type(value) in (str,bool,int,float) and (not isinstance(value,str) or len(value)<=256)
            and (type(value) is not float or math.isfinite(value)),"owned scalar state metadata required")
    return {prefix:value}

def _validate_state(state,time_s):
    _require(isinstance(state,dict) and set(state)==STATE_FIELDS,"exact numeric reset state fields required")
    for name,shape in SHAPES.items():_numeric(state[name],shape,name)
    _require(type(state["last_error"]) is int and state["last_error"]==0,"explicit actual solver error zero required")
    validate_snapshot(state["parameters"],expected_num_envs=1,expected_nodes=41,expected_time_s=time_s)
    _leaves(state)
    return state

def state_differences(before,after):
    """Exact bitwise guard plus explanatory numeric differences, never a tolerance."""
    a,b=_leaves(before),_leaves(after)
    _require(set(a)==set(b),"same invariant fields required")
    result={}
    for name in a:
        x,y=a[name],b[name]
        byte_equal=order._digest(x)==order._digest(y)
        numeric=None;maximum=None;signed_zero_only=False;changed=None
        if isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
            _require(x.shape==y.shape and x.dtype==y.dtype,"same numeric field shape/dtype required")
            numeric=bool(np.array_equal(x,y))
            delta=np.abs(x.astype(np.float64)-y.astype(np.float64))
            maximum=float(delta.max()) if delta.size else 0.
            changed=int(np.count_nonzero(x!=y))
            signed_zero_only=bool(numeric and not byte_equal and
                np.all((x==0)|(x.view(np.uint32)==y.view(np.uint32))))
        elif type(x) in (int,float) and type(y) in (int,float):
            numeric=bool(x==y);maximum=abs(float(x)-float(y));changed=int(not numeric)
            signed_zero_only=bool(numeric and not byte_equal and x==0 and y==0)
        result[name]=dict(byte_equal=byte_equal,numeric_equal=numeric,
            max_abs_difference=maximum,numerically_changed_elements=changed,signed_zero_only=signed_zero_only,
            before=order._encoded(x),after=order._encoded(y))
    return result

def _copy_velocity(rope,device):
    view=tensor=None
    try:
        import torch
        dev=torch.device(device)
        _require(dev.type=="cuda" and dev.index is not None,"explicit CUDA device index required")
        _require(torch.cuda.current_stream(dev)==torch.cuda.default_stream(dev),"default CUDA stream required")
        view=rope.velocities()
        tensor=torch.as_tensor(view)
        _require(tensor.device==dev and tensor.dtype==torch.float32,"actual float32 velocity CUDA view required")
        array=tensor.detach().clone().cpu().numpy().copy()
        _require(torch.cuda.current_stream(dev)==torch.cuda.default_stream(dev),"default CUDA stream changed")
        return _numeric(array,SHAPES["velocities"],"velocities")
    finally:rope=view=tensor=None

def _capture(rope,sim,rod,sample):
    """parameter_snapshot synchronizes the solver; endpoint helper synchronizes CUDA gather."""
    state={}
    try:
        sample["clock_before"]=order._clock(sim,rod)
        sample["source_instance_id"]=id(rope)
        _require(rope is rod.rope and rope.num_envs==1 and rope.verts_per_rope==41,
                 "same actual 41-node owner required")
        sample["solver_time_before"]=float(rope.time)
        state["parameters"]=deepcopy(rope.parameter_snapshot())
        state["last_error"]=rope.last_error
        # Reuse the existing CAI-aware reader, not np.array(CudaView).
        positions,quaternions=_copy_endpoint(rope,str(sim.device))
        state["positions"]=_numeric(positions,SHAPES["positions"],"positions")
        state["quaternions"]=_numeric(quaternions,SHAPES["quaternions"],"quaternions")
        state["velocities"]=_copy_velocity(rope,str(sim.device))
        sample["clock_after"]=order._clock(sim,rod)
        sample["solver_time_after"]=float(rope.time)
        _require(sample["clock_before"]==sample["clock_after"]
            and sample["solver_time_before"]==sample["solver_time_after"]
            ==sample["clock_before"]["rod"]["solver_time_s"],"actual clock changed during numeric reset capture")
        _validate_state(state,sample["solver_time_after"])
        sample["state"]=deepcopy(state)
        return state
    except Exception as error:
        sample["capture_error"]=type(error).__name__+": "+str(error)
        # Only bounded, already owned numeric leaves survive a failed capture.
        sample["partial_fields"]=list(state)
        raise
    finally:rope=sim=rod=state=None

def _finalize(audit):
    samples=audit["samples"]
    audit["capture_errors"]=[s["capture_error"] for s in samples if s["capture_error"] is not None]
    audit["complete_samples"]=sum(s["state"] is not None for s in samples)
    complete=len(samples)==2 and audit["complete_samples"]==2 and not audit["capture_errors"]
    audit["field_differences"]=state_differences(samples[0]["state"],samples[1]["state"]) if complete else None
    audit["byte_equal"]=(all(v["byte_equal"] for v in audit["field_differences"].values()) if complete else None)
    return audit

@contextmanager
def numeric_reset_state_scope(*,sim,rod,record=None):
    """Bind only reset_order._state, yield owned audit; record once on scope exit."""
    audit=getter=None
    try:
        _require(order._state is ORIGINAL_STATE,"unexpected or nested reset state reader binding")
        profile.verify_runtime(rod)
        basis=profile.source_basis()
        audit=dict(version=VERSION,sources=basis,samples=[],capture_errors=[],complete_samples=0,
            field_differences=None,byte_equal=None,operation_completed=False,operation_error=None,
            state_reader="existing_CAI_tensor_clone_blocking_CPU_copy",quaternion_order="xyzw",
            segment_director_count=40,padding_quaternion_count=1,maximum_samples=MAX_SAMPLES,
            new_sdk_steps=0,new_solver_steps=0,native_qualification=False,teacher_admission=False)
        def getter(rope):
            sample=None
            try:
                _require(len(audit["samples"])<MAX_SAMPLES,"at most two numeric reset state samples")
                sample=dict(clock_before=None,clock_after=None,source_instance_id=None,
                    solver_time_before=None,solver_time_after=None,state=None,capture_error=None)
                audit["samples"].append(sample)
                return _capture(rope,sim,rod,sample)
            finally:rope=sample=None
        try:
            with bindings((order,"_state",getter)):
                yield audit
            _finalize(audit)
            _require(audit["complete_samples"]==2 and audit["byte_equal"] is True,
                     "two unchanged numeric reset samples required")
            audit["operation_completed"]=True
        except Exception as error:
            audit["operation_error"]=type(error).__name__+": "+str(error)
            raise
        finally:
            _finalize(audit)
            if record is not None:record(deepcopy(audit))
    finally:sim=rod=record=getter=None

def validate_numeric_reset_audit(audit,*,reset_receipt):
    """Disk replay binds typed state to the original successful reset and mesh readback."""
    value=decode_owned(audit);reset=decode_owned(reset_receipt)
    _require(value["version"]==VERSION and value["sources"]==profile.source_basis(),"numeric reset source profile differs")
    _require(value["operation_completed"] is True and value["operation_error"] is None
        and value["native_qualification"] is False and value["teacher_admission"] is False
        and type(value["new_sdk_steps"]) is int and value["new_sdk_steps"]==0
        and type(value["new_solver_steps"]) is int and value["new_solver_steps"]==0
        and value["maximum_samples"]==2 and value["quaternion_order"]=="xyzw"
        and value["segment_director_count"]==40 and value["padding_quaternion_count"]==1,
        "completed unqualified zero-step numeric reset evidence required")
    samples=value["samples"]
    _require(len(samples)==2,"two actual state samples required")
    mesh=reset["mesh_reset"];direct=reset["mesh_state_readback"]
    for i,(name,clock) in enumerate((("native_state_before","before_clock"),("native_state_after","mutation_after_clock"))):
        sample=samples[i]
        _require(sample["capture_error"] is None and sample["clock_before"]==sample["clock_after"]==mesh[clock],
                 "numeric sample actual reset clock differs")
        _require(type(sample["source_instance_id"]) is int and sample["source_instance_id"]>0
            and sample["source_instance_id"]==samples[0]["source_instance_id"]
            ==direct["before"]["source_instance_id"]==direct["after"]["source_instance_id"],
            "numeric state actual owner differs")
        _require(sample["solver_time_before"]==sample["solver_time_after"]==mesh[clock]["rod"]["solver_time_s"],
                 "numeric native solver time differs")
        _validate_state(sample["state"],sample["solver_time_after"])
        _require(order._digest(sample["state"])==order._digest(mesh[name]),"numeric state differs from original guard receipt")
    _require(samples[0]["clock_before"]["rod"]["clock_valid"] is True
        and samples[1]["clock_before"]["rod"]["clock_valid"] is False,"actual mutation validity transition required")
    normalized=deepcopy(samples[1]["clock_before"]);normalized["rod"]["clock_valid"]=True
    _require(normalized==samples[0]["clock_before"],"mesh reset advanced clock or changed identity")
    rebuilt=_finalize(deepcopy(value))
    for key in ("capture_errors","complete_samples","field_differences","byte_equal"):
        _require(order._digest(rebuilt[key])==order._digest(value[key]),"numeric reset difference summary differs")
    _require(value["byte_equal"] is True,"actual typed reset state changed")
    return dict(version=VERSION,numeric_state_unchanged=True,actual_samples=2,
        source_instance_id=samples[0]["source_instance_id"],new_sdk_steps=0,new_solver_steps=0,
        native_qualification=False,teacher_admission=False)
