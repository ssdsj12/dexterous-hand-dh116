"""Explicit per-instance typed numeric Reset reader; no module monkeypatch."""
from copy import deepcopy
import math
from . import tabletop_mesh_observation_source as profile
from . import tabletop_mesh_reset_order as order
from .tabletop_mesh_reset_numeric_state import (
    _require, _numeric, _validate_state as _legacy_validate_state, _copy_velocity, state_differences, MAX_SAMPLES, SHAPES)
from .tabletop_material_director_readback import _copy_endpoint
from .tabletop_deformable_r0_replay import decode_owned
from .tabletop_native_training_clock import read_clock_v2
from .tabletop_dual_hand_reset_v2 import verify_source_lineage, _validate_clock_packet

VERSION = "dh116-mesh-reset-numeric-state-v2"
source_basis = profile.source_basis
verify_runtime_source = profile.verify_runtime

def _finite_number(value,label):
    try:
        valid=type(value) in (int,float) and math.isfinite(value)
    except OverflowError:
        valid=False
    _require(valid,"finite primitive numeric "+label+" required; bool is not a measurement")
    return value


def _validate_state(state,time_s):
    _finite_number(time_s,"solver time")
    _require(type(state) is dict,"owned typed numeric state required")
    parameters=state["parameters"]
    _require(type(parameters) is dict,"owned numeric parameters required")
    for key in ("time_s","contact_radius_m","contact_mu_static","contact_mu_kinetic",
            "contact_v_slip_m_s","contact_kappa"):
        _finite_number(parameters[key],"parameter "+key)
    return _legacy_validate_state(state,time_s)


def capture_numeric_state_v2(rope,sim,rod,*,clock_policy):
    """parameter_snapshot synchronizes the solver; endpoint helper synchronizes CUDA gather."""
    state={}
    sample=dict(clock_before=None,clock_after=None,source_instance_id=None,
        solver_time_before=None,solver_time_after=None,state=None,capture_error=None)
    try:
        verify_runtime_source(rod)
        sample["clock_before"]=read_clock_v2(sim,rod,policy=clock_policy)
        sample["source_instance_id"]=id(rope)
        _require(rope is rod.rope and rope.num_envs==1 and rope.verts_per_rope==41,
                 "same actual 41-node owner required")
        sample["solver_time_before"]=_finite_number(rope.time,"actual solver time")
        state["parameters"]=deepcopy(rope.parameter_snapshot())
        state["last_error"]=rope.last_error
        # Reuse the existing CAI-aware reader, not np.array(CudaView).
        positions,quaternions=_copy_endpoint(rope,str(sim.device))
        state["positions"]=_numeric(positions,SHAPES["positions"],"positions")
        state["quaternions"]=_numeric(quaternions,SHAPES["quaternions"],"quaternions")
        state["velocities"]=_copy_velocity(rope,str(sim.device))
        sample["clock_after"]=read_clock_v2(sim,rod,policy=clock_policy)
        sample["solver_time_after"]=_finite_number(rope.time,"actual solver time")
        _require(sample["clock_before"]==sample["clock_after"]
            and sample["solver_time_before"]==sample["solver_time_after"]
            ==sample["clock_before"]["rod"]["solver_time_s"],"actual clock changed during numeric reset capture")
        _validate_state(state,sample["solver_time_after"])
        sample["state"]=deepcopy(state)
        return sample
    except Exception as error:
        sample["capture_error"]=type(error).__name__+": "+str(error)
        # Only bounded, already owned numeric leaves survive a failed capture.
        sample["partial_fields"]=list(state)
        return sample
    finally:rope=sim=rod=state=None

def _finalize(audit):
    samples=audit["samples"]
    audit["capture_errors"]=[s["capture_error"] for s in samples if s["capture_error"] is not None]
    audit["complete_samples"]=sum(s["state"] is not None for s in samples)
    complete=len(samples)==2 and audit["complete_samples"]==2 and not audit["capture_errors"]
    audit["field_differences"]=state_differences(samples[0]["state"],samples[1]["state"]) if complete else None
    audit["byte_equal"]=(all(v["byte_equal"] for v in audit["field_differences"].values()) if complete else None)
    return audit

def validate_numeric_sample_v2(sample, *, clock_policy):
    _require(type(sample) is dict and set(sample)=={
        "clock_before","clock_after","source_instance_id","solver_time_before",
        "solver_time_after","state","capture_error"},"complete typed numeric sample required")
    _require(sample["capture_error"] is None,"numeric capture failed: "+str(sample["capture_error"]))
    _validate_clock_packet(sample["clock_before"],clock_policy)
    _validate_clock_packet(sample["clock_after"],clock_policy)
    _require(sample["clock_before"]==sample["clock_after"],"numeric capture actual clock changed")
    _require(type(sample["source_instance_id"]) is int and sample["source_instance_id"]>0,
        "actual numeric owner identity required")
    for key in ("solver_time_before","solver_time_after"):
        _finite_number(sample[key],key)
        _require(sample[key]>=0,"nonnegative actual solver time required")
    _require(sample["solver_time_before"]==sample["solver_time_after"]
        ==sample["clock_before"]["rod"]["solver_time_s"],"numeric solver time changed")
    _validate_state(sample["state"],sample["solver_time_after"])
    return sample["state"]

def new_numeric_audit_v2(*, clock_policy):
    return dict(version=VERSION,sources=source_basis(),source_lineage=verify_source_lineage(),
        clock_policy_sha256=clock_policy.sha256,reset_epoch=None,samples=[],
        capture_errors=[],complete_samples=0,field_differences=None,byte_equal=None,
        operation_completed=False,operation_error=None,
        state_reader="existing_CAI_tensor_clone_blocking_CPU_copy",quaternion_order="xyzw",
        segment_director_count=40,padding_quaternion_count=1,maximum_samples=2,
        new_sdk_steps=None,new_solver_steps=None,native_qualification=False,teacher_admission=False)

def validate_numeric_reset_audit_v2(audit,*,reset_receipt,clock_policy):
    """Disk replay binds typed state to the original successful reset and mesh readback."""
    value=decode_owned(audit);reset=decode_owned(reset_receipt)
    _require(type(value) is dict and set(value)=={
        "version","sources","source_lineage","clock_policy_sha256","reset_epoch",
        "samples","capture_errors","complete_samples","field_differences","byte_equal",
        "operation_completed","operation_error","state_reader","quaternion_order",
        "segment_director_count","padding_quaternion_count","maximum_samples",
        "new_sdk_steps","new_solver_steps","native_qualification","teacher_admission"},
        "exact v2 numeric audit fields required")
    _require(value["state_reader"] in ("existing_CAI_tensor_clone_blocking_CPU_copy",
        "explicit_instance_state_reader_unqualified"),"explicit numeric reader provenance required")
    _require(value["version"]==VERSION and value["sources"]==source_basis(),"numeric reset source profile differs")
    for key,expected in (("maximum_samples",2),("complete_samples",2),
            ("segment_director_count",40),("padding_quaternion_count",1),
            ("new_sdk_steps",0),("new_solver_steps",0)):
        _require(type(value[key]) is int and value[key]==expected,"exact integer numeric "+key+" required")
    _require(type(reset["reset_epoch"]) is int and reset["reset_epoch"]>=1,
        "exact reset epoch identity required")
    for key in ("new_sdk_steps","new_solver_steps"):
        _require(type(reset[key]) is int and reset[key]==0,"exact zero-step reset counter required")
    _require(value["operation_completed"] is True and value["operation_error"] is None
        and value["native_qualification"] is False and value["teacher_admission"] is False
        and type(value["new_sdk_steps"]) is int and value["new_sdk_steps"]==0
        and type(value["new_solver_steps"]) is int and value["new_solver_steps"]==0
        and value["maximum_samples"]==2 and value["quaternion_order"]=="xyzw"
        and value["segment_director_count"]==40 and value["padding_quaternion_count"]==1,
        "completed unqualified zero-step numeric reset evidence required")
    _require(value["clock_policy_sha256"]==clock_policy.sha256
        ==reset["clock_policy_sha256"] and value["source_lineage"]==verify_source_lineage()
        ==reset["source_lineage"],"numeric v2 clock/source identity differs")
    _require(type(value["reset_epoch"]) is int and value["reset_epoch"]==reset["reset_epoch"]
        ==reset["after_clock"]["reset_id"],"numeric v2 reset epoch differs")
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
        validate_numeric_sample_v2(sample,clock_policy=clock_policy)
        _require(order._digest(sample["state"])==order._digest(mesh[name]),"numeric state differs from original guard receipt")
    _require(samples[0]["clock_before"]["rod"]["clock_valid"] is True
        and samples[1]["clock_before"]["rod"]["clock_valid"] is False,"actual mutation validity transition required")
    normalized=deepcopy(samples[1]["clock_before"]);normalized["rod"]["clock_valid"]=True
    _require(normalized==samples[0]["clock_before"],"mesh reset advanced clock or changed identity")
    rebuilt=_finalize(deepcopy(value))
    for key in ("capture_errors","complete_samples","field_differences","byte_equal"):
        _require(order._digest(rebuilt[key])==order._digest(value[key]),"numeric reset difference summary differs")
    _require(value["byte_equal"] is True,"actual typed reset state changed")
    return dict(version=VERSION,reset_epoch=value["reset_epoch"],clock_policy_sha256=clock_policy.sha256,numeric_state_unchanged=True,actual_samples=2,
        source_instance_id=samples[0]["source_instance_id"],new_sdk_steps=0,new_solver_steps=0,
        native_qualification=False,teacher_admission=False)
