"""Opt-in zero-step reset plus direct mesh-state evidence; no native launch.

All borrowed SDK/solver objects are cleared from exception frames. Returned
packets contain owned arrays/scalars only. An unavailable getter never qualifies.
"""
from copy import deepcopy
import math
import numpy as np
from . import tabletop_mesh_observation_source as profile
from . import tabletop_mesh_reset_order as original
from .tabletop_deformable_r0_replay import decode_owned
from .tabletop_bimanual_table_contacts import IDENTITY, _clock as shared_clock

VERSION="dh116-direct-eleven-mesh-reset-readback-v1"
QUAT_COMPONENT_ATOL=8*float(np.finfo(np.float32).eps)
CAPACITY=41*11*12

def _require(ok,message):
    if not ok:raise ValueError(message)

def _array(value,shape,name):
    a=np.asarray(value)
    _require(a.shape==shape and a.dtype==np.dtype(np.float32) and np.isfinite(a).all(),
        "finite actual "+name+" shape required")
    _require(np.array_equal(a,a.astype(np.float32).astype(a.dtype)),
        "actual float32 "+name+" bytes required")
    return a

def _validate_clock(value):
    _require(isinstance(value,dict) and set(value)=={
        "sdk_physics_step","sdk_timestamp_s","sdk_dt_s","rod"},"exact actual clock fields required")
    step,stamp,dt=value["sdk_physics_step"],value["sdk_timestamp_s"],value["sdk_dt_s"]
    _require(type(step) is int and 0<=step<=240 and type(stamp) in (int,float)
        and math.isfinite(stamp) and type(dt) in (int,float)
        and math.isclose(dt,1/480,abs_tol=1e-12,rel_tol=0),"actual bounded SDK clock required")
    tolerance=step*abs(float(np.float32(dt))-dt)+8*np.finfo(float).eps*max(1.,step*dt)
    _require(tolerance<min(2e-7,dt/2) and abs(stamp-step*dt)<=tolerance,
        "actual SDK timestamp exceeds callback accumulation envelope")
    row=value["rod"]
    _require(isinstance(row,dict) and set(row)==set(IDENTITY) and type(row["clock_valid"]) is bool,
        "exact typed actual rod clock required")
    if row["physics_step"] is None or row["timestamp_s"] is None:
        # Actual constructor clock is not yet registered to an SDK step. Preserve
        # both unavailable fields; never manufacture SDK0 or a valid rod clock.
        _require(row["physics_step"] is None and row["timestamp_s"] is None
            and row["clock_valid"] is False
            and type(row["env_id"]) is int and row["env_id"]>=0
            and all(type(row[k]) is int and row[k]==0 for k in ("reset_id","rod_tick"))
            and all(isinstance(row[k],str) and row[k] for k in ("run_uuid","episode_uid","rod_uid"))
            and type(row["solver_time_s"]) in (int,float) and row["solver_time_s"]==0.,
            "explicit unregistered constructor clock required")
    else:
        checked=deepcopy(row);checked["clock_valid"]=True
        # A later poisoned owner retains its actual, still typed clock.
        shared_clock(lambda:checked)

def validate_capture(packet):
    """Validate one owned same-clock native snapshot, including pre-reset states."""
    _require(set(packet)=={"clock_before","clock_after","source_instance_id","snapshot","capture_error"},
        "exact mesh capture fields required")
    _require(packet["capture_error"] is None,"mesh capture failed: "+str(packet["capture_error"]))
    before,after,raw=packet["clock_before"],packet["clock_after"],packet["snapshot"]
    _validate_clock(before);_validate_clock(after)
    _require(isinstance(before,dict) and before==after,"mesh getter changed actual clock")
    _require(type(packet["source_instance_id"]) is int and packet["source_instance_id"]>0,
        "explicit actual mesh owner identity required")
    expected={"schema","num_envs","mesh_count","current_positions_world_m","current_quaternions_xyzw",
        "target_positions_world_m","target_quaternions_xyzw","solver_positions_world_m",
        "solver_quaternions_xyzw","pending_teleport","device_state_source","pending_set","moving",
        "pair_capacity","pair_ready","flags_source","owner_time_s_before","owner_time_s_after",
        "owner_time_source","last_error"}
    _require(isinstance(raw,dict) and set(raw)==expected,"exact native mesh snapshot fields required")
    _require(raw["schema"]=="dh116.mesh_state_snapshot.v1" and type(raw["num_envs"]) is int
        and raw["num_envs"]==1 and type(raw["mesh_count"]) is int and raw["mesh_count"]==11,
        "actual single-environment eleven slots required")
    _require(raw["device_state_source"]=="cuda_device_copy"
        and raw["flags_source"]=="native_host_sdf_store"
        and raw["owner_time_source"]=="native_host_owner_metadata_at_readback",
        "explicit device state and host metadata sources required")
    for prefix in ("current","target","solver"):
        _array(raw[prefix+"_positions_world_m"],(1,11,3),prefix+" position")
        q=_array(raw[prefix+"_quaternions_xyzw"],(1,11,4),prefix+" XYZW")
        _require(np.allclose(np.linalg.norm(q,axis=-1),1.,atol=2e-6,rtol=0),
            "unit native mesh XYZW required")
    tele=np.asarray(raw["pending_teleport"])
    _require(tele.shape==(1,11) and tele.dtype==np.dtype(np.uint8) and np.isin(tele,[0,1]).all(),
        "eleven native teleport flags required")
    _require(all(type(raw[k]) is bool for k in ("pending_set","moving","pair_ready")),
        "typed native host flags required")
    _require(type(raw["pair_capacity"]) is int and raw["pair_capacity"]==CAPACITY,
        "original bounded pair allocation must survive reset")
    _require(type(raw["last_error"]) is int and raw["last_error"]==0,
        "actual mesh getter solver error is not zero")
    time=before["rod"]["solver_time_s"]
    _require(type(time) in (int,float) and math.isfinite(time)
        and type(raw["owner_time_s_before"]) in (int,float)
        and type(raw["owner_time_s_after"]) in (int,float)
        and raw["owner_time_s_before"]==raw["owner_time_s_after"]==time,
        "exact native owner metadata time differs")
    return raw

def capture_mesh_state(sim,rod):
    """Call only the existing read-only getter; failures preserve any acquired bytes."""
    packet=dict(clock_before=None,clock_after=None,source_instance_id=None,snapshot=None,capture_error=None)
    rope=None
    try:
        try:
            profile.verify_runtime(rod)
            rope=rod.rope
            packet["source_instance_id"]=id(rope)
            packet["clock_before"]=original._clock(sim,rod)
            packet["snapshot"]=deepcopy(rope.mesh_state_snapshot())
            packet["clock_after"]=original._clock(sim,rod)
            _require(rod.rope is rope,"mesh owner changed during readback")
            validate_capture(packet)
        except Exception as error:
            packet["capture_error"]=type(error).__name__+": "+str(error)
        return packet
    finally:sim=rod=rope=None

def validate_reset_readback(packet,*,reset_receipt,mesh_plan):
    """Disk-replayable reconciliation with original source grids and actual hand poses."""
    packet=decode_owned(packet);reset_receipt=decode_owned(reset_receipt);mesh_plan=decode_owned(mesh_plan)
    basis=profile.source_basis()
    original.verify_plan_sources(mesh_plan)
    p=deepcopy(mesh_plan);digest=p.pop("sha256")
    _require(digest==original._digest(p),"mesh plan digest differs")
    _require(set(packet)=={"version","sources","mesh_plan_sha256","mesh_names","before","after",
        "complete","capture_error","new_sdk_steps","new_solver_steps","native_qualification","teacher_admission"},
        "exact reset mesh evidence fields required")
    _require(packet["version"]==VERSION and packet["sources"]==basis
        and mesh_plan["source_basis"]==basis and packet["mesh_plan_sha256"]==digest
        and packet["mesh_names"]==list(original.MESH_NAMES),"mesh reset source/slot identity differs")
    _require(packet["complete"] is True and packet["capture_error"] is None
        and type(packet["new_sdk_steps"]) is int and packet["new_sdk_steps"]==0
        and type(packet["new_solver_steps"]) is int and packet["new_solver_steps"]==0
        and packet["native_qualification"] is False and packet["teacher_admission"] is False,
        "completed zero-step observation without native promotion required")
    before=validate_capture(packet["before"]);after=validate_capture(packet["after"])
    _require(packet["before"]["source_instance_id"]==packet["after"]["source_instance_id"],
        "reset must preserve the same native mesh owner")
    b,a=packet["before"]["clock_before"],packet["after"]["clock_after"]
    _require(b==reset_receipt["before_clock"] and a==reset_receipt["after_actual_clocks"]
        and a==reset_receipt["mesh_reset"]["after_clock"],
        "mesh readback is not the actual original reset boundary")
    original._no_step(b,a)
    _require(all(a["rod"][k]==b["rod"][k] for k in ("run_uuid","env_id","rod_uid"))
        and a["rod"]["reset_id"]==b["rod"]["reset_id"]+1
        and a["rod"]["episode_uid"]!=b["rod"]["episode_uid"]
        and a["rod"]["physics_step"]==a["sdk_physics_step"],"fresh same-owner reset identity required")
    _require(a["rod"]["clock_valid"] is True and a["rod"]["rod_tick"]==0
        and reset_receipt["mesh_reset"]["mutation_after_clock"]["rod"]["clock_valid"] is False,
        "actual poisoned mutation and completed reset clocks required")
    _require(after["pending_set"] is False and after["moving"] is False
        and after["pair_ready"] is False and not np.asarray(after["pending_teleport"]).any(),
        "reset mesh commit left pending motion/teleport/contact capture")
    for prefix in ("target","solver"):
        for suffix in ("positions_world_m","quaternions_xyzw"):
            _require(np.array_equal(after["current_"+suffix],after[prefix+"_"+suffix]),
                "native reset current/target/solver representations disagree")
    _require(tuple(row["name"] for row in mesh_plan["entries"])==original.MESH_NAMES,
        "original mesh order required")
    for slot,entry in enumerate(mesh_plan["entries"]):
        name=entry["name"]
        if name=="table":pos,quat=entry["table_pose"]
        else:
            side,body=name.split(".");hand=reset_receipt["hands"][side]
            names=hand["body_names"];index=names.index(body)
            _require(len(names)==23 and len(set(names))==23
                and mesh_plan["bindings"][name]["body_index"]==index
                and hand["link_paths"][0][index]==mesh_plan["bindings"][name]["prim_path"],
                "actual reset body path and original pad identity differ")
            measured=np.asarray(hand["body_poses_xyzw"])[0,index]
            pos,quat=reset_receipt["pad_poses_xyzw"][name]
            _require(np.array_equal(np.concatenate((pos,quat)),measured),
                "mesh input pose differs from actual post-reset link readback")
        pos=np.asarray(pos,np.float32);q=np.asarray(quat,np.float32)
        _require(pos.shape==(3,) and q.shape==(4,) and np.isfinite(pos).all()
            and np.isfinite(q).all() and np.linalg.norm(q)>0,"finite actual reset XYZW required")
        q=q/np.linalg.norm(q)
        _require(np.array_equal(np.asarray(after["current_positions_world_m"])[0,slot],pos),
            "native current mesh position differs from actual reset link/table")
        _require(np.allclose(np.asarray(after["current_quaternions_xyzw"])[0,slot],q,
            atol=QUAT_COMPONENT_ATOL,rtol=0),"native current XYZW differs from reset normalization")
    return dict(version=VERSION,eleven_reset_meshes_observed=True,pending_reset_targets=False,
        pair_capacity_preserved=True,new_sdk_steps=0,new_solver_steps=0,
        quaternion_component_atol=QUAT_COMPONENT_ATOL,gpu_bitwise_equivalence_claimed=False,
        native_qualification=False,teacher_admission=False,slip_qualified=False)

class MeasuredMeshReset(original.RebuiltMeshReset):
    """Original reset/rebuild, then direct validation; any failure poisons the rod.

    Optional record_readback receives owned progress (including the original
    completed reset if available). The caller must durably preserve failure
    evidence before teardown; this class owns no output directory or SDK owner.
    """
    def run(self,sim,rod,bindings,*,owner_resets,record_readback=None):
        receipt=None;emitted=False;started=False
        try:
            if self.status!="new":raise RuntimeError("reset transaction cannot be reused")
            started=True
            self.completed_reset_receipt=None
            self.readback_receipt=dict(version=VERSION,sources=None,mesh_plan_sha256=self.mesh_plan["sha256"],
                mesh_names=list(original.MESH_NAMES),before=None,after=None,complete=False,capture_error=None,
                new_sdk_steps=0,new_solver_steps=0,native_qualification=False,teacher_admission=False)
            self.readback_receipt["sources"]=profile.source_basis()
            original._plan(self.mesh_plan,rod)  # No hand mutation before source validation.
            self.readback_receipt["before"]=capture_mesh_state(sim,rod)
            validate_capture(self.readback_receipt["before"])
            receipt=super().run(sim,rod,bindings,owner_resets=owner_resets)
            self.completed_reset_receipt=deepcopy(receipt)
            self.readback_receipt["after"]=capture_mesh_state(sim,rod)
            validate_capture(self.readback_receipt["after"])
            self.readback_receipt["complete"]=True
            validate_reset_readback(self.readback_receipt,reset_receipt=receipt,mesh_plan=self.mesh_plan)
            if record_readback is not None:
                emitted=True
                record_readback(dict(deepcopy(self.readback_receipt),
                    completed_reset_receipt=deepcopy(self.completed_reset_receipt)))
            receipt["mesh_state_readback"]=deepcopy(self.readback_receipt)
            return receipt
        except Exception as error:
            if started:
                self.readback_receipt["complete"]=False
                self.readback_receipt["capture_error"]=type(error).__name__+": "+str(error)
                self.status="failed";rod._needs_reset=True;rod._base.last=None
                if record_readback is not None and not emitted:
                    record_readback(dict(deepcopy(self.readback_receipt),
                        completed_reset_receipt=deepcopy(self.completed_reset_receipt)))
            raise
        finally:sim=rod=bindings=owner_resets=record_readback=receipt=None
