"""Independent v2 reset CPU fixtures: no native runtime qualification."""
from copy import deepcopy
from types import SimpleNamespace
import importlib

import numpy as np
import pytest


def test_rebuilt_v2_interface_exists():
    from go2_pvcnn.control.dh116.tabletop_mesh_reset_order_v2 import RebuiltMeshResetV2
    assert callable(RebuiltMeshResetV2)


def test_old_nested_readback_clock_keeps_258_rejection():
    from go2_pvcnn.control.dh116 import tabletop_mesh_reset_readback as old
    packet = dict(sdk_physics_step=258, sdk_timestamp_s=258/480,
        sdk_dt_s=1/480, rod={})
    with pytest.raises(ValueError, match="bounded"):
        old._validate_clock(packet)


def modules():
    stem = "go2_pvcnn.control.dh116.tabletop_"
    return tuple(importlib.import_module(stem+n) for n in (
        "dual_hand_reset_v2", "mesh_reset_order_v2",
        "mesh_reset_readback_v2", "mesh_reset_numeric_state_v2"))


def long_mesh_fixture(monkeypatch):
    """Actual Python hand/rod Reset with explicit cpu_fixture transports."""
    from test_mesh_reset_order import native_semantics
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import lumped_masses, verify_mass_readback
    from go2_pvcnn.control.dh116.tabletop_native_training_clock import ClockPolicyV2
    from go2_pvcnn.control.dh116 import tabletop_mesh_reset_order as old
    dual, order, readback, numeric = modules()
    c = native_semantics(monkeypatch)
    c.fixture_scope = "cpu_fixture"
    nodes = np.zeros((41,3),np.float32)
    nodes[:,0] = np.linspace(0,.5,41); nodes[:,2] = .2
    c.rod._base.nodes = nodes.copy()
    c.rod._base.rest = np.linalg.norm(np.diff(nodes,axis=0),axis=1)
    c.rope.nodes = nodes[None].copy(); c.rope.force = np.zeros_like(c.rope.nodes)
    c.rod._base.mass_readback = verify_mass_readback(c.rope.inv_masses(),
        lumped_masses(nodes,c.rope.density))
    c.rope.num_envs = 1; c.rope.verts_per_rope = 41
    c.v = np.zeros((1,41,3),np.float32)
    c.q = np.zeros((1,41,4),np.float32); c.q[...,3] = 1
    c.rope.velocities = lambda:c.v
    c.rope.quaternions = lambda:c.q
    r = c.rod._base.cfg["rod"]
    rest = np.linalg.norm(np.diff(nodes,axis=0),axis=1)[None].astype(np.float32)
    raw = dict(schema="dh116.native_parameter_snapshot.v1",num_envs=1,nodes_per_env=41,time_s=0.,
        metadata_source="native_host_metadata",segment_source="cuda_device_copy",hardware_measured=False,
        contact_radius_m=r["radius_m"],gravity_world_m_s2=np.array(r["gravity_world_m_s2"],np.float32),
        contact_mu_static=r["mu_s"],contact_mu_kinetic=r["mu_k"],contact_v_slip_m_s=r["v_slip_m_s"],
        contact_kappa=r["kappa"],rest_lengths_m=rest,stored_k_stretch_n_m=rest*r["axial_stiffness_n"],
        rest_quaternion_norm_n_m=4*r["bend_stiffness_n_m2"]/rest,
        stretch_ea_n=np.full((1,40),r["axial_stiffness_n"],np.float32),
        bend_ei_n_m2=np.full((1,40),r["bend_stiffness_n_m2"],np.float32),
        inverse_masses_kg_inv=np.array(c.rope.inv_masses(),np.float32).reshape(1,41))
    c.rope.parameter_snapshot = lambda:deepcopy(dict(raw,time_s=c.rope.time))
    c.sim.device = "cpu_fixture"
    def sdk_step():
        c.sim.current_time_step_index += 1
        c.sim.current_time += float(np.float32(1/480))
    c.sim.step = sdk_step
    c.policy = ClockPolicyV2(max_sdk_step=9600,timestamp_mode="fp32_dt_float64_sum")
    basis = dict(source_tree_sha256="cpu_fixture",source_scope="cpu_fixture")
    entries = []
    for i,(args,kw) in enumerate(c.slots):
        entries.append(dict(name=c.rod.mesh_names[i],values=np.asarray(args[0],np.float32),
            origin=np.asarray(args[1],np.float32),spacing=float(args[2]),mu_s=kw["mu_s"],mu_k=kw["mu_k"],
            table_pose=None if i!=5 else (np.asarray(kw["pos"],np.float32),np.array([0,0,0,1],np.float32))))
    c.mesh_plan = old.seal_plan(dict(version=old.PLAN_VERSION,entries=entries,
        bindings=deepcopy(c.rod.bindings),cfg=deepcopy(c.rod._base.cfg),
        source_basis=basis,source_scope="cpu_fixture"))
    def verify_runtime(rod):
        assert rod is c.rod and c.fixture_scope=="cpu_fixture"
    monkeypatch.setattr(order,"verify_plan_sources",lambda p:None)
    monkeypatch.setattr(order,"verify_runtime_source",verify_runtime)
    for m in (readback,numeric):
        monkeypatch.setattr(m,"verify_runtime_source",verify_runtime)
        monkeypatch.setattr(m,"source_basis",lambda:deepcopy(basis))
    c.copies = []
    def endpoints(rope,device):
        assert rope is c.rope and device=="cpu_fixture"
        c.copies.append("endpoint")
        return rope.positions().copy(), rope.quaternions().copy()
    def velocity(rope,device):
        c.copies.append("velocity")
        return rope.velocities().copy()
    monkeypatch.setattr(numeric,"_copy_endpoint",endpoints)
    monkeypatch.setattr(numeric,"_copy_velocity",velocity)
    c.mesh_epoch = 0
    clear = c.rope.clear_meshes
    def clear_epoch():
        c.mesh_epoch += 1
        clear()
    c.rope.clear_meshes = clear_epoch
    def snapshot():
        def arr(src,k):return np.stack([src[i][k] for i in range(11)])[None].astype(np.float32)
        cur,tgt = c.rope.current,c.rope.targets
        return dict(schema="dh116.mesh_state_snapshot.v1",num_envs=1,mesh_count=11,
            current_positions_world_m=arr(cur,0),current_quaternions_xyzw=arr(cur,1),
            target_positions_world_m=arr(tgt,0),target_quaternions_xyzw=arr(tgt,1),
            solver_positions_world_m=arr(cur,0),solver_quaternions_xyzw=arr(cur,1),
            pending_teleport=np.array([[c.rope.tele[i] for i in range(11)]],np.uint8),
            device_state_source="cuda_device_copy",pending_set=c.rope.pending,moving=False,
            pair_capacity=c.rope.capacity,pair_ready=c.rope.pair_ready,flags_source="native_host_sdf_store",
            owner_time_s_before=c.rope.time,owner_time_s_after=c.rope.time,
            owner_time_source="native_host_owner_metadata_at_readback",last_error=c.rope.last_error)
    c.rope.mesh_state_snapshot = snapshot
    c.raw_parameters = raw
    return c


def execute(c,uid="cpu:reset:1",**kwargs):
    _, _, rb, _ = modules()
    t = rb.MeasuredMeshResetV2(c.seed,episode_uid=uid,mesh_plan=c.mesh_plan,clock_policy=c.policy)
    return t,t.run(c.sim,c.rod,c.bindings,owner_resets=c.owners,**kwargs)


def advance(c,steps):
    for _ in range(steps):
        c.sim.step()
        poses={s+"."+n:(h.link_poses[0,i,:3].numpy().copy(),h.link_poses[0,i,3:].numpy().copy())
            for s,h in zip(("left","right"),c.hands) for i,n in enumerate(c.rod._base.names[:-1])}
        c.rod.step(poses,{n:p[0] for n,p in poses.items()},physics_step=c.sim.current_time_step_index)


@pytest.mark.parametrize("steps",[0,256,2400])
def test_complete_typed_reset_keeps_actual_clock_and_original_slots(monkeypatch,steps):
    c=long_mesh_fixture(monkeypatch);dual,order,rb,num=modules()
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_reset as old_dual
    from go2_pvcnn.control.dh116 import tabletop_mesh_reset_order as old_order
    clock,state=old_dual._clock,old_order._state
    first,_=execute(c)
    advance(c,steps)
    before=(c.sim.current_time_step_index,c.sim.current_time,c.rope.time,c.rope.steps)
    old_epoch=c.mesh_epoch
    txn,receipt=execute(c,"cpu:reset:2")
    assert before==(c.sim.current_time_step_index,c.sim.current_time,c.rope.time,c.rope.steps)
    assert c.mesh_epoch==old_epoch+1 and len(c.rope.slots)==11
    assert receipt["clock_policy_sha256"]==c.policy.sha256
    assert receipt["reset_epoch"]==receipt["after_clock"]["reset_id"]==2
    assert receipt["mesh_reset"]["mutation_after_clock"]["rod"]["clock_valid"] is False
    assert receipt["mesh_reset"]["after_clock"]["rod"]["clock_valid"] is True
    assert receipt["mesh_reset"]["borrowed_mesh_views_invalidated"] is True
    assert receipt["numeric_reset_audit"]["byte_equal"] is True
    assert receipt["numeric_reset_audit"]["complete_samples"]==2
    assert receipt["mesh_state_readback"]["complete"] is True
    assert receipt["new_sdk_steps"]==receipt["new_solver_steps"]==0
    assert c.copies.count("endpoint")==4 and c.copies.count("velocity")==4
    assert old_dual._clock is clock and old_order._state is state
    assert not receipt["native_qualification"] and not receipt["teacher_admission"]
    checked=rb.validate_reset_readback_v2(receipt["mesh_state_readback"],
        reset_receipt=receipt,mesh_plan=c.mesh_plan,clock_policy=c.policy)
    assert checked["eleven_reset_meshes_observed"] and checked["reset_epoch"]==2
    assert num.validate_numeric_reset_audit_v2(receipt["numeric_reset_audit"],
        reset_receipt=receipt,clock_policy=c.policy)["numeric_state_unchanged"]
    assert all(not composer["active"] and not composer["force"].any()
        for row in receipt["hands"].values() for composer in row["composers"].values())
    with pytest.raises(RuntimeError):txn.run(c.sim,c.rod,c.bindings,owner_resets=c.owners)


@pytest.mark.parametrize("fault",["clear","add","slot","positions","velocity","quaternion","signed_zero","sdk","solver","epoch","record"])
def test_partial_failure_is_preserved_and_no_extra_step(monkeypatch,fault):
    c=long_mesh_fixture(monkeypatch);_,order,rb,num=modules()
    original=c.rope.add_mesh
    primary=OSError("primary injected")
    def fail(*a,**kw):
        slot=original(*a,**kw)
        if slot==3:
            if fault=="add":raise primary
            if fault=="slot":return 7
            if fault=="positions":c.rope.nodes[0,0,0]+=.01
            if fault=="velocity":c.v[0,0,0]+=.01
            if fault=="quaternion":c.q[0,0,0]+=.01
            if fault=="signed_zero":c.v[0,0,0]=-0.
            if fault=="sdk":c.sim.step()
            if fault=="solver":c.rope.time+=1/480
            if fault=="epoch":c.rod._base.reset_id+=1
        return slot
    c.rope.add_mesh=fail
    if fault=="clear":c.rope.clear_meshes=lambda:(_ for _ in ()).throw(primary)
    saved=[]
    def record(row):
        saved.append(row)
        if fault in ("record","add","clear"):raise OSError("secondary recorder")
    txn=rb.MeasuredMeshResetV2(c.seed,episode_uid="bad",mesh_plan=c.mesh_plan,clock_policy=c.policy)
    with pytest.raises((ValueError,OSError,RuntimeError)) as caught:
        txn.run(c.sim,c.rod,c.bindings,owner_resets=c.owners,record_readback=record)
    if fault in ("add","clear"):assert caught.value is primary
    assert c.rope.steps==0
    assert c.sim.current_time_step_index==(3 if fault=="sdk" else 2)
    assert txn.readback_receipt["new_sdk_steps"]==(1 if fault=="sdk" else 0)
    assert txn.readback_receipt["failure_clock_progress"]["sdk_step_after"]==c.sim.current_time_step_index
    assert txn.status=="failed" and c.rod._needs_reset
    assert saved and saved[0]["complete"] is False if fault!="record" else saved[0]["complete"] is True
    if fault in ("add","clear"):assert txn.cleanup_failures
    with pytest.raises(RuntimeError,match="reset"):
        c.rod.step({}, {}, physics_step=c.sim.current_time_step_index+1)
    if fault in ("velocity","quaternion","signed_zero","positions"):
        assert txn.numeric_reset_audit["byte_equal"] is False


@pytest.mark.parametrize("owner",["controller","filter","contact","camera_history"])
def test_each_owner_failure_remains_primary_when_failure_recorder_also_throws(monkeypatch,owner):
    c=long_mesh_fixture(monkeypatch);_,_,rb,_=modules()
    primary=RuntimeError("owner "+owner)
    c.owners[owner]=lambda context:(_ for _ in ()).throw(primary)
    t=rb.MeasuredMeshResetV2(c.seed,episode_uid="ownerfail",mesh_plan=c.mesh_plan,clock_policy=c.policy)
    with pytest.raises(RuntimeError) as caught:
        t.run(c.sim,c.rod,c.bindings,owner_resets=c.owners,
            record_readback=lambda row:(_ for _ in ()).throw(OSError("secondary")))
    assert caught.value is primary and t.cleanup_failures
    assert c.sim.current_time_step_index==2 and c.rope.steps==0 and c.rod._needs_reset


@pytest.mark.parametrize("damage",["policy","epoch","clock","state","owner","diff","flag"])
def test_replay_rejects_foreign_policy_epoch_and_typed_payload(monkeypatch,damage):
    c=long_mesh_fixture(monkeypatch);_,_,rb,num=modules()
    _,receipt=execute(c)
    packet=deepcopy(receipt["numeric_reset_audit"])
    if damage=="policy":packet["clock_policy_sha256"]="0"*64
    elif damage=="epoch":packet["reset_epoch"]+=1
    elif damage=="clock":packet["samples"][1]["clock_after"]["sdk_timestamp_s"]+=.01
    elif damage=="state":packet["samples"][1]["state"]["velocities"][0,0,0]=1
    elif damage=="owner":packet["samples"][1]["source_instance_id"]+=1
    elif damage=="diff":packet["field_differences"]["positions"]["max_abs_difference"]=2
    else:packet["native_qualification"]=True
    with pytest.raises(ValueError):
        num.validate_numeric_reset_audit_v2(packet,reset_receipt=receipt,clock_policy=c.policy)


def test_old_replay_validator_rejects_new_schema(monkeypatch):
    c=long_mesh_fixture(monkeypatch);_,receipt=execute(c)
    from go2_pvcnn.control.dh116 import tabletop_mesh_reset_readback as old
    with pytest.raises(ValueError):
        old.validate_capture(receipt["mesh_state_readback"]["after"])


def test_state_reader_is_explicit_and_never_patches_old_module(monkeypatch):
    c=long_mesh_fixture(monkeypatch);_,order,_,num=modules();calls=[]
    from go2_pvcnn.control.dh116 import tabletop_mesh_reset_order as old
    saved=old._state
    def reader(rope,sim,rod,*,clock_policy):
        calls.append(rod.clock()["clock_valid"])
        return num.capture_numeric_state_v2(rope,sim,rod,clock_policy=clock_policy)
    t=order.RebuiltMeshResetV2(c.seed,episode_uid="injected",mesh_plan=c.mesh_plan,
        clock_policy=c.policy,state_reader=reader)
    receipt=t.run(c.sim,c.rod,c.bindings,owner_resets=c.owners)
    assert receipt["numeric_reset_audit"]["state_reader"]=="explicit_instance_state_reader_unqualified"
    assert calls==[True,False] and old._state is saved


def test_source_lock_failure_happens_before_mutation(monkeypatch):
    c=long_mesh_fixture(monkeypatch);dual,order,rb,num=modules()
    primary=ValueError("source changed")
    monkeypatch.setattr(dual,"verify_frozen_source",lambda *a:(_ for _ in ()).throw(primary))
    with pytest.raises(ValueError) as caught:execute(c)
    assert caught.value is primary and c.rope.resets==0 and not c.events


def test_entry_rechecks_source_before_any_capture(monkeypatch):
    c=long_mesh_fixture(monkeypatch);dual,order,rb,num=modules()
    t=rb.MeasuredMeshResetV2(c.seed,episode_uid="late-source",mesh_plan=c.mesh_plan,clock_policy=c.policy)
    def source_fail(*a):raise ValueError("source changed after construction")
    monkeypatch.setattr(dual,"verify_frozen_source",source_fail)
    calls=[]
    c.rope.mesh_state_snapshot=lambda:calls.append("native getter") or {}
    with pytest.raises(ValueError,match="source changed"):
        t.run(c.sim,c.rod,c.bindings,owner_resets=c.owners)
    assert not calls and not c.events


@pytest.mark.parametrize("bad",[None,{},{"state":object()}])
def test_invalid_instance_reader_retains_original_validation_error(monkeypatch,bad):
    c=long_mesh_fixture(monkeypatch);_,order,_,_=modules()
    t=order.RebuiltMeshResetV2(c.seed,episode_uid="bad-reader",mesh_plan=c.mesh_plan,
        clock_policy=c.policy,state_reader=lambda *a,**k:bad)
    with pytest.raises(ValueError,match="complete typed numeric sample"):
        t.run(c.sim,c.rod,c.bindings,owner_resets=c.owners)
    assert t.status=="failed" and c.rod._needs_reset and c.rope.steps==0


def test_interrupt_is_recorded_and_poisoned_without_step(monkeypatch):
    c=long_mesh_fixture(monkeypatch);_,order,rb,num=modules()
    primary=KeyboardInterrupt("fixture stop")
    c.owners["filter"]=lambda x:(_ for _ in ()).throw(primary)
    t=rb.MeasuredMeshResetV2(c.seed,episode_uid="interrupt",mesh_plan=c.mesh_plan,clock_policy=c.policy)
    saved=[]
    with pytest.raises(KeyboardInterrupt) as caught:
        t.run(c.sim,c.rod,c.bindings,owner_resets=c.owners,record_readback=saved.append)
    assert caught.value is primary and saved and "KeyboardInterrupt" in saved[0]["capture_error"]
    assert c.rod._needs_reset and c.rope.steps==0


def test_failed_typed_reader_drops_borrowed_native_arguments(monkeypatch):
    c=long_mesh_fixture(monkeypatch);_,order,rb,num=modules()
    t=order.RebuiltMeshResetV2(c.seed,episode_uid="frames",mesh_plan=c.mesh_plan,
        clock_policy=c.policy,state_reader=lambda *a,**k:None)
    with pytest.raises(ValueError) as caught:t.run(c.sim,c.rod,c.bindings,owner_resets=c.owners)
    frame=caught.value.__traceback__
    while frame:
        if frame.tb_frame.f_code.co_name=="_state" and frame.tb_frame.f_code.co_filename.endswith("_v2.py"):
            for name in ("rope","sim","rod"):
                assert frame.tb_frame.f_locals[name] is None
        frame=frame.tb_next


def test_failed_owner_drops_last_loop_binding_in_v2_frame(monkeypatch):
    c=long_mesh_fixture(monkeypatch);_,order,rb,num=modules()
    c.owners["filter"]=lambda x:(_ for _ in ()).throw(ValueError("frame owner"))
    t=order.RebuiltMeshResetV2(c.seed,episode_uid="owner-frame",mesh_plan=c.mesh_plan,clock_policy=c.policy)
    with pytest.raises(ValueError) as caught:t.run(c.sim,c.rod,c.bindings,owner_resets=c.owners)
    frame=caught.value.__traceback__
    while frame:
        if frame.tb_frame.f_code.co_filename.endswith("tabletop_dual_hand_reset_v2.py"):
            assert frame.tb_frame.f_locals["binding"] is None
        frame=frame.tb_next


def test_numeric_replay_rejects_fields_outside_its_independent_schema(monkeypatch):
    c=long_mesh_fixture(monkeypatch);_,_,_,num=modules()
    _,receipt=execute(c)
    audit=deepcopy(receipt["numeric_reset_audit"]);audit["unbound_source"]=7
    with pytest.raises(ValueError,match="exact.*numeric"):
        num.validate_numeric_reset_audit_v2(audit,reset_receipt=receipt,clock_policy=c.policy)


def test_owner_solver_call_before_failure_is_unknown_not_initial_zero(monkeypatch):
    c=long_mesh_fixture(monkeypatch);_,_,rb,_=modules()
    primary=RuntimeError("owner failed after real step")
    def one_step_then_fail(context):
        advance(c,1)
        raise primary
    c.owners["filter"]=one_step_then_fail
    t=rb.MeasuredMeshResetV2(c.seed,episode_uid="solver-failure",mesh_plan=c.mesh_plan,clock_policy=c.policy)
    saved=[]
    with pytest.raises(RuntimeError) as caught:
        t.run(c.sim,c.rod,c.bindings,owner_resets=c.owners,record_readback=saved.append)
    assert caught.value is primary and c.rope.steps==1 and c.sim.current_time_step_index==3
    assert t.numeric_reset_audit["new_solver_steps"] is None
    assert t.readback_receipt["new_solver_steps"] is None
    assert t.readback_receipt["failure_clock_progress"]["solver_step_count_status"]=="unknown_no_actual_call_counter"
    assert saved[0]["new_solver_steps"] is None
    assert c.rod._needs_reset and t.readback_receipt["new_sdk_steps"]==1


def test_numeric_replay_rejects_bool_solver_times_at_real_zero(monkeypatch):
    c=long_mesh_fixture(monkeypatch);_,_,_,num=modules()
    _,receipt=execute(c);audit=deepcopy(receipt["numeric_reset_audit"])
    for sample in audit["samples"]:
        sample["solver_time_before"]=sample["solver_time_after"]=False
    with pytest.raises(ValueError):
        num.validate_numeric_reset_audit_v2(audit,reset_receipt=receipt,clock_policy=c.policy)


def test_numeric_replay_rejects_bool_padding_count(monkeypatch):
    c=long_mesh_fixture(monkeypatch);_,_,_,num=modules()
    _,receipt=execute(c);audit=deepcopy(receipt["numeric_reset_audit"])
    audit["padding_quaternion_count"]=True
    with pytest.raises(ValueError):
        num.validate_numeric_reset_audit_v2(audit,reset_receipt=receipt,clock_policy=c.policy)


def test_replaced_default_reader_is_refused_before_any_mutation(monkeypatch):
    c=long_mesh_fixture(monkeypatch);_,order,_,num=modules()
    t=order.RebuiltMeshResetV2(c.seed,episode_uid="replace-reader",mesh_plan=c.mesh_plan,clock_policy=c.policy)
    original=t.state_reader
    t.state_reader=lambda *a,**k:original(*a,**k)
    with pytest.raises(ValueError,match="reader.*binding"):
        t.run(c.sim,c.rod,c.bindings,owner_resets=c.owners)
    assert not c.events and c.rope.resets==0 and c.rope.steps==0


@pytest.mark.parametrize("field,value",[
    ("maximum_samples",2.),("segment_director_count",40.),
    ("padding_quaternion_count",True),("padding_quaternion_count",1.),
    ("new_sdk_steps",False),("new_solver_steps",False),("complete_samples",2.),
    ("reset_epoch",True)])
def test_numeric_counts_require_exact_integers(monkeypatch,field,value):
    c=long_mesh_fixture(monkeypatch);_,_,_,num=modules()
    _,receipt=execute(c);audit=deepcopy(receipt["numeric_reset_audit"]);audit[field]=value
    with pytest.raises(ValueError):
        num.validate_numeric_reset_audit_v2(audit,reset_receipt=receipt,clock_policy=c.policy)


@pytest.mark.parametrize("value",[False,np.float64(0),float("nan"),float("inf"),None,"0"])
def test_numeric_sample_time_requires_finite_primitive_number(monkeypatch,value):
    c=long_mesh_fixture(monkeypatch);_,_,_,num=modules()
    _,receipt=execute(c);sample=deepcopy(receipt["numeric_reset_audit"]["samples"][0])
    sample["solver_time_before"]=sample["solver_time_after"]=value
    with pytest.raises(ValueError):
        num.validate_numeric_sample_v2(sample,clock_policy=c.policy)


@pytest.mark.parametrize("field",["time_s","contact_mu_static","contact_mu_kinetic"])
def test_native_parameter_numeric_scalars_cannot_be_boolean(monkeypatch,field):
    c=long_mesh_fixture(monkeypatch);_,_,_,num=modules()
    _,receipt=execute(c);sample=deepcopy(receipt["numeric_reset_audit"]["samples"][0])
    sample["state"]["parameters"][field]=False
    with pytest.raises(ValueError):
        num.validate_numeric_sample_v2(sample,clock_policy=c.policy)


@pytest.mark.parametrize("field",["owner_time_s_before","owner_time_s_after"])
def test_mesh_readback_owner_time_cannot_be_boolean(monkeypatch,field):
    c=long_mesh_fixture(monkeypatch);_,_,rb,_=modules()
    _,receipt=execute(c);packet=deepcopy(receipt["mesh_state_readback"]["after"])
    packet["snapshot"][field]=False
    with pytest.raises(ValueError):rb.validate_capture_v2(packet,clock_policy=c.policy)


def test_capture_does_not_coerce_bool_solver_time_to_zero(monkeypatch):
    c=long_mesh_fixture(monkeypatch);_,_,_,num=modules()
    c.rope.time=False
    packet=num.capture_numeric_state_v2(c.rope,c.sim,c.rod,clock_policy=c.policy)
    assert packet["capture_error"] is not None


@pytest.mark.parametrize("field",["spacing","mu_s","mu_k"])
def test_mesh_plan_numeric_fields_reject_boolean_before_mutation(monkeypatch,field):
    c=long_mesh_fixture(monkeypatch);_,order,_,_=modules()
    from go2_pvcnn.control.dh116.tabletop_mesh_reset_order import seal_plan
    c.mesh_plan.pop("sha256");c.mesh_plan["entries"][0][field]=True
    c.mesh_plan=seal_plan(c.mesh_plan)
    with pytest.raises(ValueError):execute(c)
    assert not c.events and c.rope.resets==0


def test_cached_default_reader_flag_does_not_determine_provenance(monkeypatch):
    c=long_mesh_fixture(monkeypatch);_,order,_,num=modules()
    reader=lambda *a,**k:num.capture_numeric_state_v2(*a,**k)
    t=order.RebuiltMeshResetV2(c.seed,episode_uid="reader-flag",mesh_plan=c.mesh_plan,
        clock_policy=c.policy,state_reader=reader)
    t.state_reader_is_default=True
    receipt=t.run(c.sim,c.rod,c.bindings,owner_resets=c.owners)
    assert receipt["numeric_reset_audit"]["state_reader"]=="explicit_instance_state_reader_unqualified"


def test_reader_changed_mid_rebuild_is_not_used(monkeypatch):
    c=long_mesh_fixture(monkeypatch);_,order,_,num=modules()
    t=order.RebuiltMeshResetV2(c.seed,episode_uid="reader-late-change",mesh_plan=c.mesh_plan,clock_policy=c.policy)
    clear=c.rope.clear_meshes;calls=[]
    def replaced(*a,**k):
        calls.append("unbound reader")
        return num.capture_numeric_state_v2(*a,**k)
    def clear_then_replace():
        clear();t.state_reader=replaced
    c.rope.clear_meshes=clear_then_replace
    with pytest.raises(ValueError,match="reader.*binding"):
        t.run(c.sim,c.rod,c.bindings,owner_resets=c.owners)
    assert not calls and c.rod._needs_reset
    assert t.numeric_reset_audit["new_solver_steps"] is None


@pytest.mark.parametrize("bad_recorder",[False,True])
@pytest.mark.parametrize("plan_kind",["missing_sha","none"])
def test_early_plan_failure_has_owned_packet_without_masking_primary(monkeypatch,bad_recorder,plan_kind):
    c=long_mesh_fixture(monkeypatch);_,_,rb,_=modules()
    plan=deepcopy(c.mesh_plan)
    if plan_kind=="missing_sha":plan.pop("sha256")
    else:plan=None
    t=rb.MeasuredMeshResetV2(c.seed,episode_uid="early-plan",mesh_plan=plan,clock_policy=c.policy)
    saved=[]
    def record(row):
        saved.append(row)
        if bad_recorder:raise OSError("early secondary recorder")
    expected=KeyError if plan_kind=="missing_sha" else TypeError
    with pytest.raises(expected) as caught:
        t.run(c.sim,c.rod,c.bindings,owner_resets=c.owners,record_readback=record)
    assert type(caught.value) is expected
    assert t.status=="failed" and t.readback_receipt["complete"] is False
    assert t.readback_receipt["mesh_plan_sha256"] is None
    assert t.readback_receipt["new_solver_steps"] is None
    assert saved and saved[0]["complete"] is False and saved[0]["before"] is None
    assert not c.events and c.rope.resets==0 and c.rope.steps==0
    assert c.sim.current_time_step_index==2
    if bad_recorder:assert t.cleanup_failures[-1]["error_type"]=="OSError"


def test_early_plan_access_preserves_same_exception_object_and_failed_unknown_clock(monkeypatch):
    c=long_mesh_fixture(monkeypatch);_,_,rb,_=modules()
    t=rb.MeasuredMeshResetV2(c.seed,episode_uid="early-object",mesh_plan=c.mesh_plan,clock_policy=c.policy)
    primary=RuntimeError("same primary from malformed plan")
    class BadPlan:
        def __getitem__(self,key):raise primary
    t.mesh_plan=BadPlan()
    c.sim.current_time_step_index=None
    def bad_record(row):raise OSError("secondary")
    with pytest.raises(RuntimeError) as caught:
        t.run(c.sim,c.rod,c.bindings,owner_resets=c.owners,record_readback=bad_record)
    assert caught.value is primary
    assert t.status=="failed" and t.cleanup_failures
    assert t.readback_receipt["new_sdk_steps"] is None
    assert t.readback_receipt["new_solver_steps"] is None
    assert t.readback_receipt["failure_clock_progress"]["observed_sdk_steps"] is None
    assert not c.events and c.rope.resets==0 and c.rope.steps==0
