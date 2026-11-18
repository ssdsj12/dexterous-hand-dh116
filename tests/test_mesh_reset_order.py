"""CPU state-machine doubles use real original reset/runtime, never a native reset claim."""
from copy import deepcopy
import hashlib,importlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from test_dual_hand_reset import setup
from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import DualHandResetTransaction

ROOT=Path(__file__).resolve().parents[1]
def api():
    path=ROOT/"src/go2_pvcnn/control/dh116/tabletop_mesh_reset_order.py"
    assert path.exists(),"independent reset mesh rebuild adapter is missing"
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_mesh_reset_order")

def native_semantics(monkeypatch):
    events,rod,hands,bindings,sim,seed,owners=setup(monkeypatch)
    rope=rod.rope
    rope.last_error=0
    # Existing CPU fixture did not model d_cur/d_tgt/deferred teleport.
    rope.current={i:(np.array(k["pos"],np.float32),np.array(k.get("quat",[0,0,0,1]),np.float32))
                  for i,(_,k) in enumerate(rope.slots)}
    rope.targets=deepcopy(rope.current);rope.tele={i:False for i in rope.current}
    rope.capacity=41*11*12;rope.pair_ready=False;rope.pending=False;rope.add_calls=0
    original_add=rope.add_mesh
    def add(*a,**k):
        events.append("add_mesh")
        i=original_add(*a,**k)
        q=np.array(k.get("quat",[0,0,0,1]),np.float32);q/=np.linalg.norm(q)
        pose=(np.array(k["pos"],np.float32),q)
        rope.current[i]=deepcopy(pose);rope.targets[i]=deepcopy(pose);rope.tele[i]=False
        rope.add_calls+=1;rope.pair_ready=False
        return i
    def clear():
        events.append("clear_meshes")
        rope.slots.clear();rope.current.clear();rope.targets.clear();rope.tele.clear()
        rope.wrenches=np.zeros((1,0,6));rope.pending=False;rope.pair_ready=False
    def targets(i,p,q,*,teleport):
        q=np.asarray(q,np.float32).copy();q/=np.linalg.norm(q)
        rope.targets[i]=(np.asarray(p,np.float32).copy(),q)
        rope.tele[i]=teleport;rope.pending=True
    def advance(dt,n):
        rope.started_from={}
        for i in rope.current:
            if rope.tele[i]:rope.current[i]=deepcopy(rope.targets[i])
            rope.started_from[i]=deepcopy(rope.current[i])
            rope.current[i]=deepcopy(rope.targets[i]);rope.tele[i]=False
        rope.pending=False;rope.steps+=1;rope.time+=dt;rope.pair_ready=True
        rope.force[:]=0;rope.wrenches[:]=0
    rope.add_mesh=add;rope.clear_meshes=clear;rope.set_mesh_poses=targets;rope.step_free=advance
    def parameters():
        events.append("parameter_snapshot_sync")
        return dict(time_s=rope.time,contact_radius_m=rod._base.cfg["rod"]["radius_m"],
                    inverse_masses_kg_inv=rope.inv_masses().copy())
    rope.parameter_snapshot=parameters
    rope.quaternions=lambda:np.tile(np.array([0,0,0,1.],np.float32),(1,len(rod._base.nodes),1))
    slots=deepcopy(rope.slots)
    return SimpleNamespace(events=events,rod=rod,rope=rope,hands=hands,bindings=bindings,
        sim=sim,seed=seed,owners=owners,slots=slots)

def plan(c,monkeypatch):
    m=api()
    entries=[]
    for i,(args,kw) in enumerate(c.slots):
        entries.append(dict(name=c.rod.mesh_names[i],values=np.asarray(args[0],np.float32),
            origin=np.asarray(args[1],np.float32),spacing=float(args[2]),
            mu_s=kw["mu_s"],mu_k=kw["mu_k"],
            table_pose=None if i!=5 else (np.asarray(kw["pos"],np.float32),np.array([0,0,0,1.],np.float32))))
    value=dict(version=m.PLAN_VERSION,entries=entries,bindings=deepcopy(c.rod.bindings),
        cfg=deepcopy(c.rod._base.cfg),source_basis={"source_tree_sha256":"CPU fixture"},
        source_scope="CPU fixture inputs")
    monkeypatch.setattr(m,"verify_plan_sources",lambda p:None)
    monkeypatch.setattr(m,"verify_runtime_source",lambda p:None)
    return m.seal_plan(value)

def run(c,p,uid):
    return api().RebuiltMeshReset(c.seed,episode_uid=uid,mesh_plan=p).run(
        c.sim,c.rod,c.bindings,owner_resets=c.owners)

def next_step(c):
    poses={s+"."+n:(h.link_poses[0,i,:3].numpy()+[.001,0,0],h.link_poses[0,i,3:].numpy())
        for s,h in zip(("left","right"),c.hands) for i,n in enumerate(c.rod._base.names[:-1])}
    c.sim.current_time_step_index+=1;c.sim.current_time=c.sim.current_time_step_index/480
    c.rod.step(poses,{n:p[0] for n,p in poses.items()},physics_step=c.sim.current_time_step_index)

def test_original_deferred_teleport_is_overwritten_by_first_target(monkeypatch):
    c=native_semantics(monkeypatch)
    start=deepcopy(c.rope.current[0])
    r=DualHandResetTransaction(c.seed,episode_uid="old").run(c.sim,c.rod,c.bindings,owner_resets=c.owners)
    assert not np.array_equal(start[0],r["pad_poses_xyzw"]["left.finger14_link"][0])
    next_step(c)
    np.testing.assert_array_equal(c.rope.started_from[0][0],start[0])

def test_reset_readds_all_original_meshes_after_actual_refresh_and_before_first_step(monkeypatch):
    c=native_semantics(monkeypatch);p=plan(c,monkeypatch)
    r=run(c,p,"fresh")
    assert c.events.index("clear_meshes")>c.events.index("kinematic_refresh")
    assert c.events.count("add_mesh")==11 and c.rope.capacity==5412
    assert c.rope.steps==0 and c.sim.current_time_step_index==2
    assert r["mesh_reset"]["new_sdk_steps"]==r["mesh_reset"]["new_solver_steps"]==0
    assert r["mesh_reset"]["native_current_pose_readback"] is None
    assert r["mesh_reset"]["mutation_after_clock"]["rod"]["clock_valid"] is False
    assert r["mesh_reset"]["after_clock"]["rod"]["clock_valid"] is True
    assert not r["mesh_reset"]["native_qualification"]
    expected=deepcopy(r["pad_poses_xyzw"])
    for i,name in enumerate(c.rod.mesh_names):
        if name=="table":continue
        np.testing.assert_array_equal(c.rope.current[i][0],expected[name][0].astype(np.float32))
        assert not c.rope.tele[i]
    for i,(a,k) in enumerate(c.rope.slots):
        original_args,original_kwargs=c.slots[i]
        for x,y in zip(a,original_args):np.testing.assert_array_equal(x,y)
        assert k["mu_s"]==original_kwargs["mu_s"] and k["mu_k"]==original_kwargs["mu_k"]
    assert all(h.native_zero_wrench_writes==1 for h in c.hands)
    assert all(not h.permanent_wrench_composer.active for h in c.hands)
    next_step(c)
    np.testing.assert_array_equal(c.rope.started_from[0][0],expected["left.finger14_link"][0].astype(np.float32))

def test_second_episode_after_measured_step_rebuilds_current_without_resetting_global_time(monkeypatch):
    c=native_semantics(monkeypatch);p=plan(c,monkeypatch)
    first=run(c,p,"first");next_step(c);before=c.rope.time
    c.seed=deepcopy(c.seed)
    # Same authored seed is valid; physical current moved in the preceding step.
    second=run(c,p,"second")
    assert c.rope.time==before and c.rope.steps==1 and c.rope.resets==2
    assert second["after_clock"]["reset_id"]==2 and second["after_clock"]["rod_tick"]==0
    assert c.rope.add_calls==22 and c.rope.capacity==5412 and not c.rope.pair_ready
    np.testing.assert_array_equal(c.rope.current[0][0],second["pad_poses_xyzw"]["left.finger14_link"][0].astype(np.float32))
    next_step(c)
    np.testing.assert_array_equal(c.rope.started_from[0][0],second["pad_poses_xyzw"]["left.finger14_link"][0].astype(np.float32))

@pytest.mark.parametrize("fault",["clear","add","slot","sdk","solver","parameters","positions","reset_id","episode","rod_tick","clock_valid"])
def test_partial_rebuild_poisoned_without_extra_step_or_reusable_transaction(monkeypatch,fault):
    c=native_semantics(monkeypatch);p=plan(c,monkeypatch);m=api()
    original=c.rope.add_mesh
    def damaged(*a,**k):
        i=original(*a,**k)
        if i==3:
            if fault=="add":raise RuntimeError("injected add failure")
            if fault=="slot":return 9
            if fault=="sdk":c.sim.current_time_step_index+=1
            if fault=="solver":c.rope.time+=.001
            if fault=="parameters":c.rope.inv_masses=lambda:np.ones(3)
            if fault=="positions":c.rope.nodes[0,0,0]+=.01
            if fault=="reset_id":c.rod._base.reset_id+=1
            if fault=="episode":c.rod._base.episode_uid="different"
            if fault=="rod_tick":c.rod._base.rod_tick=12
            if fault=="clock_valid":c.rod._needs_reset=False
        return i
    c.rope.add_mesh=damaged
    if fault=="clear":
        c.rope.clear_meshes=lambda:(_ for _ in ()).throw(RuntimeError("injected clear failure"))
    t=m.RebuiltMeshReset(c.seed,episode_uid="failure",mesh_plan=p)
    with pytest.raises((ValueError,RuntimeError)):t.run(c.sim,c.rod,c.bindings,owner_resets=c.owners)
    assert t.status=="failed" and not c.rod.clock()["clock_valid"] and c.rope.steps==0
    with pytest.raises(RuntimeError):t.run(c.sim,c.rod,c.bindings,owner_resets=c.owners)
    with pytest.raises(RuntimeError):next_step(c)

def test_tampered_plan_fails_before_hand_or_mesh_mutation(monkeypatch):
    c=native_semantics(monkeypatch);p=plan(c,monkeypatch);p["entries"][0]["mu_s"]+=1
    with pytest.raises(ValueError):run(c,p,"bad")
    assert c.rope.resets==0 and not c.events


def test_plan_uses_actual_locked_npz_and_original_cpu_table_sampler(monkeypatch):
    import importlib.util,sys
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import prepare_scene
    from go2_pvcnn.control.dh116.tabletop_native_parameter_endpoints import BUILD,CONFIG
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import load_candidate,box_sdf
    prepared=prepare_scene(ROOT);cfg=load_candidate(ROOT/CONFIG,ROOT)
    path=BUILD/"build/python/ropesim/sdf.py"
    spec=importlib.util.spec_from_file_location("dh116_reset_cpu_sdf",path)
    sdf=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,spec.name,sdf);spec.loader.exec_module(sdf)
    m=api();monkeypatch.setattr(m,"verify_runtime_source",lambda rod:None)
    rod=SimpleNamespace(_base=SimpleNamespace(cfg=cfg),bindings=deepcopy(prepared["pads"]))
    result=m.prepare_mesh_reset_plan(ROOT,prepared,rod,sdf_module=sdf)
    m.verify_plan_sources(result)
    assert result["source_basis"]["source_file_count"]==72
    for entry in result["entries"]:
        if entry["name"]!="table":
            with np.load(prepared["pads"][entry["name"]]["sdf_path"],allow_pickle=False) as raw:
                np.testing.assert_array_equal(entry["values"],raw["values"])
                np.testing.assert_array_equal(entry["origin"],raw["origin"])
                assert entry["spacing"]==float(raw["spacing"])
        else:
            size=np.asarray(prepared["table"]["size_m"]);half=size/2;t=cfg["table"]
            original=sdf.from_function(lambda points:box_sdf(points,size),
                -half-t["sdf_margin_m"],half+t["sdf_margin_m"],t["sdf_spacing_m"])
            np.testing.assert_array_equal(entry["values"],original.values)
            np.testing.assert_array_equal(entry["origin"],original.origin)
            assert entry["mu_s"]==t["mu_s"] and entry["mu_k"]==t["mu_k"]
    assert result["entries"][0]["mu_s"]==1. and result["entries"][0]["mu_k"]==.8
    for fault in ("mu_s","values","table_pose"):
        bad=deepcopy(result);bad.pop("sha256")
        if fault=="mu_s":bad["entries"][0]["mu_s"]+=.1
        elif fault=="values":bad["entries"][0]["values"].flat[0]+=.01
        else:bad["entries"][5]["table_pose"][0][2]+=.1
        with pytest.raises(ValueError):m.verify_plan_sources(m.seal_plan(bad))
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import _digest as scene_digest
    for fault in ("table_source","pad_source"):
        bad=deepcopy(result);bad.pop("sha256")
        prepared_bad=bad["prepared_scene"]
        if fault=="table_source":
            prepared_bad["table"]["center_m"][2]+=.1
            bad["entries"][5]["table_pose"][0][2]+=.1
        else:
            prepared_bad["pads"]["left.finger14_link"]["sdf_path"]=prepared_bad["pads"]["right.finger14_link"]["sdf_path"]
            prepared_bad["pads"]["left.finger14_link"]["sdf_sha256"]=prepared_bad["pads"]["right.finger14_link"]["sdf_sha256"]
        prepared_bad.pop("fingerprint_sha256")
        prepared_bad["fingerprint_sha256"]=scene_digest(prepared_bad)
        with pytest.raises(ValueError):m.verify_plan_sources(m.seal_plan(bad))

def test_locked_native_clear_add_state_machine_matches_cpu_assumptions():
    from go2_pvcnn.control.dh116.tabletop_native_parameter_endpoints import BUILD,_locked_source_tree
    inventory,_=_locked_source_tree()
    src=(BUILD/"source/csrc/sdf.cu").read_text()
    clear=src.split("void SdfStore::clear() {",1)[1].split("\n}",1)[0]
    assert "invalidate_pair_capture();" in clear and "cudaFree" in clear
    assert "pairCapacity" not in clear and "d_pairs" not in clear
    assert "pendingSet = false;" in clear and "moving = false;" in clear
    add=src.split("int SdfStore::add(",1)[1].split("void SdfStore::set_friction",1)[0]
    assert "set_targets_host(idx, pos, q, rows, nullptr, true, s);" in add
    assert "commitMeshK<<<" in add and "cudaStreamSynchronize(s)" in add
    assert "pendingSet = pending;" in add
    assert "tele[slot] = teleAll ? 1 : (teleFlags ? teleFlags[e] : 0);" in src
    assert inventory["csrc/sdf.cu"]=="e5b3b1b00881aa1aff1697214f36b4e12310f8d62655455be7ce6b6df0369c12"

def test_exception_frames_release_native_aliases(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_reset as original
    saved_clock=original._clock
    c=native_semantics(monkeypatch);p=plan(c,monkeypatch)
    c.rope.clear_meshes=lambda:(_ for _ in ()).throw(RuntimeError("failure"))
    with pytest.raises(RuntimeError) as caught:run(c,p,"frame")
    assert original._clock is saved_clock
    tb=caught.value.__traceback__
    while tb:
        if tb.tb_frame.f_code.co_filename.endswith("tabletop_mesh_reset_order.py"):
            for name in ("sim","rod","bindings","owner_resets","receipt","rope"):
                if name in tb.tb_frame.f_locals:assert tb.tb_frame.f_locals[name] is None
        tb=tb.tb_next


@pytest.mark.parametrize("bad",[7,True,None])
def test_nonzero_or_missing_actual_solver_error_cannot_rebuild(monkeypatch,bad):
    c=native_semantics(monkeypatch);p=plan(c,monkeypatch);c.rope.last_error=bad
    with pytest.raises(ValueError,match="solver error"):run(c,p,"error-code")
    assert "clear_meshes" not in c.events and not c.rod.clock()["clock_valid"]


def test_next_reset_accepts_actual_sdk_accumulation_without_rewriting_time(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_reset as original
    saved=original._clock
    c=native_semantics(monkeypatch);p=plan(c,monkeypatch)
    run(c,p,"first")
    c.sim.current_time_step_index=234
    c.sim.current_time=.4875000254251063  # Actual immutable projected-Pinch SDK234 readback.
    c.rod._base.physics_step=234;c.rod._base.rod_tick=2784
    c.rope.time=.48333296179771423;c.rod._last_solver_time=c.rope.time
    r=run(c,p,"second-real-time")
    assert r["after_actual_clocks"]["sdk_timestamp_s"]==.4875000254251063
    assert r["mesh_reset"]["before_clock"]==r["mesh_reset"]["after_clock"]
    assert original._clock is saved and c.rope.steps==0


@pytest.mark.parametrize("step,offset",[(241,0),(234,1e-6)])
def test_reset_clock_does_not_claim_long_run_or_accept_bad_source_time(monkeypatch,step,offset):
    c=native_semantics(monkeypatch);p=plan(c,monkeypatch)
    c.sim.current_time_step_index=step;c.sim.current_time=step/480+offset
    with pytest.raises(ValueError,match="clock|240"):run(c,p,"bounded")
    assert c.rope.resets==0 and not c.events

def test_nested_or_external_reset_clock_binding_is_rejected_without_mutation(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_reset as original
    c=native_semantics(monkeypatch);p=plan(c,monkeypatch)
    monkeypatch.setattr(original,"_clock",lambda *a:None)
    with pytest.raises(ValueError,match="binding"):run(c,p,"nested")
    assert c.rope.resets==0 and not c.events
