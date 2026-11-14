"""Exact paired source data, real reset/runtime/episode and CPU SDK doubles."""
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
MODULE="go2_pvcnn.control.dh116.tabletop_dual_pinch_builder"
PAIR=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/paired-pinch-reset-BaP0hx/result.json")
PAIR_SHA="996bdb33ed2d6f9abddb9171245142b5147af222b976fc799b5f89711c4f6a37"
SEED_SHA="647247fe31219f04337b6f6e29ddefaa2c81bcee27dbed8dd3d6f962e406230d"

def module():return importlib.import_module(MODULE)

def test_real_process_byte_loader_has_no_early_usd():
    code="""
import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('pxr','PhysxSchema','isaaclab','isaacsim','ropesim'):
            raise RuntimeError('forbidden pre-App import:'+name)
sys.meta_path.insert(0,Guard())
from go2_pvcnn.control.dh116.tabletop_dual_pinch_builder import load_pair_proposal
p=load_pair_proposal()
assert p['reset_seed']['sha256']=='647247fe31219f04337b6f6e29ddefaa2c81bcee27dbed8dd3d6f962e406230d'
print('BYTE_ONLY')
"""
    done=subprocess.run([sys.executable,"-c",code],cwd=ROOT,
        env={**os.environ,"CUDA_VISIBLE_DEVICES":""},capture_output=True,text=True,timeout=15)
    assert done.returncode==0,done.stdout+done.stderr
    assert done.stdout.strip()=="BYTE_ONLY"

def test_exact_private_proposal_and_tampered_bytes(tmp_path):
    assert hashlib.sha256(PAIR.read_bytes()).hexdigest()==PAIR_SHA
    record=module().load_pair_proposal()
    assert record["reset_seed"]["sha256"]==SEED_SHA
    assert np.allclose(np.asarray(record["rod_seed"]["proposed_nodes_m"])[:,2],.305,atol=0,rtol=0)
    assert record["wrist_clearance_floor_m"]==.0021
    altered=tmp_path/"altered.json";altered.write_bytes(PAIR.read_bytes()+b" ")
    with pytest.raises(ValueError,match="paired evidence"):
        module().load_pair_proposal(altered)

@pytest.fixture(scope="module")
def prepared():
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import prepare_scene
    return prepare_scene(ROOT)

@pytest.fixture(scope="module")
def geometry():
    from scripts.audit_dh116_pinch_initial_pose import load_geometry
    return {s:load_geometry(ROOT,side=s) for s in ("left","right")}

class View:pass
class Sim:pass

def setup(monkeypatch,tmp_path,prepared,geometry,fault=None):
    from test_dual_hand_reset import Hand
    from test_dual_hand_rod_runtime import FakeRope,Grid
    from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import RESET_OWNERS
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import author_source_stage
    from go2_pvcnn.control.dh116.tabletop_native_lifetime_audit import snapshot_weakrefs
    from go2_pvcnn.control.dh116.tabletop_deformable_rod_bridge import NativeOwnershipTrace
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,expand
    from scripts.audit_dh116_pinch_initial_pose import pose_matrix
    from scipy.spatial.transform import Rotation
    from pxr import Usd
    events=[]; refs={}; rows=[]; metadata={}
    sim=Sim();sim.stage=Usd.Stage.CreateInMemory()
    sim.current_time_step_index=0;sim.current_time=0.
    sim.get_physics_dt=lambda:1/480
    sim.physics_sim_view=View()
    sim.physics_sim_view.get_gravity=lambda:np.array([0,0,-9.81])
    sim.is_playing=lambda:True
    live=[]
    def refresh():
        events.append("sdk_refresh")
        for hand in live:
            side=hand.side;g=geometry[side]
            pose=hand.data.root_pose_w[0].numpy().copy()
            q=hand.measured[0,[hand.joint_names.index(n) for n in ACTIVE]].numpy()
            delta=pose_matrix(pose)@np.linalg.inv(pose_matrix(g["pose"]))
            poses={name.rsplit("/",1)[-1]:delta@matrix for name,matrix in g["fk"].poses(q).items()}
            for index,name in enumerate(hand.body_names):
                matrix=poses.get(name,pose_matrix(pose))
                hand.link_poses[0,index,:3]=torch.tensor(matrix[:3,3],dtype=torch.float32)
                hand.link_poses[0,index,3:]=torch.tensor(Rotation.from_matrix(matrix[:3,:3]).as_quat(),dtype=torch.float32)
            if fault=="clearance" and side=="right":
                hand.link_poses[0,:,:3]-=torch.tensor([0,0,.001])
        if fault=="refresh_tick":
            sim.current_time_step_index+=1;sim.current_time+=1/480
    sim.physics_sim_view.update_articulations_kinematic=refresh
    def reset():
        events.append("setup_reset")
        sim.current_time_step_index+=3 if fault=="setup_steps" else 2
        sim.current_time=sim.current_time_step_index/480
        refresh()
    sim.reset=reset
    def spawn(simulation,args,layout,sources,prefix,origin,*,external_wire):
        assert simulation is sim and external_wire is True
        assert args.load_parameters.effort_limit_nm==.15
        assert args.position_iterations==64 and args.velocity_iterations==16
        assert args.palm_transport=="driven_carriage" and args.free_wrist_grasp is True
        assert set(sources)=={"left","right"} and np.array_equal(origin,np.zeros(3))
        events.append("spawn")
        sim.stage=author_source_stage(ROOT,prepared)
        for side in ("left","right"):
            class LazyHand(Hand):
                def __getattribute__(self,name):
                    if name in ("permanent_wrench_composer","instantaneous_wrench_composer") and sim.current_time_step_index<2:
                        raise AttributeError("SDK composer not initialized before reset")
                    return super().__getattribute__(name)
            hand=LazyHand(side,events)
            hand.body_names=list(prepared["hands"][side]["source_colliders"])+[
                "wrist_anchor","wrist_slide_x","wrist_slide_y","wrist_slide_z","wrist_rotate_x","wrist_rotate_y"]
            old=hand.root_physx_view;view=View();view.__dict__.update(vars(old))
            hand.root_physx_view=view
            view.prim_paths=[prefix+"/"+side+"Hand/wrist_fixed"]
            view.link_paths=[[prefix+"/"+side+"Hand/"+n for n in hand.body_names]]
            view.get_coms=lambda:np.zeros((1,23,7))
            hand._physics_sim_view=sim.physics_sim_view
            pose=np.asarray(prepared["hands"][side]["builder_initial_root_pose_w"])
            hand.data.root_pose_w[0]=torch.tensor(pose,dtype=torch.float32)
            hand.root_native[:]=hand.data.root_pose_w[:,[0,1,2,4,5,6,3]]
            full=expand(prepared["hands"][side]["source_active_q"])
            for j,name in enumerate(JOINTS):
                k=hand.joint_names.index(name)
                hand.measured[0,k]=float(full[j]);hand.data.joint_pos[0,k]=float(full[j])
            refs[side]=weakref.ref(hand);refs[side+"_view"]=weakref.ref(view)
            if fault=="link_identity" and side=="right":view.link_paths[0][0]+="_wrong"
            live.append(hand)
        return dict(hands=list(live),sensors=[],wire=None,prefix=prefix,layout=layout,origin=origin)
    monkeypatch.setitem(sys.modules,"go2_pvcnn.control.dh116.tabletop_vec_isaac",SimpleNamespace(spawn_scene=spawn))
    class Rope(FakeRope):
        @classmethod
        def from_positions(cls,nodes,**kwargs):
            obj=super().from_positions(nodes,**kwargs)
            refs["rope"]=weakref.ref(obj)
            metadata["constructor_nodes"]=nodes.copy();metadata["pinned"]=kwargs["pinned"].copy()
            return obj
        def enable_contact_pair_capture(self,capacity):
            metadata["pair_capacity"]=capacity
        def reset_envs(self,ids):
            super().reset_envs(ids)
            if fault=="solver_reset":raise RuntimeError("CPU solver reset failure")
    import importlib.util
    sdf_spec=importlib.util.spec_from_file_location("_dh116_cpu_sdf",module().BACKEND.parent/"sdf.py")
    cpu_sdf=importlib.util.module_from_spec(sdf_spec)
    monkeypatch.setitem(sys.modules,sdf_spec.name,cpu_sdf)
    sdf_spec.loader.exec_module(cpu_sdf)  # Installed pure-NumPy dataclass, no CUDA/ropesim import.
    solver=SimpleNamespace(__file__=str(module().BACKEND))
    if fault=="backend_identity":solver.__file__="/wrong/solver.so"
    monkeypatch.setitem(sys.modules,"ropesim",SimpleNamespace(RopeBatch=Rope,_ropesim=solver,
        sdf=SimpleNamespace(SdfGrid=cpu_sdf.SdfGrid,
            from_function=lambda *a:cpu_sdf.SdfGrid(np.ones((2,2,2),dtype=np.float32),np.zeros(3,dtype=np.float32),.01))))
    owners={name:(lambda context,n=name:dict(owner=n,status="not_present",
        source="CPU builder fixture",episode_uid=context["episode_uid"])) for name in RESET_OWNERS}
    if fault=="owner":owners["filter"]=lambda context:(_ for _ in ()).throw(RuntimeError("CPU owner failure"))
    trace=NativeOwnershipTrace(tmp_path/"ownership.jsonl")
    def sync():events.append("synchronize")
    def record(row):
        rows.append(deepcopy(row))
        if row.get("kind")=="reset":
            metadata["rod_steps"]=refs["rope"]().steps
            metadata["rod_resets"]=refs["rope"]().resets
        if fault=="record" and row.get("kind")=="reset":raise RuntimeError("CPU recorder failure")
    def drop_fixture_handles():
        live.clear()
    return sim,trace,owners,sync,record,rows,events,refs,metadata,drop_fixture_handles

def call(sim,trace,owners,sync,record,prepared):
    return module().run_preparation(sim,project=ROOT,prepared=prepared,ownership=trace,
        synchronize=sync,record=record,owner_resets=owners,
        run_uuid="cpu-run",episode_uid="cpu-run:paired",physics_budget_end=10)

def test_real_composed_reset_uses_exact_paired_seed_and_one_free_rod(monkeypatch,tmp_path,prepared,geometry):
    sim,trace,owners,sync,record,rows,events,refs,meta,drop=setup(monkeypatch,tmp_path,prepared,geometry)
    result=call(sim,trace,owners,sync,record,prepared)
    assert result["setup_steps"]==2 and result["measured_steps"]==result["solver_steps"]==0
    assert result["reset_seed_sha256"]==SEED_SHA and result["paired_evidence_sha256"]==PAIR_SHA
    assert result["clearance_floor_m"]==.0021
    assert np.all(np.asarray(result["clearance_m"])>=.0021)
    assert np.allclose(result["clearance_m"],[[.002110872,.002116391]],atol=3e-7,rtol=0)
    proposal=module().load_pair_proposal()
    expected=[proposal[s]["path"]["initial_all_shape_table_clearance_m"] for s in ("left","right")]
    assert np.allclose(result["clearance_m"][0],expected,atol=1e-7,rtol=0)
    assert min(result["locked_pad_table_clearance_m"])>=.0021
    geometry_rows=[r for r in rows if r.get("kind")=="initial_geometry"]
    assert [r["side"] for r in geometry_rows]==["left","right"]
    assert all(len(r["evidence"]["source_colliders"])==17 and len(r["evidence"]["locked_pads"])==5
               and r["evidence"]["contact_free_proven"] is False for r in geometry_rows)
    assert meta["rod_resets"]==1 and meta["rod_steps"]==0
    assert meta["pair_capacity"]==41*11*12 and not meta["pinned"].any()
    np.testing.assert_array_equal(meta["constructor_nodes"][0],np.asarray(
        module().load_pair_proposal()["rod_seed"]["proposed_nodes_m"],dtype=np.float32))
    assert result["mesh_names"][5]=="table" and len(result["mesh_names"])==11
    assert result["native_qualification"] is result["teacher_admission"] is result["grasp_qualified"] is False
    reset=next(r for r in rows if r.get("kind")=="reset")["reset"]
    assert reset["new_sdk_steps"]==reset["new_solver_steps"]==0
    assert reset["seed"]==module().load_pair_proposal()["reset_seed"]
    assert events.index("spawn")<events.index("setup_reset")<events.index("left:reset")
    assert events[-1]=="synchronize"
    drop()

@pytest.mark.parametrize("fault",["backend_identity","setup_steps","link_identity","solver_reset","owner","record","clearance","refresh_tick"])
def test_failed_preparation_never_steps_solver_or_retries(monkeypatch,tmp_path,prepared,geometry,fault):
    args=setup(monkeypatch,tmp_path,prepared,geometry,fault)
    sim,trace,owners,sync,record,rows,events,refs,meta,drop=args
    with pytest.raises((RuntimeError,ValueError)):
        call(sim,trace,owners,sync,record,prepared)
    assert events.count("setup_reset")<=1
    assert meta.get("rod_steps",0)==0
    assert not any(r.get("kind")=="physical" for r in rows)
    drop()

@pytest.mark.parametrize("failure",[False,True])
def test_deliberately_retained_builder_frames_drop_native_aliases(monkeypatch,tmp_path,prepared,geometry,failure):
    sim,trace,owners,sync,record,rows,events,refs,meta,drop=setup(
        monkeypatch,tmp_path,prepared,geometry,"owner" if failure else None)
    frames=[]
    filename=module().__file__
    # Reproduce the observed retained builder-return frame. Retaining arbitrary
    # callback frames also retains unmodified dependency caller frames via f_back;
    # that broader SDK callback-lifetime question requires actual native evidence.
    def profiler(frame,event,arg):
        if (event=="return" and frame.f_code.co_filename==filename
                and frame.f_code.co_name in ("build_reset_owned","run_preparation","_initial_pads")):
            frames.append(frame)
    sys.setprofile(profiler)
    try:
        if failure:
            with pytest.raises(RuntimeError):call(sim,trace,owners,sync,record,prepared)
        else:call(sim,trace,owners,sync,record,prepared)
    finally:sys.setprofile(None)
    drop()
    import gc;gc.collect()
    assert frames
    assert all(ref() is None for name,ref in refs.items() if name!="rope"),[
        (name,ref()) for name,ref in refs.items() if ref() is not None]
    assert refs.get("rope",lambda:None)() is None
    assert all(not any(type(v).__name__ in ("LazyHand","Hand","Rope","DualHandPhysicalEpisode","DualHandRodRuntime")
                       for v in f.f_locals.values()) for f in frames)

@pytest.mark.parametrize("fail",[False,True])
def test_all_composers_are_reset_even_if_one_fails(fail):
    events=[]
    class Composer:
        def __init__(self,index):self.index=index
        def reset(self):
            events.append(self.index)
            if fail and self.index==1:raise RuntimeError("CPU composer reset failure")
    group=module()._Composers([Composer(i) for i in range(4)])
    if fail:
        with pytest.raises(RuntimeError,match="CPU composer reset failure"):group.reset()
    else:group.reset()
    assert events==[0,1,2,3]

@pytest.mark.parametrize("step",[None,2])
def test_missing_or_nonfresh_context_rejected_without_native_import(step):
    sim=None if step is None else SimpleNamespace(current_time_step_index=step,
        current_time=step/480,get_physics_dt=lambda:1/480)
    with pytest.raises(ValueError,match="fresh"):
        module().build_reset_owned(sim,None,project=ROOT,prepared=None,ownership=None,
            record=lambda row:None,owner_resets={},run_uuid="cpu",episode_uid="cpu:x",
            physics_budget_end=10)

def test_module_compiles_on_runtime_python():
    path=Path(module().__file__)
    compile(path.read_text(),str(path),"exec")
