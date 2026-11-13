"""CPU-only Pinch v2 pair, quiet-window, transport and ownership contracts."""
import ast
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module():
    return importlib.import_module("scripts.dh116_pinch_evidence_v2")


def packet(*,pad=False):
    return dict(substeps=12,mesh_count=6,capacity=41*6*12,attempted=int(pad),overflow=False,
        indices=np.array([[0,0,0]],dtype=np.int32) if pad else np.empty((0,3),dtype=np.int32),
        vectors=np.array([[0,0,0,0,0,1,0,0,12,0,0,0]],dtype=float) if pad else np.empty((0,12)),
        node_forces_on_rod_world_n=np.pad([[0,0,1]],((0,40),(0,0))) if pad else np.zeros((41,3)),
        mesh_wrenches_on_mesh_world=np.pad([[0,0,-1,0,0,0]],((0,5),(0,0))) if pad else np.zeros((6,6)))


def row_for(raw):
    from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS
    return dict(phase_step=1,physics_step=3,rod_tick=12,episode_uid="cpu-episode",reset_id=1,
        env_id=0,rod_uid="rod0",timestamp_s=3/480,
        rod_node_contact_forces_w=raw["node_forces_on_rod_world_n"].tolist(),
        rod_mesh_wrenches_w=dict(zip([*DISTAL_PAD_LINKS,"table"],
                                      raw["mesh_wrenches_on_mesh_world"].tolist())))


@pytest.mark.parametrize("pad",[False,True])
def test_pairs_roundtrip_empty_and_nonempty_without_losing_array_shape(pad):
    m=module();raw=packet(pad=pad);row=row_for(raw)
    record=m.phase_pair_record(raw,row)
    disk=json.loads(json.dumps(record))
    summary=m.validate_phase_pair_record(disk,row)
    assert len(summary["pairs"])==int(pad)
    assert disk["raw"]["indices_shape"]==[int(pad),3]
    assert disk["raw"]["vectors_shape"]==[int(pad),12]


@pytest.mark.parametrize("damage",["clock","shape","overflow","force","slot"])
def test_pair_disk_audit_rejects_inconsistent_raw_evidence(damage):
    m=module();raw=packet(pad=True);row=row_for(raw)
    record=m.phase_pair_record(raw,row)
    if damage=="clock":record["physics_step"]+=1
    if damage=="shape":record["raw"]["indices_shape"]=[1,2]
    if damage=="overflow":record["raw"]["overflow"]=True
    if damage=="force":record["raw"]["node_forces_on_rod_world_n"][0][2]+=1
    if damage=="slot":record["raw"]["indices"][0][2]=6
    with pytest.raises(ValueError):m.validate_phase_pair_record(record,row)


class ContactView:
    sensor_count=17
    filter_count=1
    max_contact_data_count=128
    def __init__(self,count=0):
        self.count=count
    def get_contact_data(self,dt):
        assert dt==1/480
        counts=np.zeros((17,1),dtype=np.uint32);counts[0,0]=self.count
        return (np.zeros((128,1)),np.zeros((128,3)),np.tile([0.,0.,1.],(128,1)),
                np.zeros((128,1)),counts,np.zeros((17,1),dtype=np.uint32))


def test_quiet_requires_actual_contact_counts_and_empty_pad_pairs():
    m=module();row=row_for(packet())
    contacts=m.read_table_contacts(ContactView(),row)
    assert contacts["total_contacts"]==0
    assert m.require_quiet(m.phase_pair_record(packet(),row),contacts,row) is True
    # A contact can exist with zero force; it must still block.
    contacts=m.read_table_contacts(ContactView(1),row)
    with pytest.raises(ValueError,match="hand-table"):
        m.require_quiet(m.phase_pair_record(packet(),row),contacts,row)
    contacts=m.read_table_contacts(ContactView(),row)
    with pytest.raises(ValueError,match="quiet"):
        m.require_quiet(m.phase_pair_record(packet(pad=True),row_for(packet(pad=True))),contacts,row_for(packet(pad=True)))


@pytest.mark.parametrize("damage",["count","capacity","sensors","filter","negative"])
def test_contact_buffer_bounds_and_identity_are_mandatory(damage):
    m=module();view=ContactView()
    if damage=="count":view.count=129
    if damage=="capacity":view.count=128
    if damage=="sensors":view.sensor_count=16
    if damage=="filter":view.filter_count=0
    if damage=="negative":
        original=view.get_contact_data
        def wrong(dt):
            values=list(original(dt));values[-1]=np.full((17,1),-1,dtype=np.int32);return tuple(values)
        view.get_contact_data=wrong
    with pytest.raises(ValueError):m.read_table_contacts(view,row_for(packet()))


def test_motion_starts_only_after_48_measured_quiet_steps():
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import candidate_action
    cfg=json.loads((ROOT/"configs/dh116_tasks/flexible_pinch_candidate_v1.json").read_text())
    assert all(not candidate_action(cfg,i).any() for i in range(1,49))
    assert np.flatnonzero(candidate_action(cfg,49)).tolist()==[1,2]
    assert sum(bool(candidate_action(cfg,i).any()) for i in range(1,239))==160

def test_builder_and_measurement_retained_frames_clear_all_native_aliases():
    path=ROOT/"scripts/dh116_pinch_acquisition_v2.py"
    assert path.exists()
    tree=ast.parse(path.read_text())
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    for name,required in {
        "build_native":{"sim","hand","view","composer","runtime","provider","asset_sim_view",
            "manager_sim_view","snapshot","mesh_poses","contact_view","geometry","geometry_capture"},
        "measure_owned":{"sim","bundle","hand","view","composer","runtime","provider","contact_view",
            "snapshot","mesh_poses"},
    }.items():
        f=functions[name]
        final=next(n for n in f.body if isinstance(n,ast.Try)).finalbody
        cleared={target.id for n in ast.walk(ast.Module(body=final,type_ignores=[]))
            if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and n.value.value is None
            for target in n.targets if isinstance(target,ast.Name)}
        assert required<=cleared
        # Execute the exact cleanup suite while deliberately keeping its frame.
        cleanup=ast.FunctionDef(name="retained",args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],
            kw_defaults=[],defaults=[]),body=[
            ast.Assign(targets=[ast.Name(id=n,ctx=ast.Store())],value=ast.Call(func=ast.Name(id="Owner",ctx=ast.Load()),args=[],keywords=[]))
            for n in sorted(required)]+[
            ast.Assign(targets=[ast.Name(id="refs",ctx=ast.Store())],value=ast.Dict(
                keys=[ast.Constant(n) for n in sorted(required)],
                values=[ast.Call(func=ast.Name(id="ref",ctx=ast.Load()),args=[ast.Name(id=n,ctx=ast.Load())],keywords=[])
                        for n in sorted(required)])),
            ast.Return(value=ast.Tuple(elts=[ast.Call(func=ast.Name(id="frame",ctx=ast.Load()),args=[],keywords=[]),
                ast.Name(id="refs",ctx=ast.Load())],ctx=ast.Load()))],decorator_list=[])
        # Put the return under finally so the returned actual-shaped frame survives.
        cleanup.body=[ast.Try(body=cleanup.body,handlers=[],orelse=[],finalbody=[
            n for n in final if isinstance(n,ast.Assign)])]
        import sys,weakref
        ns={"Owner":type("Owner",(),{}),"ref":weakref.ref,"frame":sys._getframe}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cleanup],type_ignores=[])),"<cleanup-fixture>","exec"),ns)
        held,refs=ns["retained"]()
        assert all(r() is None for r in refs.values())
        assert all(held.f_locals[n] is None for n in required)

def test_frozen_recipe_uses_same_240_total_clock_and_all_stop_debugger():
    path=ROOT/"configs/dh116_execution/deformx_pinch_left_candidate_v2.json"
    assert path.exists()
    cfg=json.loads(path.read_text())
    assert cfg["budget"]["physics_steps"]==240 and cfg["budget"]["updates"]==0
    assert cfg["budget"]["gpu_index"]==3 and cfg["budget"]["environments"]==1
    assert cfg["debugger"]=="gdb_on_segv"
    assert "scripts/dh116_physx_lifetime_debugger_stopped.py" in cfg["source_files"]
    assert cfg["validator"]["line_counts"]["physics.jsonl"]==238
    assert cfg["validator"]["line_counts"]["pairs.jsonl"]==238
    assert cfg["validator"]["line_counts"]["table_contacts.jsonl"]==238
    assert not any("WORKER" in name for name in cfg["environment"])


@pytest.mark.parametrize("stop",["none","table","solver"])
def test_actual_measurement_loop_counts_238_and_blocks_quiet_contact_before_action(tmp_path,monkeypatch,stop):
    import torch
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,expand
    from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import NativeOwners
    from scripts import dh116_pinch_acquisition_v2 as native
    from scripts import verify_dh116_deformx_thumb_contact as thumb
    cfg=json.loads((ROOT/"configs/dh116_tasks/flexible_pinch_candidate_v1.json").read_text())
    cfg["initial_pose"]=json.loads((ROOT/"configs/dh116_tasks/flexible_pinch_pose_candidate_v2.json").read_text())
    start=np.asarray(cfg["initial_pose"]["active_q_rad"])
    sim=SimpleNamespace(current_time_step_index=2,current_time=2/480)
    class Owner:pass
    class Composer:
        def __init__(self):
            self.composed_force_as_torch=torch.zeros((1,17,3))
            self.composed_torque_as_torch=torch.zeros((1,17,3))
        def reset(self):
            self.composed_force_as_torch.zero_();self.composed_torque_as_torch.zero_()
        def set_forces_and_torques(self,forces,torques,body_ids,is_global):
            assert is_global
            self.composed_force_as_torch[:,body_ids]=forces
            self.composed_torque_as_torch[:,body_ids]=torques
    class Hand:
        joint_names=list(JOINTS)
        def __init__(self):
            self.data=SimpleNamespace(joint_pos=torch.tensor(expand(start)[None],dtype=torch.float32),
                joint_vel=torch.zeros((1,11)),body_link_pos_w=torch.zeros((1,17,3)),
                body_link_quat_w=torch.tensor([1.,0,0,0]).repeat(1,17,1),
                body_com_pos_w=torch.zeros((1,17,3)))
            self.targets=self.data.joint_pos.clone()
        def set_joint_position_target(self,value,joint_ids):
            self.targets[:,joint_ids]=value
            self.data.joint_pos=torch.tensor(expand(value[0].numpy())[None],dtype=torch.float32)
        def write_data_to_sim(self):pass
        def update(self,dt):assert dt==1/480
    hand=Hand()
    view=Owner();view.get_dof_position_targets=lambda:hand.targets
    class Rope:
        def contact_pair_samples(self):return packet()
    class Runtime:
        rope=Rope();rod_tick=0;time_origin=0.;episode_uid="cpu-episode";reset_id=1
        def step(self,poses,physics_step):
            if stop=="solver":raise RuntimeError("CPU solver failure")
            self.rod_tick+=12
            return dict(solver_time_s=self.rod_tick/12/480,node_positions=np.zeros((1,41,3)),
                velocities=np.zeros((1,41,3)),distributed_contact=np.zeros((1,41,3)),
                named_wrenches={n:np.zeros(6) for n in [*DISTAL_PAD_LINKS,"table"]})
    provider=Owner();provider.snapshot=lambda **kwargs:dict(version="CPU fixture",backend_sha256="0"*64)
    bundle=SimpleNamespace(hand=hand,hand_view=view,composer=Composer(),runtime=Runtime(),
        provider=provider,asset_sim_view=Owner(),manager_sim_view=Owner(),
        contact_view=ContactView(1 if stop=="table" else 0))
    bundle.rope=bundle.runtime.rope
    bundle.metadata=dict(actual_pad_bindings={n:dict(body_path="/World/DeformXPinch/leftHand/"+n,
        body_index=i,mesh_index=i) for i,n in enumerate(DISTAL_PAD_LINKS)},
        joint_names=list(JOINTS),active_ids=[JOINTS.index(n) for n in ACTIVE],pad_body_ids=list(range(5)),physics_step=2)
    monkeypatch.setattr(native,"build_native",lambda *a:bundle)
    def step(sim,expected,budget_end):
        assert sim.current_time_step_index==expected and budget_end==240
        sim.current_time_step_index+=1;sim.current_time=sim.current_time_step_index/480
        return sim.current_time_step_index
    monkeypatch.setattr(thumb,"step_thumb_physics",step)
    trace=SimpleNamespace(watch=lambda *a,**k:None,checkpoint=lambda *a:None)
    args=SimpleNamespace(output=tmp_path,device="cpu")
    contract=dict(code_config_sha256="a"*64)
    if stop=="solver":
        with pytest.raises(RuntimeError,match="CPU solver failure"):
            native.measure_owned(sim,args,contract,cfg,NativeOwners(),trace,ROOT)
        assert json.loads((tmp_path/"measurement_frame_detached.json").read_text())["operation_raised"] is True
        assert sim.current_time_step_index==3
    else:
        result=native.measure_owned(sim,args,contract,cfg,NativeOwners(),trace,ROOT)
        expected=238 if stop=="none" else 1
        assert result["physics_rows"]==expected and sim.current_time_step_index==expected+2
        assert result["failure"] is None if stop=="none" else "hand-table" in result["failure"]
        for name in ("physics.jsonl","pairs.jsonl","table_contacts.jsonl"):
            assert len((tmp_path/name).read_text().splitlines())==expected
        assert json.loads((tmp_path/"measurement_frame_detached.json").read_text())["clearing_completed"] is True

ENTRY="scripts/verify_dh116_pinch_candidate_v2.py"
WRAPPER="scripts/run_dh116_pinch_work_packages_v2.py"

def test_direct_native_cli_rejects_before_preflight_mkdir_or_isaac(tmp_path):
    import os,subprocess,sys
    env={**os.environ,"CUDA_VISIBLE_DEVICES":""}
    env.pop("DH116_PINCH_WORKER_V2",None)
    output=tmp_path/"native"
    done=subprocess.run([sys.executable,"-m","scripts.verify_dh116_pinch_candidate_v2",
        "--cleanup","explicit","--output",str(output),"--headless","--execute-native"],
        cwd=ROOT,env=env,text=True,capture_output=True,timeout=20)
    assert done.returncode!=0 and "dedicated Pinch v2 worker required" in done.stderr
    assert not output.exists()

def test_wrapper_restores_all_bindings_on_exception(monkeypatch):
    w=importlib.import_module("scripts.run_dh116_pinch_work_packages_v2")
    names=("__file__","ENTRY","WRAPPER","VERSION","WORKER_ENV","AUDIT_NAME",
           "AUDIT_SOURCES","cleanup_case","render_gdb_commands","audit_outcome")
    before={n:getattr(w.base,n) for n in names}
    def fail():
        assert w.base.ENTRY==ENTRY and w.base.WRAPPER==WRAPPER
        assert "set non-stop off" in w.base.render_gdb_commands(Path("/tmp/CPU-only"))
        raise RuntimeError("CPU wrapper failure")
    monkeypatch.setattr(w.base,"main",fail)
    with pytest.raises(RuntimeError,match="CPU wrapper failure"):w.main()
    assert before=={n:getattr(w.base,n) for n in names}

@pytest.mark.parametrize("reject",[False,True])
def test_real_cpu_worker_reentry_keeps_v2_marker_stopped_capture_and_postexit_gate(tmp_path,reject):
    import os,shutil,subprocess
    w=importlib.import_module("scripts.run_dh116_pinch_work_packages_v2")
    scripts=tmp_path/"scripts";scripts.mkdir();(scripts/"__init__.py").touch()
    shutil.copytree(ROOT/"scripts",scripts,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__"))
    for name in set(w.AUDIT_SOURCES):
        (tmp_path/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,tmp_path/name)
    (scripts/"validate_dh116_pinch_v2.py").write_text("""
from pathlib import Path
VERSION='cpu-fixture-old-validator'
def audit_outcome(output,outer,cleanup):
    assert not (Path(outer).parent/'receipt.json').exists()
    return {'lifecycle_qualified':not (Path(__file__).parents[1]/'reject').exists(),
            'teacher_admission':False,'grasp_qualified':False,'cpu_fixture':True}
""")
    if reject:(tmp_path/"reject").touch()
    source=(ROOT/"scripts/run_dh116_work_packages.py").read_text()
    stub="""
def _cpu_native(self,path,spec,output,result,**kwargs):
    from types import SimpleNamespace
    from scripts import verify_dh116_pinch_candidate_v2 as entry
    output.mkdir(parents=True)
    (path/'launch_argv.json').write_text(json.dumps({'argv':['--output',str(output)]}))
    native_debugger_wrapper(path,result)
    capture=(path/'debugger.gdb').read_text()
    assert 'set non-stop off' in capture and 'stopped_thread_count' in capture and str(output) in capture
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    os.environ['DH116_PHYSX_LIFETIME_GDB']='1' # CPU substitute for actual debugger environment.
    entry.require_worker(SimpleNamespace(cleanup='explicit',output=output),self.root)
    (output/'gdb_finalize.json').write_text(json.dumps({'stopped_thread_count':2}))
    result.update(native_qualification=True,native_execution_performed=False,
        worker_root=str(self.root),worker_private=str(self.private),worker_pid=os.getpid(),
        worker_cli_entry=str(Path(sys.argv[0]).resolve()))
Repository._native_child=_cpu_native
Repository.verify_local_target=lambda self:None
Repository.head=lambda self:'b'*40
Repository.registry=lambda self:json.loads((self.private/'registry.json').read_text())
Repository.native_recipe=lambda self,package:json.loads((self.private/'recipe.json').read_text())
Repository.native_plan=lambda self,**kwargs:{'native_qualified':[],'native_ready':['frame']}
Repository.native_resource_blocker=lambda *a,**k:None
NATIVE_SCHEDULER_LOCK=Path(__file__).resolve().parents[1]/'private'/'reservation.lock'
"""
    marker="if __name__ == '__main__':";assert source.count(marker)==1
    (scripts/"run_dh116_work_packages.py").write_text(source.replace(marker,stub+"\n"+marker))
    private=tmp_path/"private";private.mkdir()
    identity=dict(command_spec_sha256="c"*64,code_config_sha256="d"*64)
    spec=dict(entrypoint=ENTRY,argv=["--cleanup","explicit","--output","{output}",
        "--approved-code-config-sha256","a"*64],
        source_files=list(w.AUDIT_SOURCES),debugger="gdb_on_segv",budget={"timeout_seconds":180,"physics_steps":240,"gpu_index":3,"environments":1,"updates":0},
        validator={"equals":{"case":"explicit","version":w.entry.VERSION,"code_config_sha256":"a"*64}})
    (private/"recipe.json").write_text(json.dumps(dict(spec=spec,**identity)))
    package=dict(executor="native_bounded",evidence_scope="cpu_fixture_only",dependencies=[],dependency_scopes={})
    (private/"registry.json").write_text(json.dumps({"packages":{"frame":package}}))
    parent="""
import json
from pathlib import Path
from scripts import run_dh116_pinch_work_packages_v2 as w
root=Path.cwd()
def parent_main():
    repo=w.base.LifetimeRepository(root,evidence_dir=root/'private')
    approval={'version':1,'batch_id':'cpu','output_root':str(root/'artifacts'/'output'),
       'packages':{'frame':{'command_spec_sha256':'c'*64,'code_config_sha256':'d'*64}}}
    print(json.dumps(repo.start_native('frame',approval)))
    return 0
w.base.scheduler.main=parent_main
raise SystemExit(w.main())
"""
    env={**os.environ,"PYTHONPATH":os.pathsep.join((str(tmp_path),str(ROOT/"src"))),
        "CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1"}
    done=subprocess.run([sys.executable,"-c",parent],cwd=tmp_path,env=env,text=True,capture_output=True,timeout=25)
    assert done.returncode==0,done.stdout+done.stderr
    child=json.loads(done.stdout.splitlines()[-1])
    assert child["worker_cli_entry"]==str(tmp_path/WRAPPER)
    assert child["worker_root"]==str(tmp_path) and child["worker_private"]==str(private)
    assert child["native_execution_performed"] is False
    assert child["native_qualification"] is (not reject)
    receipt=next((private/"native-attempts").glob("*/*/*/receipt.json"))
    assert json.loads(receipt.read_text())["native_qualification"] is (not reject)


def test_contact_reporting_authors_only_measurement_api_on_original_17_bodies(monkeypatch):
    # Actual USD body hierarchy with CPU schema doubles; no Kit/schema plugin boot.
    from pxr import Usd,UsdGeom,UsdPhysics,Sdf
    import pxr
    class Rigid:
        def __init__(self,prim):self.prim=prim
        def GetSleepThresholdAttr(self):return self.prim.GetAttribute("physxRigidBody:sleepThreshold")
    class Report:
        def __init__(self,prim):self.prim=prim
        @classmethod
        def Apply(cls,prim):return cls(prim)
        def CreateThresholdAttr(self):return self.prim.CreateAttribute("physxContactReport:threshold",Sdf.ValueTypeNames.Float)
        def GetThresholdAttr(self):return self.prim.GetAttribute("physxContactReport:threshold")
    PhysxSchema=SimpleNamespace(PhysxRigidBodyAPI=Rigid,PhysxContactReportAPI=Report)
    monkeypatch.setattr(pxr,"PhysxSchema",PhysxSchema,raising=False)
    from scripts.verify_dh116_cooked_convex_hulls_native import _sources
    from scripts.dh116_pinch_acquisition_v2 import author_contact_reports_only
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    path="/World/DeformXPinch/leftHand"
    prim=stage.DefinePrim(path,"Xform")
    prim.GetReferences().AddReference(_sources(ROOT)["left"]["path"])
    before={str(p.GetPath()):PhysxSchema.PhysxRigidBodyAPI(p).GetSleepThresholdAttr().Get()
        for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    assert len(before)==17
    records=author_contact_reports_only(stage,path)
    assert set(records)==set(before)
    for name,old in before.items():
        assert records[name]["authored_sleep_threshold_before"]==old
        assert PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(name)).GetSleepThresholdAttr().Get()==old
        assert PhysxSchema.PhysxContactReportAPI(stage.GetPrimAtPath(name)).GetThresholdAttr().Get()==0.


@pytest.mark.parametrize("failure",[False,True])
def test_actual_lifecycle_releases_owned_contacts_before_stop_and_close(tmp_path,failure):
    import sys,weakref
    from scripts import dh116_pinch_acquisition_v2 as native
    class Sim:
        current_time_step_index=0
        current_time=0.
    class Owner:
        def reset(self):pass
    held=[];refs={};events=[]
    def acquire(sim,owners,trace,cleanup):
        value=None
        try:
            for name in native.OWNED_NAMES:
                value=Owner();owners.keep(name,value);refs[name]=weakref.ref(value)
                trace.watch(name,value,owner="CPU fixture",release="CPU scope")
            trace.checkpoint("resources_acquired")
            sim.current_time_step_index=240;sim.current_time=240/480
            held.append(sys._getframe())
            if failure:raise RuntimeError("CPU acquisition failure")
            return dict(physics_rows=238,code_config_sha256="a"*64,failure=None)
        finally:
            value=sim=None
    def close():
        assert all(ref() is None for ref in refs.values())
        assert held[0].f_locals.get("value") is None
        events.append("close")
    kwargs=dict(make_sim=Sim,acquire=acquire,synchronize=lambda:events.append("sync"),
        clear_sim=lambda value:events.append("clear"),stop=lambda:events.append("STOP"),
        manager_is_none=lambda:True,freeze=lambda:False,output=tmp_path,cleanup="explicit")
    app=SimpleNamespace(update=lambda:events.append("update"),close=close)
    if failure:
        with pytest.raises(RuntimeError,match="CPU acquisition failure"):native.run_lifecycle(app,**kwargs)
    else:
        result=native.run_lifecycle(app,**kwargs)
        assert result["measurement_steps"]==result["rod_steps"]==238 and result["physics_steps"]==240
        assert result["lifetime_released_before_close"] is True
    assert events.index("sync")<events.index("STOP")<events.index("close")
    assert events.count("update")==events.count("close")==1


@pytest.mark.parametrize("advance",[False,True])
def test_actual_kinematic_refresh_cannot_advance_sdk_clock(advance):
    from scripts.dh116_pinch_acquisition_v2 import refresh_kinematics_without_step
    sim=SimpleNamespace(current_time_step_index=2,current_time=2/480)
    events=[]
    def refresh():
        events.append("refresh")
        if advance:sim.current_time_step_index+=1
    hand=SimpleNamespace(_physics_sim_view=SimpleNamespace(update_articulations_kinematic=refresh),
        update=lambda dt:events.append(("cache",dt)))
    if advance:
        with pytest.raises(RuntimeError,match="clock"):refresh_kinematics_without_step(sim,hand)
    else:
        refresh_kinematics_without_step(sim,hand)
        assert events==["refresh",("cache",1/480)]
        assert sim.current_time_step_index==2


@pytest.mark.parametrize("damage",[None,"table","clock","live","frame","geometry","stopped","action","target","reaction"])
def test_independent_disk_audit_accepts_complete_cpu_fixture_and_rejects_damage(tmp_path,monkeypatch,damage):
    import sys
    from tests.test_dh116_physx_lifetime_validation import evidence
    from scripts import validate_dh116_pinch_v2 as audit
    from scripts import dh116_pinch_acquisition_v2 as native
    from scripts import audit_dh116_pinch_initial_pose as geometry_audit
    out=evidence(tmp_path)
    test_actual_measurement_loop_counts_238_and_blocks_quiet_contact_before_action(out,monkeypatch,"none")
    def read(name):return json.loads((out/name).read_text())
    def write(name,value):(out/name).write_text(json.dumps(value))
    acquisition=read("acquisition.json")
    measurement=read("measurement.json")
    plan=read("plan.json");plan.update(acquisition)
    plan["physics_budget"]=measurement["physics_budget"]
    plan["contact_reporting"]=[dict(body_path=f"/World/DeformXPinch/leftHand/body{i}",
        contact_report_enabled=True,threshold=0,authored_sleep_threshold_before=None,
        authored_sleep_threshold_after=None) for i in range(17)]
    write("plan.json",plan)
    clocks=[dict(phase=p,clock_available=True,sdk_step=240,sdk_time_s=.5)
        for p in ("before_stop","after_stop","before_app_close","after_app_close")]
    result=read("result.json");result.update(version=native.VERSION,case="explicit",
        acquisition=measurement,clocks=clocks,physics_steps=240,measurement_steps=238,rod_steps=238,
        code_config_sha256=measurement["code_config_sha256"],physics_clock=measurement["physics_clock"],
        physics_budget=measurement["physics_budget"])
    write("result.json",result)
    value=read("before_app_close.json");value.update(case="explicit",clocks=clocks[:3]);write("before_app_close.json",value)
    value=read("app_close_returned.json");value["case"]="explicit";write("app_close_returned.json",value)
    for phase in audit.PHASES:
        value=read(phase+".refs.json")
        value["resources"]["contact_view"]={"alive":phase=="resources_acquired"}
        write(phase+".refs.json",value)
    native.record_detachment(out,False)
    value=read("gdb_finalize.json");value["stopped_thread_count"]=1;write("gdb_finalize.json",value)
    write("initial_geometry.json",{"cpu_fixture":True})
    monkeypatch.setattr(geometry_audit,"load_geometry",lambda *a,**k:dict(reference=dict(reference_layout=dict(
        table_center=[0,0,0],table_size=[1,1,1],wire_points=[[0,0,1],[1,0,1]]))))
    monkeypatch.setitem(sys.modules,"go2_pvcnn.control.dh116.tabletop_pinch_initial_geometry_v2",
        SimpleNamespace(validate_pinch_initial_geometry=lambda *a,**k:dict(valid=damage!="geometry",
            initial_geometry_passed=damage!="geometry")))
    if damage=="table":
        rows=[json.loads(v) for v in (out/"physics.jsonl").read_text().splitlines()]
        data=(out/"table_contacts.jsonl").read_text().splitlines()
        data[0]=json.dumps(module().read_table_contacts(ContactView(1),rows[0]))
        (out/"table_contacts.jsonl").write_text("\n".join(data)+"\n")
    if damage=="clock":
        result["clocks"][-1]["sdk_step"]=241;write("result.json",result)
    if damage=="live":
        value=read("before_app_close.refs.json");value["resources"]["contact_view"]["alive"]=True
        write("before_app_close.refs.json",value)
    if damage=="frame":
        value=read("measurement_frame_detached.json");value["operation_raised"]=True;write("measurement_frame_detached.json",value)
    if damage=="stopped":
        value=read("gdb_finalize.json");value["stopped_thread_count"]=0;write("gdb_finalize.json",value)
    if damage in ("action","target","reaction"):
        data=[json.loads(v) for v in (out/"physics.jsonl").read_text().splitlines()]
        if damage=="action":data[0]["canonical_action24"][1]=.25
        if damage=="target":data[0]["sdk_joint_target_rad"][plan["active_ids"][1]]+=.01
        if damage=="reaction":data[0]["post_state"]["com_pos"][0][0]=float("nan")
        (out/"physics.jsonl").write_text("".join(json.dumps(v)+"\n" for v in data))
    result=audit.audit_outcome(out,tmp_path/"outer_exit.json","explicit")
    assert result["lifecycle_qualified"] is (damage is None),result.get("reason")
    assert result["teacher_admission"] is result["grasp_qualified"] is False

def test_builder_keeps_original_articulation_and_free_rod_physics_calls():
    old=ast.parse((ROOT/"scripts/dh116_physx_lifetime_frame_acquisition_v2.py").read_text())
    new=ast.parse((ROOT/"scripts/dh116_pinch_acquisition_v2.py").read_text())
    def calls(tree,name):
        return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Call)
            and ((isinstance(n.func,ast.Name) and n.func.id==name)
                or (isinstance(n.func,ast.Attribute) and n.func.attr==name))]
    for name in ("Articulation","FreeRodRuntime","install_hand_mimic","configure_mimic_compliance",
                 "bind_hand_materials","hand_material_contract"):
        assert calls(old,name)==calls(new,name)
    assert len(calls(new,"record_reset"))==1
    assert len(calls(new,"step_thumb_physics"))==1
    assert len(calls(new,"step"))==1  # Only runtime.step; no direct SDK step hidden in setup.
    assert len(calls(new,"enable_contact_pair_capture"))==1

def test_cpu_preflight_binds_exact_successful_pair_receipt_without_importing_native(tmp_path):
    import os,subprocess
    code="""
from pathlib import Path
import json,sys
from scripts.verify_dh116_pinch_candidate_v2 import preflight,PAIR_RECEIPT_SHA
contract,cfg=preflight(Path.cwd())
assert contract['pair_calibration_receipt_sha256']==PAIR_RECEIPT_SHA
assert cfg['initial_pose']['version']=='dh116-pinch-pose-candidate-v2'
assert 'ropesim' not in sys.modules and 'isaaclab' not in sys.modules
assert contract['native_execution_performed'] is False and contract['teacher_admission'] is False
print(json.dumps({'cpu_preflight':True,'cuda_constructed':False,'isaac_launched':False}))
"""
    done=subprocess.run([sys.executable,"-c",code],cwd=ROOT,
        env={**os.environ,"CUDA_VISIBLE_DEVICES":""},text=True,capture_output=True,timeout=30)
    assert done.returncode==0,done.stdout+done.stderr
    assert json.loads(done.stdout)["cuda_constructed"] is False

def test_generic_scheduler_also_accepts_complete_cpu_fixture(tmp_path,monkeypatch):
    from scripts import run_dh116_work_packages as scheduler
    test_independent_disk_audit_accepts_complete_cpu_fixture_and_rejects_damage(tmp_path,monkeypatch,None)
    output=tmp_path/"native"
    spec=json.loads((ROOT/"configs/dh116_execution/deformx_pinch_left_candidate_v2.json").read_text())
    expected=spec["validator"]["equals"]["code_config_sha256"]
    for name in ("measurement.json","plan.json","result.json"):
        path=output/name;record=json.loads(path.read_text());record["code_config_sha256"]=expected
        if name=="result.json":record["acquisition"]["code_config_sha256"]=expected
        path.write_text(json.dumps(record))
    result=scheduler.validate_native_artifacts(spec,output)
    assert result["ledger_counts"]["physics.jsonl"]==238
    assert result["ledger_counts"]["pairs.jsonl"]==238

def test_recipe_candidate_digest_and_registered_source_set_match_current_preflight():
    from scripts import verify_dh116_pinch_candidate_v2 as entry
    from scripts import run_dh116_work_packages as scheduler
    from scripts import run_dh116_pinch_work_packages_v2 as wrapper
    spec=json.loads((ROOT/"configs/dh116_execution/deformx_pinch_left_candidate_v2.json").read_text())
    contract,cfg=entry.preflight(ROOT)
    flag="--approved-code-config-sha256"
    assert spec["argv"][spec["argv"].index(flag)+1]==contract["code_config_sha256"]
    assert spec["validator"]["equals"]["code_config_sha256"]==contract["code_config_sha256"]
    assert set(wrapper.AUDIT_SOURCES)<=set(spec["source_files"])
    recipe=scheduler.Repository(ROOT).native_recipe({"command_spec":
        "configs/dh116_execution/deformx_pinch_left_candidate_v2.json","tests":[]})
    assert len(recipe["code_config_sha256"])==64
