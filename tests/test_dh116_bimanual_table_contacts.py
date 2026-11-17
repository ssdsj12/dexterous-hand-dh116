"""Installed API CPU execution and explicit USD reporting, never an Isaac App."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib
import json
import numpy as np
import pytest

MODULE="go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts"
SDK=Path("/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extsPhysics/omni.physics.tensors/omni/physics/tensors/impl/api.py")
def api():return importlib.import_module(MODULE)

def clock(step=2):
    return dict(run_uuid="cpu",env_id=0,episode_uid="cpu:episode",reset_id=1,rod_uid="rod",
        physics_step=step,rod_tick=(step-2)*12,timestamp_s=step/480,solver_time_s=(step-2)/480,clock_valid=True)

@pytest.fixture
def context():
    from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation import _load_offline_usd_schema
    _load_offline_usd_schema()
    from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS
    from go2_pvcnn.control.dh116.tabletop_joint_target_readback import WRIST_LINEAR,WRIST_ANGULAR
    # The installed class is executed against CPU buffers, not imported through Kit.
    tree=ast.parse(SDK.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="RigidContactView")
    scope=dict(float32=np.float32,uint32=np.uint32)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(SDK),"exec"),scope)
    class Front:
        def create_tensor(self,shape,dtype):
            a=np.zeros(shape,dtype=dtype);return a,a
    backends=[]
    class SimView:
        def create_rigid_contact_view(self,paths,filter_patterns,max_contact_data_count):
            assert len(paths)==17 and filter_patterns==[[prefix+"/Table"]]*17
            backend=SimpleNamespace(sensor_count=17,filter_count=1,
                max_contact_data_count=max_contact_data_count,
                sensor_paths=list(reversed(paths)),filter_paths=deepcopy(filter_patterns))
            def get_data(force,point,normal,separation,counts,starts,dt):
                backend.dt=dt;counts[:]=0;starts[:]=0
                counts[0,0]=1;force[0,0]=.004/dt;normal[0]=[0,0,1]
                return True
            backend.get_contact_data=get_data
            backends.append(backend)
            return scope["RigidContactView"](backend,Front())
    stage=Usd.Stage.CreateInMemory();prefix="/World/DeformXDualHand"
    table=UsdGeom.Cube.Define(stage,prefix+"/Table").GetPrim();UsdPhysics.CollisionAPI.Apply(table)
    # Synthetic dynamic partner only exercises API parsing. Production table is
    # static and has an explicitly unsupported GPU filter capability.
    UsdPhysics.RigidBodyAPI.Apply(table)
    prepared=dict(prefix=prefix,hands={});hands=[];sim_view=SimView()
    names=["base_link"]+[f"finger{i}_link" for i in (11,12,13,14,21,22,23,31,32,33,41,42,43,51,52,53)]
    for side in ("left","right"):
        root=prefix+"/"+side+"Hand";prepared["hands"][side]=dict(source_colliders=dict.fromkeys(names))
        for name in names:
            prim=UsdGeom.Xform.Define(stage,root+"/"+name).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.)
        hands.append(SimpleNamespace(body_names=names+["extra"+str(i) for i in range(6)],
            joint_names=list(JOINTS+WRIST_LINEAR+WRIST_ANGULAR),_physics_sim_view=sim_view,
            root_physx_view=SimpleNamespace(prim_paths=[root+"/wrist_fixed"],
                link_paths=[[root+"/"+n for n in names+["extra"+str(i) for i in range(6)]]])))
    state=dict(clock=clock())
    return stage,hands,prepared,lambda:deepcopy(state["clock"]),state,backends

def open_source(context):
    stage,hands,prepared,read,state,backends=context
    return api().open_bimanual_table_contacts(stage,hands,prepared,read_clock=read)

def test_real_sdk_six_buffers_dt_and_actual_reordered_sensor_paths(context):
    source=open_source(context);context[4]["clock"]=clock(3)
    result=api().read_bimanual_table_contacts(source,read_clock=context[3])
    assert result["available"] and result["total_contacts"]==2
    assert result["dt_s"]==1/480 and result["native_qualified"] is False
    for side,backend in zip(("left","right"),context[5]):
        row=result["hands"][side];contact=row["contacts"][0]
        assert backend.dt==1/480 and row["sensor_count"]==17
        assert contact["body_path"]==backend.sensor_paths[0]
        assert contact["table_path"]==context[2]["prefix"]+"/Table"
        assert contact["normal_force_n"]==pytest.approx(1.92)
    json.dumps(result,allow_nan=False)

def test_missing_source_is_na_not_zero():
    result=api().read_bimanual_table_contacts(None,read_clock=lambda:clock(3))
    assert result["available"] is False and result["total_contacts"] is None
    assert result["hands"] is None

def test_absent_reporter_does_not_create_views_or_invent_zero(context):
    from pxr import PhysxSchema
    prim=context[0].GetPrimAtPath(context[1][1].root_physx_view.link_paths[0][0])
    prim.RemoveAPI(PhysxSchema.PhysxContactReportAPI)
    source=open_source(context);context[4]["clock"]=clock(3)
    result=api().read_bimanual_table_contacts(source,read_clock=context[3])
    assert result["available"] is False and result["total_contacts"] is None
    assert context[5]==[]

@pytest.mark.parametrize("fault",["wrong_sensor","wrong_table","saturated","overlap","nan","clock","reset","stale"])
def test_invalid_source_or_buffers_rejected(context,fault):
    source=open_source(context);context[4]["clock"]=clock(3);backend=context[5][0]
    if fault=="wrong_sensor":backend.sensor_paths[0]="/World/Rod"
    elif fault=="wrong_table":backend.filter_paths[0]=["/World/Rod"]
    elif fault in ("saturated","overlap","nan"):
        prior=backend.get_contact_data
        def bad(*args):
            prior(*args)
            if fault=="saturated":args[4][0,0]=backend.max_contact_data_count
            elif fault=="overlap":args[4][1,0]=1
            else:args[0][0,0]=np.nan
            return True
        backend.get_contact_data=bad
    elif fault=="clock":
        prior=backend.get_contact_data
        def bad(*args):
            prior(*args);context[4]["clock"]=clock(4);return True
        backend.get_contact_data=bad
    elif fault=="reset":context[4]["clock"]["reset_id"]=2
    elif fault=="stale":context[4]["clock"]=clock(2)
    with pytest.raises(ValueError):api().read_bimanual_table_contacts(source,read_clock=context[3])

def test_original_dual_spawn_already_requests_reporting():
    p=Path(__file__).resolve().parents[1]/"src/go2_pvcnn/control/dh116/tabletop_vec_isaac.py"
    tree=ast.parse(p.read_text());spawn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="spawn_scene")
    calls=[n for n in ast.walk(spawn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="UsdFileCfg"]
    assert len(calls)==1 and any(k.arg=="activate_contact_sensors" and isinstance(k.value,ast.Constant) and k.value.value is True for k in calls[0].keywords)


def test_import_in_fresh_process_does_not_load_usd_or_isaac():
    import os,subprocess,sys
    code="""import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('pxr','isaaclab','isaacsim','omni','ropesim'):
            raise RuntimeError('forbidden early import:'+name)
sys.meta_path.insert(0,Guard())
from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts import read_bimanual_table_contacts
print('BYTE_IMPORT')
"""
    result=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,
        cwd=Path(__file__).resolve().parents[1],env={**os.environ,"CUDA_VISIBLE_DEVICES":""},timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.strip()=="BYTE_IMPORT"


def test_reporting_removed_after_open_is_na(context):
    from pxr import PhysxSchema
    source=open_source(context);context[4]["clock"]=clock(3)
    context[0].GetPrimAtPath(context[1][0].root_physx_view.link_paths[0][0]).RemoveAPI(PhysxSchema.PhysxContactReportAPI)
    result=api().read_bimanual_table_contacts(source,read_clock=context[3])
    assert result["available"] is False and result["total_contacts"] is None


def test_release_drops_views_and_never_reuses_previous_zero(context):
    import weakref,gc
    source=open_source(context);views=[weakref.ref(v) for v in source.views]
    source.release();gc.collect();context[4]["clock"]=clock(3)
    assert all(ref() is None for ref in views)
    result=api().read_bimanual_table_contacts(source,read_clock=context[3])
    assert result["available"] is False and result["total_contacts"] is None


@pytest.mark.parametrize("layout",["broadcast","flat"])
def test_resolved_single_table_layouts_preserve_raw_metadata(context,layout):
    source=open_source(context);context[4]["clock"]=clock(3)
    for backend in context[5]:backend.filter_paths=[context[2]["prefix"]+"/Table"]*(1 if layout=="broadcast" else 17)
    result=api().read_bimanual_table_contacts(source,read_clock=context[3])
    assert result["available"] and len(result["hands"]["left"]["filter_paths"])==(1 if layout=="broadcast" else 17)


def test_zero_count_is_only_reported_from_valid_fresh_buffers(context):
    source=open_source(context);context[4]["clock"]=clock(3)
    for backend in context[5]:
        def zero(force,point,normal,separation,counts,starts,dt):
            counts[:]=0;starts[:]=0;return True
        backend.get_contact_data=zero
    result=api().read_bimanual_table_contacts(source,read_clock=context[3])
    assert result["available"] and result["total_contacts"]==0
    with pytest.raises(ValueError,match="fresh"):
        api().read_bimanual_table_contacts(source,read_clock=context[3])


@pytest.mark.parametrize("what",["body","table"])
def test_deactivated_actual_prim_is_na(context,what):
    source=open_source(context);context[4]["clock"]=clock(3)
    path=context[1][0].root_physx_view.link_paths[0][0] if what=="body" else context[2]["prefix"]+"/Table"
    context[0].GetPrimAtPath(path).SetActive(False)
    result=api().read_bimanual_table_contacts(source,read_clock=context[3])
    assert result["available"] is False and result["total_contacts"] is None


def test_original_static_table_gpu_filter_is_na_before_view_creation(context):
    from pxr import UsdPhysics
    table=context[0].GetPrimAtPath(context[2]["prefix"]+"/Table")
    table.RemoveAPI(UsdPhysics.RigidBodyAPI)
    source=open_source(context);context[4]["clock"]=clock(3)
    result=api().read_bimanual_table_contacts(source,read_clock=context[3])
    assert not context[5]
    assert result["available"] is False and result["total_contacts"] is None
    assert "static" in result["reason"] and "GPU" in result["reason"]


def test_table_source_accepts_original_container_backend_instance(context):
    from pxr import UsdPhysics
    table=context[0].GetPrimAtPath(context[2]["prefix"]+"/Table")
    table.RemoveAPI(UsdPhysics.RigidBodyAPI)
    for hand in context[1]:
        view=hand.root_physx_view
        view.prim_paths=[view.prim_paths[0].removesuffix("/wrist_fixed")]
    source=open_source(context)
    context[4]["clock"]=clock(3)
    result=api().read_bimanual_table_contacts(source,read_clock=context[3])
    assert result["total_contacts"] is None and not context[5]
