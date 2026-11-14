"""Execute original spawn_scene, with only CPU SDK construction boundaries."""
import ast
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
MODULE="go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation_v2"

@pytest.fixture(scope="module")
def prepared():
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    scene._load_offline_usd_schema()
    return scene.prepare_scene(ROOT)

@pytest.fixture
def original_spawn(prepared):
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,expand
    from go2_pvcnn.control.dh116.runtime_mimic import install_hand_mimic,configure_mimic_compliance
    from go2_pvcnn.control.dh116.tabletop_load_parameters import LEGACY_LOAD_PARAMETERS
    from go2_pvcnn.control.dh116.tabletop_velocity_limits import ACTIVE_VELOCITY_LIMIT_RAD_S,passive_velocity_limit
    stage=Usd.Stage.CreateInMemory();created=[]
    def config(**kwargs):return SimpleNamespace(**kwargs)
    def cuboid(**kwargs):
        cfg=config(**kwargs)
        def spawn(path,cfg,*,translation):
            created.append(("table",path,cfg))
            cube=UsdGeom.Cube.Define(stage,path);cube.CreateSizeAttr(1.)
            cube.AddTranslateOp().Set(Gf.Vec3d(*translation))
            cube.AddScaleOp().Set(Gf.Vec3f(*cfg.size))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        cfg.func=spawn;return cfg
    def material(**kwargs):
        cfg=config(**kwargs)
        def spawn(path,cfg):
            created.append(("material",path,kwargs.copy()))
            scene._material(stage,path,kwargs)
        cfg.func=spawn;return cfg
    def articulation(cfg):
        created.append(("hand",cfg.prim_path,cfg))
        prim=UsdGeom.Xform.Define(stage,cfg.prim_path).GetPrim()
        prim.GetReferences().AddReference(cfg.spawn.usd_path)
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*cfg.init_state.pos))
        q=cfg.init_state.rot;orientation=prim.GetAttribute("xformOp:orient")
        quaternion=Gf.Quatf(q[0],Gf.Vec3f(*q[1:])) if str(orientation.GetTypeName())=="quatf" else Gf.Quatd(q[0],Gf.Vec3d(*q[1:]))
        orientation.Set(quaternion)
        return SimpleNamespace(cfg=cfg)
    def forbidden(*args,**kwargs):raise AssertionError("external_wire must not construct rigid wire or ContactSensor")
    class ArticulationCfg(SimpleNamespace):InitialStateCfg=staticmethod(config)
    utils=SimpleNamespace(CuboidCfg=cuboid,CollisionPropertiesCfg=config,
        UsdFileCfg=config,RigidBodyPropertiesCfg=config,ArticulationRootPropertiesCfg=config,RigidBodyMaterialCfg=material)
    source=ROOT/"src/go2_pvcnn/control/dh116/tabletop_vec_isaac.py"
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=="spawn_scene")
    scope=dict(__name__="go2_pvcnn.control.dh116._cpu_spawn_ast",__package__="go2_pvcnn.control.dh116",
        copy=copy,np=np,UsdGeom=UsdGeom,sim_utils=utils,ArticulationCfg=ArticulationCfg,Articulation=articulation,
        ImplicitActuatorCfg=config,ACTIVE=ACTIVE,JOINTS=JOINTS,expand=expand,
        LEGACY_LOAD_PARAMETERS=LEGACY_LOAD_PARAMETERS,ACTIVE_VELOCITY_LIMIT_RAD_S=ACTIVE_VELOCITY_LIMIT_RAD_S,
        passive_velocity_limit=passive_velocity_limit,install_hand_mimic=install_hand_mimic,
        configure_mimic_compliance=configure_mimic_compliance,RigidObject=forbidden,
        RigidObjectCfg=forbidden,ContactSensor=forbidden,ContactSensorCfg=forbidden)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),"exec"),scope)
    return scope["spawn_scene"],SimpleNamespace(stage=stage),created

def test_original_external_spawn_reproduces_missing_wire_center_before_any_creation(prepared,original_spawn):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    spawn,sim,created=original_spawn
    args,layout,sources=scene.spawn_parameters(ROOT,prepared)
    assert "wire_center" not in layout
    with pytest.raises(KeyError,match="wire_center"):
        spawn(sim,args,layout,sources,prepared["prefix"],np.zeros(3),external_wire=True)
    assert created==[]

def test_successor_runs_complete_original_external_spawn_without_rigid_wire(prepared,original_spawn):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    from go2_pvcnn.control.dh116.linear_coupling import JOINTS,expand
    module=importlib.import_module(MODULE);spawn,sim,created=original_spawn
    old=scene.spawn_parameters;original_args,original_layout,original_sources=old(ROOT,prepared)
    with module.external_wire_layout():
        args,layout,sources=scene.spawn_parameters(ROOT,prepared)
        np.testing.assert_allclose(layout["wire_center"],np.asarray(original_layout["wire_points"]).mean(0),rtol=0,atol=0)
        assert vars(args)==vars(original_args) and sources==original_sources
        for key in original_layout:
            if key not in ("hand_roots","active_joints"):assert layout[key]==original_layout[key]
        result=spawn(sim,args,layout,sources,prepared["prefix"],np.zeros(3),external_wire=True)
    assert scene.spawn_parameters is old
    assert result["wire"] is None and result["sensors"]==[] and len(result["hands"])==2
    assert not sim.stage.GetPrimAtPath(prepared["prefix"]+"/Wire")
    assert len([row for row in created if row[0]=="table"])==1
    assert len([row for row in created if row[0]=="hand"])==2
    assert len([row for row in created if row[0]=="material"])==2
    for side,hand in zip(("left","right"),result["hands"]):
        cfg=hand.cfg
        assert cfg.spawn.activate_contact_sensors is True
        assert cfg.spawn.articulation_props.solver_position_iteration_count==64
        assert cfg.spawn.articulation_props.solver_velocity_iteration_count==16
        assert cfg.actuators["active"].effort_limit_sim==.15
        assert set(cfg.actuators)=={"active","passive","wrist","wrist_rotation"}
        np.testing.assert_allclose([cfg.init_state.joint_pos[n] for n in JOINTS],
            expand(original_layout["active_joints"][side]),atol=0,rtol=0)
        assert sim.stage.GetPrimAtPath(prepared["prefix"]+"/"+side+"Hand/wrist_fixed")
    # The authored bookkeeping key cannot mutate the prepared seed or points.
    assert "wire_center" not in original_layout

def test_binding_restores_original_after_error(prepared):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    old=scene.spawn_parameters
    with pytest.raises(RuntimeError,match="intentional"):
        with importlib.import_module(MODULE).external_wire_layout():
            assert scene.spawn_parameters is not old
            raise RuntimeError("intentional")
    assert scene.spawn_parameters is old

@pytest.mark.parametrize("points",[[],[[0,0]],[[0,0,0],[float("nan"),0,0]]])
def test_invalid_authored_wire_points_are_refused(monkeypatch,points):
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    previous=lambda project,prepared:(SimpleNamespace(),dict(wire_points=points),{})
    monkeypatch.setattr(scene,"spawn_parameters",previous)
    with importlib.import_module(MODULE).external_wire_layout():
        with pytest.raises(ValueError,match="wire_points"):scene.spawn_parameters(ROOT,{})
    assert scene.spawn_parameters is previous

def test_module_import_has_no_early_sdk_dependency():
    import os,subprocess,sys
    code="""import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('pxr','isaaclab','isaacsim','omni','ropesim'):
            raise RuntimeError('early SDK import:'+name)
sys.meta_path.insert(0,Guard())
from go2_pvcnn.control.dh116.tabletop_dual_hand_scene_preparation_v2 import external_wire_layout
print('SAFE_IMPORT')
"""
    result=subprocess.run([sys.executable,"-c",code],cwd=ROOT,env={**os.environ,"CUDA_VISIBLE_DEVICES":""},
        capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=="SAFE_IMPORT"
