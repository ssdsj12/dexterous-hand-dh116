"""Original installed Cuboid hierarchy on a CPU USD stage; no Kit or physics."""
import ast
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from test_dh116_bimanual_table_contacts import context, clock

MODULE = "go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts_v2"
SHAPES = Path("/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab/sim/spawners/shapes/shapes.py")
SHAPES_SHA = "053227b88000f63f707f6719d81b342bfa59c04ba78126e4e8b6f3760c7df4ca"

def api():
    return importlib.import_module(MODULE)

def installed_table(context):
    """Execute both actual shape function bodies; replace only USD transport glue."""
    from pxr import UsdGeom, UsdPhysics
    stage, _, prepared, *_ = context
    table = prepared["prefix"]+"/Table"
    stage.RemovePrim(table)
    assert hashlib.sha256(SHAPES.read_bytes()).hexdigest() == SHAPES_SHA
    tree = ast.parse(SHAPES.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ("spawn_cuboid", "_spawn_geom_from_prim_type")]
    assert len(nodes) == 2
    for node in nodes:
        node.decorator_list = []
        node.returns = None
        for arg in node.args.args+node.args.kwonlyargs:
            arg.annotation = None
    def create_prim(path, prim_type, *, stage, attributes=None, **kwargs):
        prim = stage.DefinePrim(path, prim_type)
        for key, value in (attributes or {}).items():
            prim.GetAttribute(key).Set(value)
        return prim
    def collision(path, cfg, *, stage):
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(path)).CreateCollisionEnabledAttr().Set(True)
    scope = dict(get_current_stage=lambda: stage, create_prim=create_prim,
                 schemas=SimpleNamespace(define_collision_properties=collision))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SHAPES), "exec"), scope)
    cfg = SimpleNamespace(size=(.8, .6, .05), collision_props=object(),
                          visual_material=None, physics_material=None, mass_props=None, rigid_props=None)
    scope["spawn_cuboid"](table, cfg)
    assert stage.GetPrimAtPath(table).GetTypeName() == "Xform"
    assert not stage.GetPrimAtPath(table).HasAPI(UsdPhysics.CollisionAPI)
    assert stage.GetPrimAtPath(table+"/geometry/mesh").HasAPI(UsdPhysics.CollisionAPI)
    return table

def test_installed_cuboid_reproduces_old_root_failure(context):
    table = installed_table(context)
    from go2_pvcnn.control.dh116 import tabletop_bimanual_table_contacts as old
    with pytest.raises(ValueError, match="actual original table collider required"):
        old.open_bimanual_table_contacts(*context[:3], read_clock=context[3])
    assert not context[5]

def test_successor_resolves_installed_descendant_without_views_or_physics_change(context):
    table = installed_table(context)
    before = context[0].GetRootLayer().ExportToString()
    source = api().open_bimanual_table_contacts(*context[:3], read_clock=context[3])
    assert context[0].GetRootLayer().ExportToString() == before
    assert not source.views and not context[5]
    assert source.metadata["table_path"] == table
    assert source.metadata["table_colliders"]["collider_paths"] == [table+"/geometry/mesh"]
    assert source.metadata["table_colliders"]["colliders"][0]["rigid_body_ancestors"] == []
    assert all(len(paths)==17 for paths in source.metadata["paths"].values())
    context[4]["clock"] = clock(3)
    from go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts import read_bimanual_table_contacts
    result = read_bimanual_table_contacts(source, read_clock=context[3])
    assert result["available"] is False and result["total_contacts"] is None and result["hands"] is None
    assert "static GPU" in result["reason"]
    assert not result["native_qualified"] and not result["teacher_admission"]
    json.dumps(result, allow_nan=False)
    source.release()
    assert source.stage is None

@pytest.mark.parametrize("where", ["root", "geometry", "mesh", "above_table"])
def test_rigid_body_any_ancestor_rejected_without_scene_conversion(context, where):
    from pxr import UsdPhysics
    table = installed_table(context)
    path = {"root":table, "geometry":table+"/geometry", "mesh":table+"/geometry/mesh",
            "above_table":context[2]["prefix"]}[where]
    UsdPhysics.RigidBodyAPI.Apply(context[0].GetPrimAtPath(path))
    with pytest.raises(ValueError, match="static"):
        api().open_bimanual_table_contacts(*context[:3], read_clock=context[3])
    assert not context[5]

@pytest.mark.parametrize("fault", ["missing", "inactive", "no_collision", "disabled"])
def test_missing_actual_enabled_collision_is_not_na_success(context, fault):
    from pxr import UsdPhysics
    table = installed_table(context)
    stage = context[0]
    if fault == "missing": stage.RemovePrim(table)
    elif fault == "inactive": stage.GetPrimAtPath(table).SetActive(False)
    elif fault == "no_collision": stage.GetPrimAtPath(table+"/geometry/mesh").RemoveAPI(UsdPhysics.CollisionAPI)
    else: UsdPhysics.CollisionAPI(stage.GetPrimAtPath(table+"/geometry/mesh")).CreateCollisionEnabledAttr().Set(False)
    with pytest.raises(ValueError, match="table"):
        api().open_bimanual_table_contacts(*context[:3], read_clock=context[3])
    assert not context[5]

def test_all_actual_descendants_preserved_and_disabled_not_counted(context):
    from pxr import UsdGeom, UsdPhysics
    table = installed_table(context)
    second = UsdGeom.Cube.Define(context[0],table+"/other/shape").GetPrim()
    UsdPhysics.CollisionAPI.Apply(second)
    disabled = UsdGeom.Cube.Define(context[0],table+"/disabled").GetPrim()
    UsdPhysics.CollisionAPI.Apply(disabled).CreateCollisionEnabledAttr().Set(False)
    record = api().resolve_static_table_colliders(context[0], table)
    assert record["collider_paths"] == [table+"/geometry/mesh", table+"/other/shape"]
    assert record["disabled_collider_paths"] == [table+"/disabled"]

@pytest.mark.parametrize("fault", ["shared_hand", "shared_view", "different_sim", "wrong_link", "wrong_joints"])
def test_existing_hand_identity_checks_retained(context, fault):
    installed_table(context)
    hands = context[1]
    if fault == "shared_hand": hands[1] = hands[0]
    elif fault == "shared_view": hands[1].root_physx_view = hands[0].root_physx_view
    elif fault == "different_sim": hands[1]._physics_sim_view = object()
    elif fault == "wrong_link": hands[1].root_physx_view.link_paths[0][0] += "_wrong"
    else: hands[1].joint_names[0] = "wrong"
    with pytest.raises(ValueError):
        api().open_bimanual_table_contacts(*context[:3], read_clock=context[3])
    assert not context[5]

def test_clock_change_while_reading_is_rejected(context):
    installed_table(context)
    calls = []
    def read():
        calls.append(1)
        return clock(2 if len(calls)==1 else 3)
    with pytest.raises(ValueError, match="clock changed"):
        api().open_bimanual_table_contacts(*context[:3], read_clock=read)

def test_module_import_has_no_isaac_or_usd_side_effect():
    import os, subprocess, sys
    code = """import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('pxr','isaaclab','isaacsim','omni','ropesim'):
   raise RuntimeError('forbidden:'+name)
sys.meta_path.insert(0,Guard())
import go2_pvcnn.control.dh116.tabletop_bimanual_table_contacts_v2
print('CLEAN')
"""
    result = subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,
                            env={**os.environ,"CUDA_VISIBLE_DEVICES":""},timeout=15)
    assert result.returncode == 0, result.stdout+result.stderr
    assert result.stdout.strip() == "CLEAN"
