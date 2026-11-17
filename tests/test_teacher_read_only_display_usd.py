"""CPU USD checks only; these do not count as native grasp experiments."""
from pathlib import Path
import math
import numpy as np
import pytest
import torch
pytest.importorskip("pxr.Usd",reason="run with installed SDK USD libraries, without Kit")
from pxr import Usd,UsdGeom,UsdPhysics,Gf

def fixture_stage(unsafe=False):
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage,"/World/Physical")
    body=UsdGeom.Xform.Define(stage,"/World/Physical/base_link")
    body.AddTranslateOp().Set(Gf.Vec3d(3,4,5))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdGeom.Xform.Define(stage,"/World/Physical/base_link/visuals")
    mesh=UsdGeom.Mesh.Define(stage,"/World/Physical/base_link/visuals/mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(0,0,0),Gf.Vec3f(1,0,0),Gf.Vec3f(0,1,0)])
    mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2])
    if unsafe:UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return stage

def binding():
    return dict(env_id=0,hand="left",source_root="/World/Physical",body_names=["base_link"])

def test_copies_visual_geometry_only_and_never_adds_physics():
    from go2_pvcnn.control.dh116.tabletop_teacher_read_only_display import ReadOnlyHandDisplay
    stage=fixture_stage();owner=ReadOnlyHandDisplay(stage,[binding()])
    source=stage.GetPrimAtPath("/World/Physical/base_link")
    assert source.HasAPI(UsdPhysics.RigidBodyAPI)
    assert source.GetAttribute("xformOp:translate").Get()==Gf.Vec3d(3,4,5)
    row=owner.links[0]
    mesh=UsdGeom.Mesh(stage.GetPrimAtPath(row["display_path"]+"/visuals/mesh"))
    assert list(mesh.GetPointsAttr().Get())==[Gf.Vec3f(0,0,0),Gf.Vec3f(1,0,0),Gf.Vec3f(0,1,0)]
    assert owner.audit_no_physics()["passed"] is True

def test_measured_world_pose_matches_display_under_rotated_parent():
    from go2_pvcnn.control.dh116.tabletop_teacher_read_only_display import ReadOnlyHandDisplay
    stage=fixture_stage();owner=ReadOnlyHandDisplay(stage,[binding()])
    parent=UsdGeom.Xformable(stage.GetPrimAtPath(owner.links[0]["display_path"]).GetParent())
    parent.AddTranslateOp().Set(Gf.Vec3d(8,2,-1))
    parent.AddOrientOp().Set(Gf.Quatf(math.cos(.3),Gf.Vec3f(0,0,math.sin(.3))))
    poses={(0,"left"):torch.tensor([[.5,.2,.4,math.cos(.5),0,0,math.sin(.5)]],dtype=torch.float64)}
    result=owner.sync(poses,read_clock=lambda:(8,8/480))
    assert result["max_position_error_m"]<1e-10
    assert result["max_rotation_error_rad"]<1e-6
    assert result["physics_clock_unchanged"] is True
    assert stage.GetPrimAtPath("/World/Physical/base_link").GetAttribute("xformOp:translate").Get()==Gf.Vec3d(3,4,5)

def test_refuses_physics_in_visual_subtree_before_hiding_original():
    from go2_pvcnn.control.dh116.tabletop_teacher_read_only_display import ReadOnlyHandDisplay
    stage=fixture_stage(unsafe=True)
    with pytest.raises(ValueError,match="physics"):
        ReadOnlyHandDisplay(stage,[binding()])
    assert not stage.GetPrimAtPath("/World/Physical/base_link").GetAttribute("visibility").HasAuthoredValueOpinion()

def test_sync_rejects_invalid_quaternion_without_touching_source():
    from go2_pvcnn.control.dh116.tabletop_teacher_read_only_display import ReadOnlyHandDisplay
    owner=ReadOnlyHandDisplay(fixture_stage(),[binding()])
    with pytest.raises(ValueError,match="quaternion"):
        owner.sync({(0,"left"):torch.zeros((1,7))},read_clock=lambda:(0,0.))

@pytest.mark.parametrize("filename",["d134680ab619_DH116_left.usd","3046ad2ed855_DH116_right.usd"])
def test_real_instanced_dh116_visuals_keep_original_mesh_points(filename):
    from go2_pvcnn.control.dh116.tabletop_teacher_read_only_display import ReadOnlyHandDisplay
    root=Path(__file__).resolve().parents[1]
    path=root/"assets/fixed_dual_panda_dh116_v1/dependencies"/filename
    stage=Usd.Stage.Open(str(path));source_root=str(stage.GetDefaultPrim().GetPath())
    names=[p.GetName() for p in stage.GetDefaultPrim().GetChildren()
        if p.GetName()=="base_link" or p.GetName().startswith("finger")]
    owner=ReadOnlyHandDisplay(stage,[dict(env_id=0,hand="left",source_root=source_root,body_names=names)])
    assert len(owner.links)==17
    assert owner.audit_no_physics()["passed"]
    for row in owner.links:
        def meshes(prefix):
            return [p for p in Usd.PrimRange(stage.GetPrimAtPath(prefix),Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
        source=meshes(row["source_visuals"])
        copied=meshes(row["display_path"]+"/visuals")
        assert len(source)==len(copied)>0
        for a,b in zip(source,copied):
            np.testing.assert_array_equal(UsdGeom.Mesh(a).GetPointsAttr().Get(),UsdGeom.Mesh(b).GetPointsAttr().Get())


def test_close_restores_visibility_without_changing_physics():
    from go2_pvcnn.control.dh116.tabletop_teacher_read_only_display import ReadOnlyHandDisplay
    stage=fixture_stage();owner=ReadOnlyHandDisplay(stage,[binding()])
    prefix=owner.prefix
    owner.close();owner.close()
    assert not stage.GetPrimAtPath(prefix)
    source=stage.GetPrimAtPath("/World/Physical/base_link")
    assert source.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not source.GetAttribute("visibility").HasAuthoredValueOpinion()
    assert source.GetAttribute("xformOp:translate").Get()==Gf.Vec3d(3,4,5)

def test_display_owner_rejects_physics_clock_advancement():
    from go2_pvcnn.control.dh116.tabletop_teacher_read_only_display import ReadOnlyHandDisplay
    owner=ReadOnlyHandDisplay(fixture_stage(),[binding()])
    clock=iter([(0,0.),(1,1/480)])
    poses={(0,"left"):torch.tensor([[0.,0.,0.,1.,0.,0.,0.]])}
    with pytest.raises(RuntimeError,match="clock"):
        owner.sync(poses,read_clock=lambda:next(clock))


def test_display_edits_do_not_touch_asset_root_layer_or_user_edit_target():
    from go2_pvcnn.control.dh116.tabletop_teacher_read_only_display import ReadOnlyHandDisplay
    stage=fixture_stage()
    before=stage.GetRootLayer().ExportToString()
    target=stage.GetEditTarget().GetLayer()
    owner=ReadOnlyHandDisplay(stage,[binding()])
    owner.sync({(0,"left"):torch.tensor([[.4,.2,.3,1.,0.,0.,0.]])},read_clock=lambda:(0,0.))
    assert stage.GetRootLayer().ExportToString()==before
    assert stage.GetEditTarget().GetLayer()==target
    owner.close()
    assert stage.GetRootLayer().ExportToString()==before

def test_closing_display_preserves_concurrent_user_visibility_change():
    from go2_pvcnn.control.dh116.tabletop_teacher_read_only_display import ReadOnlyHandDisplay
    stage=fixture_stage();owner=ReadOnlyHandDisplay(stage,[binding()])
    with Usd.EditContext(stage,stage.GetRootLayer()):
        # MakeInvisible short-circuits while the stronger display layer is
        # already invisible; Set authors a real concurrent root-layer edit.
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/Physical/base_link")).GetVisibilityAttr().Set("invisible")
    owner.close()
    assert UsdGeom.Imageable(stage.GetPrimAtPath("/World/Physical/base_link")).GetVisibilityAttr().Get()=="invisible"
    assert stage.GetPrimAtPath("/World/Physical/base_link").GetAttribute("visibility").HasAuthoredValueOpinion()
