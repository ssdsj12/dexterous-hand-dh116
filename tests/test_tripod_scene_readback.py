"""Synthetic USD fixtures validate read-only scene traversal, not Isaac runs."""
from types import SimpleNamespace as NS
import numpy as np
import pytest


def scene():
    from pxr import Usd,UsdGeom,UsdPhysics,Gf,Sdf
    from go2_pvcnn.control.dh116.linear_coupling import MIMIC,JOINTS
    stage=Usd.Stage.CreateInMemory();prefix='/World/envs/env_0'
    UsdGeom.Xform.Define(stage,prefix)
    root=UsdGeom.Xform.Define(stage,prefix+'/Table');root.AddTranslateOp().Set(Gf.Vec3d(0.,0.,.15))
    cube=UsdGeom.Cube.Define(stage,prefix+'/Table/geometry/mesh')
    cube.CreateSizeAttr(1.);cube.AddScaleOp().Set(Gf.Vec3d(1.,1.,.3));UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    visual=UsdGeom.Cube.Define(stage,prefix+'/Table/visual_only');visual.CreateSizeAttr(100.)
    for side in ('left','right'):
        for follower,master,ratio,offset in MIMIC:
            p=stage.DefinePrim(f'{prefix}/{side}Hand/joints/{follower}','PhysicsRevoluteJoint')
            p.CreateAttribute('physxMimicJoint:rotX:gearing',Sdf.ValueTypeNames.Float).Set(-ratio)
            p.CreateAttribute('physxMimicJoint:rotX:offset',Sdf.ValueTypeNames.Float).Set(-float(np.degrees(offset)))
            p.CreateAttribute('physxMimicJoint:rotX:referenceJointAxis',Sdf.ValueTypeNames.Token).Set('rotX')
            p.CreateRelationship('physxMimicJoint:rotX:referenceJoint').SetTargets([f'{prefix}/{side}Hand/joints/{master}'])
    world=NS(sim=NS(stage=stage),paths=[prefix],origins=np.zeros((1,3)),end_support_scene=None,
        layout=dict(table_center=[0.,0.,.15],table_size=[1.,1.,.3]))
    device=NS(hands=[NS(joint_names=JOINTS),NS(joint_names=JOINTS)])
    return world,device,cube


def verify(world,device):
    from go2_pvcnn.control.dh116.tabletop_tripod_staged_experiment import _verify_original_scene
    return _verify_original_scene(world,device)


def test_actual_collision_descendant_is_validated_without_requiring_api_on_parent():
    world,device,cube=scene();before=world.sim.stage.GetRootLayer().ExportToString()
    out=verify(world,device)
    assert out['static_colliders'][0]['collider_paths']==['/World/envs/env_0/Table/geometry/mesh']
    assert world.sim.stage.GetRootLayer().ExportToString()==before
    assert np.allclose(out['static_colliders'][0]['bounds_w_m'],[[-.5,-.5,0.],[.5,.5,.3]])


def test_disabled_only_collider_cannot_be_certified_as_original_table():
    from pxr import UsdPhysics
    world,device,cube=scene();UsdPhysics.CollisionAPI(cube).GetCollisionEnabledAttr().Set(False)
    with pytest.raises(ValueError,match='collider'):verify(world,device)


def test_wrong_physical_collision_bounds_are_rejected_even_if_visual_bounds_are_large():
    from pxr import Gf
    world,device,cube=scene();cube.GetOrderedXformOps()[0].Set(Gf.Vec3d(1.,1.,.4))
    with pytest.raises(ValueError,match='bounds'):verify(world,device)
