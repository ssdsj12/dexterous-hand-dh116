from pathlib import Path

import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade


def test_default_material_does_not_change_historical_contract():
    from go2_pvcnn.control.dh116.tabletop_material import material_contract
    assert material_contract('default') == {}
    with pytest.raises(ValueError):
        material_contract('unknown')


def test_high_friction_is_versioned_and_has_no_bounce():
    from go2_pvcnn.control.dh116.tabletop_material import material_contract
    a=material_contract('high_friction')
    assert a['object_contact_material'] == dict(profile='high_friction_v1',
        static_friction=1.2,dynamic_friction=1.,restitution=0.,
        friction_combine_mode='max',restitution_combine_mode='min',scope='object_colliders')
    a['object_contact_material']['static_friction']=99
    assert material_contract('high_friction')['object_contact_material']['static_friction']==1.2


def test_every_straw_collision_shape_is_bound_without_touching_other_objects():
    from go2_pvcnn.control.dh116.tabletop_material import bind_object_colliders
    stage=Usd.Stage.CreateInMemory()
    root=stage.DefinePrim('/World/Wire','Xform')
    asset=Path('assets/dh116_hollow_straw_v1/straw.usda').resolve()
    root.GetReferences().AddReference(str(asset))
    material=UsdShade.Material.Define(stage,'/World/GripMaterial')
    visual=stage.GetPrimAtPath('/World/Wire/Visual')
    visual_binding=UsdShade.MaterialBindingAPI(visual).ComputeBoundMaterial()[0].GetPath()
    visual_physics=UsdShade.MaterialBindingAPI(visual).ComputeBoundMaterial('physics')[0].GetPath()
    other=UsdGeom.Cube.Define(stage,'/World/Table').GetPrim()
    UsdPhysics.CollisionAPI.Apply(other)
    paths=bind_object_colliders(stage,'/World/Wire',material)
    assert len(paths)==32
    for path in paths:
        bound,_=UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial('physics')
        assert bound.GetPath()==material.GetPath()
    assert not other.HasAPI(UsdShade.MaterialBindingAPI)
    assert UsdShade.MaterialBindingAPI(visual).ComputeBoundMaterial()[0].GetPath()==visual_binding
    assert UsdShade.MaterialBindingAPI(visual).ComputeBoundMaterial('physics')[0].GetPath()==visual_physics


def test_missing_colliders_cannot_silently_skip_material():
    from go2_pvcnn.control.dh116.tabletop_material import bind_object_colliders
    stage=Usd.Stage.CreateInMemory();stage.DefinePrim('/Wire','Xform')
    material=UsdShade.Material.Define(stage,'/Material')
    with pytest.raises(ValueError,match='collider'):
        bind_object_colliders(stage,'/Wire',material)


def test_live_material_audit_rejects_one_unmodified_shape():
    from go2_pvcnn.control.dh116.tabletop_material import verify_material_properties
    values=np.tile([1.2,1.,0.],(1,32,1))
    assert verify_material_properties(values,'high_friction',32)['shapes']==32
    values[0,20,1]=.5
    with pytest.raises(ValueError,match='material'):
        verify_material_properties(values,'high_friction',32)
    with pytest.raises(ValueError):
        verify_material_properties(np.ones((1,1,3)),'high_friction',32)


def test_rough_rod_and_hand_profiles_are_explicit_training_assumptions():
    from go2_pvcnn.control.dh116.tabletop_material import material_contract, hand_material_contract
    rod = material_contract('rough_rod')['object_contact_material']
    hand = hand_material_contract('grip')['hand_contact_material']
    assert (rod['static_friction'], rod['dynamic_friction']) == (1.5, 1.2)
    assert (hand['static_friction'], hand['dynamic_friction']) == (1., .8)
    assert rod['calibration'] == hand['calibration'] == 'training_assumption_not_measured'
    assert rod['friction_combine_mode'] == hand['friction_combine_mode'] == 'max'
    assert hand_material_contract('default') == {}


def test_actual_hand_colliders_receive_physics_binding_and_modes_are_checked():
    import json
    from go2_pvcnn.control.dh116.tabletop_material import (
        bind_object_colliders, verify_collider_bindings, hand_material_contract)
    from pxr import Sdf
    spec = hand_material_contract('grip')['hand_contact_material']
    manifest = json.loads(Path('assets/fixed_dual_panda_dh116_v1/manifest.json').read_text())
    stage = Usd.Stage.CreateInMemory()
    source = next(x['path'] for x in manifest['sources'] if x['kind'] == 'dh116')
    stage.DefinePrim('/Hand', 'Xform').GetReferences().AddReference(source)
    mat = UsdShade.Material.Define(stage, '/Grip')
    physics = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    physics.CreateStaticFrictionAttr(spec['static_friction'])
    physics.CreateDynamicFrictionAttr(spec['dynamic_friction'])
    physics.CreateRestitutionAttr(0.)
    combine = mat.GetPrim().CreateAttribute('physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token)
    combine.Set('max')
    mat.GetPrim().CreateAttribute('physxMaterial:restitutionCombineMode', Sdf.ValueTypeNames.Token).Set('min')
    paths = bind_object_colliders(stage, '/Hand', mat)
    assert len(paths) == 17
    assert verify_collider_bindings(stage, paths, spec)['colliders'] == 17
    combine.Set('average')
    with pytest.raises(ValueError, match='combine'):
        verify_collider_bindings(stage, paths, spec)


def test_live_hand_audit_checks_all_clones_and_rejects_one_default_shape():
    from go2_pvcnn.control.dh116.tabletop_material import verify_hand_material_properties
    values = np.tile([1., .8, 0.], (4, 17, 1))
    report = verify_hand_material_properties(values, 'grip', 4, 17)
    assert report['environments'] == 4 and report['shapes_per_hand'] == 17
    values[3, 16, 0] = .5
    with pytest.raises(ValueError, match='material'):
        verify_hand_material_properties(values, 'grip', 4, 17)
    with pytest.raises(ValueError):
        verify_hand_material_properties(values[:, :16], 'grip', 4, 17)
