"""Soft contact belongs to the manufacturer's silicone pads, not the hand frame."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf

PADS=('finger14_link','finger23_link','finger33_link','finger43_link','finger53_link')


def test_compliance_is_versioned_and_explicitly_estimated():
    from go2_pvcnn.control.dh116.tabletop_material import hand_material_contract
    base=hand_material_contract('grip')['hand_contact_material']
    spec=hand_material_contract('grip_compliant')['hand_contact_material']
    assert spec['profile']=='grip_compliant_pads_v1'
    assert spec['static_friction']==base['static_friction']==1.
    assert spec['pad_compliance']==dict(links=list(PADS),stiffness_n_m=10000.,damping_n_s_m=30.,
        calibration='simulation_estimate_not_force_displacement_calibrated')
    assert 'pad_compliance' not in base


def test_only_pad_colliders_get_soft_material_and_binding_audit_rejects_wrong_k():
    from go2_pvcnn.control.dh116.tabletop_material import bind_hand_materials,hand_material_contract
    stage=Usd.Stage.CreateInMemory();stage.DefinePrim('/Hand','Xform')
    for link in ('palm_link',*PADS):
        prim=UsdGeom.Cube.Define(stage,'/Hand/'+link+'/collider').GetPrim();UsdPhysics.CollisionAPI.Apply(prim)
    spec=hand_material_contract('grip_compliant')['hand_contact_material']
    def mat(path,k):
        material=UsdShade.Material.Define(stage,path);api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(1.);api.CreateDynamicFrictionAttr(.8);api.CreateRestitutionAttr(0.)
        for n,v in [('frictionCombineMode','max'),('restitutionCombineMode','min')]:
            material.GetPrim().CreateAttribute('physxMaterial:'+n,Sdf.ValueTypeNames.Token).Set(v)
        for n,v in [('compliantContactStiffness',k),('compliantContactDamping',30. if k else 0.)]:
            material.GetPrim().CreateAttribute('physxMaterial:'+n,Sdf.ValueTypeNames.Float).Set(v)
        return material
    hard=mat('/Hard',0.);soft=mat('/Soft',10000.)
    paths,report=bind_hand_materials(stage,'/Hand',hard,spec,soft)
    assert len(paths)==6 and report['compliant_pad_colliders']==5
    for path in paths:
        binding=UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial('physics')[0]
        assert str(binding.GetPath())==('/Hard' if 'palm_link' in path else '/Soft')
    soft.GetPrim().GetAttribute('physxMaterial:compliantContactStiffness').Set(500.)
    with pytest.raises(ValueError,match='compliance'):
        bind_hand_materials(stage,'/Hand',hard,spec,soft)


def test_live_material_encoding_and_per_link_audit():
    from go2_pvcnn.control.dh116.tabletop_material import verify_hand_material_properties,verify_hand_link_material
    values=np.tile([1.,.8,0.],(4,17,1));values[:,:5,2]=-10000.
    report=verify_hand_material_properties(values,'grip_compliant',4,17)
    assert report['compliant_shapes_per_hand']==5
    for link in ('palm_link',*PADS):
        v=np.tile([1.,.8,-10000. if link in PADS else 0.],(4,1,1))
        verify_hand_link_material(v,'grip_compliant',link,4)
        v[-1,0,2]=0. if link in PADS else -10000.
        with pytest.raises(ValueError,match='material'):
            verify_hand_link_material(v,'grip_compliant',link,4)
    values[-1,0,2]=0.
    with pytest.raises(ValueError,match='material'):
        verify_hand_material_properties(values,'grip_compliant',4,17)


def test_compliant_material_constructor_fields_reach_scene_spawn():
    from go2_pvcnn.control.dh116.tabletop_material import hand_material_contract,material_config_fields
    spec=hand_material_contract('grip_compliant')['hand_contact_material']
    hard=material_config_fields(spec);soft=material_config_fields(spec,compliant=True)
    assert hard.get('compliant_contact_stiffness',0.)==0.
    assert soft['compliant_contact_stiffness']==10000. and soft['compliant_contact_damping']==30.
    tree=ast.parse(Path('src/go2_pvcnn/control/dh116/tabletop_vec_isaac.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='RigidBodyMaterialCfg']
    # Exercise the actual soft-material constructor expression.
    call=next(n for n in calls if 'compliant=True' in ast.unparse(n))
    value=eval(compile(ast.Expression(call),'<material spawn>','eval'),dict(settings=spec,
        material_config_fields=material_config_fields,sim_utils=SimpleNamespace(RigidBodyMaterialCfg=lambda **kw:kw)))
    assert value==soft


def test_only_exact_compliant_pad_migration_is_allowed():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    from go2_pvcnn.control.dh116.tabletop_material import hand_material_contract
    old=dict(physics_hz=480,solver_type=0,position_iterations=64,velocity_iterations=16,
             object_kind='solid_cylinder',**hand_material_contract('grip'))
    new=dict(old,**hand_material_contract('grip_compliant'))
    assert checkpoint_contract_changes(old,new)==['compliant_silicone_pads_v1']
    assert checkpoint_contract_changes(new,new)==[]
    for bad in (dict(new,physics_hz=240),dict(new,solver_type=1),dict(new,object_kind='deformx_wire')):
        with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(old,bad)
    bad=dict(new,hand_contact_material=dict(new['hand_contact_material'],pad_compliance={'stiffness_n_m':100.}))
    with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(old,bad)
    with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(new,old)


def test_historical_solid_cylinder_contract_omits_object_kind():
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    from go2_pvcnn.control.dh116.tabletop_material import hand_material_contract
    from go2_pvcnn.control.dh116.tabletop_adaptive_initialization import object_spec
    assert object_spec(Path('.'),'solid_cylinder')[2]=={}
    old=dict(physics_hz=480,solver_type=0,position_iterations=64,velocity_iterations=16,
             **hand_material_contract('grip'))
    new=dict(old,**hand_material_contract('grip_compliant'))
    assert checkpoint_contract_changes(old,new)==['compliant_silicone_pads_v1']
    for field in ('object','object_usd','object_asset','object_backend','object_mass_kg'):
        a=dict(old,**{field:'unexpected asset'});b=dict(new,**{field:'unexpected asset'})
        with pytest.raises(ValueError,match='contract mismatch'):checkpoint_contract_changes(a,b)
