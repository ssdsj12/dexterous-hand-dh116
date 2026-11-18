"""Explicit object contact profiles and live PhysX material verification."""
import numpy as np


def material_contract(profile):
    if profile == 'default':
        return {}
    if profile == 'extreme_grip':
        return dict(object_contact_material=dict(profile='extreme_grip_v1',
            static_friction=5.,dynamic_friction=5.,restitution=0.,
            friction_combine_mode='max',restitution_combine_mode='min',scope='object_colliders',
            calibration='artificial_high_friction_experiment_not_measured'))
    if profile == 'rough_rod':
        return dict(object_contact_material=dict(profile='rough_rod_v1',
            static_friction=1.5,dynamic_friction=1.2,restitution=0.,
            friction_combine_mode='max',restitution_combine_mode='min',scope='object_colliders',
            calibration='training_assumption_not_measured'))
    if profile != 'high_friction':
        raise ValueError('unknown object contact material profile')
    return dict(object_contact_material=dict(profile='high_friction_v1',
        static_friction=1.2,dynamic_friction=1.,restitution=0.,
        friction_combine_mode='max',restitution_combine_mode='min',scope='object_colliders'))


def hand_material_contract(profile):
    if profile == 'default':
        return {}
    if profile not in ('grip','grip_compliant'):
        raise ValueError('unknown hand contact material profile')
    spec=dict(profile='grip_v1',
        static_friction=1.,dynamic_friction=.8,restitution=0.,
        friction_combine_mode='max',restitution_combine_mode='min',scope='all_hand_colliders',
        calibration='training_assumption_not_measured')
    if profile=='grip_compliant':
        # Manufacturer CSV identifies these five distal links as silicone A25.
        # No measured force-displacement curve is supplied: K/C are estimates.
        spec.update(profile='grip_compliant_pads_v1',pad_compliance=dict(
            links=['finger14_link','finger23_link','finger33_link','finger43_link','finger53_link'],
            stiffness_n_m=10000.,damping_n_s_m=30.,
            calibration='simulation_estimate_not_force_displacement_calibrated'))
    return dict(hand_contact_material=spec)


def material_config_fields(spec, *, compliant=False):
    fields={k:spec[k] for k in ('static_friction','dynamic_friction','restitution',
                               'friction_combine_mode','restitution_combine_mode')}
    if compliant:
        pad=spec['pad_compliance']
        fields.update(compliant_contact_stiffness=pad['stiffness_n_m'],
                      compliant_contact_damping=pad['damping_n_s_m'])
    return fields


def bind_hand_materials(stage, root_path, hard_material, spec, soft_material=None):
    paths=bind_object_colliders(stage,root_path,hard_material)
    if 'pad_compliance' not in spec:
        return paths,verify_collider_bindings(stage,paths,spec)
    if soft_material is None:raise ValueError('compliant pad material required')
    pad=spec['pad_compliance'];soft_paths=[]
    for link in pad['links']:
        soft_paths.extend(bind_object_colliders(stage,root_path+'/'+link,soft_material))
    soft_spec=dict(spec,compliant_contact_stiffness=pad['stiffness_n_m'],
                   compliant_contact_damping=pad['damping_n_s_m'])
    verify_collider_bindings(stage,soft_paths,soft_spec)
    hard_paths=[path for path in paths if path not in soft_paths]
    hard_spec=dict(spec,compliant_contact_stiffness=0.,compliant_contact_damping=0.)
    if hard_paths:verify_collider_bindings(stage,hard_paths,hard_spec)
    return paths,dict(colliders=len(paths),compliant_pad_colliders=len(soft_paths),
                      hard_colliders=len(hard_paths),pad_compliance=pad)


def verify_collider_bindings(stage, paths, spec):
    """Check resolved physics-purpose binding, including combination modes."""
    from pxr import UsdPhysics, UsdShade
    if not paths:
        raise ValueError('no collider bindings to verify')
    for path in paths:
        material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).ComputeBoundMaterial('physics')
        if not material:
            raise ValueError(f'missing physics material: {path}')
        physics = UsdPhysics.MaterialAPI(material.GetPrim())
        values = [physics.GetStaticFrictionAttr().Get(), physics.GetDynamicFrictionAttr().Get(),
                  physics.GetRestitutionAttr().Get()]
        if not np.allclose(values, [spec[k] for k in ('static_friction','dynamic_friction','restitution')],
                           atol=1e-6,rtol=0):
            raise ValueError(f'bound collider material differs: {path}')
        prim = material.GetPrim()
        if (prim.GetAttribute('physxMaterial:frictionCombineMode').Get() != spec['friction_combine_mode']
                or prim.GetAttribute('physxMaterial:restitutionCombineMode').Get() != spec['restitution_combine_mode']):
            raise ValueError(f'bound material combine mode differs: {path}')
        for key,attribute in (('compliant_contact_stiffness','compliantContactStiffness'),
                              ('compliant_contact_damping','compliantContactDamping')):
            if key in spec:
                value=prim.GetAttribute('physxMaterial:'+attribute).Get()
                if value is None or not np.isclose(value,spec[key],atol=1e-6,rtol=0):
                    raise ValueError(f'bound material compliance differs: {path}')
    return dict(colliders=len(paths),friction_combine_mode=spec['friction_combine_mode'],
                restitution_combine_mode=spec['restitution_combine_mode'])


def verify_hand_material_properties(properties, profile, expected_envs, expected_shapes):
    values = np.asarray(properties,dtype=float)
    if values.shape != (expected_envs,expected_shapes,3) or not np.isfinite(values).all():
        raise ValueError('invalid live hand material properties')
    settings = hand_material_contract(profile)
    report=dict(environments=expected_envs,shapes_per_hand=expected_shapes,
                unique_coefficients=np.unique(values.reshape(-1,3),axis=0).tolist())
    if settings:
        spec=settings['hand_contact_material']
        expected=[spec[k] for k in ('static_friction','dynamic_friction','restitution')]
        pad=spec.get('pad_compliance')
        if pad:
            # PhysX tensor API encodes compliant stiffness as negative restitution.
            soft=np.isclose(values[...,2],-pad['stiffness_n_m'],atol=1e-6,rtol=0)
            hard=np.isclose(values[...,2],0.,atol=1e-6,rtol=0)
            if (not np.allclose(values[...,:2],expected[:2],atol=1e-6,rtol=0)
                    or not (soft|hard).all() or not (soft.sum(1)==len(pad['links'])).all()):
                raise ValueError('live hand material differs from requested pad profile')
            report['compliant_shapes_per_hand']=len(pad['links'])
        elif not np.allclose(values,expected,atol=1e-6,rtol=0):
            raise ValueError('live hand material differs from requested profile')
    return report


def verify_hand_link_material(properties,profile,link,expected_envs):
    spec=hand_material_contract(profile)['hand_contact_material']
    pad=spec.get('pad_compliance',{})
    encoded=-pad['stiffness_n_m'] if link in pad.get('links',[]) else spec['restitution']
    expected=[spec['static_friction'],spec['dynamic_friction'],encoded]
    values=np.asarray(properties,dtype=float)
    if (values.shape!=(expected_envs,1,3) or not np.isfinite(values).all()
            or not np.allclose(values,expected,atol=1e-6,rtol=0)):
        raise ValueError(f'live hand link material differs: {link}')
    return dict(link=link,environments=expected_envs,coefficients=expected)


def bind_object_colliders(stage, root_path, material):
    """Bind physics purpose on each real collider, leaving visual material alone."""
    from pxr import Usd, UsdPhysics, UsdShade
    root=stage.GetPrimAtPath(root_path)
    if not root.IsValid() or not material.GetPrim().IsValid():
        raise ValueError('valid object collider root and material required')
    colliders=[p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)
               and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    if not colliders:
        raise ValueError('no enabled object colliders found')
    for prim in colliders:
        # Imported hands instance their collision meshes. Author the binding on
        # the instance root so its read-only proxies inherit physics purpose.
        target = prim
        while target.IsInstanceProxy():
            target = target.GetParent()
        UsdShade.MaterialBindingAPI.Apply(target).Bind(material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,materialPurpose='physics')
    return [str(prim.GetPath()) for prim in colliders]


def verify_material_properties(properties, profile, expected_shapes):
    """Refuse to train if any cooked collider kept the wrong coefficients."""
    values=np.asarray(properties,dtype=float)
    if values.shape != (1,expected_shapes,3) or not np.isfinite(values).all():
        raise ValueError('invalid live object material properties')
    settings=material_contract(profile)
    if settings:
        spec=settings['object_contact_material']
        expected=[spec[k] for k in ('static_friction','dynamic_friction','restitution')]
        if not np.allclose(values,expected,atol=1e-6,rtol=0):
            raise ValueError('live object material differs from requested profile')
    return dict(shapes=expected_shapes,static_friction=values[0,:,0].tolist(),
        dynamic_friction=values[0,:,1].tolist(),restitution=values[0,:,2].tolist())
