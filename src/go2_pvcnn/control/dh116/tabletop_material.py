"""Explicit object contact profiles and live PhysX material verification."""
import numpy as np


def material_contract(profile):
    if profile == 'default':
        return {}
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
    if profile != 'grip':
        raise ValueError('unknown hand contact material profile')
    return dict(hand_contact_material=dict(profile='grip_v1',
        static_friction=1.,dynamic_friction=.8,restitution=0.,
        friction_combine_mode='max',restitution_combine_mode='min',scope='all_hand_colliders',
        calibration='training_assumption_not_measured'))


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
    return dict(colliders=len(paths),friction_combine_mode=spec['friction_combine_mode'],
                restitution_combine_mode=spec['restitution_combine_mode'])


def verify_hand_material_properties(properties, profile, expected_envs, expected_shapes):
    values = np.asarray(properties,dtype=float)
    if values.shape != (expected_envs,expected_shapes,3) or not np.isfinite(values).all():
        raise ValueError('invalid live hand material properties')
    settings = hand_material_contract(profile)
    if settings:
        spec = settings['hand_contact_material']
        expected = [spec[k] for k in ('static_friction','dynamic_friction','restitution')]
        if not np.allclose(values,expected,atol=1e-6,rtol=0):
            raise ValueError('live hand material differs from requested profile')
    return dict(environments=expected_envs,shapes_per_hand=expected_shapes,
                unique_coefficients=np.unique(values.reshape(-1,3),axis=0).tolist())


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
