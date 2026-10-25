"""Author linear DH116 mimic constraints in the runtime layer only."""
import numpy as np
from .linear_coupling import MIMIC


def install_hand_mimic(stage, hand_path):
    from pxr import UsdPhysics, PhysxSchema
    for follower, master, ratio, offset in MIMIC:
        joint = stage.GetPrimAtPath(f'{hand_path}/joints/{follower}')
        reference = stage.GetPrimAtPath(f'{hand_path}/joints/{master}')
        for prim in (joint, reference):
            if not prim.IsValid() or UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get() != 'X':
                raise ValueError(f'invalid DH116 mimic joint: {prim}')
        api = PhysxSchema.PhysxMimicJointAPI.Apply(joint, 'rotX')
        api.CreateReferenceJointRel().SetTargets([reference.GetPath()])
        api.CreateReferenceJointAxisAttr('rotX')
        api.CreateGearingAttr(-ratio)
        api.CreateOffsetAttr(-float(np.degrees(offset)))
    return len(MIMIC)


def configure_mimic_compliance(stage, hand_path, natural_frequency, damping_ratio, *, allow_rigid=False):
    """Author compliance, or opt in explicitly to zero-frequency rigid coupling.

    Isaac 5.1 / PhysX 107.3 exposes these as codeless USD attributes, not
    PhysxMimicJointAPI Create* methods. Frequency is in inverse seconds.
    """
    from pxr import Sdf, UsdPhysics
    if (not np.isfinite([natural_frequency, damping_ratio]).all()
            or natural_frequency < 0 or (natural_frequency == 0 and not allow_rigid) or damping_ratio < 1.):
        raise ValueError('finite nonnegative frequency (zero requires allow_rigid) and damping ratio >=1 required')
    joints = [stage.GetPrimAtPath(f'{hand_path}/joints/{f}') for f,_,_,_ in MIMIC]
    if any(not j.IsValid() or UsdPhysics.RevoluteJoint(j).GetAxisAttr().Get() != 'X' for j in joints):
        raise ValueError('five valid X-axis follower joints required')
    for joint in joints:
        joint.CreateAttribute('physxMimicJoint:rotX:naturalFrequency', Sdf.ValueTypeNames.Float).Set(float(natural_frequency))
        joint.CreateAttribute('physxMimicJoint:rotX:dampingRatio', Sdf.ValueTypeNames.Float).Set(float(damping_ratio))
    return len(joints)
