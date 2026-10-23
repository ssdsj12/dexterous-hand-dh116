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
