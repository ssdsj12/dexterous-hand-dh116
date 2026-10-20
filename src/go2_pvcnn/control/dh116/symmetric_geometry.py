"""Mirror calibrated functional frames, not raw left/right USD link frames.

Functional X and Z are polar directions. The lateral Y axis is reconstructed
to keep a proper rotation under reflection about world Y=0. This is a target
geometry operation, never an IK, collision or physical grasp certificate.
"""
import numpy as np


def mirror_functional_pose(position, rotation):
    p=np.asarray(position,dtype=float)
    R=np.asarray(rotation,dtype=float)
    if (p.shape!=(3,) or R.shape!=(3,3)
            or not np.isfinite(p).all() or not np.isfinite(R).all()
            or not np.allclose(R.T@R,np.eye(3),atol=1e-8,rtol=0)
            or not np.isclose(np.linalg.det(R),1.,atol=1e-8,rtol=0)):
        raise ValueError('expected finite position and proper functional rotation')
    reflection=np.diag([1.,-1.,1.])
    return reflection@p,reflection@R@reflection
