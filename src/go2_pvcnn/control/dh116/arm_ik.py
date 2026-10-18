"""Bounded Panda IK against the current USD FK; no motion authorization."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def solve_arm_pose(model, target, hand, seed, lower, upper):
    target=np.asarray(target,dtype=float)
    hand=np.asarray(hand,dtype=float)
    seed=np.asarray(seed,dtype=float)
    lower=np.asarray(lower,dtype=float);upper=np.asarray(upper,dtype=float)
    if (target.shape!=(4,4) or hand.shape!=(6,) or seed.shape!=(7,)
            or lower.shape!=(7,) or upper.shape!=(7,)
            or not all(np.isfinite(x).all() for x in (target,hand,seed,lower,upper))
            or np.any(lower>=upper) or np.any(seed<lower) or np.any(seed>upper)
            or not np.allclose(target[3],[0,0,0,1],rtol=0,atol=1e-10)
            or not np.allclose(target[:3,:3].T@target[:3,:3],np.eye(3),rtol=0,atol=1e-10)
            or abs(np.linalg.det(target[:3,:3])-1)>1e-10):
        raise ValueError('invalid pose, hand, seed or joint limits')
    def error(arm):
        pose=model.fk(np.r_[arm,hand])[0][model.bodies['base_link']]
        return np.r_[pose[:3,3]-target[:3,3],
            .1*Rotation.from_matrix(target[:3,:3].T@pose[:3,:3]).as_rotvec()]
    fit=least_squares(error,seed,bounds=(lower,upper),max_nfev=300,
        ftol=1e-12,xtol=1e-12,gtol=1e-12)
    residual=error(fit.x)
    pos=float(np.linalg.norm(residual[:3]));angle=float(np.linalg.norm(residual[3:])/.1)
    return dict(q=np.r_[fit.x,hand].tolist(),position_error_m=pos,rotation_error_rad=angle,
        converged=bool(pos<=1e-5 and angle<=1e-4),optimizer_status=int(fit.status),
        collision_qualified=False)
