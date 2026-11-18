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


def align_contact_centres(model,seed,contacts,wire_samples,arm_lower,arm_upper,*,centre_fraction=1.):
    """Offline approach-endpoint refinement, never a close-phase arm command.

    Two surface queries preserve the current wire/mesh distances while the
    calibrated pad centres move toward the closest contacts. Non-pinch finger
    coordinates stay parked. Source collision and physical tests remain external.
    """
    from .linear_coupling import active_limits
    q=np.asarray(seed,dtype=float);samples=np.asarray(wire_samples,dtype=float)
    lower=np.asarray(arm_lower,dtype=float);upper=np.asarray(arm_upper,dtype=float)
    if (q.shape!=(13,) or lower.shape!=(7,) or upper.shape!=(7,)
            or samples.ndim!=2 or samples.shape[1]!=3 or len(samples)<2
            or len(contacts)!=2 or not all(np.isfinite(a).all() for a in (q,samples,lower,upper))
            or np.any(lower>=upper) or not 0<centre_fraction<=1):
        raise ValueError('finite dual-contact alignment geometry required')
    low=np.r_[lower,active_limits()[0]];high=np.r_[upper,active_limits()[1]]
    if np.any(q<low) or np.any(q>high):raise ValueError('alignment seed outside joint limits')
    for c in contacts:
        if (c['body'] not in model.bodies or not callable(c['query'])
                or any(np.asarray(c[key]).shape!=(3,) or not np.isfinite(c[key]).all()
                       for key in ('reference_point','reference_normal','mesh_offset'))
                or not np.isclose(np.linalg.norm(c['reference_normal']),1.)):
            raise ValueError('named, calibrated contact surface required')

    def evaluate(trial):
        poses=model.fk(trial)[0];values=[]
        for c in contacts:
            t=poses[model.bodies[c['body']]];R=t[:3,:3];origin=t[:3,3]+R@c['mesh_offset']
            cp,sd,n=c['query']((samples-origin)@R);i=int(np.argmin(sd))
            point=t[:3,3]+R@c['reference_point'];normal=R@c['reference_normal']
            values.append((point,normal,origin+R@cp[i],R@n[i],float(sd[i])))
        return poses[model.bodies['base_link']],values

    palm0,start=evaluate(q)
    goals=[((1-centre_fraction)*p+centre_fraction*cp,
            ((1-centre_fraction)*n+centre_fraction*cn)/np.linalg.norm((1-centre_fraction)*n+centre_fraction*cn))
           for p,n,cp,cn,sd in start]
    ids=np.arange(10)
    def error(x):
        trial=q.copy();trial[ids]=x;palm,values=evaluate(trial);out=[]
        for i,(point,normal,cp,cn,sd) in enumerate(values):
            out.extend((point-goals[i][0])/.0005);out.extend((normal-goals[i][1])/.1)
            out.append((sd-start[i][4])/.000015)
        out.extend((palm[:3,3]-palm0[:3,3])/.002)
        out.extend(Rotation.from_matrix(palm[:3,:3]@palm0[:3,:3].T).as_rotvec()/np.deg2rad(3.))
        out.extend((x-q[ids])/.03*.1)
        return out

    fit=least_squares(error,q[ids],bounds=(np.maximum(low[ids],q[ids]-.1),np.minimum(high[ids],q[ids]+.1)),
        diff_step=1e-4,max_nfev=150,xtol=1e-9,ftol=1e-9,gtol=1e-9)
    trial=q.copy();trial[ids]=fit.x;palm,after=evaluate(trial)
    delta=palm[:3,3]-palm0[:3,3]
    angle=float(Rotation.from_matrix(palm[:3,:3]@palm0[:3,:3].T).magnitude())
    gaps=[v[4]-s[4] for v,s in zip(after,start)]
    return dict(q=trial.tolist(),palm_delta_world_m=delta.tolist(),palm_rotation_delta_rad=angle,
        gap_changes_m=gaps,max_gap_change_m=float(np.max(np.abs(gaps))),
        reference_offset_before_m=[float(np.linalg.norm(v[0]-v[2])) for v in start],
        reference_offset_after_m=[float(np.linalg.norm(v[0]-v[2])) for v in after],
        converged=bool(fit.success and np.max(np.abs(gaps))<=1e-5 and np.linalg.norm(delta)<=.002
                       and angle<=np.deg2rad(3.)),collision_qualified=False,physics_qualified=False)
