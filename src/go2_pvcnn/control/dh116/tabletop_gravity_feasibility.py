"""Static six-axis gravity-wrench screening, never dynamic success evidence.

Normals and returned forces act ON ROD. Jacobians are actual hand contact point
Jacobians already folded into A active motors, including mimic ratios. Required
motor effort is J.T @ f_rod + gravity_torques: balancing the hand's -f_rod
reaction. gravity_torques denotes required compensation, not applied gravity.
"""
import numpy as np
from scipy.optimize import linprog


def gravity_feasibility(points, normals, jacobians, center, mass, friction, drive_caps, *,
                        hand_ids=None, digit_ids=None, face_ids=None, gravity=(0.,0.,-9.81),
                        gravity_torques=None, cone_sides=16, tolerance=1e-7):
    """Minimize total normal load with an inscribed polygonal friction cone.

    There is no target or fixed bound on total N. drive_caps are the original
    physical active motor limits (DH116: .2 Nm); this function does not alter
    them. If grouping ids are supplied, each represented hand needs two faces;
    face totals match and digit TOTALS within each face match across patches.
    Missing contacts cannot be invented. Invalid data returns a finite failure.
    On infeasibility, diagnostics describe the zero-contact-force candidate,
    never a feasible solution; its gravity residual is explicitly nonzero.
    """
    parsed=True
    failure_target=None
    failure_motor=None
    failure_caps=None
    try:
        p=np.asarray(points,dtype=float);normal=np.asarray(normals,dtype=float)
        j=np.asarray(jacobians,dtype=float);com=np.asarray(center,dtype=float)
        caps=np.asarray(drive_caps,dtype=float);g=np.asarray(gravity,dtype=float)
        c=len(p);a=caps.size
    except (TypeError,ValueError):
        p=np.empty((0,3));c=0;a=0
        parsed=False

    def fail(status,message):
        return dict(static_feasible=False,dynamic_success=False,status=status,message=message,
                    forces_on_rod_n=np.zeros((c,3)),normal_loads_n=np.zeros(c),
                    motor_torques_nm=np.zeros(a) if failure_motor is None else failure_motor.copy(),
                    motor_cap_residual_nm=np.zeros(a) if failure_caps is None else failure_caps.copy(),
                    wrench_residual=np.zeros(6) if failure_target is None else -failure_target,
                    group_residual=np.empty(0),residuals_available=failure_target is not None,
                    model='static_inscribed_friction_cone_lp')
    if not parsed:
        return fail('invalid_input','numeric geometry and limits required')
    try:
        if (p.shape!=(c,3) or normal.shape!=p.shape or j.shape!=(c,3,a) or caps.shape!=(a,) or
                com.shape!=(3,) or g.shape!=(3,) or cone_sides<4 or int(cone_sides)!=cone_sides or
                not np.isfinite(mass) or mass<0 or not np.isfinite(tolerance) or tolerance<=0 or
                any(not np.isfinite(v).all() for v in (p,normal,j,com,caps,g)) or (caps<0).any()):
            return fail('invalid_input','finite compatible geometry and physical limits required')
        norm=np.linalg.norm(normal,axis=-1)
        if (norm<1e-12).any(): return fail('invalid_input','nonzero rod normals required')
        normal=normal/norm[:,None]
        mu=np.broadcast_to(np.asarray(friction,dtype=float),(c,))
        gt=np.zeros(a) if gravity_torques is None else np.asarray(gravity_torques,dtype=float)
        if gt.shape!=(a,) or not np.isfinite(gt).all() or not np.isfinite(mu).all() or (mu<0).any():
            return fail('invalid_input','finite nonnegative friction and active gravity compensation required')
        failure_target=np.r_[-float(mass)*g,np.zeros(3)]
        failure_motor=gt
        failure_caps=caps-np.abs(gt)
        if c==0: return fail('no_contacts','no native effective pad contacts supplied')
        supplied=[v is not None for v in (hand_ids,digit_ids,face_ids)]
        groups=[]
        if any(supplied):
            if not all(supplied): return fail('invalid_input','all three grouping id arrays required together')
            ids=[np.asarray(v) for v in (hand_ids,digit_ids,face_ids)]
            if any(v.shape!=(c,) or not np.issubdtype(v.dtype,np.integer) or (v<0).any() for v in ids):
                return fail('invalid_input','nonnegative integer grouping ids required')
            hi,di,fi=ids
            for hand in np.unique(hi):
                selected=hi==hand;faces=np.unique(fi[selected])
                if len(faces)!=2: return fail('unpaired_faces','each represented hand requires two faces')
                face_masks=[selected&(fi==f) for f in faces]
                groups.append(face_masks[0].astype(float)-face_masks[1])
                for mask in face_masks:
                    digits=np.unique(di[mask]);first=mask&(di==digits[0])
                    for digit in digits[1:]: groups.append(first.astype(float)-(mask&(di==digit)))
        # Positive combinations of boundary rays stay INSIDE Coulomb's cone.
        k=int(cone_sides);basis=np.eye(3)[np.abs(normal).argmin(-1)]
        t1=np.cross(normal,basis);t1/=np.linalg.norm(t1,axis=-1)[:,None]
        t2=np.cross(normal,t1);theta=2*np.pi*np.arange(k)/k
        rays=normal[:,None,:]+mu[:,None,None]*(np.cos(theta)[None,:,None]*t1[:,None,:]+
                                                        np.sin(theta)[None,:,None]*t2[:,None,:])
        wrench=np.concatenate((rays,np.cross((p-com)[:,None,:],rays)),axis=-1).reshape(c*k,6).T
        target=failure_target
        group_matrix=np.repeat(np.asarray(groups).reshape(-1,c),k,axis=1)
        equality=np.vstack((wrench,group_matrix));rhs=np.r_[target,np.zeros(len(groups))]
        motor=np.einsum('cia,cki->ack',j,rays).reshape(a,c*k)
        inequality=np.vstack((motor,-motor));upper=np.r_[caps-gt,caps+gt]
        result=linprog(np.ones(c*k),A_ub=inequality if a else None,b_ub=upper if a else None,
                       A_eq=equality,b_eq=rhs,bounds=(0.,None),method='highs',
                       options={'primal_feasibility_tolerance':max(1e-10,min(tolerance,1e-7))})
        if not result.success: return fail('infeasible' if result.status==2 else 'solver_failure',result.message)
        coeff=result.x.reshape(c,k)
        forces=np.einsum('ck,cki->ci',coeff,rays)
        torque=motor@result.x+gt
        residual=wrench@result.x-target;group_residual=group_matrix@result.x
        cap_residual=caps-np.abs(torque)
        feasible=bool(np.isfinite(result.x).all() and np.max(np.abs(residual))<=tolerance and
                      np.all(cap_residual>=-tolerance) and np.all(np.abs(group_residual)<=tolerance))
        return dict(static_feasible=feasible,dynamic_success=False,status='optimal' if feasible else 'residual_failure',
                    message=result.message,forces_on_rod_n=forces,normal_loads_n=coeff.sum(-1),
                    motor_torques_nm=torque,motor_cap_residual_nm=cap_residual,wrench_residual=residual,
                    group_residual=group_residual,residuals_available=True,model='static_inscribed_friction_cone_lp',
                    cone_sides=k,total_normal_load_n=float(coeff.sum()))
    except (TypeError,ValueError,OverflowError):
        return fail('invalid_input','invalid geometry, friction, grouping or solver parameters')
