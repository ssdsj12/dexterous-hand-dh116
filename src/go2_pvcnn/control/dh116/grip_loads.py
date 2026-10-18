"""Conservative quasistatic support screen, NOT a deformable grasp proof.

64 nonnegative rays per contact inscribe (never enlarge) the circular Coulomb
cone. Contact normals point from the pad toward the wire. Force and moment
balance are independently recomputed from the returned forces. No binding,
controller gain, material property, or hardware limit is inferred here.
"""
import numpy as np
from scipy.optimize import linprog


def solve_support(points,normals,com,*,weight_n,mu,max_load_n):
    p=np.asarray(points,dtype=float);n=np.asarray(normals,dtype=float);c=np.asarray(com,dtype=float)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or len(p)<2 or n.shape!=p.shape or c.shape!=(3,)
            or any(not np.isfinite(x).all() for x in (p,n,c))
            or not np.allclose(np.linalg.norm(n,axis=1),1.,atol=1e-6,rtol=0)
            or not np.isfinite([weight_n,mu,max_load_n]).all()
            or weight_n<=0 or mu<0 or max_load_n<=0):
        raise ValueError('invalid quasistatic support inputs')
    theta=np.arange(64)*2*np.pi/64
    rays=[]
    for normal in n:
        seed=np.eye(3)[np.argmin(np.abs(normal))]
        t1=np.cross(normal,seed);t1/=np.linalg.norm(t1);t2=np.cross(normal,t1)
        rays.append(normal+mu*(np.cos(theta)[:,None]*t1+np.sin(theta)[:,None]*t2))
    rays=np.asarray(rays)
    moments=np.cross((p-c)[:,None,:],rays)
    matrix=np.concatenate([rays,moments],axis=2).reshape(-1,6).T
    upper=np.zeros((len(p),len(p)*64))
    for i in range(len(p)):upper[i,i*64:(i+1)*64]=1.
    required=np.array([0.,0.,weight_n,0.,0.,0.])
    solution=linprog(np.ones(len(p)*64),A_eq=matrix,b_eq=required,
                    A_ub=upper,b_ub=np.full(len(p),max_load_n),bounds=(0,None),method='highs')
    result=dict(feasible=False,forces_w_n=None,normal_loads_n=None,grasp_qualified=False,
                cone_rays=64,solver_status=int(solution.status),message=solution.message,
                weight_n=float(weight_n),mu=float(mu),max_load_n=float(max_load_n))
    if not solution.success:return result
    forces=np.einsum('ij,ijk->ik',solution.x.reshape(len(p),64),rays)
    loads=np.sum(forces*n,axis=1);tangent=forces-loads[:,None]*n
    ferr=float(np.linalg.norm(forces.sum(0)-required[:3]))
    merr=float(np.linalg.norm(np.cross(p-c,forces).sum(0)))
    cone=float(np.max(np.linalg.norm(tangent,axis=1)-mu*loads))
    result.update(feasible=bool(ferr<1e-7 and merr<1e-7 and cone<1e-7
        and np.min(loads)>=-1e-7 and np.max(loads)<=max_load_n+1e-7),
        forces_w_n=forces.tolist(),normal_loads_n=loads.tolist(),force_error_n=ferr,
        moment_error_nm=merr,max_cone_violation_n=cone)
    return result
