"""Local front-patch gate for measured closed-hull closest-point queries.

This does not infer successful grasp from proximity, and does not waive the
remainder of a distal body. All inputs must share the same coordinate frame.
"""
import numpy as np


def check_pad_front(points,normals,distances,*,reference_point,reference_normal,
                    wire_radius,patch_radius,normal_cosine):
    p,n,d,r,v=(np.asarray(x,dtype=float) for x in
               (points,normals,distances,reference_point,reference_normal))
    if (p.ndim!=2 or p.shape[1:]!=(3,) or len(p)==0 or n.shape!=p.shape
            or d.shape!=(len(p),) or r.shape!=(3,) or v.shape!=(3,)
            or any(not np.isfinite(x).all() for x in (p,n,d,r,v))):
        raise ValueError('invalid measured front-patch geometry')
    if (not np.allclose(np.linalg.norm(n,axis=1),1.,atol=1e-6,rtol=0)
            or not np.isclose(np.linalg.norm(v),1.,atol=1e-6,rtol=0)):
        raise ValueError('front normals must be unit vectors')
    if (not np.isfinite([wire_radius,patch_radius,normal_cosine]).all()
            or wire_radius<=0 or not 0<patch_radius<=.01 or not .5<=normal_cosine<=1):
        raise ValueError('invalid explicit front-patch configuration')
    front=(np.linalg.norm(p-r,axis=1)<=patch_radius)&(n@v>=normal_cosine)
    clearance=d-wire_radius
    failures=[]
    if np.any(clearance[~front]<.001):failures.append('non-front wire clearance below 1 mm')
    if np.any(clearance<-.001):failures.append('pad penetration exceeds 1 mm')
    return dict(passed=not failures,failures=failures,front_samples=int(front.sum()),
                max_penetration_m=float(np.maximum(-clearance,0).max()),
                min_nonfront_clearance_m=float(clearance[~front].min()) if (~front).any() else None)
