"""Deterministic first-hit source-triangle selection, not a contact solver."""
import numpy as np
from .contact_frames import surface_reference


def ray_pick(points, faces, origin, direction):
    points=np.asarray(points,dtype=float); faces=np.asarray(faces)
    origin=np.asarray(origin,dtype=float); direction=np.asarray(direction,dtype=float)
    if (points.ndim!=2 or points.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3
        or not np.issubdtype(faces.dtype,np.integer) or np.any(faces<0)
        or np.any(faces>=len(points)) or origin.shape!=(3,) or direction.shape!=(3,)
        or not all(np.isfinite(x).all() for x in (points,origin,direction))):
        raise ValueError('invalid ray or triangles')
    length=np.linalg.norm(direction)
    if not np.isfinite(length) or length<1e-12: raise ValueError('invalid ray direction')
    direction=direction/length
    triangles=points[faces]; edge1=triangles[:,1]-triangles[:,0]; edge2=triangles[:,2]-triangles[:,0]
    cross=np.cross(direction,edge2); det=np.einsum('ij,ij->i',edge1,cross)
    valid=np.abs(det)>1e-14
    inv=np.zeros_like(det); inv[valid]=1./det[valid]
    offset=origin-triangles[:,0]
    u=np.einsum('ij,ij->i',offset,cross)*inv
    q=np.cross(offset,edge1)
    v=(q@direction)*inv
    t=np.einsum('ij,ij->i',edge2,q)*inv
    valid &= (u>=0)&(v>=0)&(u+v<=1)&(t>=0)&np.isfinite(t)
    if not valid.any(): raise ValueError('ray missed surface')
    index=int(np.argmin(np.where(valid,t,np.inf)))
    weights=np.array([1.-u[index]-v[index],u[index],v[index]])
    point,normal=surface_reference(triangles[index],weights)
    if normal@direction>=0: raise ValueError('first hit has inward winding; audit source orientation')
    return dict(triangle_index=index,weights=weights.tolist(),point=point.tolist(),
                normal=normal.tolist(),ray_distance_m=float(t[index]))
