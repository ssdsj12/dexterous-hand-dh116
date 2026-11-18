"""Explicit source-mesh curved front cap; all remaining triangles stay guarded.

The 12 mm / 60 degree outer cap is a simulation surface classification, not a
relaxed collision margin. It is distinct from the small measured-load patch.
Only the connected cap containing the calibrated surface point is selected.
"""
from collections import defaultdict
import numpy as np


def classify_front_cap(vertices,faces,point,normal):
    v=np.asarray(vertices,dtype=float);raw=np.asarray(faces);p=np.asarray(point,dtype=float);n=np.asarray(normal,dtype=float)
    if (v.ndim!=2 or v.shape[1:]!=(3,) or raw.ndim!=2 or raw.shape[1:]!=(3,)
            or len(raw)==0 or not np.issubdtype(raw.dtype,np.integer)
            or p.shape!=(3,) or n.shape!=(3,) or not np.isfinite(v).all()
            or not np.isfinite(p).all() or not np.isfinite(n).all()
            or not np.isclose(np.linalg.norm(n),1,atol=1e-6,rtol=0)
            or np.any(raw<0) or np.any(raw>=len(v))):
        raise ValueError('invalid front-cap geometry')
    tri=v[raw];a=tri[:,1]-tri[:,0];b=tri[:,2]-tri[:,0];cross=np.cross(a,b)
    length=np.linalg.norm(cross,axis=1)
    if np.any(length<=1e-14):raise ValueError('degenerate front-cap triangle')
    normals=cross/length[:,None]
    candidate=(normals@n>=.5)&(np.linalg.norm(tri-p,axis=2).max(1)<=.012)
    # Surface point must project inside an eligible triangle, not merely near
    # an unrelated parallel face. 100 nm derived hull quantization is < .1mm.
    d=p-tri[:,0];aa=np.sum(a*a,axis=1);bb=np.sum(b*b,axis=1);ab=np.sum(a*b,axis=1)
    ad=np.sum(a*d,axis=1);bd=np.sum(b*d,axis=1);den=length**2
    u=(bb*ad-ab*bd)/den;w=(aa*bd-ab*ad)/den
    seeds=np.flatnonzero(candidate&(np.abs(np.sum(d*normals,axis=1))<=1e-4)
                        &(u>=-1e-6)&(w>=-1e-6)&(u+w<=1+1e-6))
    if len(seeds)==0:raise ValueError('reference surface point not in a local front cap')
    edges=defaultdict(list)
    for index in np.flatnonzero(candidate):
        f=raw[index]
        for j in range(3):edges[tuple(sorted((int(f[j]),int(f[(j+1)%3]))))].append(int(index))
    adjacent=defaultdict(set)
    for group in edges.values():
        for i in group:adjacent[i].update(group)
    selected=set();stack=[int(seeds[0])]
    while stack:
        i=stack.pop()
        if i in selected:continue
        selected.add(i);stack.extend(adjacent[i]-selected)
    mask=np.array([i in selected for i in range(len(raw))])
    if mask.all():raise ValueError('cannot exempt the entire collision surface')
    return mask


def sample_wire(nodes,*,spacing=.0005):
    p=np.asarray(nodes,dtype=float)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or len(p)<2 or not np.isfinite(p).all()
            or not np.isfinite(spacing) or not 0<spacing<=.0005):
        raise ValueError('invalid wire sampling input')
    points=[];allowance=0.
    for a,b in zip(p[:-1],p[1:]):
        length=float(np.linalg.norm(b-a))
        # Rigid transforms can move an exact integer ratio a few ulps upward.
        # Stabilize only that rounding boundary; keep the ACTUAL half-spacing
        # below as the distance allowance, so no geometric bound is discarded.
        ratio=length/spacing;nearest=float(np.rint(ratio))
        if abs(ratio-nearest)<=1e-10:ratio=nearest
        count=max(1,int(np.ceil(ratio)))
        points.append(np.linspace(a,b,count+1));allowance=max(allowance,length/(2*count))
    return np.concatenate(points),allowance
