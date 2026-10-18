"""Conservative, source-traceable clipping of the existing 12 mm / 60° cap.

The clipping polytope is INSCRIBED in the sphere. Only its connected, normal-
eligible intersection is front. Every exterior fragment remains protected.
This changes a guard query mesh, never the physical contact surface.
"""
from collections import defaultdict
from functools import lru_cache
import numpy as np
from scipy.spatial import ConvexHull


@lru_cache(maxsize=1)
def _cap_planes():
    phi=(1.+np.sqrt(5.))/2.
    vertices=np.array([(0,a,b*phi) for a in (-1.,1.) for b in (-1.,1.)]+
                      [(a,b*phi,0) for a in (-1.,1.) for b in (-1.,1.)]+
                      [(b*phi,0,a) for a in (-1.,1.) for b in (-1.,1.)])
    vertices/=np.linalg.norm(vertices,axis=1)[:,None]
    for _ in range(3):
        faces=ConvexHull(vertices).simplices
        edges=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
        mid=vertices[edges].mean(1);mid/=np.linalg.norm(mid,axis=1)[:,None]
        vertices=np.r_[vertices,mid]
    vertices*=.012
    planes=ConvexHull(vertices).equations.copy()
    if np.linalg.norm(vertices,axis=1).max()>.012+1e-14:
        raise ValueError('cap polytope must be inscribed')
    planes.setflags(write=False)
    return planes


def _clip(poly,plane):
    signed=poly@plane[:3]+plane[3];inside=[];outside=[]
    for i in range(len(poly)):
        a,b=poly[i],poly[(i+1)%len(poly)];da,db=signed[i],signed[(i+1)%len(poly)]
        (inside if da<=0 else outside).append(a)
        if (da<=0)!=(db<=0):
            cross=a+(b-a)*(da/(da-db));inside.append(cross);outside.append(cross)
    return np.asarray(inside).reshape(-1,3),np.asarray(outside).reshape(-1,3)


def _triangulate(poly):
    if len(poly)<3:return []
    tri=np.array([[poly[0],poly[i],poly[i+1]] for i in range(1,len(poly)-1)])
    area2=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    return list(tri[area2>1e-18])


def _shared_segment_inside(a,b,planes):
    offset=planes[:,:3]@a+planes[:,3];slope=planes[:,:3]@(b-a)
    parallel=np.abs(slope)<=1e-15
    if np.any(offset[parallel]>0):return False
    positive=slope>1e-15;negative=slope<-1e-15
    hi=min(1.,float(np.min(-offset[positive]/slope[positive],initial=np.inf)))
    lo=max(0.,float(np.max(-offset[negative]/slope[negative],initial=-np.inf)))
    return (hi-lo)*np.linalg.norm(b-a)>1e-10


def partition_front_cap(vertices,faces,point,normal):
    v=np.asarray(vertices,dtype=float);f=np.asarray(faces)
    p=np.asarray(point,dtype=float);n=np.asarray(normal,dtype=float)
    if (v.ndim!=2 or v.shape[1:]!=(3,) or f.ndim!=2 or f.shape[1:]!=(3,) or len(f)==0
            or not np.issubdtype(f.dtype,np.integer) or np.any(f<0) or np.any(f>=len(v))
            or p.shape!=(3,) or n.shape!=(3,) or not np.isfinite(v).all()
            or not np.isfinite(p).all() or not np.isfinite(n).all()
            or not np.isclose(np.linalg.norm(n),1.,atol=1e-6,rtol=0)):
        raise ValueError('invalid front partition geometry')
    tri=v[f]-p;a=tri[:,1]-tri[:,0];b=tri[:,2]-tri[:,0]
    cross=np.cross(a,b);length=np.linalg.norm(cross,axis=1)
    if np.any(length<=1e-14):raise ValueError('degenerate source triangle')
    normals=cross/length[:,None];eligible=normals@n>=.5
    planes=_cap_planes();inner={};outer={}
    for i in np.flatnonzero(eligible):
        poly=tri[i].copy();pieces=[]
        # Planes containing the original triangle also contain every fragment.
        active=planes[np.max(tri[i]@planes[:,:3].T+planes[:,3],axis=0)>0]
        for plane in active:
            poly,cut=_clip(poly,plane);pieces.extend(_triangulate(cut))
            if len(poly)<3:break
        if _triangulate(poly):inner[int(i)]=poly;outer[int(i)]=pieces
    d=-tri[:,0];aa=np.sum(a*a,axis=1);bb=np.sum(b*b,axis=1);ab=np.sum(a*b,axis=1)
    ad=np.sum(a*d,axis=1);bd=np.sum(b*d,axis=1)
    u=(bb*ad-ab*bd)/length**2;w=(aa*bd-ab*ad)/length**2
    plane_distance=np.abs(np.sum(d*normals,axis=1))
    seeds=[i for i in inner if plane_distance[i]<=1e-4 and u[i]>=-1e-6 and w[i]>=-1e-6 and u[i]+w[i]<=1+1e-6]
    if not seeds:raise ValueError('reference point is not on a normal-eligible front surface')
    edges=defaultdict(list)
    for i in inner:
        for j in range(3):edges[tuple(sorted((int(f[i,j]),int(f[i,(j+1)%3]))))].append(i)
    adjacent=defaultdict(set)
    for (a_id,b_id),group in edges.items():
        if len(group)>1 and _shared_segment_inside(v[a_id]-p,v[b_id]-p,planes):
            for i in group:adjacent[i].update(group)
    selected=set();stack=[min(seeds,key=lambda i:plane_distance[i])]
    while stack:
        i=stack.pop()
        if i in selected:continue
        selected.add(i);stack.extend(adjacent[i]-selected)
    front=[];remainder=[];front_ids=[];other_ids=[]
    for i in range(len(f)):
        ft=_triangulate(inner[i]) if i in selected else []
        rt=outer[i] if i in selected else [tri[i]]
        front.extend(ft);front_ids.extend([i]*len(ft));remainder.extend(rt);other_ids.extend([i]*len(rt))
    if not front or not remainder:raise ValueError('front and protected remainder must both exist')
    return dict(front_triangles=np.asarray(front)+p,remainder_triangles=np.asarray(remainder)+p,
        front_source_ids=np.asarray(front_ids,dtype=int),remainder_source_ids=np.asarray(other_ids,dtype=int),
        method='connected-inscribed-icosphere3-clipped-cap',radius_m=.012,normal_cosine=.5,
        inscribed_radial_error_bound_m=float(.012-np.min(-planes[:,3])))
