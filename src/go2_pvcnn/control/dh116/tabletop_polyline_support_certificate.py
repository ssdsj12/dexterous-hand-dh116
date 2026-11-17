"""Conservative full-3D segment support certificates, independent of PhysX.

For any unit-or-shorter normal n, min(V.n)-max(a.n,b.n) is a lower
bound on distance between the complete source shape and segment [a,b].
The minimum over the original adjacent segments covers the entire polyline.
No gap is a contact measurement. Negative bounds mean not proven clear.
"""
import hashlib
import math
import numpy as np
from .tabletop_measured_closure_geometry import rod_support_lower as legacy_lower

VERSION="dh116-polyline-support-certificate-v1"
RADIUS=.005
NUMERIC_GUARD=2e-6

def arrays(vertices,nodes):
    v=np.asarray(vertices,dtype=np.float64);p=np.asarray(nodes,dtype=np.float64)
    if (v.ndim!=2 or v.shape[1:]!=(3,) or len(v)<1 or p.ndim!=2
            or p.shape[1:]!=(3,) or len(p)<2 or not np.isfinite(v).all()
            or not np.isfinite(p).all()):
        raise ValueError("complete finite shape vertices and original adjacent nodes required")
    return v,p

def sha(a):
    a=np.ascontiguousarray(a,dtype="<f8")
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()

def directions(v,p):
    from scipy.spatial import ConvexHull,QhullError
    angle=np.arange(64)*2*np.pi/64
    n=[np.stack((np.cos(angle),np.zeros(64),np.sin(angle)),axis=1),
       np.eye(3),-np.eye(3),v.mean(0)[None]-(p[:-1]+p[1:])/2]
    hull_faces=0
    if len(v)>=4:
        try:
            faces=ConvexHull(v).equations[:,:3]
            hull_faces=len(faces);n.extend((faces,-faces))
        except QhullError:
            # Degenerate input still has valid axis/centroid separating candidates.
            pass
    n=np.concatenate(n);length=np.linalg.norm(n,axis=1)
    n=n[length>1e-15];length=length[length>1e-15]
    n=n/(length[:,None]*(1+1e-12))
    return np.unique(n,axis=0),hull_faces

def certify(vertices,nodes):
    v,p=arrays(vertices,nodes);n,hull_faces=directions(v,p)
    projection_min=np.concatenate([(v@n[i:i+256].T).min(0)
                                  for i in range(0,len(n),256)])
    projected=p@n.T
    pair_max=np.maximum(projected[:-1],projected[1:])
    support=projection_min[None]-pair_max
    winner=support.argmax(1)
    gaps=np.maximum(0,support[np.arange(len(p)-1),winner])-RADIUS-NUMERIC_GUARD
    old=float(legacy_lower(v,p))
    segments=[dict(segment=i,normal=n[j].tolist(),
                   shape_min_projection_m=float(projection_min[j]),
                   segment_max_projection_m=float(pair_max[i,j]),lower_m=float(gaps[i]))
              for i,j in enumerate(winner)]
    return dict(version=VERSION,vertices_sha256=sha(v),nodes_sha256=sha(p),
                original_vertex_count=len(v),original_node_count=len(p),
                rod_radius_m=RADIUS,numeric_guard_m=NUMERIC_GUARD,
                old_global_xz_lower_m=old,segment_lower_m=float(gaps.min()),
                lower_m=max(old,float(gaps.min())),segments=segments,
                candidate_direction_count=len(n),hull_face_count=hull_faces,
                geometry_scope="original_source_vertices_and_adjacent_node_capsules",
                actual_contact_measurement=False,native_qualification=False)

def validate(packet,vertices,nodes):
    v,p=arrays(vertices,nodes)
    if (packet.get("version")!=VERSION or packet.get("vertices_sha256")!=sha(v)
            or packet.get("nodes_sha256")!=sha(p) or packet.get("original_vertex_count")!=len(v)
            or packet.get("original_node_count")!=len(p) or packet.get("rod_radius_m")!=RADIUS
            or packet.get("numeric_guard_m")!=NUMERIC_GUARD
            or packet.get("actual_contact_measurement") is not False
            or packet.get("native_qualification") is not False
            or len(packet.get("segments",[]))!=len(p)-1):
        raise ValueError("certificate source, budget or semantics differ")
    lower=[]
    for i,s in enumerate(packet["segments"]):
        n=np.asarray(s["normal"],dtype=float)
        if s["segment"]!=i or n.shape!=(3,) or not np.isfinite(n).all() or np.linalg.norm(n)>1.:
            raise ValueError("ordered original segment and non-expansive finite normal required")
        vm=float((v@n).min());pm=float(max(p[i]@n,p[i+1]@n))
        gap=max(0.,vm-pm)-RADIUS-NUMERIC_GUARD
        for key,wanted in (("shape_min_projection_m",vm),("segment_max_projection_m",pm),("lower_m",gap)):
            if not math.isclose(s[key],wanted,rel_tol=0,abs_tol=1e-12):
                raise ValueError("full original vertex/segment support recomputation differs")
        lower.append(gap)
    old=float(legacy_lower(v,p))
    for key,wanted in (("old_global_xz_lower_m",old),("segment_lower_m",min(lower)),("lower_m",max(old,min(lower)))):
        if not math.isclose(packet[key],wanted,rel_tol=0,abs_tol=1e-12):
            raise ValueError("aggregate conservative bound differs")
    return True

def refined_lower(vertices,nodes):
    v,p=arrays(vertices,nodes);old=float(legacy_lower(v,p))
    if old>=0:return old
    result=certify(v,p)
    validate(result,v,p)
    return result["lower_m"]
