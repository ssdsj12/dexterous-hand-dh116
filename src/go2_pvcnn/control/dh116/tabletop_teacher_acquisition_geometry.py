"""Offline actual-node SDF acquisition geometry. Never predicts contact force."""
import numpy as np
from scipy.ndimage import map_coordinates

def sample_nodes(grid,nodes,pose,radius):
    nodes=np.asarray(nodes,dtype=float);pose=np.asarray(pose,dtype=float)
    if nodes.shape!=(41,3) or not np.isfinite(nodes).all():
        raise ValueError("exactly 41 finite actual rod nodes required")
    if (pose.shape!=(4,4) or not np.isfinite(pose).all()
            or not np.isfinite(radius) or radius<=0):
        raise ValueError("finite rigid transform and positive finite radius required")
    rotation=pose[:3,:3]
    if (not np.allclose(pose[3],[0.,0.,0.,1.],atol=1e-8,rtol=0)
            or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6,rtol=0)
            or not np.isclose(np.linalg.det(rotation),1.,atol=1e-6,rtol=0)):
        raise ValueError("proper rigid transform required")
    values=np.asarray(grid["values"]);origin=np.asarray(grid["origin"])
    spacing=float(grid["spacing"])
    if (values.ndim!=3 or min(values.shape)<2 or not np.isfinite(values).all()
            or origin.shape!=(3,) or not np.isfinite(origin).all()
            or not np.isfinite(spacing) or spacing<=0):
        raise ValueError("finite 3D grid, three-vector origin and positive finite spacing required")
    p=(nodes-pose[:3,3])@pose[:3,:3]
    coords=(p-origin)/spacing
    inside=np.all((coords>=0)&(coords<=np.array(values.shape)-1),axis=1)
    gaps=map_coordinates(values,coords[inside].T,order=1,output=np.float64,prefilter=False,mode="constant",cval=np.nan)-radius
    return dict(nodes_inside_grid=int(inside.sum()),nodes_outside_grid=int((~inside).sum()),
        nearest_sdf_gap_m=float(gaps.min()) if len(gaps) else None,
        maximum_sampled_penetration_m=float(max(0,-gaps.min())) if len(gaps) else None)

def required_reachable(rows,required,budget=.001):
    return all(rows[k]["nearest_sdf_gap_m"] is not None and
        0<=rows[k]["nearest_sdf_gap_m"]<=budget for k in required)

def freeze_decision(reachable,source_passed,all_cooked_available):
    blockers=[]
    if not reachable:blockers.append("required pad unknown, penetrating, or outside 1mm recovery budget")
    if not source_passed:blockers.append("source geometry/table clearance checks do not pass")
    if not all_cooked_available:blockers.append("twelve non-pad live cooked shapes per hand unavailable")
    return dict(freeze_s0=not blockers,native_admission=False,blockers=blockers)

def checked_geometry(call):
    try:return call()
    except ValueError as error:
        return dict(status="unknown",intersects_capsule=None,error=str(error))

def translated_poses(poses,translation):
    out={k:np.array(v,copy=True) for k,v in poses.items()}
    for t in out.values():t[:3,3]+=translation
    return out

def source_radius(scene):
    radius=float(scene["wire_radius_m"])
    if not np.isfinite(radius) or radius<=0:raise ValueError("finite actual scene radius required")
    return radius

def capsule_plane_bound(vertices,nodes,radius):
    from scipy.spatial import ConvexHull
    planes=ConvexHull(vertices).equations
    normals=planes[:,:3];norms=np.linalg.norm(normals,axis=1)
    distances=(np.asarray(nodes)@normals.T+planes[:,3])/norms
    # Each segment is separated if both its ends clear one supporting plane.
    bound=float(np.min(np.max(np.minimum(distances[:-1],distances[1:]),axis=1))-radius)
    return dict(method="continuous_segment_supporting_plane_lower_bound",
        capsule_clearance_lower_bound_m=bound,
        intersects_capsule=False if bound>0 else None,
        status="separated" if bound>0 else "unknown")

def proposal_nodes(original,observed):
    original=np.asarray(original,dtype=float);observed=np.asarray(observed,dtype=float)
    if (original.shape!=(41,3) or observed.shape!=(41,3)
            or not np.isfinite(original).all() or not np.isfinite(observed).all()):
        raise ValueError("41 finite constructor and observed nodes required")
    return dict(constructor_nodes=original.copy().tolist(),
        observed_reference_nodes=observed.copy().tolist())
