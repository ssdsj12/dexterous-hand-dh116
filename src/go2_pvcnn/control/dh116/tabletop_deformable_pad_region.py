"""Original effective-pad region applied to same-boundary solver contact points.

Only the frame/sign adapter is new. The original atlas and _on_patch remain
authoritative. Patch membership alone cannot supply depth, slip or success.
"""
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .tabletop_effective_pad_runtime import _on_patch
from .tabletop_deformable_bimanual_step import CLOCK_KEYS

VERSION="dh116-deformable-original-pad-region-v1"
def array(v,shape):
    a=np.asarray(v,dtype=float)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError("finite exact-shape measured array")
    return a

def rigid(pose):
    p=array(pose,(4,4));R=p[:3,:3]
    if (not np.allclose(p[3],[0,0,0,1],atol=1e-12,rtol=0)
        or not np.allclose(R.T@R,np.eye(3),atol=1e-7,rtol=0)
        or not math.isclose(np.linalg.det(R),1.,abs_tol=1e-7)):
        raise ValueError("proper rigid link transform required")
    return p

def independent_gates(checked):
    if type(checked) is not bool:raise ValueError("explicit patch membership")
    return dict(patch_membership_checked=checked,contact_depth_m=None,
        fixed_material_slip_m=None,positive_effective_contact_qualified=False,
        native_qualification=False,success_qualified=False,teacher_admission=False)

def classify_pair(pair,link_to_world,atlas):
    T=rigid(link_to_world)
    p=array(pair["contact_point_world_m"],(3,));n=array(pair["outward_normal_world"],(3,))
    fn=pair["normal_force_on_rod_n"]
    if abs(np.linalg.norm(n)-1)>2e-5 or type(fn) not in (int,float) or not math.isfinite(fn) or fn<0:
        raise ValueError("actual unit obstacle normal and nonnegative load required")
    if (atlas["half_widths_m"]!=[.004,.0035] or atlas["surface_tolerance_m"]!=.001
        or atlas["normal_cosine_min"]!=.8):
        raise ValueError("original unexpanded effective-pad atlas contract required")
    c=array(atlas["center_local_m"],(3,));u=array(atlas["tangent_u"],(3,));v=array(atlas["tangent_v"],(3,))
    planes=np.asarray(atlas["planes_local"],dtype=float)
    if (planes.ndim!=2 or planes.shape[1]!=4 or not len(planes) or not np.isfinite(planes).all()
        or not np.allclose(np.linalg.norm(planes[:,:3],axis=1),1.,atol=1e-8,rtol=0)):
        raise ValueError("original finite unit supporting planes required")
    local=(p-T[:3,3])@T[:3,:3]
    # Solver stores obstacle outward / normal on ROD. Original native sampler
    # consumes force ON HAND, hence one explicit sign conversion, no mirror.
    on_hand=-n@T[:3,:3]
    uv=np.array([(local-c)@u,(local-c)@v])
    ellipse=float(np.square(uv/atlas["half_widths_m"]).sum())
    signed=planes[:,:3]@local+planes[:,3];j=int(signed.argmax())
    cosine=float(on_hand@planes[j,:3]/np.linalg.norm(on_hand))
    reasons=[k for k,b in (
        ("no_positive_compression",fn<=0),
        ("outside_inner_ellipse",ellipse>1.),
        ("outside_source_surface",abs(signed[j])>.001),
        ("wrong_normal",cosine>-.8)) if b]
    row={k:torch.as_tensor(atlas[k],dtype=torch.float64) for k in
         ("center_local_m","tangent_u","tangent_v","half_widths_m","planes_local")}
    row.update(surface_tolerance_m=.001,normal_cosine_min=.8)
    accepted=fn>0 and bool(_on_patch(torch.as_tensor(local[None]),torch.as_tensor(on_hand[None]),row)[0])
    if accepted!= (not reasons):raise ValueError("audit differs from original runtime predicate")
    return dict(version=VERSION,point_link_m=local.tolist(),normal_on_hand_local=on_hand.tolist(),
        uv_m=uv.tolist(),ellipse_value=ellipse,plane_signed_distance_m=float(signed[j]),
        normal_cosine_on_hand=cosine,accepted_by_original_patch=accepted,
        first_rejection=None if accepted else reasons[0],overlap_rejections=reasons,
        **independent_gates(True))

def bind_geometry(raw,clock,mesh_name,binding):
    if any(raw["identity"].get(k)!=clock.get(k) for k in CLOCK_KEYS):
        raise ValueError("same actual SDK/rod/policy/episode boundary required")
    parts=mesh_name.split(".")
    if len(parts)!=2 or parts[0] not in ("left","right"):raise ValueError("own side and pad name")
    side,name=parts;h=raw["hands"][side];names=h["body_names"]
    if len(names)!=len(set(names)) or name not in names:raise ValueError("unique native body name")
    i=names.index(name);batches=h["link_paths"]
    if len(batches)!=1 or len(batches[0])!=len(names):raise ValueError("one complete native link-path batch required")
    paths=batches[0]
    if (binding["side"]!=side or binding["body_name"]!=name or binding["body_index"]!=i
        or binding["frame"]!="PhysX_link_origin" or binding["unit"]!="m"
        or paths[i]!=binding["prim_path"]
        or side+"Hand" not in paths[i].split("/") or paths[i].split("/")[-1]!=name):
        raise ValueError("native own-side link/body/SDF frame binding differs")
    poses=np.asarray(h["link_poses_xyzw"])
    if poses.shape!=(1,len(names),7):raise ValueError("one measured complete articulation")
    pose=array(poses[0,i],(7,))
    if abs(np.linalg.norm(pose[3:])-1)>1e-5:raise ValueError("actual unit XYZW quaternion")
    T=np.eye(4);T[:3,:3]=Rotation.from_quat(pose[3:]).as_matrix();T[:3,3]=pose[:3]
    return rigid(T)
