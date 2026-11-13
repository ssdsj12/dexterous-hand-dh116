"""Analytic fixed Cosserat material-marker transport, never a native slip source.

The existing solver stores one segment director q_i (xyzw) whose local +X is
the material tangent. A reference marker uses fixed rest s and fixed local YZ
offset. This is a piecewise linear centerline/rigid cross-section reconstruction
hypothesis, not the node-sphere collision surface or a measured contact station.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import numpy as np

VERSION="dh116-cosserat-material-marker-reference-cpu-v1"
SOURCE_HASHES={
 "third_party/yarnball/KittenEngine/YarnBall/sim/cosserat.cu":"7c9151c1c4df64c4735b7df69de605d030bbbb99e8b9638095ef99fb3dbddb03",
 "third_party/yarnball/KittenEngine/YarnBall/YarnBall.cu":"06af0339837c196cc8f716b6095b50e5409cbe7544c01d9da8a9067f0b801e68",
 "csrc/bindings.cpp":"89b3928faf49ec26e2d53614c0e30c5d6242892e54c8345c45f4d5702077d8d5"}


def source_contract(root):
    sources={}
    for name,expected in SOURCE_HASHES.items():
        path=(Path(root)/name).resolve(strict=True)
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected:raise ValueError("installed pair-build source differs: "+name)
        sources[name]=dict(path=str(path),sha256=actual)
    return dict(version=VERSION,sources=sources,segment_director_axis=[1,0,0],
        quaternion_order="xyzw",segment_index_meaning="i_to_i_plus_1",
        last_entry_is_padding=True,director_source="cosserat.cu:76-83",
        getter_source="bindings.cpp:380-389,856-860",native_qualification=False)


def _array(value,shape,label):
    a=np.asarray(value,dtype=np.float64)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError("finite "+label+" shape required")
    return a


def _rest(value):
    rest=np.asarray(value,dtype=np.float64)
    if rest.ndim!=1 or len(rest)<1 or not np.isfinite(rest).all() or np.any(rest<=0):
        raise ValueError("positive finite material rest segments required")
    with np.errstate(over="ignore"):
        if not np.isfinite(rest.sum()):raise ValueError("finite total material length required")
    return rest


def _digest(rest):
    # Numeric values, order and number are fixed; current geometric lengths
    # are never substituted for original material coordinates.
    return hashlib.sha256(np.asarray(rest,dtype="<f8").tobytes()).hexdigest()


@dataclass(frozen=True)
class MaterialMarker:
    material_reference_uid: str
    rest_sha256: str
    segment_index: int
    alpha: float
    rest_s_m: float
    offset_yz_m: tuple


@dataclass(frozen=True)
class FixedPadReference:
    marker: MaterialMarker
    point_pad: tuple
    normal_pad: tuple


def make_marker(rest_lengths_m,*,segment_index,alpha,offset_yz_m,material_reference_uid):
    rest=_rest(rest_lengths_m)
    if (type(segment_index) is not int or not 0<=segment_index<len(rest)
            or type(alpha) not in (int,float) or not math.isfinite(alpha) or not 0<=alpha<=1
            or not isinstance(material_reference_uid,str) or not material_reference_uid):
        raise ValueError("explicit fixed segment/fraction/material reference required")
    offset=_array(offset_yz_m,(2,),"cross-section offset")
    s=float(rest[:segment_index].sum()+alpha*rest[segment_index])
    return MaterialMarker(material_reference_uid,_digest(rest),segment_index,float(alpha),s,tuple(offset))


def _rotation(q):
    x,y,z,w=_array(q,(4,),"XY ZW segment director")
    length=math.sqrt(x*x+y*y+z*z+w*w)
    if abs(length-1)>1e-4:raise ValueError("unit material director required")
    x,y,z,w=np.asarray([x,y,z,w])/length
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def marker_position(marker,rest_lengths_m,nodes_w,segment_quaternions_xyzw):
    """Evaluate x(s,t)+Q_i(t)*(0,rho_y,rho_z), retaining director spin."""
    if type(marker) is not MaterialMarker:raise ValueError("explicit fixed material marker required")
    rest=_rest(rest_lengths_m)
    rebuilt=make_marker(rest,segment_index=marker.segment_index,alpha=marker.alpha,
        offset_yz_m=marker.offset_yz_m,material_reference_uid=marker.material_reference_uid)
    if marker!=rebuilt:raise ValueError("material rest coordinate/reference changed")
    nodes=_array(nodes_w,(len(rest)+1,3),"rod nodes")
    directors=_array(segment_quaternions_xyzw,(len(rest)+1,4),"all segment directors with final padding")
    i,alpha=marker.segment_index,marker.alpha
    center=(1-alpha)*nodes[i]+alpha*nodes[i+1]
    offset=_rotation(directors[i])@np.asarray([0.,*marker.offset_yz_m])
    result=center+offset
    if not np.isfinite(result).all():raise ValueError("material marker arithmetic overflow")
    return result


def _pad(value):
    pose=_array(value,(4,4),"pad pose")
    R=pose[:3,:3]
    if (not np.allclose(pose[3],[0,0,0,1],atol=1e-10,rtol=0)
            or not np.allclose(R.T@R,np.eye(3),atol=1e-7,rtol=0)
            or not np.isclose(np.linalg.det(R),1,atol=1e-7,rtol=0)):
        raise ValueError("proper rigid pad transform required")
    return pose


def bind_pad_reference(marker,rest_lengths_m,nodes_w,segment_quaternions_xyzw,pad_to_world,
                       *,fixed_contact_normal_pad):
    """Freeze a supplied reference normal; do not infer/reselect from geometry."""
    point=marker_position(marker,rest_lengths_m,nodes_w,segment_quaternions_xyzw)
    pose=_pad(pad_to_world)
    normal=_array(fixed_contact_normal_pad,(3,),"reference contact normal")
    norm=float(np.linalg.norm(normal))
    if abs(norm-1)>1e-5:raise ValueError("unit fixed reference contact normal required")
    return FixedPadReference(marker,tuple((point-pose[:3,3])@pose[:3,:3]),tuple(normal/norm))


def measure_transport(reference,rest_lengths_m,nodes_w,segment_quaternions_xyzw,pad_to_world):
    """Relative material displacement at the original pad-plane reference.

    No contact eligibility, continuous contact s, uncertainty or native clock
    is supplied by this analytic helper. No force integration is performed.
    """
    if type(reference) is not FixedPadReference:raise ValueError("frozen pad/material reference required")
    point=marker_position(reference.marker,rest_lengths_m,nodes_w,segment_quaternions_xyzw)
    pose=_pad(pad_to_world)
    origin=_array(reference.point_pad,(3,),"fixed reference point")
    normal=_array(reference.normal_pad,(3,),"fixed reference normal")
    if not np.isclose(np.linalg.norm(normal),1,atol=1e-10,rtol=0):
        raise ValueError("fixed reference normal changed")
    local=(point-pose[:3,3])@pose[:3,:3]
    delta=local-origin
    normal_delta=float(delta@normal)
    tangent=delta-normal_delta*normal
    distance=float(np.linalg.norm(tangent))
    if not np.isfinite(delta).all() or not math.isfinite(distance):
        raise ValueError("relative material transport arithmetic overflow")
    return dict(version=VERSION,scope="analytic_fixed_material_marker_transport_only",
        material_reference_uid=reference.marker.material_reference_uid,
        fixed_rest_s_m=reference.marker.rest_s_m,world_material_marker_m=point.tolist(),
        pad_local_material_marker_m=local.tolist(),fixed_reference_point_pad_m=origin.tolist(),
        fixed_reference_normal_pad=normal.tolist(),relative_displacement_pad_m=delta.tolist(),
        marker_tangent_displacement_vector_pad_m=tangent.tolist(),
        marker_tangent_displacement_m=distance,marker_normal_displacement_m=normal_delta,
        current_contact_s_m=None,r0_tangential_slip_m=None,physical_error_bound_m=None,
        native_clock_alignment_verified=False,contact_marker_correspondence_verified=False,
        collision_surface_model_equivalence_verified=False,native_qualification=False,
        teacher_admission=False,training_reward=None)
