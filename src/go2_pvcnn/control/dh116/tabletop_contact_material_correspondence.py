"""Diagnose node-sphere contact versus fixed Cosserat surface material.

This does not invent continuous contact s or a physical slip/error bound.
Only final-substep contact points can be compared with endpoint directors.
All candidate residuals are retained, including incompatible sphere contacts.
"""
import math
import numpy as np
from . import tabletop_cosserat_material_marker_reference as reference

VERSION="dh116-contact-material-endpoint-correspondence-v1"
CENTER_CONSISTENCY_M=2e-6
ORIGINAL_SLIP_LIMIT_M=.0005

def real_directors(rest,nodes,quaternions):
    rest=reference._rest(rest)
    nodes=reference._array(nodes,(len(rest)+1,3),"actual node positions")
    q=reference._array(quaternions,(len(rest),4),"actual real segment directors")
    if not np.allclose(np.linalg.norm(q,axis=-1),1.,atol=1e-4,rtol=0):
        raise ValueError("all actual real directors must be unit XYZW")
    # Legacy analytic helper accepts an unused trailing row. This sentinel is
    # never measured, indexed, returned as source data or treated as a segment.
    unused=np.zeros((1,4))
    return rest,nodes,np.concatenate((q,unused),axis=0)

def contact_candidate(pair,rest_lengths,nodes,quaternions,*,radius_m,material_reference_uid):
    if (type(radius_m) not in (float,int) or not math.isfinite(radius_m) or radius_m<=0
            or not isinstance(material_reference_uid,str) or not material_reference_uid):
        raise ValueError("native positive radius and explicit episode material identity required")
    rest,nodes,q=real_directors(rest_lengths,nodes,quaternions)
    node=pair["node_index"]
    if (type(pair["substep"]) is not int or pair["substep"]!=11
            or type(node) is not int or not 0<=node<len(nodes)):
        raise ValueError("final solver substep and original integer material node required")
    point=reference._array(pair["contact_point_world_m"],(3,),"actual sphere contact")
    normal=reference._array(pair["outward_normal_world"],(3,),"actual world normal")
    norm=float(np.linalg.norm(normal))
    if abs(norm-1.)>2e-5:raise ValueError("unit actual obstacle normal required")
    normal=normal/norm
    residual=float(np.linalg.norm(point+radius_m*normal-nodes[node]))
    if residual>CENTER_CONSISTENCY_M:
        raise ValueError("contact sphere center and same-endpoint node disagree")
    segment=min(node,len(rest)-1);alpha=1. if node==len(rest) else 0.
    R=reference._rotation(q[segment])
    offset=R.T@(point-nodes[node])
    marker=reference.make_marker(rest,segment_index=segment,alpha=alpha,
        offset_yz_m=tuple(offset[1:]),material_reference_uid=material_reference_uid)
    projected=reference.marker_position(marker,rest,nodes,q)
    model_residual=float(np.linalg.norm(projected-point))
    mapping=dict(node_index=node,segment_index=segment,alpha=alpha,
        segment_axis_world=R[:,0].tolist(),material_local_contact_offset_m=offset.tolist(),
        sphere_center_residual_m=residual,axial_cross_section_residual_m=float(abs(offset[0])),
        radial_surface_residual_m=float(abs(np.linalg.norm(offset[1:])-radius_m)),
        surface_model_candidate_residual_m=model_residual,
        original_slip_limit_m=ORIGINAL_SLIP_LIMIT_M,
        exceeds_original_slip_threshold_as_model_residual=model_residual>ORIGINAL_SLIP_LIMIT_M,
        consistency_tolerance_m=CENTER_CONSISTENCY_M,
        consistency_tolerance_is_not_physical_error_bound=True,
        radial_offset_not_rescaled=True,continuous_contact_s_m=None)
    return dict(version=VERSION,marker=marker,mapping=mapping,
        reference_normal_world=normal.copy(),contact_marker_correspondence_verified=False,
        r0_tangential_slip_m=None,physical_error_bound_m=None,
        native_qualification=False,teacher_admission=False)

def bind_candidate(candidate,rest,nodes,quaternions,pad_to_world):
    if candidate.get("version")!=VERSION:
        raise ValueError("explicit diagnostic candidate version required")
    rest,nodes,q=real_directors(rest,nodes,quaternions)
    pad=reference._pad(pad_to_world)
    normal=reference._array(candidate["reference_normal_world"],(3,),"reference world normal")
    return reference.bind_pad_reference(candidate["marker"],rest,nodes,q,pad,
        fixed_contact_normal_pad=pad[:3,:3].T@normal)

def transport(fixed_reference,rest,nodes,quaternions,pad_to_world,*,endpoint_contact_present):
    if type(endpoint_contact_present) is not bool:raise ValueError("explicit measured endpoint contact flag required")
    rest,nodes,q=real_directors(rest,nodes,quaternions)
    result=reference.measure_transport(fixed_reference,rest,nodes,q,pad_to_world)
    result.update(correspondence_version=VERSION,endpoint_contact_present=endpoint_contact_present,
        substep_path_bound_available=False,contact_reference_reselected=False,
        contact_loss_resets_material_reference=False,rigid_cross_section_reconstruction_hypothesis=True)
    return result
