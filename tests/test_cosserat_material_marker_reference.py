"""Analytic material-marker kinematics only; no solver or native contact source."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import math
import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_cosserat_material_marker_reference import (
    make_marker,marker_position,bind_pad_reference,measure_transport,source_contract)

REST=np.array([.1,.1])
NODES=np.array([[0.,0,0],[.1,0,0],[.2,0,0]])
DIRECTORS=np.tile([0.,0,0,1.],(3,1))
IDENTITY=np.eye(4)


def marker(offset=(.005,0.)):
    return make_marker(REST,segment_index=0,alpha=.5,offset_yz_m=offset,
                       material_reference_uid="analytic-rod-reset-1")


def reference(m=None,normal=(0.,0,1.)):
    return bind_pad_reference(m or marker(),REST,NODES,DIRECTORS,IDENTITY,
                             fixed_contact_normal_pad=normal)


def test_rest_coordinate_and_cross_section_axes_follow_original_x_director():
    m=marker()
    assert m.rest_s_m==pytest.approx(.05)
    np.testing.assert_allclose(marker_position(m,REST,NODES,DIRECTORS),[.05,.005,0])
    with pytest.raises(FrozenInstanceError):m.alpha=.6


def test_relative_translation_uses_fixed_reference_pad_plane():
    ref=reference()
    translated=NODES+[.001,.002,.003]
    result=measure_transport(ref,REST,translated,DIRECTORS,IDENTITY)
    assert result["marker_tangent_displacement_m"]==pytest.approx(math.sqrt(5)*.001)
    assert result["marker_normal_displacement_m"]==pytest.approx(.003)
    assert result["current_contact_s_m"] is result["r0_tangential_slip_m"] is None
    assert result["physical_error_bound_m"] is None
    assert result["native_qualification"] is result["teacher_admission"] is False


def test_common_rigid_rotation_translation_cancels_in_pad_material_frame():
    ref=reference()
    angle=.7;c,s=math.cos(angle),math.sin(angle)
    R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    shift=np.array([.2,-.3,.4])
    pad=np.eye(4);pad[:3,:3]=R;pad[:3,3]=shift
    q=np.tile([0,0,math.sin(angle/2),math.cos(angle/2)],(3,1))
    result=measure_transport(ref,REST,NODES@R.T+shift,q,pad)
    assert result["marker_tangent_displacement_m"]<1e-15
    assert abs(result["marker_normal_displacement_m"])<1e-15


def test_material_director_spin_changes_surface_marker_with_identical_centerline():
    # Spin about original material x/tangent; the geometry-only tangent is unchanged.
    angle=.2
    q=np.tile([math.sin(angle/2),0,0,math.cos(angle/2)],(3,1))
    m=marker();ref=reference(m,normal=(0,1.,0))
    result=measure_transport(ref,REST,NODES,q,IDENTITY)
    assert result["marker_tangent_displacement_m"]==pytest.approx(.005*math.sin(angle))
    assert result["marker_tangent_displacement_m"]>.0005
    np.testing.assert_allclose(marker_position(marker((0,0)),REST,NODES,q),[.05,0,0])


def test_stretch_does_not_change_fixed_rest_s_or_offset():
    m=marker();ref=reference(m)
    result=measure_transport(ref,REST,NODES*2,DIRECTORS,IDENTITY)
    assert m.rest_s_m==pytest.approx(.05)
    assert result["marker_tangent_displacement_m"]==pytest.approx(.05)
    assert result["fixed_rest_s_m"]==pytest.approx(.05)
    assert result["current_contact_s_m"] is None


def test_bending_uses_selected_segment_director_without_neighbor_blending():
    m=make_marker(REST,segment_index=1,alpha=.5,offset_yz_m=(.005,0),
        material_reference_uid="analytic")
    nodes=np.array([[0.,0,0],[.1,0,0],[.1,.1,0]])
    q=DIRECTORS.copy();q[1]=[0,0,2**-.5,2**-.5]
    np.testing.assert_allclose(marker_position(m,REST,nodes,q),[.095,.05,0],atol=1e-15)


def test_quaternion_sign_equivalence_and_unused_padding():
    np.testing.assert_allclose(marker_position(marker(),REST,NODES,DIRECTORS),
                               marker_position(marker(),REST,NODES,-DIRECTORS))
    q=DIRECTORS.copy();q[-1]=[0,0,0,0]  # installed final entry is unused padding
    np.testing.assert_allclose(marker_position(marker(),REST,NODES,q),[.05,.005,0])


@pytest.mark.parametrize("fault",["rest","director","position","pad","normal","alpha","offset"])
def test_malformed_or_changed_reference_sources_rejected(fault):
    m=marker();ref=reference(m);rest=REST.copy();q=DIRECTORS.copy();p=NODES.copy();pad=IDENTITY.copy()
    if fault=="rest":rest[0]+=.001
    if fault=="director":q[0]=[0,0,0,0]
    if fault=="position":p[0,1]=np.nan
    if fault=="pad":pad[0,0]=2.
    with pytest.raises(ValueError):
        if fault=="normal":reference(normal=(0,0,0))
        elif fault=="alpha":make_marker(rest,segment_index=0,alpha=1.01,offset_yz_m=(0,0),material_reference_uid="x")
        elif fault=="offset":make_marker(rest,segment_index=0,alpha=.5,offset_yz_m=(np.nan,0),material_reference_uid="x")
        else:measure_transport(ref,rest,p,q,pad)


def test_fixed_reference_cannot_reselect_normal_or_material_station():
    ref=reference()
    with pytest.raises(FrozenInstanceError):ref.normal_pad=(1.,0,0)
    with pytest.raises(FrozenInstanceError):ref.marker=marker((0,0))


def test_installed_source_hashes_and_director_convention_are_exact():
    root=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/teacher-pair-build-zQEenV/source")
    if not root.exists():pytest.skip("existing Ubuntu pair-build source unavailable")
    result=source_contract(root)
    assert result["segment_director_axis"]==[1,0,0]
    assert result["quaternion_order"]=="xyzw"
    assert result["last_entry_is_padding"] is True
    assert result["native_qualification"] is False


def test_source_guard_rejects_modified_source_before_any_reference_use(tmp_path):
    with pytest.raises((OSError,ValueError)):source_contract(tmp_path)

def test_material_director_is_not_replaced_with_sheared_geometric_tangent():
    nodes=NODES.copy();nodes[1]=[.1,.03,0.]
    # Material +X stays x while the centerline edge has a y component.
    value=marker_position(marker(),REST,nodes,DIRECTORS)
    np.testing.assert_allclose(value,[.05,.02,0.],atol=1e-15)


def test_node_station_does_not_choose_between_adjacent_segment_material_frames():
    left=make_marker(REST,segment_index=0,alpha=1.,offset_yz_m=(.005,0),
        material_reference_uid="same-rod")
    right=make_marker(REST,segment_index=1,alpha=0.,offset_yz_m=(.005,0),
        material_reference_uid="same-rod")
    q=DIRECTORS.copy();q[1]=[0,0,2**-.5,2**-.5]
    assert left.rest_s_m==right.rest_s_m
    assert np.linalg.norm(marker_position(left,REST,NODES,q)-marker_position(right,REST,NODES,q))>.005

def test_accumulated_material_coordinate_overflow_is_rejected():
    with pytest.raises(ValueError,match="total material length"):
        make_marker([1e308,1e308],segment_index=1,alpha=.5,offset_yz_m=(0,0),
                    material_reference_uid="invalid")
