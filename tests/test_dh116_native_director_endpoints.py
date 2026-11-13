"""CPU geometry and immutable native-archive replay, never new Isaac execution."""
from copy import deepcopy
import numpy as np
import pytest
from scripts import audit_dh116_native_director_endpoints as audit


def straight():
    nodes=np.zeros((41,3)); nodes[:,0]=np.linspace(0,.5,41)
    q=np.zeros((41,4));q[:,3]=1
    rest=np.linalg.norm(np.diff(nodes,axis=0),axis=1)
    return nodes,q,rest


def test_identity_and_padding_are_not_spurious_material_segments():
    p,q,r=straight();q[-1]=[2,3,4,5]
    out=audit.geometry_metrics(p,q,r,p,q)
    assert out["real_segments"]==40
    assert out["node_displacement_m"]["max"]==0
    assert out["director_rotation_from_reset_deg"]["max"]==0
    assert out["director_centerline_angle_deg"]["max"]==0
    assert out["relative_segment_length_change"]["max"]==0
    assert out["r0_tangential_slip_m"] is None
    assert out["contact_qualified"] is False


def test_rigid_translation_and_axial_twist_are_distinct_from_contact_slip():
    p,q,r=straight()
    moved=p+np.array([0,.002,0])
    turned=q.copy();turned[:-1]=[np.sin(.1),0,0,np.cos(.1)]
    out=audit.geometry_metrics(moved,turned,r,p,q)
    assert out["node_displacement_m"]["max"]==pytest.approx(.002)
    assert out["director_rotation_from_reset_deg"]["max"]==pytest.approx(np.degrees(.2))
    assert out["director_centerline_angle_deg"]["max"]==pytest.approx(0)
    assert out["r0_tangential_slip_m"] is None


def test_quaternion_sign_does_not_change_metrics():
    p,q,r=straight()
    a=audit.geometry_metrics(p,q,r,p,q)
    b=audit.geometry_metrics(p,-q,r,p,q)
    assert a==b


def test_misalignment_and_axial_strain_have_explicit_units():
    p,q,r=straight();m=p*1.01
    q[:-1]=[0,0,np.sin(np.pi/4),np.cos(np.pi/4)]
    out=audit.geometry_metrics(m,q,r,p,straight()[1])
    assert out["director_centerline_angle_deg"]["p95"]==pytest.approx(90)
    assert out["relative_segment_length_change"]["p50"]==pytest.approx(.01)
    assert out["node_displacement_m"]["max"]==pytest.approx(.005)


@pytest.mark.parametrize("fault",["nan","short_q","nonunit","zero_segment","zero_rest","bad_baseline"])
def test_missing_or_inconsistent_geometry_rejected(fault):
    p,q,r=straight();p0=p.copy();q0=q.copy()
    if fault=="nan":p[0,0]=np.nan
    if fault=="short_q":q=q[:-1]
    if fault=="nonunit":q[1]=0
    if fault=="zero_segment":p[2]=p[1]
    if fault=="zero_rest":r[0]=0
    if fault=="bad_baseline":q0[4]=[1,1,1,1]
    with pytest.raises(ValueError):audit.geometry_metrics(p,q,r,p0,q0)


def test_archived_native_endpoints_and_no_new_training(tmp_path):
    out=audit.summarize()
    assert out["source_native_receipt_sha256"]==audit.RECEIPT_SHA
    assert out["source_native_physics_steps"]==18
    assert out["this_audit_new_physics_steps"]==out["ppo_updates"]==0
    assert len(out["endpoints"])==17
    assert [r["identity"]["physics_step"] for r in out["endpoints"]]==list(range(2,19))
    assert out["native"]["outer_exit_code"]==0
    assert out["native"]["material_packets"]==17
    assert out["native"]["hand_rod_pair_samples"]==0
    assert out["native"]["table_pair_samples"]==7872
    assert out["teacher_admission"] is out["grasp_qualified"] is False
    assert out["r0_tangential_slip_m"] is None
    assert out["comparison"]["clocks_and_action_schedule_equal"] is True
    assert all(v>=0 for v in out["comparison"]["maximum_abs_readback_difference"].values())


def test_archive_wrong_receipt_hash_rejected():
    with pytest.raises(ValueError,match="SHA"):
        audit.summarize(receipt_sha="0"*64)
