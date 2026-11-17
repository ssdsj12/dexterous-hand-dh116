import pytest
from go2_pvcnn.control.dh116.tabletop_teacher_video_delivery import measured_annotation

def sample(eligible=True):
    frame=dict(sdk_step=99,sdk_time=1.0,policy_step=3,net_lift=.008)
    row=dict(sdk_step=99,sdk_time=1.,policy_step=3,
        clearance=[dict(eligible=eligible,status="clear" if eligible else "supported",
        net_lift_m=.008 if eligible else None,reason="qualified" if eligible else "support")],
        support_loads=[dict(contact_data_valid=True,step_id=99,total_support_force_norm_n=0 if eligible else 1)])
    return frame,row

def test_supported_height_is_not_qualified_lift():
    frame,row=sample(False)
    a=measured_annotation(frame,row,0)
    assert a["height_delta_m"]==.008 and a["net_lift_m"] is None
    assert not a["counts_as_grasp_success"]

def test_cleared_support_height_retains_actual_net_lift():
    frame,row=sample()
    a=measured_annotation(frame,row,0)
    assert a["net_lift_m"]==pytest.approx(.008)
    assert a["support_status"]=="clear"

def test_stale_ledger_rejected():
    frame,row=sample();row["sdk_step"]=98
    with pytest.raises(ValueError,match="identity"):measured_annotation(frame,row,0)

def test_missing_contact_validity_never_qualified():
    frame,row=sample();row["support_loads"][0]["contact_data_valid"]=False
    assert measured_annotation(frame,row,0)["net_lift_m"] is None
