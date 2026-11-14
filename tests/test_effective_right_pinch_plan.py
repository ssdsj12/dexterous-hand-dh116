"""CPU admission/identity of new right-only original-pad approach, no Isaac."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from scripts import prepare_dh116_effective_right_pinch as plan
from scripts import prepare_dh116_right_pinch_owned_contact as previous
from go2_pvcnn.control.dh116.linear_coupling import expand
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
ROOT=Path(__file__).resolve().parents[1]

def test_new_reset_preserves_left_rod_limits_and_independent_right_sources():
    p=plan.derive_pair();old=previous.load_pair()
    assert p["left"]==old["left"] and p["rod_seed"]==old["rod_seed"]
    assert p["reset_seed"]["joint_positions"][0]==old["reset_seed"]["joint_positions"][0]
    assert p["reset_seed"]["reset_root_pose_w"][0]==old["reset_seed"]["reset_root_pose_w"][0]
    assert p["right"]["candidate"]["side"]=="right"
    assert not p["native_qualified"] and not p["teacher_admission"]
    assert p["required_pad_min_gap_m"]==-.0003 and p["wrist_clearance_floor_m"]==.0021
    expand(p["right"]["candidate"]["active_q_rad"])

def test_24d_twenty_control_actions_reconstruct_all232_boundaries():
    p=plan.derive_pair();a=plan.action_schedule(p).numpy()
    assert a.shape==(29,24) and np.max(abs(a))<=1
    assert np.array_equal(a[:,:6],np.zeros((29,6)))
    assert np.array_equal(a[:,12:15],np.zeros((29,3)))
    assert np.array_equal(a[:,18:],np.zeros((29,6)))
    q=np.array(p["right"]["candidate"]["active_q_rad"]);offset=np.zeros(3)
    for ordinal in range(1,233):
        row=a[(ordinal-1)//8]
        q+=row[6:12]*ACTION_CONTRACT["joint_rate_rad_s"]/480
        offset+=row[15:18]*ACTION_CONTRACT["translation_rate_m_s"]/480
        target=plan.expected_boundary(p,ordinal)
        np.testing.assert_allclose(q,target["right_q_rad"],atol=1e-12,rtol=0)
        np.testing.assert_allclose(offset,target["right_offset_m"],atol=1e-12,rtol=0)
    assert a[6:26,15].max()>0 and a[6:26,15].min()<0
    assert abs(offset[0])<1e-16 and offset[2]==pytest.approx(-.0005)

def test_modified_plan_or_out_of_budget_step_rejected():
    p=plan.derive_pair()
    for ordinal in (-1,233,True):
        with pytest.raises(ValueError):plan.expected_boundary(p,ordinal)
    p["right"]["candidate"]["active_delta_rad"][0]+=.01
    with pytest.raises(ValueError):plan.action_schedule(p)

def test_original_cpu_control_kernels_follow_new_targets_without_projection():
    from scripts import audit_dh116_right_approach_action_contract as audit
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    rows=[];out={}
    with bindings((audit,"operation",plan)):
        audit.run(out,rows.append)
    assert len(rows)==232
    assert not any(r["projection_clipped"] for r in rows)
    assert max(r["q_error_rad"] for r in rows)<1e-7
    assert max(r["root_error_m"] for r in rows)<1e-8
