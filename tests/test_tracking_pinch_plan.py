"""New candidate plan and original action/geometry kernels, CPU only."""
from copy import deepcopy
import importlib
from pathlib import Path
import numpy as np
import pytest
from scripts import prepare_dh116_effective_right_pinch as old
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from go2_pvcnn.control.dh116.linear_coupling import expand
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def plan():return importlib.import_module("scripts.prepare_dh116_tracking_pinch")

def test_independent_reset_keeps_left_free_rod_original_limits(plan):
    p=plan.derive_pair();baseline=old.load_pair()
    assert p["left"]==baseline["left"] and p["rod_seed"]==baseline["rod_seed"]
    assert p["reset_seed"]["joint_positions"][0]==baseline["reset_seed"]["joint_positions"][0]
    assert p["reset_seed"]["reset_root_pose_w"][0]==baseline["reset_seed"]["reset_root_pose_w"][0]
    assert p["wrist_clearance_floor_m"]==.0021 and p["required_pad_min_gap_m"]==-.0003
    assert p["right"]["candidate"]["active_delta_rad"]==baseline["right"]["candidate"]["active_delta_rad"]
    assert p["right"]["candidate"]["base_pose_wxyz"]!=baseline["right"]["candidate"]["base_pose_wxyz"]
    assert p["native_qualified"] is p["teacher_admission"] is False
    expand(p["right"]["candidate"]["active_q_rad"])
    assert plan.load_pair()==p

def test_actual_twenty_actions_reconstruct_all232_boundaries(plan):
    p=plan.derive_pair();a=plan.action_schedule(p).numpy()
    assert a.shape==(29,24)
    assert np.array_equal(a[:,:6],np.zeros((29,6)))
    assert np.array_equal(a[:,12:15],np.zeros((29,3)))
    assert np.array_equal(a[:,18:],np.zeros((29,6)))
    np.testing.assert_allclose(a[:,6:12],old.action_schedule().numpy()[:,6:12],rtol=0,atol=1e-12)
    q=np.array(p["right"]["candidate"]["active_q_rad"]);offset=np.zeros(3)
    for ordinal in range(1,233):
        row=a[(ordinal-1)//8]
        q+=row[6:12]*ACTION_CONTRACT["joint_rate_rad_s"]/480
        offset+=row[15:18]*ACTION_CONTRACT["translation_rate_m_s"]/480
        target=plan.expected_boundary(p,ordinal)
        np.testing.assert_allclose(q,target["right_q_rad"],atol=1e-12,rtol=0)
        np.testing.assert_allclose(offset,target["right_offset_m"],atol=1e-12,rtol=0)
        assert abs(offset[0])<=.0001+1e-15
    assert np.max(abs(a[:,15]*ACTION_CONTRACT["translation_rate_m_s"]))<=.00114+1e-15
    assert offset[2]==pytest.approx(-.0005)

def test_foreign_plan_and_budget_rejected(plan):
    p=plan.derive_pair();p["right"]["candidate"]["active_delta_rad"][0]+=.01
    with pytest.raises(ValueError):plan.action_schedule(p)
    for k in (-1,233,True):
        with pytest.raises(ValueError):plan.expected_boundary(plan.derive_pair(),k)

def test_original_cpu_controller_and_projection_follow_new_plan(plan):
    from scripts import audit_dh116_right_approach_action_contract as audit
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    rows=[];out={}
    with bindings((audit,"operation",plan)):audit.run(out,rows.append)
    assert len(rows)==232 and not any(r["projection_clipped"] for r in rows)
    assert max(r["q_error_rad"] for r in rows)<1e-7
    assert max(r["root_error_m"] for r in rows)<1e-8

def test_new_geometry_uses_new_plan_without_mutating_old(plan):
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    from go2_pvcnn.control.dh116 import tabletop_effective_right_pinch_geometry as original
    fixture=importlib.import_module("test_effective_right_pinch_geometry")
    m=importlib.import_module("go2_pvcnn.control.dh116.tabletop_tracking_pinch_geometry")
    old_plan=original.plan;before=old.expected_boundary(old.load_pair(),149)
    with bindings((fixture,"plan",plan)):p,x=fixture.packet(149)
    assert m.validate_projection(p,x)["ordinal"]==149
    assert original.plan is old_plan
    np.testing.assert_array_equal(old.expected_boundary(old.load_pair(),149)["right_offset_m"],before["right_offset_m"])
    x["wrist_projection"]["safe_z_clipped"][0,1]=True
    with pytest.raises(ValueError):m.validate_projection(p,x)
    assert original.plan is old_plan
