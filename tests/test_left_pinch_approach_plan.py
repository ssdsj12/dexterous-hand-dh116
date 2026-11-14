from copy import deepcopy
import importlib
import numpy as np
import pytest
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
from go2_pvcnn.control.dh116.linear_coupling import expand

@pytest.fixture(scope="module")
def m():return importlib.import_module("scripts.prepare_dh116_left_pinch_approach")

def test_independent_own_left_preserves_right_rod_and_physics(m):
    p=m.derive_pair();old=m.previous.load_pair()
    assert p["right"]==old["right"]
    assert p["rod_seed"]==old["rod_seed"]
    assert p["wrist_clearance_floor_m"]==.0021
    assert p["required_pad_min_gap_m"]==-.0003
    assert p["left"]["candidate"]["active_delta_rad"]==old["left"]["candidate"]["active_delta_rad"]
    assert p["left"]["candidate"]["static_reset_only"] is False
    assert p["reset_seed"]["joint_positions"][1]==old["reset_seed"]["joint_positions"][1]
    assert p["native_qualified"] is p["teacher_admission"] is False

def test_all232_boundaries_reconstruct_both_hands(m):
    p=m.derive_pair();a=m.action_schedule(p).numpy();old=m.previous.action_schedule().numpy()
    np.testing.assert_array_equal(a[:,6:12],old[:,6:12])
    np.testing.assert_array_equal(a[:,15:],old[:,15:])
    q=np.array([p[s]["candidate"]["active_q_rad"] for s in ("left","right")]);off=np.zeros((2,3))
    for ordinal in range(1,233):
        u=a[(ordinal-1)//8];q+=u[:12].reshape(2,6)*ACTION_CONTRACT["joint_rate_rad_s"]/480
        off+=u[12:18].reshape(2,3)*ACTION_CONTRACT["translation_rate_m_s"]/480
        e=m.expected_boundary(p,ordinal)
        np.testing.assert_allclose(q[0],e["left_q_rad"],atol=1e-12,rtol=0)
        np.testing.assert_allclose(q[1],e["right_q_rad"],atol=1e-12,rtol=0)
        np.testing.assert_allclose(off,np.array([e["left_offset_m"],e["right_offset_m"]]),atol=1e-12,rtol=0)
    assert off[0,2]==pytest.approx(-.0015)

def test_original_control_kernel_both_joint_targets_and_projection(m):
    from scripts import audit_dh116_right_approach_action_contract as audit
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    rows=[];out={}
    with bindings((audit,"operation",m)):audit.run(out,rows.append)
    p=m.load_pair()
    for row in rows:
        e=m.expected_boundary(p,row["ordinal"])
        np.testing.assert_allclose(row["projected_fingers"][0,:6],e["left_q_rad"],atol=1e-7)
        np.testing.assert_allclose(row["projected_fingers"][0,6:],e["right_q_rad"],atol=1e-7)
    assert len(rows)==232 and not any(row["projection_clipped"] for row in rows)
