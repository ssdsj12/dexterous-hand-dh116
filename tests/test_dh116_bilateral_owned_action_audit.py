import importlib
from pathlib import Path
import numpy as np
import pytest

def module():
    assert (Path(__file__).parents[1]/"scripts/audit_dh116_bilateral_owned_action_contract.py").exists(),"bilateral control replay missing"
    return importlib.import_module("scripts.audit_dh116_bilateral_owned_action_contract")

@pytest.mark.parametrize("side",["left","right"])
def test_either_hand_target_error_is_detected(side):
    m=module();p=m.operation.derive_pair();e=m.operation.expected_boundary(p,100)
    q=np.r_[e["left_q_rad"],e["right_q_rad"]]
    assert m.target_errors(p,q,e["root_poses_w"],100)==pytest.approx((0.,0.))
    q[0 if side=="left" else 7]+=.01
    assert m.target_errors(p,q,e["root_poses_w"],100)[0]==pytest.approx(.01)

def test_root_error_is_not_hidden_by_correct_joint_targets():
    m=module();p=m.operation.derive_pair();e=m.operation.expected_boundary(p,100)
    q=np.r_[e["left_q_rad"],e["right_q_rad"]];r=e["root_poses_w"].copy();r[1,2]+=.001
    assert m.target_errors(p,q,r,100)[1]==pytest.approx(.001)
