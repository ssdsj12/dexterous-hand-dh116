"""Independent reset search may widen pose proposals, never closure amplitude."""
from copy import deepcopy
import numpy as np
import pytest
from scripts import solve_dh116_left_pinch_tracking as m

def test_expanded_reset_preserves_physical_pose_for_rescaled_seed():
    p=m.load_packet();x=np.array([.2,-.2,.3,.1,-.1,.2,.3,-.2,.2])
    old=m.decode_candidate(p,x)
    new=m.decode_candidate(p,x/2,config=m.RESET_CONFIG)
    np.testing.assert_allclose(old["root_pose_wxyz"],new["root_pose_wxyz"],atol=1e-14)
    np.testing.assert_allclose(old["q_rad"],new["q_rad"],atol=1e-14)
    start=m.start_for(p,new)
    np.testing.assert_allclose(np.array(new["q_rad"])-start["q_rad"],p["original_joint_delta_rad"],atol=1e-14)

def test_new_reset_config_cannot_change_original_guards():
    p=m.load_packet();g=m.Geometry(p,config=m.RESET_CONFIG)
    v=g.evaluate(np.zeros(9),p["scenarios"][-1:])
    assert m.RESET_CONFIG["original_motion_unchanged"] is True
    assert v["native_ready"] is False
    bad=deepcopy(m.RESET_CONFIG);bad["active_scales_rad"]=[10.]*3
    with pytest.raises(ValueError):m.decode_candidate(p,np.zeros(9),config=bad)
