"""Raise only the independent left approach start; retain existing motion limits."""
import numpy as np
import pytest
from scripts import audit_dh116_left_pinch_tracking_path as m

def test_raised_start_converges_to_same_endpoint_and_right_unchanged():
    r,g=m.load_candidate();pair=m.left.plan.load_pair(m.left.ROOT)
    for f in (0.,.25,.5,.75,1.):
        old=m.commanded_pair(r,pair,f);new=m.commanded_pair(r,pair,f,left_start_raise_m=.0015)
        assert new["right"]==old["right"]
        np.testing.assert_allclose(new["left"]["q_rad"],old["left"]["q_rad"],atol=0)
        expected=np.asarray(old["left"]["root_pose_wxyz"]);expected[2]+=.0015*(1-f)
        np.testing.assert_allclose(new["left"]["root_pose_wxyz"],expected,atol=1e-14)
    assert m.validate_approach(m.APPROACH_CONFIG)["left_z_rate_m_s"]==pytest.approx(-.0045)

def test_unfrozen_or_over_speed_approach_rejected():
    bad=dict(m.APPROACH_CONFIG,left_start_raise_m=1.)
    with pytest.raises(ValueError):m.validate_approach(bad)

def test_path_auditor_override_does_not_mutate_original_plan(tmp_path):
    from types import SimpleNamespace
    original_parent=m.prior.path.targets.parent
    original_pair=original_parent.load_pair(m.left.ROOT)
    before=original_pair["left"]["candidate"].get("root_translation_delta_m",[0,0,0])
    auditor=m.make_auditor(tmp_path/"unused.json","a"*64,.0015)
    custom=auditor.targets.parent.load_pair(m.left.ROOT)
    assert custom["left"]["candidate"]["root_translation_delta_m"]==[0.,0.,-.0015]
    assert m.prior.path.targets.parent is original_parent
    assert original_parent.load_pair(m.left.ROOT)["left"]["candidate"].get("root_translation_delta_m",[0,0,0])==before
