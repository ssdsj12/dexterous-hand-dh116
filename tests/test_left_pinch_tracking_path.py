from copy import deepcopy
import importlib
import numpy as np
import pytest

@pytest.fixture(scope="module")
def m():return importlib.import_module("scripts.audit_dh116_left_pinch_tracking_path")

def test_frozen_own_left_candidate(m):
    r,g=m.load_candidate()
    assert r["qualification"]["empirical_endpoint_passed"]
    assert g.side=="left"
    p=m.parent_record(r,g)
    assert p["sides"]["left"]["search"]["best"]["admission"]["static_constraints_passed"]
    assert not p["sides"]["right"]["search"]["best"]["admission"]["static_constraints_passed"]

def test_right_path_unchanged_and_left_original_delta(m):
    r,g=m.load_candidate();pair=m.left.plan.load_pair(m.left.ROOT)
    end=r["all_observed_validation"]["endpoint"]
    for ordinal in (0,48,49,100,208,232):
        old=m.left.plan.expected_boundary(pair,ordinal)
        poses=m.commanded_pair(r,pair,old["fraction"])
        np.testing.assert_allclose(poses["right"]["root_pose_wxyz"],old["root_poses_w"][1],atol=1e-14)
        np.testing.assert_allclose(poses["right"]["q_rad"],old["right_q_rad"],atol=1e-14)
    start=m.commanded_pair(r,pair,0.)["left"]
    np.testing.assert_allclose(np.asarray(end["q_rad"])-start["q_rad"],r["source"]["original_joint_delta_rad"],atol=1e-14)

def test_interhand_bound_counts_both_moving_hands(m):
    assert m.combined_gap(.01,.002,.004,3)==pytest.approx(.0085)
    with pytest.raises(ValueError):m.combined_gap(.01,-.002,.004,3)

def test_only_complete_actual_guard_evidence_passes(m):
    result=dict(continuous_static_path={"sides":{"left":{"static_path_passed":True,"path_samples":1025}}},
        continuous_interhand={"passed":True,"samples":1025},
        observed_path={"passed":True,"samples":232,"first_failure":None})
    assert m.qualification(result)["conditional_path_passed"]
    for section in ("continuous_interhand","observed_path"):
        bad=deepcopy(result);bad[section]["passed"]=False
        assert not m.qualification(bad)["conditional_path_passed"]
    assert not m.qualification(result)["teacher_admission"]
