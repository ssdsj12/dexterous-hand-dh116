import numpy as np
import pytest
from scripts.prepare_dh116_right_pinch_owned_contact import derive_pair, action_schedule, expected_boundary
from go2_pvcnn.control.dh116.tabletop_dual_hand_reset import _seed
from scripts.prepare_dh116_pinch_margin_contact import load_pair as old_pair

def test_right_only_original24d_schedule_and_exact_closed_budget():
    p=derive_pair();a=action_schedule(p).numpy()
    assert a.shape==(29,24) and np.all(a[:6]==0) and np.all(a[26:]==0)
    assert set(np.flatnonzero(np.any(a!=0,axis=0)))=={6,7,8,17}
    assert np.max(abs(a))<=1
    np.testing.assert_allclose(a[:,6:12].sum(0)*.3*8/480,p["right"]["candidate"]["active_delta_rad"],atol=1e-16)
    assert a[:,17].sum()*.01*8/480==pytest.approx(-.0005)
    assert p["schedule"]["total_steps"]==234 and not p["native_qualified"]

def test_original_reset_contract_and_other_hand_rod_are_preserved():
    p=derive_pair();old=old_pair()
    assert _seed(p["reset_seed"])==p["reset_seed"]
    assert p["left"]==old["left"] and p["rod_seed"]==old["rod_seed"]
    for name in ("hand_roots","authored_root_seed_w","reset_root_pose_w","joint_positions","joint_velocities"):
        assert p["reset_seed"][name][0]==old["reset_seed"][name][0]
    assert p["right"]["source_sha256"]==old["right"]["source_sha256"]
    assert p["right"]["candidate"]["side"]=="right"
    assert p["right"]["candidate"]["native_ready"] is False

def test_expected_physical_boundary_not_policy_only():
    p=derive_pair()
    assert expected_boundary(p,48)["fraction"]==0
    assert expected_boundary(p,49)["fraction"]==pytest.approx(1/160)
    assert expected_boundary(p,208)["fraction"]==1
    assert expected_boundary(p,232)["fraction"]==1
    with pytest.raises(ValueError):expected_boundary(p,233)

def test_config_cannot_silently_change_right_seed(tmp_path):
    import json
    from scripts.prepare_dh116_right_pinch_owned_contact import CONFIG,load_pair
    p=derive_pair();p["reset_seed"]["joint_positions"][1][0]+=.001
    path=tmp_path/CONFIG;path.parent.mkdir(parents=True);path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match="frozen"):load_pair(tmp_path)

def test_explicit_float64_schedule_roundtrip_is_not_default_float32():
    p=derive_pair();a=action_schedule(p)
    import torch
    assert a.dtype==torch.float64
    np.testing.assert_allclose(a[6,6:12].numpy()*.1,p["right"]["candidate"]["active_delta_rad"],rtol=0,atol=1e-16)
