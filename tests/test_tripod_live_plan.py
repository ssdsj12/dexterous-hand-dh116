"""Independent left Tripod plan preserves original right-side identity and limits."""
import importlib.util
import numpy as np
import pytest

def module():
    name="scripts.prepare_dh116_tripod_owned_approach"
    assert importlib.util.find_spec(name) is not None,"independent Tripod plan missing"
    return __import__(name,fromlist=["derive_pair"])

def test_independent_seed_and_234_step_action_schedule():
    m=module();p=m.derive_pair();a=m.action_schedule(p).numpy()
    assert p["left"]["candidate"]["independent_style"]=="tripod"
    assert p["schedule"]==dict(setup_steps=2,quiet_steps=8,closure_steps=200,hold_steps=24,
        total_steps=234,policy_steps=29,physics_hz=480,control_hz=60,ppo_updates=0,environments=1)
    assert a.shape==(29,24) and np.max(abs(a))<=1
    assert not a[0].any() and not a[26:].any()
    assert not a[:,6:12].any() and not a[:,15:18].any() and not a[:,21:24].any()
    assert a[1:26,:6].any() and a[1:26,12:15].any() and a[1:26,18:21].any()
    assert p["native_qualified"] is p["teacher_admission"] is False

def test_original_right_usd_sdf_seed_and_free_rod_preserved():
    m=module();p=m.derive_pair();old=m.previous.load_pair()
    assert p["right"]==old["right"]
    for key in ("hand_roots","authored_root_seed_w","reset_root_pose_w","joint_positions","joint_velocities"):
        assert p["reset_seed"][key][1]==old["reset_seed"][key][1]
    assert p["rod_seed"]==old["rod_seed"]
    assert p["required_pads"]=={"left":["finger14_link","finger23_link","finger33_link"],
                               "right":["finger14_link","finger23_link"]}
    assert p["inactive_side_role"]=="right_independent_pinch_zero_action_not_tripod_validation"

def test_schedule_does_not_trust_modified_config():
    m=module();p=m.derive_pair();p["schedule"]["quiet_steps"]=0
    with pytest.raises(ValueError,match="frozen"):m.action_schedule(p)

def test_expected_boundary_phase_and_actual_xyz_order():
    m=module();p=m.derive_pair()
    start=m.expected_boundary(p,8);end=m.expected_boundary(p,208);hold=m.expected_boundary(p,232)
    assert start["phase"]=="quiet" and end["phase"]=="approach" and hold["phase"]=="observe"
    assert start["fraction"]==0 and end["fraction"]==hold["fraction"]==1
    assert np.array_equal(end["root_poses_w"],hold["root_poses_w"])
    assert np.allclose(start["root_poses_w"],p["reset_seed"]["reset_root_pose_w"])
    assert np.allclose(end["root_poses_w"][0],p["left"]["candidate"]["end_base_pose_wxyz"],atol=1e-12)
    assert np.array_equal(end["root_poses_w"][1],start["root_poses_w"][1])

@pytest.mark.parametrize("ordinal",[-1,233,8.5,True])
def test_outside_stage_identity_refused(ordinal):
    m=module()
    with pytest.raises(ValueError):m.expected_boundary(m.derive_pair(),ordinal)
