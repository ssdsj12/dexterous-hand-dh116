import numpy as np
import pytest


def test_pinch_material_alignment_keeps_transformed_pair_midpoint_and_tangent():
    from scripts.screen_dh116_pinch_supported_reset import material_transfer

    old = np.array([[0., 0., 0.], [0., 1., 0.]])
    new = np.array([[0., 1., 2.], [0., 1., 3.]])
    rotation, center_old, center_new = material_transfer(old,new)
    np.testing.assert_allclose(rotation.apply(old-center_old)+center_new,new,atol=1e-12)
    with pytest.raises(ValueError):
        material_transfer(old,np.zeros((2,3)))


def test_supported_pinch_seed_cannot_enter_default_power_reset():
    from scripts.verify_dh116_pinch_supported_reset import REQUEST
    from go2_pvcnn.control.dh116.tabletop_power_supported_task import prepare_initial_state
    import json

    raw = json.loads(REQUEST.read_bytes())
    config = raw["config"]
    table = {"center_m":[.65,0.,.275], "size_m":[.8,1.,.05]}
    with pytest.raises(ValueError):
        prepare_initial_state(config,table)
    seed, active, rod, poses = prepare_initial_state(config,table,expected_scope=config["scope"])
    assert active.shape == (2,6) and rod.shape == (41,3) and set(poses)=={"support_0","support_1"}

def test_supported_pinch_controls_require_own_settled_physx_joint_order():
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
    from scripts.screen_dh116_pinch_supported_approach import settled_control_state

    identity = {'physics_step': 240}
    root = [1.0, 0.0, .45, 1.0, 0.0, 0.0, 0.0]
    names = ['wrist_x', *ACTIVE]
    hand = {'names': names, 'position': np.ones((1, 7)), 'target': np.zeros((1, 7))}
    sample = {'rod': identity, 'native_joint_state': {'left': hand.copy(), 'right': hand.copy()},
              'hand_link_poses': {}}
    seed = {'hands': {side: {'pose_wxyz': root} for side in ('left', 'right')}}
    post = settled_control_state(sample, seed)
    assert post['action_readback']['post']['hands']['left']['active_indices'] == list(range(1, 7))
    np.testing.assert_array_equal(post['wrist_projection']['projected_root_poses'], [[root, root]])
    sample['native_joint_state']['left']['names'] = names[:-1]+['unregistered']
    with pytest.raises(ValueError):
        settled_control_state(sample, seed)
