"""Fixed bases must not be teleported again when their submitted pose is unchanged."""
from types import SimpleNamespace
import pytest
import torch
from go2_pvcnn.control.dh116.tabletop_tensor_world import TensorWorld


class Hand:
    def __init__(self):
        self.calls = []

    def write_root_pose_to_sim(self, pose, env_ids=None):
        self.calls.append((pose.clone(), None if env_ids is None else env_ids.clone()))


def world():
    w = TensorWorld.__new__(TensorWorld)
    w.hands = [Hand(), Hand()]
    w._root_pose_written = torch.zeros((3, 2, 7), dtype=torch.float32)
    w._root_pose_written[:, :, 3] = 1.
    return w


def test_unchanged_bases_are_not_written_even_below_float32_resolution():
    w = world()
    p = w._root_pose_written[:, 0].double().clone()
    p[:, 3] += 1e-10
    w.write_root_pose(0, p)
    assert w.hands[0].calls == []


def test_only_changed_environments_are_written_and_cached_by_value():
    w = world()
    p = w._root_pose_written[:, 1].clone()
    p[2, 0] = .01
    w.write_root_pose(1, p)
    assert len(w.hands[1].calls) == 1
    pose, ids = w.hands[1].calls[0]
    assert ids.tolist() == [2]
    torch.testing.assert_close(pose, p[2:3])
    w.write_root_pose(1, p)
    assert len(w.hands[1].calls) == 1
    # The caller can mutate its buffer, including moving back to the old pose.
    p[2, 0] = 0.
    w.write_root_pose(1, p)
    assert len(w.hands[1].calls) == 2
    assert not w.hands[0].calls


def test_explicit_reset_always_writes_selected_rows_and_updates_cache():
    w = world()
    ids = torch.tensor([2, 0])
    p = w._root_pose_written[ids, 0].clone()
    w.write_root_pose(0, p, env_ids=ids)
    w.write_root_pose(0, p, env_ids=ids)
    assert len(w.hands[0].calls) == 2
    p[:, 2] = torch.tensor([.03, .07])
    w.write_root_pose(0, p, env_ids=ids)
    full = w._root_pose_written[:, 0].clone()
    assert full[:, 2].tolist() == [p[1, 2].item(), 0., p[0, 2].item()]
    w.write_root_pose(0, full)
    assert len(w.hands[0].calls) == 3


def test_root_write_checkpoint_migration_is_exact_and_one_way():
    from go2_pvcnn.control.dh116 import tabletop_tensor_world as module
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    expected = module.ROOT_POSE_WRITE_CONTRACT
    old = dict(num_envs=4096, physics_hz=480, velocity_iterations=16)
    new = dict(old, root_pose_write=expected)
    assert checkpoint_contract_changes(old, new) == ['skip_unchanged_fixed_root_pose']
    assert checkpoint_contract_changes(new, new) == []
    for bad in (dict(new, num_envs=1024), dict(new, root_pose_write='other')):
        with pytest.raises(ValueError, match='contract mismatch'):
            checkpoint_contract_changes(old, bad)
    with pytest.raises(ValueError, match='contract mismatch'):
        checkpoint_contract_changes(new, old)


def test_advance_does_not_teleport_paused_hands_but_applies_lift_motion():
    w = world()
    w.n, w.dt = 3, 1 / 480
    w.palm_limit = torch.tensor([.003, .003, .004])
    w.palms = torch.zeros(3, 2, 3)
    w.clearances = torch.full((3, 2), .003)
    w.gains = torch.full((3, 2), 10.)
    w.parameters = SimpleNamespace(stiffness_rate=10., stiffness_max=20.)
    w.root_seed = w._root_pose_written.clone()
    w.root_targets = w.root_seed.clone()
    w.commands = torch.zeros(3, 2, 6)
    w.refresh = lambda: None
    w.w = SimpleNamespace(ids=[list(range(6))] * 2,
        args=SimpleNamespace(headless=True), sim=SimpleNamespace(step=lambda **kw: None))
    for hand in w.hands:
        hand.set_joint_position_target = lambda *a, **kw: None
        hand.write_joint_stiffness_to_sim = lambda *a, **kw: None
        hand.write_data_to_sim = lambda: None
        hand.actuators = {'active': SimpleNamespace(stiffness=torch.zeros(3, 6))}
    task = SimpleNamespace(palm_actions=torch.ones(3, 2, 3),
        offset=torch.zeros(3), command=torch.zeros(3, 12))
    w.advance(task, torch.zeros(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool))
    assert all(not hand.calls for hand in w.hands)
    task.offset[1] = .0001
    w.advance(task, torch.zeros(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool))
    for hand in w.hands:
        assert len(hand.calls) == 1
        pose, ids = hand.calls[0]
        assert ids.tolist() == [1]
        assert pose[0, 2].item() == task.offset[1].item()


def test_reset_forces_unchanged_base_writes_with_unsorted_environment_ids():
    w = world()
    w.device = 'cpu'
    w.seed_q = [torch.zeros(3, 6) for _ in range(2)]
    w.seed_qd = [torch.zeros(3, 6) for _ in range(2)]
    w.root_seed = w._root_pose_written.clone()
    w.root_seed[:, :, 2] = torch.tensor([.1, .2, .3])[:, None]
    w.initial_offset = .01
    w.root_targets = w.root_seed.clone()
    w.commands = torch.zeros(3, 2, 6, dtype=torch.float64)
    w.palms, w.gains = torch.zeros(3, 2, 3), torch.zeros(3, 2)
    w.seed_wire, w.seed_velocity = torch.zeros(3, 7), torch.zeros(3, 6)
    w.link_loads, w.forces = torch.zeros(3, 2, 17), torch.zeros(3, 4)
    w.vertical_components=torch.ones(3,2,2)
    w.contact_center_w=torch.ones(3,2,3)
    w.contact_normal_w=torch.ones(3,2,3)
    w.contact_normal_load_n=torch.ones(3,2)
    w.contact_valid=torch.ones(3,2,dtype=torch.bool)
    w.contact_all_below_centerline=torch.ones(3,2,dtype=torch.bool)
    w.contact_count=torch.ones(3,2,dtype=torch.long)
    noop = lambda *args, **kw: None
    w.refresh, w.measure = noop, noop
    w.wire = SimpleNamespace(reset=noop, write_root_pose_to_sim=noop, write_root_velocity_to_sim=noop)
    w.w = SimpleNamespace(ids=[list(range(6))] * 2, sensors=[],
        sim=SimpleNamespace(physics_sim_view=SimpleNamespace(update_articulations_kinematic=noop)))
    for hand in w.hands:
        for name in ('reset', 'write_joint_state_to_sim', 'write_root_velocity_to_sim',
                     'set_joint_position_target', 'write_joint_stiffness_to_sim', 'write_data_to_sim'):
            setattr(hand, name, noop)
        hand.actuators = {'active': SimpleNamespace(stiffness=torch.zeros(3, 6))}
    ids = torch.tensor([2, 0])
    w.reset(ids)
    assert not w.vertical_components[ids].any()
    assert w.vertical_components[1].eq(1).all()
    for values in (w.contact_center_w,w.contact_normal_w):
        assert values[ids].isnan().all()
        assert values[1].eq(1).all()
    for values in (w.contact_normal_load_n,w.contact_valid,
                   w.contact_all_below_centerline,w.contact_count):
        assert not values[ids].any()
        assert values[1].eq(1).all()
    w.reset(ids)
    for hi, hand in enumerate(w.hands):
        assert len(hand.calls) == 2
        pose, written_ids = hand.calls[-1]
        assert written_ids.tolist() == [2, 0]
        expected = w.root_seed[ids, hi].clone()
        expected[:, 2] += w.initial_offset
        torch.testing.assert_close(pose, expected)
        torch.testing.assert_close(w._root_pose_written[ids, hi], expected)
        assert w._root_pose_written[1, hi, 2] == 0.


@pytest.mark.parametrize("headless", [False, True])
def test_advance_renders_once_per_control_tick_without_skipping_physics(headless):
    w = world()
    w.n, w.dt = 3, 1 / 480
    w.palm_limit = torch.tensor([.003, .003, .004])
    w.palms = torch.zeros(3, 2, 3)
    w.clearances = torch.full((3, 2), .003)
    w.gains = torch.full((3, 2), 10.)
    w.parameters = SimpleNamespace(stiffness_rate=10., stiffness_max=20.)
    w.root_seed = w._root_pose_written.clone()
    w.root_targets = w.root_seed.clone()
    w.commands = torch.zeros(3, 2, 6)
    w.refresh = lambda: None
    w.w = SimpleNamespace(ids=[list(range(6))] * 2,
        args=SimpleNamespace(headless=headless), sim=SimpleNamespace(step=lambda **kw: None))
    for hand in w.hands:
        hand.set_joint_position_target = lambda *a, **kw: None
        hand.write_joint_stiffness_to_sim = lambda *a, **kw: None
        hand.write_data_to_sim = lambda: None
        hand.actuators = {'active': SimpleNamespace(stiffness=torch.zeros(3, 6))}
    task = SimpleNamespace(palm_actions=torch.ones(3, 2, 3),
        offset=torch.zeros(3), command=torch.zeros(3, 12))
    w.decimation = 8
    w._physics_steps = 0
    renders, samples = [], []
    w.w.sim.step = lambda **kw: renders.append(kw['render'])
    w.refresh = lambda: samples.append(len(renders))
    for _ in range(16):
        w.advance(task, torch.zeros(3, dtype=torch.bool), torch.ones(3, dtype=torch.bool))
    assert len(renders) == len(samples) == 16
    assert [i for i,r in enumerate(renders) if r] == ([] if headless else [7,15])
