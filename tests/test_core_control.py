"""CPU contracts at the mounted Articulation and tensor command boundaries."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from dh116.env.contracts import JOINTS
from dh116.env.contracts import expand
from dh116.hand.kinematics import JOINT_NAMES
from dh116.hand.kinematics import ANGULAR_JOINT_NAMES
from dh116.hand.articulation import RegexArticulationSide
from dh116.hand.safety import TensorControl
from dh116.rl.teacher import SharedSourceIdentity as SharedSourceIdentity
from dh116.env.contracts import surfa_contr__ACTION_CONTRACT as ACTION_CONTRACT
NAMES = list(JOINTS + JOINT_NAMES + ANGULAR_JOINT_NAMES)

class Composer:

    def __init__(self):
        self.force = torch.full((3, 23, 3), 7.0)
        self.torque = self.force.clone()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.force[ids] = 0
        self.torque[ids] = 0

class View:

    def __init__(self):
        self.prim_paths = [f'/World/envs/env_{i}/leftHand/wrist_fixed' for i in (10, 0, 2)]
        self.q = torch.tensor([10.0, 0.0, 2.0])[:, None].repeat(1, 17)
        self.qd = self.q + 0.25
        self.targets = self.q + 100
        self.velocity = self.q + 200
        self.effort = self.q + 300
        self.root = torch.zeros(3, 7)
        self.root[:, 0] = torch.tensor([10.0, 0.0, 2.0])
        self.root[:, 6] = 1
        self.root_velocity = torch.ones(3, 6)
        self.calls = []

    def get_dof_positions(self):
        return self.q

    def get_dof_velocities(self):
        return self.qd

    def get_dof_position_targets(self):
        return self.targets

    def get_root_transforms(self):
        return self.root

    def get_root_velocities(self):
        return self.root_velocity

    def get_link_transforms(self):
        return self.root[:, None].repeat(1, 23, 1)

    def get_link_velocities(self):
        return self.root_velocity[:, None].repeat(1, 23, 1)

    def write(self, key, data, indices):
        target = getattr(self, key)
        assert data.shape == target.shape, 'SDK selected setters require a full-view payload'
        target[indices] = data[indices]
        self.calls.append((key, indices.clone()))

    def set_dof_position_targets(self, data, indices):
        self.write('targets', data, indices)

    def set_dof_velocity_targets(self, data, indices):
        self.write('velocity', data, indices)

    def set_dof_actuation_forces(self, data, indices):
        self.write('effort', data, indices)

    def apply_forces_and_torques_at_position(self, force_data, torque_data, position_data, indices, is_global):
        assert force_data.shape == torque_data.shape == (3 * 23, 3)
        self.calls.append(('force', indices.clone()))

class Hand:
    device = 'cpu'
    joint_names = NAMES
    body_names = ['base_link'] + [f'link_{i}' for i in range(22)]
    num_instances = 3
    is_initialized = True

    def __init__(self):
        self.root_physx_view = View()
        v = self.root_physx_view
        self.data = SimpleNamespace(joint_pos_target=v.targets.clone(), joint_vel_target=v.velocity.clone(), joint_effort_target=v.effort.clone())
        self.permanent_wrench_composer = Composer()
        self.instantaneous_wrench_composer = Composer()
        self.actuators = {'active': SimpleNamespace(stiffness=torch.ones(3, 6))}

    def reset(self, env_ids):
        self.permanent_wrench_composer.reset(env_ids)
        self.instantaneous_wrench_composer.reset(env_ids)

    def cache(self, key, value, joint_ids, env_ids):
        ids = torch.arange(17) if joint_ids is None else torch.as_tensor(joint_ids)
        getattr(self.data, key)[env_ids[:, None], ids] = value

    def set_joint_position_target(self, value, joint_ids=None, env_ids=None):
        self.cache('joint_pos_target', value, joint_ids, env_ids)

    def set_joint_velocity_target(self, value, joint_ids=None, env_ids=None):
        self.cache('joint_vel_target', value, joint_ids, env_ids)

    def set_joint_effort_target(self, value, joint_ids=None, env_ids=None):
        self.cache('joint_effort_target', value, joint_ids, env_ids)

    def write_joint_state_to_sim(self, q, qd, env_ids):
        self.root_physx_view.q[env_ids] = q
        self.root_physx_view.qd[env_ids] = qd

    def write_root_pose_to_sim(self, pose, env_ids):
        self.root_physx_view.root[env_ids] = pose[:, [0, 1, 2, 4, 5, 6, 3]]

    def write_root_velocity_to_sim(self, velocity, env_ids):
        self.root_physx_view.root_velocity[env_ids] = velocity

    def write_joint_stiffness_to_sim(self, *args, **kwargs):
        pass

    def write_joint_effort_limit_to_sim(self, *args, **kwargs):
        pass

    def write_data_to_sim(self):
        raise AssertionError('selected reset must preserve other instantaneous wrenches')

def articulation():
    hand = Hand()
    ids = (0, 2, 10)
    return (RegexArticulationSide(hand, ids, tuple((f'/World/envs/env_{i}' for i in ids)), '/wrist_fixed'), hand)

def test_actual_regex_row_mapping_owns_reads_and_orders_selected_targets():
    (side, hand) = articulation()
    state = side.read()
    assert state['q'][:, 0].tolist() == [0, 2, 10] and state['root_pose_wxyz'][:, 3].tolist() == [1, 1, 1]
    hand.root_physx_view.q.add_(40)
    assert state['q'][:, 0].tolist() == [0, 2, 10]
    side.set_joint_position_target(torch.tensor([[31.0, 32.0], [41.0, 42.0]]), joint_ids=[0, 3], env_ids=[10, 0])
    assert hand.data.joint_pos_target[:2, [0, 3]].tolist() == [[31, 32], [41, 42]]
    for invalid in ([2, 2], [3], [2.5]):
        with pytest.raises(ValueError):
            side.rows(invalid)

def test_selected_reset_preserves_peer_state_targets_and_pending_reactions():
    (side, hand) = articulation()
    before = side.read()
    force = hand.instantaneous_wrench_composer.force.clone()
    q = torch.zeros(1, 17)
    q[:, :11] = torch.tensor(expand([0.2] * 6))
    qd = torch.zeros_like(q)
    pose = torch.tensor([[0.3, 0.4, 0.5, 1.0, 0.0, 0.0, 0.0]])
    side.reset(env_ids=[2], root_pose_wxyz=pose, joint_pos=q, joint_vel=qd, active_effort_limit_nm=0.2)
    after = side.read()
    for key in ('q', 'qd', 'backend_targets', 'root_pose_wxyz', 'root_velocity'):
        torch.testing.assert_close(after[key][[0, 2]], before[key][[0, 2]], rtol=0, atol=0)
    torch.testing.assert_close(after['backend_targets'][1], q[0], rtol=0, atol=0)
    torch.testing.assert_close(after['root_pose_wxyz'][1], pose[0], rtol=0, atol=0)
    assert torch.equal(hand.instantaneous_wrench_composer.force[:2], force[:2]) and (not hand.instantaneous_wrench_composer.force[2].any())
    assert all((rows.tolist() == [2] for (_, rows) in hand.root_physx_view.calls))

def control_fixture():
    writes = []
    sides = {}
    for name in ('left', 'right'):
        sides[name] = SimpleNamespace(joint_names=NAMES, hand=SimpleNamespace(permanent_wrench_composer=SimpleNamespace(reset=lambda : writes.append('reset'))), set_wrenches=lambda **kw: writes.append(kw), set_joint_position_target=lambda *a, **kw: writes.append('target'))
    batch = SimpleNamespace(sides=sides, write_data_to_sim=lambda : writes.append('write'))
    roots = torch.zeros(2, 2, 7, dtype=torch.float64)
    roots[:, :, 3] = 1
    roots[:, :, 2] = 0.6
    roots[:, 0, 0] = -0.1
    roots[:, 1, 0] = 0.1
    q = torch.zeros(2, 2, 17, dtype=torch.float32)
    q[:, :, :11] = torch.tensor(expand([0.2] * 6))
    state = dict(q=q, qd=torch.zeros_like(q), backend_targets=q.clone(), root_pose_wxyz=roots.clone(), link_pose_xyzw=torch.zeros(2, 2, 1, 7, dtype=torch.float64), body_com_pose_w=torch.zeros(2, 2, 1, 7, dtype=torch.float64))
    control = TensorControl(batch, roots, {'left.pad': dict(side='left', body_index=0), 'right.pad': dict(side='right', body_index=0)}, ('left.pad', 'right.pad', 'table'))
    stamp = SharedSourceIdentity(1, 2, (0, 1), 'run', ('a', 'b'), ('rod0', 'rod1'), (1, 1), (0, 0), 2, 2 / 480, 1 / 480, 1 / 480, 2 / 480, 4.0, 'a' * 64, 'cpu_fixture')
    action = torch.full((2, 24), 0.2)
    action[1] *= -1
    tr = control.prepare(action, state, torch.full((2, 2), 0.01), roots, stamp=stamp)
    w = torch.zeros(2, 3, 6, dtype=torch.float64)
    source = dict(source_solver_id=1, parent_num_envs=2, run_uuid='run', source_physics_step=2, applied_physics_step=3, rod_ticks=(0, 0), episode_uids=('a', 'b'), reset_ids=(1, 1), solver_time_s=4.0, mesh_wrenches=w)
    return (control, stamp, state, tr, w, source, writes)

def test_all24_commands_and_com_wrench_reach_one_owned_sdk_transaction():
    (control, stamp, state, tr, w, source, writes) = control_fixture()
    finger = 0.2 + torch.tensor([0.2, -0.2])[:, None] * ACTION_CONTRACT['joint_rate_rad_s'] / 480
    for expected in tr['expected']:
        torch.testing.assert_close(expected[:, :6], finger.expand(-1, 6).float(), atol=1e-07, rtol=0)
        for (lo, rate) in ((6, 'translation_rate_m_s'), (9, 'rotation_rate_rad_s')):
            delta = torch.tensor([0.2, -0.2])[:, None].expand(-1, 3) * ACTION_CONTRACT[rate] / 480
            torch.testing.assert_close(expected[:, lo:lo + 3], delta, atol=1e-09, rtol=1e-05)
    state['link_pose_xyzw'][..., 0] = 48.125
    state['body_com_pose_w'][..., 0] = 48.13500000023
    w[:, :, 1] = 2.0
    w[:, :, 5] = 0.125
    control.submit(tr, stamp=stamp, wrenches=w, state=state, reaction_source=source)
    issued = [row for row in writes if isinstance(row, dict)]
    for row in issued:
        torch.testing.assert_close(row['torques'][:, :, 2], torch.full((2, 1), 0.10499999954).float(), rtol=0, atol=0)
    assert writes.count('write') == 1 and len(issued) == 2
    for s in (0, 1):
        state['backend_targets'][:, s, control.write_ids[s]] = tr['expected'][s]
    after = replace(stamp, sdk_step=3, rod_ticks=(12, 12), solver_time_s=4 + 1 / 480)
    assert not control.finish(tr, state, stamp=after).any()
    with pytest.raises(ValueError):
        control.submit(tr, stamp=stamp, wrenches=w, state=state, reaction_source=source)

@pytest.mark.parametrize('fault', ['source', 'buffer', 'force_limit', 'nonfinite_com'])
def test_reaction_faults_are_rejected_before_any_sdk_write(fault):
    (control, stamp, state, tr, w, source, writes) = control_fixture()
    if fault == 'source':
        source['reset_ids'] = (2, 1)
    elif fault == 'buffer':
        source['mesh_wrenches'] = w.clone()
    elif fault == 'force_limit':
        w[0, 0, 0] = 30 + 1e-07
    else:
        state['body_com_pose_w'][0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        control.submit(tr, stamp=stamp, wrenches=w, state=state, reaction_source=source)
    assert not writes
