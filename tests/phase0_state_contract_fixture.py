"""Explicit CPU synthetic fixtures; no simulation, hardware, training, or new data.

TensorTask.__new__ supplies state only; all observations use production methods.
AST extraction executes unchanged source nodes, avoiding unavailable training-only
tensordict/rsl_rl imports. TensorDictFixture tests routing, not those libraries.
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / 'src/go2_pvcnn/control/dh116'


def cpu_task():
    from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask
    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    from go2_pvcnn.control.dh116.tabletop_surface_feedback import SurfaceFeedback
    t = TensorTask.__new__(TensorTask)
    t.n = 2
    t.device = torch.device('cpu')
    def values(*tail, offset=0., scale=.001):
        count = math.prod((t.n, *tail))
        return (torch.arange(count, dtype=torch.float64).reshape(t.n, *tail) + 1.) * scale + offset
    t.low, t.high = [torch.tensor(list(x) * 2, dtype=torch.float64) for x in active_limits()]
    t.parameters = SimpleNamespace(max_force_n=12.)
    t.m = dict(q=(t.low + .4 * (t.high-t.low))[None].repeat(t.n, 1),
               qd=values(12), forces=values(4), clearance=values(offset=.003),
               object_relative=values(3), object_quat=values(4), object_velocity=values(6),
               mimic_errors=values(10, scale=.0001), adaptive_palm_offset=values(2, 3),
               grip_stiffness=values(2, offset=10.), bottom=values(offset=.36),
               pad_relative=values(4, 3), grouped_object_loads=values(2, 6, offset=.1),
               hand_object_loads=values(2, offset=1.),
               wrist_pose=values(2, 7), wrist_velocity=values(2, 6), wrist_joint_angles=values(2, 3),
               contact_valid=torch.ones(t.n, 2, dtype=torch.bool),
               contact_all_below_centerline=torch.ones(t.n, 2, dtype=torch.bool),
               contact_center_relative_m=values(2, 3),
               effective_pad_loads_n=values(2, 5, offset=.1),
               effective_pad_vectors_w=values(2, 5, 3),
               effective_pad_valid=torch.ones(t.n, 2, 5, dtype=torch.bool),
               rejected_contact_loads_n=values(2), pad_reference_points_w=values(2, 5, 3))
    t.m['object_quat'][:, 0] = 1.
    t.m['wrist_pose'][..., 3] = 1.
    # Include two opposed contacts so allocation branches have nonzero evidence.
    t.m['effective_pad_vectors_w'][..., 1, :] *= -1.
    t.reference = t.m['q'] - .01
    t.previous = t.m['q'] + .01
    t.command = t.m['q'] + .02
    t.filtered_residual = values(12)
    t.elapsed = values(offset=.2)
    t.contact_time = values(offset=.05)
    t.success_time = values(offset=.1)
    t.offset = values()
    t.close_progress = values()
    t.phase = torch.full((t.n,), 3, dtype=torch.long)
    t.descent_stop = 1
    t.grip = values(3)
    t.grip_valid = torch.tensor([True, False])
    for name in ('offset', 'filtered', 'target', 'boost', 'requested'):
        setattr(t, 'f_' + name, values(2))
    t.f_last_slip = values()
    t.free_grasp_lift = t.free_wrist_grasp = t.surface_force_sharing = t.force_region = True
    t.contact_reward_window = 'close_start'
    t.force_reward_start = 'downward_after_close'
    t.contact_reward_clock = values()
    t.grasp_ready_awarded = torch.tensor([False, True])
    t.force_scoring_started = torch.ones(t.n, dtype=torch.bool)
    t.shaping = None
    t.actions = values(24)
    t.rotation_targets = values(2, 3)
    t.free_wrist_world = SimpleNamespace(root_seed=values(2, 7), wrist_angles=values(2, 3), palms=values(2, 3))
    t.surface_feedback = SurfaceFeedback(t.n, 'cpu')
    # Diagnostic fixture publication only: no claim that begin/sample happened.
    t.surface_feedback._features[:] = values(20)
    t.surface_feedback._valid[:] = True
    t.success = torch.zeros(t.n, dtype=torch.bool)
    t.failure = torch.zeros(t.n, dtype=torch.long)
    t.pressure_limited = torch.zeros(t.n, 2, dtype=torch.bool)
    return t


def observation_context(t):
    from go2_pvcnn.control.dh116.tabletop_policy_force_region import observe as force_observe
    from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import observe as wrist_observe
    from go2_pvcnn.control.dh116.tabletop_dexhands_feedback import geometry_features
    from go2_pvcnn.control.dh116.tabletop_surface_exploration import allocation, observation as surface_observe
    return dict(t=t, self=t, m=t.m, maximum=t.parameters.max_force_n,
                joints=lambda v: 2*(v-t.low)/(t.high-t.low)-1,
                stop=torch.nn.functional.one_hot(torch.full_like(t.phase, t.descent_stop), 3),
                force_observe=force_observe, wrist_observe=wrist_observe,
                geometry_features=geometry_features, surface_observe=surface_observe,
                a=allocation(t), math=math)


def sensor_types():
    class TensorDictFixture(dict):
        def __init__(self, data, batch_size, device):
            super().__init__(data)
            self.batch_size = batch_size
            self.device = device
    ns = dict(torch=torch, TensorDict=TensorDictFixture)
    tree = ast.parse((MODULES / 'tabletop_vision_student.py').read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(x, ast.Name) and x.id in {'POINTS', 'HISTORY', 'PROPRIO_DIM', 'FRAME_DIM', 'POLICY_DIM'}
                     for x in node.targets)]
    nodes += [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SensorHistory']
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MODULES / 'tabletop_vision_student.py'), 'exec'), ns)
    source = MODULES / 'tabletop_vision_training.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_observations')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
    return ns['SensorHistory'], ns['_observations'], TensorDictFixture
