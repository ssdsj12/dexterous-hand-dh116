"""One shared physical stepping/reset/sensor runtime with no policy collector."""
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import is_dataclass
from dataclasses import replace
import math
import os
from time import perf_counter
from types import MappingProxyType
import uuid
import numpy as np
import torch
from dh116.hand.kinematics import defor_wrist_proje__CLEARANCE_M as CLEARANCE_M
from dh116.hand.kinematics import defor_wrist_proje__SOURCE as SOURCE
from dh116.hand.kinematics import WRIST_ANGULAR
from dh116.hand.kinematics import palm_mount__CONTRACT as WRIST
from dh116.hand.articulation import capture_regex_state
from dh116.hand.articulation import restore_regex_seed
from dh116.env.contracts import validate_runtime_execution
from dh116.hand.safety import TensorControl
from dh116.physics.geometry import _rotation_xyzw
from dh116.physics.geometry import make_tensor_geometry
from dh116.physics.geometry import TensorMeshBridge
from dh116.rl.teacher import FULL_CONTEXT_MODE, FullTaskContext, TensorSourceOwner
from dh116.rl.reward import TensorLaneSchema
from dh116.rl.reward import TensorRewardFrame
from dh116.rl.reward import TensorTrainingReward
from dh116.rl.reward import measured_finger_errors
from dh116.rl.reward import train_rewar__ERROR_FIELDS as ERROR_FIELDS
from dh116.rl.reward import GOAL_SHAPING_FIELDS, GOAL_SOFT_COSTS, GOAL_TERMINATIONS, load_config, train_rewar__GOAL_FULL
SIDES = ('left', 'right')
_MEASURED_FIELDS = ('q', 'qd', 'backend_targets', 'root_pose_wxyz', 'root_pose_xyzw', 'root_velocity', 'link_pose_wxyz', 'link_pose_xyzw', 'body_com_velocity_w', 'body_com_pose_w', 'base_link_pose_w', 'base_link_com_velocity_w')

def pinch_geometry_inputs(geometry):
    """Select Pinch SDF roles independently of the complete reaction-mesh inventory."""
    result = dict(geometry)
    result['required_pads'] = ('finger14_link', 'finger23_link')
    return result

def probe_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if is_dataclass(value):
        return {f.name: probe_copy(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: probe_copy(v) for (k, v) in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)((probe_copy(v) for v in value))
    return value


def rod_local_wrists(nodes, palms, stations):
    """Fixed material stations with tangent-Y and gravity-referenced lateral axes."""
    index = stations.clamp(1, nodes.shape[1] - 2)
    row = torch.arange(len(nodes), device=nodes.device)[:, None]
    tangent = nodes[row, index + 1] - nodes[row, index - 1]
    tangent = tangent / tangent.norm(dim=-1, keepdim=True)
    up = tangent.new_tensor([0, 0, 1]).expand_as(tangent)
    alternate = tangent.new_tensor([1, 0, 0]).expand_as(tangent)
    up = torch.where(tangent[..., 2:].abs() > 0.99, alternate, up)
    lateral = torch.linalg.cross(tangent, up)
    lateral = lateral / lateral.norm(dim=-1, keepdim=True)
    vertical = torch.linalg.cross(lateral, tangent)
    basis = torch.stack((lateral, tangent, vertical), dim=-1)
    rel = palms[..., :3] - nodes[row, stations]
    rotation = _rotation_xyzw(palms[..., [0, 1, 2, 4, 5, 6, 3]])
    return (rel[..., None, :] @ basis).squeeze(-2), basis.transpose(-1, -2) @ rotation


class GraspContactReadout:
    """Reduce a reconciled DeformX packet at the same rod/SDK boundary."""

    SUPPORTS = ('table', 'support_0', 'support_1')
    PADS = ('finger14_link', 'finger23_link', 'finger33_link', 'finger43_link', 'finger53_link')
    MIN_PAD_LOAD_N = 0.05
    MIN_HAND_LOAD_N = 0.3
    MAX_OPPOSITION_COS = -0.25
    SUPPORT_LOAD_LIMIT_N = 1e-6

    def __init__(self, grasp, mesh_names, *, device):
        self.grasp = grasp
        self.device = torch.device(device)
        self.mesh_names = tuple(mesh_names)
        if len(self.mesh_names) != len(set(self.mesh_names)):
            raise ValueError('unique contact mesh names required')
        fingers = {'pinch': self.PADS[:2], 'tripod': self.PADS[:3],
                   'power': (*self.PADS, 'base_link')}[grasp]
        self.required = tuple((side, fingers) for side in SIDES)
        self.pad_slots = tuple(
            tuple(self.mesh_names.index(f'{side}.{name}') for name in pads)
            for side, pads in self.required
        )
        self.support_slots = tuple(self.mesh_names.index(name) for name in self.SUPPORTS)
        self.hand_slots = self.pad_slots[0] + self.pad_slots[1]
        self.non_target_slots = tuple(i for i, name in enumerate(self.mesh_names)
                                      if name.startswith(('left.', 'right.')) and i not in self.hand_slots)
        self.required_slots = torch.zeros((2, len(self.mesh_names)), device=self.device, dtype=torch.bool)
        for i, slots in enumerate(self.pad_slots):
            self.required_slots[i, list(slots)] = True
        # The actual native rod runs from -Y to +Y, with the left hand at -Y.
        self.side_by_mesh = torch.tensor(
            [-1 if name.startswith('left.') else 1 if name.startswith('right.') else 0
             for name in self.mesh_names], device=self.device,
        )

    def _opposed(self, normals, available, fraction):
        cosine = normals @ normals.transpose(-1, -2)
        opposed = (cosine <= self.MAX_OPPOSITION_COS) & available[..., :, None] & available[..., None, :]
        angular = ((0.5 - cosine) / (0.5 - self.MAX_OPPOSITION_COS)).clamp(0, 1)
        if self.grasp == 'tripod':
            thumb, index, middle = normals.unbind(-2)
            axis = thumb - (index + middle) * 0.5
            length = axis.norm(dim=-1, keepdim=True)
            face = (normals * (axis / length.clamp_min(1e-12)).unsqueeze(-2)).sum(-1)
            axis_valid = length.squeeze(-1) > 1e-12
            hard = (axis_valid & opposed[..., 0, 1] & opposed[..., 0, 2]
                    & (face[..., 0] >= 0.2) & (face[..., 1:] <= -0.2).all(-1))
            face_margin = torch.minimum(face[..., 0], -face[..., 1:].amax(-1))
            angular_pair = angular[..., 0, 1:].amin(-1)
            quality = torch.minimum(angular_pair, ((face_margin + 0.2) / 0.4).clamp(0, 1))
            return hard, quality * fraction.amin(-1) * axis_valid
        if self.grasp == 'power':
            hard = opposed[..., :5, :5].any(dim=(-1, -2)) & opposed[..., 5, :5].any(dim=-1)
            pair_load = torch.minimum(fraction[..., :, None], fraction[..., None, :])
            score = angular * pair_load
            fingers = score[..., :5, :5].amax(dim=(-1, -2))
            palm = score[..., 5, :5].amax(-1)
            return hard, torch.minimum((palm + fingers) * 0.5, fraction[..., 5])
        return opposed[..., 0, 1], angular[..., 0, 1] * fraction.amin(-1)

    def __call__(self, packet, nodes):
        # Each pair has already been checked against the owned solver reductions,
        # and the complete per-lane pair capacity and clock have been verified.
        env = packet['env_ids']
        step, node, mesh = packet['indices'].unbind(-1)
        vectors = packet['vectors']
        normal_force = packet['normal_force'].clamp_min(0)
        batch = nodes.shape[0]
        meshes = len(self.mesh_names)
        row = (env * 12 + step) * meshes + mesh
        loads = torch.zeros(batch * 12 * meshes, device=self.device, dtype=torch.float64)
        loads.index_add_(0, row, normal_force)
        weighted_normal = loads.new_zeros((loads.numel(), 3))
        weighted_normal.index_add_(0, row, vectors[:, 3:6] * normal_force[:, None])
        force_magnitude = loads.new_zeros(loads.shape)
        force_magnitude.index_add_(0, row, vectors[:, 6:9].norm(dim=-1))
        loads = loads.view(batch, 12, meshes)
        weighted_normal = weighted_normal.view(batch, 12, meshes, 3)
        support_loads = force_magnitude.view(batch, 12, meshes)[:, :, self.support_slots]
        # Signed world-Z force on the rod, from the same reconciled contact pairs.
        vertical = loads.new_zeros(batch * 12 * meshes)
        vertical.index_add_(0, row, vectors[:, 8])
        vertical = vertical.view(batch, 12, meshes)

        y = nodes[..., 1]
        center = (y[:, 0] + y[:, -1]) * 0.5
        material_orientation = y[:, -1] > y[:, 0]
        side_sign = self.side_by_mesh[mesh]
        # Power's original audit checks all hand links; the distal audits check
        # only the required fingers for wrong-side material contacts.
        if self.grasp != 'power':
            side_sign = torch.where(self.required_slots.any(0)[mesh], side_sign, 0)
        wrong = (side_sign != 0) & (((node - 20) * side_sign <= 0) |
                                    ((y[env, node] - center[env]) * side_sign <= 0))
        wrong_count = torch.zeros(batch, device=self.device, dtype=torch.long)
        wrong_count.index_add_(0, env, wrong.long())
        correct_side = (wrong_count == 0) & material_orientation

        result = {}
        effective = []
        contact_nodes = []
        for side, slots, required_slot in zip(SIDES, self.pad_slots, self.required_slots, strict=True):
            pad_loads = loads[:, :, slots]
            pad_normals = weighted_normal[:, :, slots]
            lengths = pad_normals.norm(dim=-1)
            directions = pad_normals / lengths.clamp_min(1e-12)[..., None]
            loaded = pad_loads > self.MIN_PAD_LOAD_N
            available = lengths > 1e-12
            fraction = (lengths / self.MIN_PAD_LOAD_N).clamp(0, 1).masked_fill(~available, 0)
            opposed_steps, opposition_progress = self._opposed(directions, available, fraction)
            opposed = opposed_steps.all(1) & correct_side
            effective.append(opposed & loaded.all(dim=(1, 2)) &
                             (pad_loads.sum(-1) > self.MIN_HAND_LOAD_N).all(1))
            result['coverage_' + side] = loaded.all(1).double().mean(-1) * correct_side
            pad_fraction = (pad_loads / self.MIN_PAD_LOAD_N).clamp(0, 1)
            weakest_pad = pad_fraction.amin(dim=-1).mean(1)
            pad_progress = 0.5 * (pad_fraction.mean(dim=(1, 2)) + weakest_pad)
            if self.grasp == 'power':
                finger_progress = (pad_loads[..., :5].sum(-1) / self.MIN_HAND_LOAD_N).clamp(0, 1)
                hand_progress = (finger_progress + pad_fraction[..., 5]).mean(1) * 0.5
            else:
                hand_progress = (pad_loads.sum(-1) / self.MIN_HAND_LOAD_N).clamp(0, 1).mean(1)
            load_progress = 0.5 * (pad_progress + hand_progress)
            result['load_progress_' + side] = 0.5 * (load_progress + opposition_progress.mean(1)) * correct_side
            result['opposed_' + side] = opposed
            node_load = loads.new_zeros(batch * 41)
            node_load.index_add_(0, env * 41 + node, normal_force * required_slot[mesh])
            contact_nodes.append(node_load.view(batch, 41).argmax(1))
        result['contact_nodes'] = torch.stack(contact_nodes, dim=1)
        result['bilateral_support'] = effective[0] & effective[1]
        result['off_table'] = (support_loads[:, :, 0] <= self.SUPPORT_LOAD_LIMIT_N).all(1)
        result['no_external_support'] = (support_loads <= self.SUPPORT_LOAD_LIMIT_N).all(dim=(1, 2))
        result['support_load_n'] = support_loads.amax(1)
        result['support_vertical_n'] = vertical[:, :, self.support_slots].clamp_min(0).sum(-1).amax(1)
        result['hand_vertical_n'] = vertical[:, :, self.hand_slots].sum(-1).amin(1) * correct_side
        left_up, right_up = (vertical[:, :, slots].sum(-1) for slots in self.pad_slots)
        result['balanced_hand_vertical_n'] = 2 * torch.minimum(left_up, right_up).amin(1) * correct_side
        result['non_target_normal_n'] = (loads[:, :, self.non_target_slots].amax((1, 2))
                                         if self.non_target_slots else loads.new_zeros(batch))
        return result


class FixedGraspAnchors:
    """Select loaded material nodes once, then read their motion in each palm frame."""

    def __init__(self, batch, device):
        self.present = torch.zeros(batch, dtype=torch.bool, device=device)
        self.ids = torch.zeros(batch, 2, dtype=torch.long, device=device)
        self.ticks = torch.full((batch, 4), -1, dtype=torch.long, device=device)
        self.reference = torch.full((batch, 4, 3), torch.nan, dtype=torch.float64, device=device)
        self.reference_tangent = torch.zeros_like(self.reference[:, :2])
        self.rows = torch.arange(batch, device=device)[:, None]
        self.station = torch.full((batch, 2), torch.nan, dtype=torch.float64, device=device)

    def reset(self, ids):
        rows = list(ids)
        self.present[rows] = False
        self.ids[rows] = 0
        self.ticks[rows] = -1
        self.reference[rows] = torch.nan
        self.station[rows] = torch.nan

    @staticmethod
    def _to_palm(vectors, palms):
        w = palms[..., 3:4]
        inverse_xyz = -palms[..., 4:]
        cross = 2 * torch.linalg.cross(inverse_xyz, vectors)
        return vectors + w * cross + torch.linalg.cross(inverse_xyz, cross)

    def capture(self, tick, nodes, palms, contact, contact_nodes, rest_lengths, rest_total, *, validate=True):
        nodes, palms = nodes.double(), palms.double()
        acquire = contact & ~self.present
        chosen = torch.where(acquire[:, None], contact_nodes, self.ids)
        pair = nodes[self.rows, chosen]
        lo = nodes[self.rows, (chosen - 1).clamp_min(0)]
        hi = nodes[self.rows, (chosen + 1).clamp_max(40)]
        tangent = self._to_palm(hi - lo, palms)
        length = tangent.norm(dim=-1)
        self.invalid_tangent = contact & (length <= 1e-12).any(1)
        if validate:
            require_substep_guard(self.invalid_tangent, 'contact material tangent is undefined')
        relative = self._to_palm(pair - palms[..., :3], palms)
        ref = relative.repeat_interleave(2, dim=1)
        self.reference = torch.where(acquire[:, None, None], ref, self.reference)
        self.reference_tangent = torch.where(
            acquire[:, None, None], tangent / length.clamp_min(1e-12)[..., None], self.reference_tangent,
        )
        self.ticks = torch.where(acquire[:, None], tick, self.ticks)
        self.present |= acquire
        self.ids = chosen
        # The source validates native rest lengths. Rescaling the
        # prefix at the final node preserves the authoritative total length.
        stations = torch.nn.functional.pad(rest_lengths.double().cumsum(1), (1, 0))
        stations = stations / stations[:, -1:] * rest_total[:, None]
        self.station = torch.where(acquire[:, None], stations[self.rows, chosen], self.station)
        delta = relative - self.reference[:, ::2]
        material = delta.norm(dim=-1)
        tangential = (delta * self.reference_tangent).sum(-1).abs()
        drift = torch.stack((material, tangential), -1).flatten(1)
        # Bound the FP32 readback's coordinate rounding before evaluating the
        # original slip limit. This is numerical uncertainty in simulation.
        magnitudes = torch.maximum(
            pair.abs().amax(-1), palms[..., :3].abs().amax(-1),
        ).clamp_min(1.0)
        uncertainty = (8 * torch.finfo(torch.float32).eps * magnitudes).repeat_interleave(2, dim=1)
        return (
            self.present.clone(), drift, uncertainty, self.reference.clone(), self.ticks.clone(),
        )

class RawTableClearance:
    """Full original vertex sets and raw SDK quaternion before each rod step."""

    def __init__(self, prepared, body_names, device):
        self.shapes = [(i, body_names[i].index(name), row['vertices']) for (i, side) in enumerate(SIDES) for (name, row) in prepared['hands'][side]['source_colliders'].items()]
        if len(self.shapes) != 34:
            raise ValueError('all 34 source colliders required')
        self.sides = []
        for i in range(2):
            (bodies, vertices) = zip(*[(j, v) for (side, j, v) in self.shapes if side == i])
            self.sides.append((torch.as_tensor(np.repeat(bodies, [len(v) for v in vertices]), device=device), torch.as_tensor(np.concatenate(vertices), dtype=torch.float64, device=device)))

    def __call__(self, poses, *, isolate=False):
        clear = []
        for (side, (links, vertices)) in enumerate(self.sides):
            p = poses[:, side].double()
            (w, x, y, z) = p[..., 3:].unbind(-1)
            row = torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)
            heights = (row[:, links] * vertices).sum(-1) + p[:, links, 2]
            clear.append(heights.amin(-1) - 0.3)
        clear = torch.stack(clear, -1)
        if bool((~torch.isfinite(clear)).any()) or (not isolate and bool((clear < CLEARANCE_M).any())):
            raise ValueError('original pre-rod raw source table clearance guard failed')
        return clear

def mesh_reset(solver, bridge, state, ids, *, before=None, verify=True):
    """Cold native Reset ABI, with all three actual phase and velocity readbacks."""
    (positions, quaternions) = bridge.targets(state['link_pose_xyzw'])
    idx = list(ids)
    p = positions[idx].cpu().numpy()
    q = quaternions[idx].cpu().numpy()
    clock = float(solver.time)
    solver.reset_env_mesh_poses(np.asarray(idx, dtype=np.int32), p, q)
    if float(solver.time) != clock or solver.last_error:
        raise ValueError('mesh reset advanced native time or failed')
    if not verify:
        return
    after = solver.mesh_state_snapshot()
    for phase in ('current', 'target', 'solver'):
        np.testing.assert_array_equal(after[phase + '_positions_world_m'][idx], p)
        np.testing.assert_allclose(after[phase + '_quaternions_xyzw'][idx], q, atol=2e-06, rtol=0)
    for field in ('solver_linear_velocity_world_m_s', 'solver_angular_velocity_world_rad_s', 'motion_delta_world_m', 'motion_rotation_world_rad', 'pending_teleport'):
        if np.any(after[field][idx]):
            raise ValueError('selected mesh Reset retained motion')
    if before is not None:
        peers = sorted(set(range(solver.num_envs)) - set(ids))
        for (key, value) in before.items():
            if isinstance(value, np.ndarray) or key == 'pair_counts':
                (a, b) = (np.asarray(value)[peers], np.asarray(after[key])[peers])
                if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                    raise ValueError('peer native mesh Reset readback changed: ' + key)

def require_substep_guard(failures, label):
    if bool(failures.any()):
        raise ValueError(label + ' at lanes ' + str(failures.nonzero().flatten().tolist()))

def _owned(value):
    """Copy capture payloads on their current device; reject opaque handles."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _owned(item) for (key, item) in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)((_owned(item) for item in value))
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError('capture payload must contain owned tensors and plain data')

def _versions(value):
    if isinstance(value, torch.Tensor):
        return (id(value), value._version)
    if isinstance(value, Mapping):
        return tuple(((key, _versions(item)) for (key, item) in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple((_versions(item) for item in value))
    return None

@dataclass(frozen=True)
class MeasuredPhysicsFrame:
    """Owned measured labels, separate from any actor input or reference packet."""
    phase: str
    stamp: object
    state: Mapping
    rod_positions_w: torch.Tensor
    rod_velocities_w: torch.Tensor
    clearance_m: torch.Tensor
    actions: torch.Tensor | None
    token: str | None
    policy_uids: tuple

class NativeTensorPhysicsRuntime:
    """Original physical chain and reward guards, with no policy/model/trainer."""

    def __init__(self, project, sim, batch, prepared, archive, seed, initial, bindings, runtime, profile,
                 *, episode_windows=32, wall_cap_s=180.0, full_grasp=False):
        self._configure_physics(
            project, sim, batch, prepared, archive, seed, initial, bindings, runtime, profile,
            episode_windows=episode_windows, wall_cap_s=wall_cap_s, full_grasp=full_grasp,
        )
        self._initialize_episode()

    def _configure_physics(self, project, sim, batch, prepared, archive, seed, initial, bindings, runtime, profile,
                           *, episode_windows=32, wall_cap_s=180.0, full_grasp=False):
        validate_runtime_execution()
        if type(episode_windows) is not int or episode_windows < 1:
            raise ValueError('positive independent episode window budget required')
        self.episode_windows = episode_windows
        (self.sim, self.batch, self.runtime) = (sim, batch, runtime)
        (self.initial, self.state) = (initial, initial)
        self.device = runtime.device
        (self.style, self.n) = (archive.style, runtime.solver.num_envs)
        self.ids = tuple(range(self.n))
        self.origins = initial['env_origins'] + initial['env_origins'].new_tensor(seed['origin'])
        self.config = load_config(project / f'configs/training/reward_{self.style}.json',
                                  ppo_gamma=0.995, full_grasp=full_grasp)
        self.full_grasp = self.config.profile == train_rewar__GOAL_FULL
        if self.full_grasp and self.episode_windows <= 300:
            raise ValueError('complete grasp episode must allow approach before the 300-window hold')
        source_options = {'context_mode': FULL_CONTEXT_MODE} if self.full_grasp else {}
        self.source = TensorSourceOwner(
            runtime, sim, profile, self.origins, self.device, self.style,
            self.episode_windows / 60, wall_cap_s, rod_uids=tuple(f'rod:{i}' for i in self.ids),
            **source_options,
        )
        self.joints = tuple((tuple(batch.sides[s].joint_names) for s in SIDES))
        names = tuple((tuple(batch.sides[s].body_names) for s in SIDES))
        for (side, body_names) in zip(SIDES, names, strict=True):
            owner = batch.sides[side]
            paths = owner.hand.root_physx_view.link_paths
            for (prefix, idx) in zip(batch.prefixes, owner.row_indices.tolist(), strict=True):
                if tuple(paths[idx]) != tuple((prefix + '/' + side + 'Hand/' + name for name in body_names)):
                    raise ValueError('actual link identity differs from original source geometry')
        self.reset_proofs = []
        self.prove_peer_reset = False
        self.probe_dir = None
        self.bindings = bindings
        self._views = tuple((batch.sides[s].hand.root_physx_view for s in SIDES))
        roots = initial['root_seed_w'].new_tensor(seed['reset_root_pose_w'])[None].repeat(self.n, 1, 1)
        roots[..., :3] += initial['env_origins'][:, None]
        self.ctrl = TensorControl(batch, roots, bindings, runtime.mesh_names,
                                  isolate_constraints=self.full_grasp, soft_constraints=self.full_grasp)
        runtime.strict_audit = not self.full_grasp
        if self.full_grasp:
            layout = batch.scenes[0]['layout']
            self.table_centers = roots.new_tensor(np.stack(
                [scene['layout']['table_center'] for scene in batch.scenes]
            ))
            self.table_half_size = roots.new_tensor(layout['table_size']) * 0.5
            self._hand_mesh_slots = self.ctrl.slots[0] + self.ctrl.slots[1]
            self.ctrl.fixed_fingers = tuple(i + s * 6 for s in (0, 1) for i in range(6)
                                            if i not in self.config.finger_goal.participating)
        self.effort = float(prepared['candidate_physics']['hand']['active_effort_limit_nm'])
        self.raw_clearance = RawTableClearance(prepared, names, self.device)
        self.bridge = TensorMeshBridge(runtime.mesh_names, bindings, initial['mesh_positions_world_m'], initial['mesh_quaternions_xyzw'])
        mesh_reset(runtime.solver, self.bridge, initial, self.ids)
        parameters = runtime.solver.parameter_snapshot()
        self.radius = float(parameters['contact_radius_m'])
        self.radii = torch.full((self.n,), self.radius, dtype=torch.float64, device=self.device)
        self.geometry = None
        self.support_geometry = None
        if self.full_grasp or self.style == 'power':
            from dh116.env.scene import load_anatomical_reference

            if self.full_grasp:
                reference_sha = self.config.finger_goal.reference_sha256
            else:
                reference_sha = archive.task['near_open_source_sha256']
            if self.style == 'power' and reference_sha != archive.task['near_open_source_sha256']:
                raise ValueError('Power wrist and archived scene reference SHA differ')
            reference = load_anatomical_reference(self.style, reference_sha, root=project)
            reference_poses = roots.new_tensor(reference['root_pose_wxyz'])
            reference_stations = roots.new_tensor(reference['station_nodes'], dtype=torch.long)
        geometry_backend = os.environ.get('DH116_TENSOR_GEOMETRY_BACKEND')
        if geometry_backend is None and self.n == 512 and self.device.type == 'cuda':
            geometry_backend = os.environ.setdefault('DH116_TENSOR_GEOMETRY_BACKEND', 'cuda_cabi')
        if self.style in ('pinch', 'tripod'):
            original = pinch_geometry_inputs(archive.scene_inputs['geometry'])
            if self.style == 'tripod':
                original['required_pads'] = GraspContactReadout.PADS[:3]
            expected_radius = float(archive.mesh_plan['cfg']['rod']['radius_m'])
            if not math.isclose(self.radius, expected_radius, abs_tol=1e-09, rel_tol=2e-06):
                raise ValueError('locked pad geometry material radius differs')
            for side in SIDES:
                expected_routes = 5 if len(runtime.mesh_names) == 11 and self.style == 'pinch' else 7
                if len(self.ctrl.slots[SIDES.index(side)]) != expected_routes:
                    raise ValueError('original named hand reaction route layout required')
                for (name, locked) in original['sources'][side]['sdf_binding'].items():
                    row = bindings[side + '.' + name]
                    if row['sdf_sha256'] != locked['sha256'] or row['capture_sha256'] != locked['capture_sha256']:
                        raise ValueError('original locked SDF or capture differs')
            self.geometry = make_tensor_geometry(original, names, device=self.device, backend=geometry_backend)
            if self.style == 'tripod' and not self.full_grasp:
                from dh116.physics.geometry import TripodSupportGuard
                if len(runtime.mesh_names) != 17 or archive.task is None:
                    raise ValueError('Tripod needs all 17 original mesh slots and both supports')
                self.support_geometry = TripodSupportGuard(
                    self.geometry, archive.task['support_scene']['supports'], initial['env_origins'],
                )
        elif self.style == 'power':
            if not self.full_grasp:
                self.targets = reference_poses[None, :, :3] + initial['env_origins'][:, None]
                self.stations = reference_stations.expand(self.n, -1)
        else:
            raise ValueError('unknown grasp geometry')
        self.contact = None
        if self.full_grasp:
            if self.style == 'power':
                original = archive.scene_inputs['geometry']
                grids = {s: {} for s in SIDES}
                sources = {s: {'sdf_binding': {}} for s in SIDES}
                pads = (*GraspContactReadout.PADS, 'base_link')
                for entry in archive.mesh_plan['entries']:
                    label = entry['name']
                    if label in bindings:
                        side, link = label.split('.')
                        grids[side][link] = {k: entry[k] for k in ('values', 'origin', 'spacing')}
                        sources[side]['sdf_binding'][link] = {
                            k: bindings[label][k] for k in ('sdf_sha256', 'capture_sha256')
                        }
                self.geometry = make_tensor_geometry(
                    dict(geometry=original['geometry'], grids=grids, sources=sources, required_pads=pads),
                    names, device=self.device, backend=geometry_backend,
                )
            reference_nodes = roots.new_tensor(reference['rod_nodes_w_m'])
            self.wrist_stations = reference_stations.expand(self.n, -1)
            self.wrist_reference = rod_local_wrists(
                reference_nodes[None], reference_poses[None], reference_stations[None],
            )
            if not all(bool(torch.isfinite(x).all()) for x in self.wrist_reference):
                raise ValueError('finite source-bound bilateral wrist reference required')
            if len(runtime.mesh_names) != 17 or archive.task is None:
                raise ValueError('complete grasp reward requires the original supported 17-mesh scene')
            self.contact = GraspContactReadout(self.style, runtime.mesh_names, device=self.device)
            pad_order = tuple((i, name) for i, (_, names) in enumerate(self.contact.required) for name in names)
            if tuple(self.geometry.pad_columns) != pad_order:
                raise ValueError('approach and native contact pad identities differ')
            inverse_mass = np.asarray(parameters['inverse_masses_kg_inv'], dtype=np.float64)
            weight = (1.0 / inverse_mass).sum(1) * -float(parameters['gravity_world_m_s2'][2])
            if weight.shape != (self.n,) or not (np.isfinite(weight) & (weight > 0)).all():
                raise ValueError('finite positive native rod weight required for lift shaping')
            self.rod_weight_n = torch.as_tensor(weight, device=self.device)
            self.anchors = FixedGraspAnchors(self.n, self.device)
            self._anchor_rest_lengths = self.source.rest_lengths_m
            self._anchor_rest_total = self.source.rest_length_m
            self._lift_reached = torch.zeros(self.n, dtype=torch.bool, device=self.device)
            self._hold_position = torch.full((self.n, 41, 3), torch.nan, dtype=torch.float64, device=self.device)

    def _initialize_episode(self):
        self.reward = TensorTrainingReward(self.config, num_envs=self.n, device=self.device,
                                           defer_checks=self.full_grasp)
        self.ages = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.controls = 0
        self._initialize_window_state()
        self._reset_refs(self.ids)

    def _initialize_window_state(self):
        self._physical_failed = False
        self._action_window = None
        self._reset_allowed = ()
        self._settled_boundary = None
        self._capture_hook = None
        self._capture_cadence = 'window'
        self._capture_history_length = 2
        self._capture_history = {i: [] for i in self.ids}
        self._in_capture = False

    def _assert_usable(self):
        if self._physical_failed:
            raise ValueError('failed physical runtime requires a new native scope')
        if self._in_capture:
            self._physical_failed = True
            raise ValueError('capture callback cannot enter a physical transaction')

    def begin_action_window(self, actions, *, token, policy_uids, end):
        """Bind one producer-owned action/token/policy identity to eight SDK steps."""
        try:
            self._assert_usable()
            if self._action_window is not None or self._reset_allowed:
                raise ValueError('prior action window or selected Reset is unfinished')
            if self.controls == 0:
                self.first_sdk = self.runtime.physics_step
            if not isinstance(token, str) or not token.strip():
                raise ValueError('one producer-owned action batch token required')
            self.reward.begin(tuple(self.reward.schemas), request=actions, tokens=(token,) * self.n, policy_uids=policy_uids, physics_tick=self.last.physics_tick, solver_time_s=self.last.solver_time_s, remaining_substeps=end - self.runtime.physics_step)
            self._action_window = dict(actions=actions, action_version=actions._version, token=token, policy_uids=tuple(policy_uids), start_step=self.runtime.physics_step, end=end, recorded=False)
            if self._capture_cadence == 'window':
                self._notify_capture('pre')
        except BaseException:
            self._physical_failed = True
            raise

    def _window(self, *, completed=False):
        self._assert_usable()
        window = self._action_window
        if window is None:
            raise ValueError('owned action window required')
        if window['actions']._version != window['action_version']:
            raise ValueError('owned action window request changed')
        if completed and (self.runtime.physics_step != window['start_step'] + 8 or self.sim.current_time_step_index != self.runtime.physics_step or self.source._boundary().sdk_step != self.runtime.physics_step or (self.ctrl.pending is not None)):
            raise ValueError('eight completed same-source SDK transactions required')
        return window

    def sdk(self, actions, end):
        """The single guarded substep implementation used by all action producers."""
        try:
            window = self._window()
            if actions is not window['actions'] or end != window['end'] or self.runtime.physics_step >= window['start_step'] + 8 or window['recorded']:
                raise ValueError('same producer request and open eight-step window required')
            if self._capture_cadence == 'substep':
                self._notify_capture('pre')
            ready = self._sdk_impl(actions, end)
            if not self.full_grasp and self.runtime.physics_step < window['start_step'] + 8:
                require_substep_guard(ready, 'terminal lane before end of shared SDK window')
            if self._capture_cadence == 'substep':
                self._notify_capture('post')
            return ready
        except BaseException:
            self._physical_failed = True
            raise

    def advance_action_window(self, actions, end):
        """Complete the shared SDK window and validate deferred source checks."""
        try:
            for _ in range(8):
                ready = self.sdk(actions, end)
            if self.full_grasp:
                started = perf_counter()
                self.reward.finalize_observations()
                self.clear = self.raw_clearance(self.state['link_pose_wxyz'], isolate=True)
                self.timings['window_boundary_checks'] = perf_counter() - started
            return ready
        except BaseException:
            self._physical_failed = True
            raise

    def record_window_request(self):
        """Record actual action history before a consumer computes post-state value."""
        try:
            window = self._window(completed=True)
            if window['recorded']:
                raise ValueError('action window request was already recorded')
            self.source.set_previous_request(window['actions'], window['token'])
            window['recorded'] = True
            if self._capture_cadence == 'window':
                self._notify_capture('post')
        except BaseException:
            self._physical_failed = True
            raise

    def complete_action_window(self, *, truncated=None, buffer_full=None):
        """Settle original reward once; authorize only its done rows for Reset."""
        try:
            window = self._window(completed=True)
            if not window['recorded']:
                self.record_window_request()
            self.ages += 1
            timeout = (self.ages >= self.episode_windows) & ~self.reward._terminal
            if truncated is not None:
                timeout |= truncated
            full = None if buffer_full is None else torch.as_tensor(buffer_full, device=self.device, dtype=torch.bool).expand(self.n) & ~timeout & ~self.reward._terminal
            packet = self.reward.settle_once((window['token'],) * self.n, truncated=timeout, buffer_full=full)
            self.controls += 1
            self._action_window = None
            self._reset_allowed = tuple(packet.done.nonzero().flatten().tolist())
            self._settled_boundary = self.source._boundary()
            return packet
        except BaseException:
            self._physical_failed = True
            raise

    def measurement(self, phase='inspection'):
        """Return measured device snapshots; no source references or critic fields."""
        self._assert_usable()
        window = self._action_window
        return MeasuredPhysicsFrame(phase, self.source._boundary(), MappingProxyType(_owned({key: self.state[key] for key in _MEASURED_FIELDS if key in self.state})), _owned(self.nodes), _owned(self.velocity), _owned(self.clear), None if window is None else _owned(window['actions']), None if window is None else window['token'], () if window is None else window['policy_uids'])

    def attach_capture(self, capture, *, cadence='window', history_length=2):
        """Attach an optional trusted observer before the first action window.

        Window cadence is the default. A callback receives no physical handles,
        must return a mapping of all env IDs to plain capture data, and must not
        step, Reset, write physical state or enter runtime transactions.
        """
        self._assert_usable()
        if self.controls or self._action_window is not None or self._capture_hook is not None or (cadence not in ('window', 'substep')) or (type(history_length) is not int) or (history_length < 1) or (not callable(getattr(capture, 'on_measurement', None))) or (not callable(getattr(capture, 'reset_selected', None))):
            raise ValueError('one valid startup capture observer and positive history bound required')
        (self._capture_hook, self._capture_cadence) = (capture, cadence)
        self._capture_history_length = history_length

    def capture_history(self, env_id):
        if type(env_id) is not int or env_id not in self.ids:
            raise ValueError('actual capture environment ID required')
        return tuple((_owned(item) for item in self._capture_history[env_id]))

    def _capture_checkpoint(self):
        pending = self.ctrl.pending
        caches = []
        for owner in self.batch.sides.values():
            data = owner.hand.data
            buffers = {key: value if isinstance(value, torch.Tensor) else getattr(value, 'data', None) for (key, value) in vars(data).items()}
            caches.append((id(owner.hand), id(owner.hand.root_physx_view), id(data), _versions(buffers)))
        return (self.source._boundary(), self.sim.current_time_step_index, self.runtime.physics_step, float(self.runtime.solver.time), id(self.ctrl), self.ctrl.epoch, id(pending), None if pending is None else pending.get('phase'), self.ctrl._failed, self._physical_failed, _versions((self.state, self.nodes, self.velocity, self.clear, self.initial, self.ctrl.palms, self.ctrl.angles)), tuple(caches))

    def _invoke_capture(self, callback, *args):
        before = self._capture_checkpoint()
        self._in_capture = True
        try:
            result = callback(*args)
            if self._capture_checkpoint() != before:
                raise ValueError('capture changed physical source/clock/transaction/tensor state')
            return result
        except BaseException:
            self._physical_failed = True
            raise
        finally:
            self._in_capture = False

    def _notify_capture(self, phase):
        if self._capture_hook is None:
            return
        event = self.measurement(phase)
        result = self._invoke_capture(self._capture_hook.on_measurement, event)
        if not isinstance(result, Mapping) or set(result) != set(self.ids):
            raise ValueError('capture must return one plain payload per actual environment')
        for env_id in self.ids:
            history = self._capture_history[env_id]
            history.append(_owned(result[env_id]))
            del history[:-self._capture_history_length]

    def reset(self, ids):
        """Only an unchanged, settled window may Reset its selected done rows."""
        try:
            self._assert_usable()
            ids = tuple(ids)
            if self._action_window is not None or not ids or len(set(ids)) != len(ids) or any((type(i) is not int or i not in self._reset_allowed for i in ids)) or (self._settled_boundary != self.source._boundary()):
                raise ValueError('selected Reset requires a trusted settled action-window boundary')
            self._reset_impl(ids)
            for env_id in ids:
                self._capture_history[env_id].clear()
            if self._capture_hook is not None:
                self._invoke_capture(self._capture_hook.reset_selected, ids, self.measurement('reset'))
            self._reset_allowed = tuple((i for i in self._reset_allowed if i not in ids))
            self._settled_boundary = self.source._boundary()
        except BaseException:
            self._physical_failed = True
            raise

    def source_guards(self):
        state = self.state
        joint_bad = self.ctrl.joint_failures(state)
        bad = torch.zeros_like(joint_bad)
        for name in ('backend_targets', 'root_pose_wxyz', 'body_com_velocity_w', 'link_pose_xyzw', 'body_com_pose_w'):
            bad |= ~torch.isfinite(state[name]).flatten(1).all(1)
        for (i, s) in enumerate(SIDES):
            owner = self.batch.sides[s]
            hand = owner.hand
            if hand.root_physx_view is not self._views[i] or tuple(hand.joint_names) != self.joints[i]:
                raise ValueError('actual mounted hand view/joint source changed')
            root = hand.data.root_pose_w.index_select(0, owner.row_indices).double()
            target = hand.data.joint_pos_target.index_select(0, owner.row_indices)
            if root.shape != (self.n, 7) or target.shape != state['q'][:, i].shape:
                raise ValueError('incomplete SDK root/target cache')
            bad |= ~torch.isfinite(target).all(1) | ~torch.isfinite(root).all(1)
            bad |= (root[:, 3:].norm(dim=-1) - 1).abs() > 1e-05
            bad |= (root - state['root_pose_wxyz'][:, i].double()).abs().amax(-1) > 1e-06
        bad |= ((state['link_pose_xyzw'][..., 3:].double().norm(dim=-1) - 1.0).abs() > 1e-5).any((1, 2))
        require_substep_guard(bad, 'original mounted17 numerical/source/cache guards')
        if not self.full_grasp:
            require_substep_guard(joint_bad, 'original mounted17 joint guards')
        return joint_bad

    def arm_probe(self, output):
        if self.n != 2:
            raise ValueError('raw parity snapshots are N2 only')
        self.probe_dir = output
        self.snapshot('startup', dict(state=self.state, roots=self.ctrl.roots, source=self.source._boundary(), radius=self.radius, nodes=self.nodes, velocity=self.velocity, reset=self.last, sources=self.reward.schemas, origins=self.origins, write_ids=self.ctrl.write_ids, active_ids=self.ctrl.active, contact_ids=self.ctrl.contact_bodies, mesh_slots=self.ctrl.slots, mesh_names=self.runtime.mesh_names, bindings=self.bindings, shapes=self.raw_clearance.shapes, joint_names=self.joints, body_names=tuple((self.batch.sides[s].body_names for s in SIDES))))

    def snapshot(self, name, row):
        if self.probe_dir is not None:
            with (self.probe_dir / ('tensor_n2_' + name + '.pt')).open('xb') as stream:
                torch.save(probe_copy(row), stream)

    def rod_state(self, *, validate=True):
        solver = self.runtime.solver
        (p, v) = (torch.as_tensor(x) for x in (solver.positions(), solver.velocities()))
        if any((x.device != self.device or x.shape != (self.n, 41, 3) for x in (p, v))):
            raise ValueError('actual same-device full rod position and velocity required')
        if validate:
            require_substep_guard(~(torch.isfinite(p) & torch.isfinite(v)).all((1, 2)), 'nonfinite rod')
        return (p, v)

    def _read_full_rod_state(self):
        nodes, velocity = self.rod_state(validate=False)
        valid = (torch.isfinite(nodes) & torch.isfinite(velocity)).all((1, 2))
        # Keep the last complete finite observation for unusable rows so they
        # never enter geometry or the all-lane Teacher value batch as NaNs.
        # The raw validity mask still rejects a failure on its first live step.
        self.nodes = torch.where(valid[:, None, None], nodes, self.nodes)
        self.velocity = torch.where(valid[:, None, None], velocity, self.velocity)
        return valid

    def distance(self, stamp, *, reset=False, active=None, lane_ids=None):
        if reset or not self.full_grasp:
            self.nodes, self.velocity = self.rod_state()
            if self.full_grasp:
                # The native solver may reuse its readback buffers on the next step.
                self.nodes, self.velocity = self.nodes.clone(), self.velocity.clone()
        subset = slice(None) if lane_ids is None else lane_ids
        p = self.nodes[subset]
        poses = self.state['link_pose_xyzw'][subset]
        self._geometry_bad = torch.zeros(len(p), dtype=torch.bool, device=self.device)
        if self.full_grasp:
            radii = self.radii[subset]
            gaps, valid, normals, proximity = self.geometry.policy_pad_gaps(poses, p, radii, approach_normals=True)
            nearest = gaps.masked_fill(~valid, float('inf')).amin(-1)
            min_gap = nearest.amin(-1)
            if not reset:
                self._pad_backface = min_gap < -0.003
                self._soft_pad_depth = ((-0.0008 - min_gap) / 0.0022).clamp(0, 1)
            else:
                require_substep_guard(min_gap < -0.003, 'Reset pad is behind its native SDF surface')
            # Reward facing pads before they load. Only native forces can qualify contact.
            normals = normals.reshape(len(p), 2, -1, 3)
            proximity = proximity.reshape(len(p), 2, -1)
            _, alignment = self.contact._opposed(normals, proximity > 0, proximity)
            balanced = (alignment.sum(-1) + 2 * alignment.amin(-1)) * 0.25
            distance = gaps.amin(-1).clamp_min(0).mean(-1)
            return distance.double() + radii * (1.0 - balanced.double())
        power_distance = None
        if self.style == 'power':
            rows = torch.arange(len(p), device=self.device)[:, None]
            stations = self.stations[subset]
            initial = self.runtime._initial[subset]
            target = self.targets[subset] + p[rows, stations].double() - initial[rows, stations].double()
            power_distance = (self.state['base_link_pose_w'][subset, ..., :3].double() - target).norm(dim=-1).mean(-1)
            if self.geometry is None:
                return power_distance
        geometry = self.geometry.capture(poses, p, frame=stamp, radius=self.radii[subset], active=active)
        accepted = geometry.source_complete if reset else geometry.geometry_passed
        if power_distance is not None:
            # Closed Power may have pads outside their contact lattice. Use
            # actual in-lattice depths and retain the convex/table guards.
            accepted = ~(geometry.non_target_failure.any(1) | geometry.table_failure.any(1) |
                         geometry.inter_hand_failure)
        self._geometry_bad = ~accepted
        if self.support_geometry is not None:
            live = active & ~self._geometry_bad if active is not None and not reset else active
            self._geometry_bad |= self.support_geometry.capture(
                poses, p, frame=stamp, measurement=geometry, active=live, lane_ids=lane_ids,
            )
        require_substep_guard(self._geometry_bad, 'original source geometry/SDF or Tripod clearance')
        if power_distance is not None:
            return power_distance
        return geometry.positive_pad_mean_m.mean(-1)

    def frame(self, stamp, distance, *, capture=None, constraint=None, suppress_contact=None,
              hard_causes=None, soft_costs=None, numerical_ok=None):
        q = torch.stack([self.state['q'][:, s, self.ctrl.active[s]] for s in (0, 1)], dim=1)
        errors = measured_finger_errors(q, self.config.finger_goal)
        healthy = torch.full_like(distance, self.runtime.solver.last_error == 0, dtype=torch.bool)
        if numerical_ok is not None:
            healthy &= numerical_ok
        if self.full_grasp:
            hard = hard_causes or dict.fromkeys(GOAL_TERMINATIONS, torch.zeros_like(healthy))
            soft = dict.fromkeys(GOAL_SOFT_COSTS, torch.zeros_like(distance))
            if soft_costs is not None:
                soft.update({name: value.to(distance.dtype) for name, value in soft_costs.items()})
            fields = dict(distance=distance, numerical_health=healthy,
                          constraint_failure=torch.stack(tuple(hard.values())).any(0), **hard, **soft)
        else:
            guard = self.ctrl.joint_failures(self.state) if constraint is None else constraint
            fields = dict(distance=distance, numerical_health=healthy, constraint_failure=guard,
                          original_guard_joint_force_clearance_geometry=guard)
        fields.update(zip(ERROR_FIELDS, errors.unbind(-1), strict=True))
        tick = torch.full_like(distance, stamp.sdk_step, dtype=torch.long)
        solver_time = torch.full_like(distance, stamp.solver_time_s, dtype=torch.float64)
        if self.contact is None:
            return TensorRewardFrame(tick, solver_time, fields)

        nodes = self.nodes.double()
        pos, rotation = rod_local_wrists(nodes, self.state['base_link_pose_w'].double(), self.wrist_stations)
        goal_pos, goal_rotation = self.wrist_reference
        angle_delta = (goal_rotation.transpose(-1, -2) @ rotation).diagonal(dim1=-2, dim2=-1).sum(-1)
        fields['wrist_goal_m'] = (pos - goal_pos).norm(dim=-1).amax(-1)
        fields['wrist_goal_rad'] = ((angle_delta - 1) * 0.5).clamp(-1.0, 1.0).acos().amax(-1)
        z = nodes[..., 2]
        fields.update(median_z=z.median(1).values, min_z=z.amin(1))
        if capture is None:
            # Reset has no completed 12-substep contact capture. No support-free
            # assertion or new contact is made on that boundary.
            zeros = torch.zeros_like(distance, dtype=torch.bool)
            contact = dict(
                coverage_left=distance.new_zeros(self.n), coverage_right=distance.new_zeros(self.n),
                opposed_left=zeros, opposed_right=zeros, bilateral_support=zeros,
                off_table=zeros, no_external_support=zeros,
                contact_nodes=torch.zeros((self.n, 2), device=self.device, dtype=torch.long),
            )
            contact.update({name: torch.zeros_like(distance) for name in GOAL_SHAPING_FIELDS})
        else:
            if capture['pair_owner_time_s'] != stamp.solver_time_s:
                raise ValueError('grasp contact does not belong to the elapsed solver boundary')
            contact = self.contact(capture, nodes)
            contact['support_unload'] = (1 - contact.pop('support_vertical_n') / self.rod_weight_n).clamp(0, 1)
            contact['hand_lift_load'] = (contact.pop('hand_vertical_n') / self.rod_weight_n).clamp(0, 1)
            contact['balanced_hand_lift_load'] = (
                contact.pop('balanced_hand_vertical_n') / self.rod_weight_n).clamp(0, 1)
            fields['soft_non_target_contact'] = ((contact.pop('non_target_normal_n') - 0.05) / 0.05).clamp(0, 1)
        fields.update({key: val for key, val in contact.items() if key not in ('contact_nodes', 'support_load_n')})
        if suppress_contact is not None:
            # Physical work after a lane terminates does not qualify as task contact.
            for key in ('coverage_left', 'coverage_right', 'opposed_left', 'opposed_right',
                        'bilateral_support', 'off_table', 'no_external_support', *GOAL_SHAPING_FIELDS):
                fields[key] = fields[key].masked_fill(suppress_contact, 0)
        present = fields['bilateral_support']
        fields['prohibited_support'] = self._lift_reached & ~fields['no_external_support'] if capture is not None else torch.zeros_like(present)
        if suppress_contact is not None:
            fields['prohibited_support'] &= ~suppress_contact
        fields['hold_motion'] = torch.zeros_like(present)
        if capture is not None:
            height = torch.minimum(fields['median_z'] - self.reward._median, fields['min_z'] - self.reward._min)
            lifted = present & fields['no_external_support'] & fields['off_table'] & (
                height >= self.config.success['net_lift_m']
            )
            self._lift_reached |= lifted
            has_reference = torch.isfinite(self._hold_position).all(dim=(1, 2))
            candidate = torch.where(has_reference[:, None, None], self._hold_position, nodes)
            excursion = (nodes - candidate).norm(dim=-1).amax(1)
            # The hold interval restarts if the rod moves outside the original
            # half-millimetre allowance about the fixed world hold reference.
            stable = excursion <= self.config.success['max_slip_m']
            fields['hold_motion'] = lifted & stable
            self._hold_position = torch.where(
                lifted[:, None, None],
                torch.where(stable[:, None, None], candidate, nodes),
                torch.nan,
            )
        anchors = self.anchors.capture(
            tick[:, None], nodes, self.state['base_link_pose_w'], present,
            contact['contact_nodes'], self._anchor_rest_lengths, self._anchor_rest_total,
            validate=False,
        )
        fields['numerical_health'] &= ~self.anchors.invalid_tangent
        return TensorRewardFrame(tick, solver_time, fields, *anchors)

    def task_context(self, stamp):
        if not self.full_grasp:
            return self.source.contexts()
        context = self.reward.post_context
        readouts = self.reward._last_values
        phase = torch.where(
            context['hold_s'] > 0, 3,
            torch.where(readouts['bilateral_support'], 2, torch.where(self.reward._arrived, 1, 0)),
        )
        reasons = getattr(self.reward, '_reasons', {})
        terminal = self.reward._terminal
        zeros = torch.zeros_like(terminal)
        return self.source.contexts(full=FullTaskContext(
            source=stamp, phase=phase, contact_fraction=torch.stack(
                (readouts['coverage_left'], readouts['coverage_right']), dim=-1,
            ), contact_duration_s=context['contact_s'], hold_duration_s=context['hold_s'],
            anchor_station_m=self.anchors.station, anchor_valid=self.anchors.present[:, None].expand(-1, 2),
            prohibited_support_latched=reasons.get('prohibited_support', zeros) & terminal,
            success_event_emitted=reasons.get('success', zeros) & terminal,
            trusted_failure_event_emitted=reasons.get('constraint_failure', zeros) & terminal,
        ))

    def _reset_refs(self, ids):
        stamp = self.source._boundary()
        if not self.full_grasp and self.radius != float(self.runtime.solver.parameter_snapshot()['contact_radius_m']):
            raise ValueError('Reset changed the verified contact radius')
        selected = list(ids)
        if not self.full_grasp:
            require_substep_guard(self.source_guards()[selected], 'original Reset joint guards')
        if len(selected) == self.n:
            self.clear = self.raw_clearance(self.state['link_pose_wxyz'])
            d = self.distance(stamp, reset=True)
            self.source.reset(ids, d[selected])
        else:
            self.clear[selected] = self.raw_clearance(self.state['link_pose_wxyz'][selected])
            distances = self.distance(stamp, reset=True, lane_ids=selected)
            d = self.last.fields['distance'].clone()
            d[selected] = distances
            self.source.reset(ids, distances)
        if self.full_grasp:
            self.anchors.reset(ids)
            self._lift_reached[list(ids)] = False
            self._hold_position[list(ids)] = torch.nan
        frame = self.frame(stamp, d)
        for (field, seed_field) in (('q', 'seed_q'), ('qd', 'seed_qd'), ('root_pose_wxyz', 'seed_root_pose_w')):
            if not torch.allclose(self.state[field][selected], self.initial[seed_field][selected], atol=1e-06, rtol=0):
                raise ValueError('selected hand Reset readback differs from seed: ' + field)
        if self.state['root_velocity'][selected].count_nonzero() or self.velocity[selected].count_nonzero():
            raise ValueError('selected hand or rod Reset retained velocity')
        ds = 'NativePowerPoseApproach.fixed_material_station' if self.style == 'power' else 'PolicyOwned' + self.style.title() + 'Approach.fixed_material41'
        guardsource = 'original_joint_force_clearance_and_policy_source_geometry_guards'
        if self.full_grasp:
            ds = 'locked_native_pad_grids_and_measured_approach'
            guardsource = 'native_cable_fall_joint_dislocation_pad_backface'
        specs = (('distance', 'm', ds), ('numerical_health', 'dimensionless', 'same_owner_SDK_and_RopeBatch'),
                 ('constraint_failure', 'dimensionless', guardsource),
                 *((name, 'rad', 'sdk_named_active_q') for name in ERROR_FIELDS))
        if not self.full_grasp:
            specs += (('original_guard_joint_force_clearance_geometry', 'dimensionless', guardsource),)
        schema_options = {}
        if self.full_grasp:
            from dh116.rl.reward import train_rewar___FULL_UNITS
            contact_source = 'DeformX.flat_pairs.same_SDK_and_rod_boundary'
            frame_sources = {
                'median_z': 'DeformX.actual_41_world_nodes', 'min_z': 'DeformX.actual_41_world_nodes',
                'hold_motion': 'derived:fixed_world_hold_reference',
                'prohibited_support': 'derived:post_lift_external_load_from_DeformX',
            }
            specs += tuple((key, unit, frame_sources.get(key, contact_source))
                           for key, unit in train_rewar___FULL_UNITS.items())
            specs += tuple((key, 'dimensionless', contact_source) for key in GOAL_SHAPING_FIELDS)
            guard_sources = dict(cable_fall='actual_41_rod_nodes_and_table_bounds',
                                 joint_dislocation='native_q_and_articulation_hard_limits',
                                 pad_backface='actual_41_rod_nodes_and_locked_pad_SDF',
                                 soft_pad_depth='actual_41_rod_nodes_and_locked_pad_SDF',
                                 soft_non_target_contact=contact_source,
                                 soft_wrist_tracking='actual_wrist_and_projected_pose',
                                 soft_mimic='native_q_and_mimic_contract',
                                 soft_joint_limit='native_q_and_articulation_hard_limits',
                                 soft_reaction_force='unclamped_native_mesh_wrench')
            specs += tuple((key, 'dimensionless', guard_sources[key])
                           for key in (*GOAL_TERMINATIONS, *GOAL_SOFT_COSTS))
            specs += (('slip', 'm', 'derived:fixed_material_and_tangential_anchors'),
                      ('wrist_goal_m', 'm', self.config.finger_goal.reference_sha256),
                      ('wrist_goal_rad', 'rad', self.config.finger_goal.reference_sha256))
            schema_options['required_pad_ids'] = self.contact.required
        schemas = []
        for i in ids:
            episode = stamp.episode_uids[i]
            anchor_specs = tuple(
                (f'{episode}:{side}:{kind}', 'DeformX.fixed_material_node_in_SDK_palm',
                 'bounded_fp32_coordinate_rounding')
                for side in SIDES for kind in ('material', 'tangential')
            ) if self.full_grasp else ()
            schemas.append(TensorLaneSchema(
                episode, i, str(stamp.reset_ids[i]), stamp.rod_uids[i], 'unqualified_source',
                'policy_owned_' + self.style + '_endpoint_v1', tuple(map(str, range(41))),
                specs, anchor_specs=anchor_specs, **schema_options,
            ))
        schemas = tuple(schemas)
        f = frame
        annotations = {
            key: getattr(f, key)[selected] for key in (
                'anchor_present', 'anchor_drift', 'anchor_uncertainty',
                'anchor_reference', 'anchor_reference_tick',
            )
        } if self.full_grasp else {}
        reset = replace(f, physics_tick=f.physics_tick[selected], solver_time_s=f.solver_time_s[selected],
                        fields={k: v[selected] for k, v in f.fields.items()}, **annotations)
        self.reward.reset_selected(ids, schemas, reset, previous_request=self.source.previous_request[selected], tokens=tuple((self.source.previous_request_tokens[i] for i in ids)), policy_uids=('controller_reset_zero',) * len(ids))
        self.ages[selected] = 0
        if len(selected) == self.n:
            self.last = frame
        else:
            def updated(previous, new):
                previous = previous.clone()
                previous[selected] = new
                return previous

            keep = self.last
            self.last = replace(keep,
                physics_tick=updated(keep.physics_tick, reset.physics_tick),
                solver_time_s=updated(keep.solver_time_s, reset.solver_time_s),
                fields={key: updated(keep.fields[key], value) for key, value in reset.fields.items()},
                **{key: updated(getattr(keep, key), value) for key, value in annotations.items()},
            )

    def _reaction_force_cost(self, mesh_wrenches):
        # Hold checks the native force before the independent SDK wrench cap.
        raw_force = mesh_wrenches[:, self._hand_mesh_slots, :3].norm(dim=-1).amax(1)
        return ((raw_force - 15) / 25).clamp(0, 1)

    def _sdk_impl(self, actions, end):
        from dh116.physics.solver import step_thumb_physics
        phase_start = perf_counter()
        if self.full_grasp and phase_start - self.source.started >= self.source.wall_cap_s:
            raise TimeoutError('native training wall budget expired before the next SDK step')

        def mark(name):
            nonlocal phase_start
            now = perf_counter()
            self.timings[name] = self.timings.get(name, 0.0) + now - phase_start
            phase_start = now
        pre = self.source._boundary()
        halted = self.reward._terminal
        if self.full_grasp:
            halted = halted | self.reward._window_bad
        trans = self.ctrl.prepare(actions, self.state, self.clear, self.state['base_link_pose_w'], stamp=pre, halted=halted)
        if self.controls == 0 and pre.sdk_step == self.first_sdk:
            reaction = self.runtime.initial_reaction(physics_step=pre.sdk_step + 1)
        else:
            reaction = self.runtime.consume(physics_step=pre.sdk_step + 1)
        evidence = None
        if self.probe_dir is not None:
            evidence = probe_copy(dict(pre=pre, state_pre=self.state, rod_pre=self.nodes, velocity_pre=self.velocity, pre_frame=self.last, pre_clearance=self.clear, palms=self.ctrl.palms, angles=self.ctrl.angles, action=actions, projection=trans['projection'], command=trans['expected'], reaction=reaction))
        self.ctrl.submit(trans, stamp=pre, wrenches=reaction['mesh_wrenches'], state=self.state, reaction_source=reaction)
        if evidence is not None:
            evidence['composer'] = {}
            for s in SIDES:
                owner = self.batch.sides[s]
                comp = owner.hand.permanent_wrench_composer
                evidence['composer'][s] = probe_copy({k: getattr(comp, 'composed_' + k + '_as_torch').index_select(0, owner.row_indices) for k in ('force', 'torque')})
        mark('project_reaction_submit')
        sdk = step_thumb_physics(self.sim, pre.sdk_step, budget_end=end)
        mark('physx_step')
        self.batch.update(1 / 480)
        measured = capture_regex_state(self.batch, body_com_pose_b_xyzw=self.initial['body_com_pose_b_xyzw'])
        mark('sdk_readback')
        if self.full_grasp:
            sdk_valid = trans['submit_valid']
            for name in _MEASURED_FIELDS:
                sdk_valid = sdk_valid & torch.isfinite(measured[name]).flatten(1).all(1)
            quaternion = measured['link_pose_xyzw'][..., 3:]
            sdk_valid &= ((quaternion.norm(dim=-1) - 1).abs() <= 1e-5).flatten(1).all(1)
            # Invalid SDK rows retain the last finite poses for native submission.
            # A bad readback on a live row still invalidates the whole window.
            for name in _MEASURED_FIELDS:
                value = measured[name]
                rows = sdk_valid.reshape((-1,) + (1,) * (value.ndim - 1))
                measured[name] = torch.where(rows, value, self.state[name])
            self.state = measured
            link_poses = measured['link_pose_xyzw']
        else:
            self.state = measured
            self.clear = self.raw_clearance(measured['link_pose_wxyz'])
            guard = trans['constraint_failure'] | self.source_guards()
            link_poses = measured['link_pose_xyzw']
        self.bridge.submit(self.runtime.solver, link_poses, validate_values=not self.full_grasp)
        mark('mesh_pose_submit')
        capture = self.runtime.advance(physics_step=sdk)
        if self.full_grasp:
            # An invalid owned packet can signal a global indexing, count or
            # force-balance fault. Validate it even after every lane has ended.
            self.reward._window_bad |= ~capture['valid']
            rod_valid = self._read_full_rod_state()
        mark('rod_flat_export_reduce')
        post = self.source._boundary()
        joint_bad = self.ctrl.finish(trans, measured, stamp=post)
        distance = self.distance(post)
        mark('pad_geometry')
        if self.full_grasp:
            numerical_ok = sdk_valid & capture['valid'] & rod_valid
            relative = self.nodes - self.table_centers[:, None]
            outside = (relative[..., :2].abs() > self.table_half_size[:2] + self.radius).any(-1)
            fallen = (outside & (relative[..., 2] < self.table_half_size[2] - self.radius)).any(1)
            tracking = (measured['base_link_pose_w'][..., :3] -
                        trans['projection']['projected_root_poses'][..., :3]).norm(dim=-1).amax(1)
            hard = dict(cable_fall=fallen, joint_dislocation=joint_bad, pad_backface=self._pad_backface)
            soft = dict(soft_pad_depth=self._soft_pad_depth,
                        soft_wrist_tracking=((tracking - 0.010) / 0.010).clamp(0, 1),
                        soft_mimic=trans['soft_mimic'], soft_joint_limit=trans['soft_joint_limit'],
                        soft_reaction_force=self._reaction_force_cost(capture['mesh_wrenches']))
            self.last = self.frame(post, distance, capture=capture, suppress_contact=halted,
                                   hard_causes=hard, soft_costs=soft, numerical_ok=numerical_ok)
        else:
            guard |= joint_bad | self._geometry_bad
            if self.geometry is not None:
                projection = trans['projection']
                if projection['projection_source'] != SOURCE:
                    raise ValueError('original wrist projection source differs')
                tracking = (measured['base_link_pose_w'][..., :3].double()
                            - projection['projected_root_poses'][..., :3]).norm(dim=-1)
                speed = measured['base_link_com_velocity_w'][..., :3].double().norm(dim=-1)
                angular = torch.stack([measured['q'][:, s, [self.joints[s].index(k) for k in WRIST_ANGULAR]]
                                       for s in (0, 1)], 1)
                wrench = capture['mesh_wrenches'][:, self.ctrl.slots[0] + self.ctrl.slots[1]]
                bad = ((tracking > WRIST['tracking_guard_m']) | (speed > WRIST['linear_speed_guard_m_s'])).any(1)
                bad |= (angular.double().abs() > math.pi / 2).any((1, 2))
                bad |= (wrench[..., :3].norm(dim=-1) > 30).any(1)
                require_substep_guard(bad, 'original force or wrist guard')
            self.last = self.frame(post, distance, capture=capture, constraint=guard)
        ready = self.reward.observe(self.last)
        mark('r0_and_post_guards')
        if evidence is not None:
            evidence.update(post=post, state_post=self.state, nodes_post=self.nodes, velocity_post=self.velocity, post_clearance=self.clear, capture=capture, post_frame=self.last, geometry=None if self.geometry is None else self.geometry.result)
            self.snapshot('sdk_' + str(sdk), evidence)
            mark('n2_cold_parity_evidence')
        return ready

    def peer_state(self, peers):
        """Cold probe only. Keep exact byte patterns including signed zero and NaNs."""
        tensors = {**self.state, 'rod_nodes': self.nodes, 'rod_v': self.velocity, 'palm': self.ctrl.palms, 'angles': self.ctrl.angles, 'age': self.ages, 'rest': self.source.rest_lengths_m, 'request': self.source.previous_request, 'reset_z': self.source.reset_z, 'reset_distance': self.source.reset_distance_m}
        for (name, value) in vars(self.reward).items():
            if isinstance(value, torch.Tensor) and value.ndim and (len(value) == self.n):
                tensors['reward.' + name] = value
        for name in ('counts', 'node_forces', 'mesh_wrenches'):
            tensors['reaction.' + name] = self.runtime._reaction[name]
        for s in SIDES:
            owner = self.batch.sides[s]
            for k in ('permanent', 'instantaneous'):
                composer = getattr(owner.hand, k + '_wrench_composer')
                for field in ('force', 'torque'):
                    value = getattr(composer, 'composed_' + field + '_as_torch')
                    tensors['.'.join((s, k, field))] = value.index_select(0, owner.row_indices)
        result = {k: v[list(peers)].contiguous().view(torch.uint8).clone() for (k, v) in tensors.items()}
        packet = self.runtime._reaction
        mask = torch.isin(packet['env_ids'], torch.tensor(peers, device=self.device))
        for k in ('env_ids', 'indices', 'vectors', 'normal_force'):
            result['flat_pairs.' + k] = packet[k][mask].contiguous().view(torch.uint8).clone()
        for name in ('_terms', '_last_values'):
            for (key, value) in getattr(self.reward, name).items():
                if isinstance(value, torch.Tensor):
                    result['reward.' + name + '.' + key] = value[list(peers)].contiguous().view(torch.uint8).clone()
        return result

    def _reset_impl(self, ids):
        prove_peers = self.prove_peer_reset and self.n == 2
        peers = tuple(sorted(set(self.ids) - set(ids))) if prove_peers else ()

        def peer_epochs():
            return tuple(((self.runtime.episodes[i], self.runtime.resets[i], self.runtime.ticks[i], self.reward.schemas[i], self.source.previous_request_tokens[i]) for i in peers))
        epochs = peer_epochs()
        before = self.peer_state(peers) if peers else {}
        mesh = self.runtime.solver.mesh_state_snapshot() if peers else None
        sdk = self.runtime.physics_step
        self.state = restore_regex_seed(self.sim, self.batch, self.initial, env_ids=list(ids), active_effort_limit_nm=self.effort)
        self.runtime.reset_lanes(ids, episode_uids=tuple((uuid.uuid4().hex for _ in ids)))
        mesh_reset(self.runtime.solver, self.bridge, self.state, ids, before=mesh, verify=not self.full_grasp or self.prove_peer_reset)
        self.ctrl.invalidate(list(ids))
        self._reset_refs(ids)
        if peers:
            after = self.peer_state(peers)
            if epochs != peer_epochs() or set(before) != set(after) or any((not torch.equal(before[k], after[k]) for k in before)):
                raise ValueError('selected Reset changed peer hand/rod/reaction/controller/source/reward state')
        self.snapshot('reset_' + str(sdk), dict(ids=ids, state=self.state, nodes=self.nodes, velocities=self.velocity, source=self.source._boundary(), references=self.last, schemas=tuple(self.reward.schemas)))
        if self.sim.current_time_step_index != sdk:
            raise ValueError('Reset unexpectedly advanced PhysX')
        if prove_peers:
            self.reset_proofs.append(dict(sdk_step=sdk, selected=list(ids), peers=list(peers), peer_fields=list(before), native_mesh_verified=True, reset_ids=list(self.runtime.resets)))
