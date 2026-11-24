"""One shared physical stepping/reset/sensor runtime with no policy collector."""
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import is_dataclass
from dataclasses import replace
import json
import math
from time import perf_counter
from types import MappingProxyType
import uuid
import numpy as np
import torch
from dh116.physics.geometry import collision_clearance
from dh116.hand.kinematics import defor_wrist_proje__CLEARANCE_M as CLEARANCE_M
from dh116.hand.kinematics import defor_wrist_proje__SOURCE as SOURCE
from dh116.hand.kinematics import WRIST_ANGULAR
from dh116.env.contracts import checked_file
from dh116.hand.kinematics import palm_mount__CONTRACT as WRIST
from dh116.hand.articulation import capture_regex_state
from dh116.hand.articulation import restore_regex_seed
from dh116.env.contracts import validate_runtime_execution
from dh116.hand.safety import TensorControl
from dh116.physics.geometry import DECISION_MARGIN_M
from dh116.physics.geometry import make_tensor_geometry
from dh116.physics.geometry import TensorMeshBridge
from dh116.rl.teacher import TensorSourceOwner
from dh116.rl.reward import TensorLaneSchema
from dh116.rl.reward import TensorRewardFrame
from dh116.rl.reward import TensorTrainingReward
from dh116.rl.reward import measured_finger_errors
from dh116.rl.reward import train_rewar__ERROR_FIELDS as ERROR_FIELDS
from dh116.rl.reward import load_config
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

    def __call__(self, poses):
        clear = []
        for (side, (links, vertices)) in enumerate(self.sides):
            p = poses[:, side].double()
            (w, x, y, z) = p[..., 3:].unbind(-1)
            row = torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)
            heights = (row[:, links] * vertices).sum(-1) + p[:, links, 2]
            clear.append(heights.amin(-1) - 0.3)
        clear = torch.stack(clear, -1)
        near = ((clear - CLEARANCE_M).abs() <= DECISION_MARGIN_M).any(-1)
        if bool(near.any()):
            (exact, _) = collision_clearance(poses[near].detach().cpu().numpy(), self.shapes)
            clear[near] = clear.new_tensor(exact)
        if bool((~torch.isfinite(clear) | (clear < CLEARANCE_M)).any()):
            raise ValueError('original pre-rod raw source table clearance guard failed')
        return clear

def mesh_reset(solver, bridge, state, ids, *, before=None):
    """Cold native Reset ABI, with all three actual phase and velocity readbacks."""
    (positions, quaternions) = bridge.targets(state['link_pose_xyzw'])
    idx = list(ids)
    p = positions[idx].cpu().numpy()
    q = quaternions[idx].cpu().numpy()
    clock = float(solver.time)
    solver.reset_env_mesh_poses(np.asarray(idx, dtype=np.int32), p, q)
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
    if float(solver.time) != clock or solver.last_error:
        raise ValueError('mesh reset advanced native time or failed')

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

    def __init__(self, project, sim, batch, prepared, archive, seed, initial, bindings, runtime, profile):
        self._configure_physics(project, sim, batch, prepared, archive, seed, initial, bindings, runtime, profile)
        self._initialize_episode()

    def _configure_physics(self, project, sim, batch, prepared, archive, seed, initial, bindings, runtime, profile):
        validate_runtime_execution()
        (self.sim, self.batch, self.runtime) = (sim, batch, runtime)
        (self.initial, self.state) = (initial, initial)
        self.device = runtime.device
        (self.style, self.n) = (archive.style, runtime.solver.num_envs)
        self.ids = tuple(range(self.n))
        self.origins = initial['env_origins'] + initial['env_origins'].new_tensor(seed['origin'])
        self.source = TensorSourceOwner(runtime, sim, profile, self.origins, self.device, self.style, 32 / 60, 180.0, rod_uids=tuple((f'rod:{i}' for i in self.ids)))
        self.joints = tuple((tuple(batch.sides[s].joint_names) for s in SIDES))
        names = tuple((tuple(batch.sides[s].body_names) for s in SIDES))
        for (side, body_names) in zip(SIDES, names, strict=True):
            owner = batch.sides[side]
            paths = owner.hand.root_physx_view.link_paths
            for (prefix, idx) in zip(batch.prefixes, owner.row_indices.tolist(), strict=True):
                if tuple(paths[idx]) != tuple((prefix + '/' + side + 'Hand/' + name for name in body_names)):
                    raise ValueError('actual link identity differs from original source geometry')
        self.reset_proofs = []
        self.probe_dir = None
        self.bindings = bindings
        self._views = tuple((batch.sides[s].hand.root_physx_view for s in SIDES))
        roots = initial['root_seed_w'].new_tensor(seed['reset_root_pose_w'])[None].repeat(self.n, 1, 1)
        roots[..., :3] += initial['env_origins'][:, None]
        self.ctrl = TensorControl(batch, roots, bindings, runtime.mesh_names)
        self.effort = float(prepared['candidate_physics']['hand']['active_effort_limit_nm'])
        self.raw_clearance = RawTableClearance(prepared, names, self.device)
        self.bridge = TensorMeshBridge(runtime.mesh_names, bindings, initial['mesh_positions_world_m'], initial['mesh_quaternions_xyzw'])
        mesh_reset(runtime.solver, self.bridge, initial, self.ids)
        self.radius = float(runtime.solver.parameter_snapshot()['contact_radius_m'])
        self.radii = torch.full((self.n,), self.radius, dtype=torch.float64, device=self.device)
        self.geometry = None
        if self.style == 'pinch':
            original = pinch_geometry_inputs(archive.scene_inputs['geometry'])
            expected_radius = float(archive.mesh_plan['cfg']['rod']['radius_m'])
            if not math.isclose(self.radius, expected_radius, abs_tol=1e-09, rel_tol=2e-06):
                raise ValueError('locked Pinch material radius differs')
            for side in SIDES:
                if len(self.ctrl.slots[SIDES.index(side)]) != 5:
                    raise ValueError('original Pinch reaction route layout required')
                for (name, locked) in original['sources'][side]['sdf_binding'].items():
                    row = bindings[side + '.' + name]
                    if row['sdf_sha256'] != locked['sha256'] or row['capture_sha256'] != locked['capture_sha256']:
                        raise ValueError('original locked SDF or capture differs')
            self.geometry = make_tensor_geometry(original, names, device=self.device)
        elif self.style == 'power':
            goal = json.loads(checked_file(project / 'configs/tasks/power_reference.json', archive.task['near_open_source_sha256']))
            targets = [goal['candidate'][s]['target_pose_wxyz'][:3] for s in SIDES]
            self.targets = roots.new_tensor(targets)[None] + initial['env_origins'][:, None]
            self.stations = (self.targets[:, :, None] - runtime._initial[:, None].double()).norm(dim=-1).argmin(-1)
            self.row = torch.arange(self.n, device=self.device)[:, None]
        else:
            raise ValueError('Tripod requires its complete native support and route guard')
        self.config = load_config(project / f'configs/training/reward_{self.style}.json', ppo_gamma=0.995)

    def _initialize_episode(self):
        self.reward = TensorTrainingReward(self.config, num_envs=self.n, device=self.device)
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
            if self.runtime.physics_step < window['start_step'] + 8:
                require_substep_guard(ready, 'terminal lane before end of shared SDK window')
            if self._capture_cadence == 'substep':
                self._notify_capture('post')
            return ready
        except BaseException:
            self._physical_failed = True
            raise

    def advance_action_window(self, actions, end):
        """Retain the original early-terminal stop between the eight substeps."""
        try:
            for substep in range(8):
                ready = self.sdk(actions, end)
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
            timeout = (self.ages >= 32) & ~self.reward._terminal
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
        bad = self.ctrl.joint_failures(state)
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
        require_substep_guard(bad, 'original mounted17 source/joint/cache guards')

    def arm_probe(self, output):
        if self.n != 2:
            raise ValueError('raw parity snapshots are N2 only')
        self.probe_dir = output
        self.snapshot('startup', dict(state=self.state, roots=self.ctrl.roots, source=self.source._boundary(), radius=self.radius, nodes=self.nodes, velocity=self.velocity, reset=self.last, sources=self.reward.schemas, origins=self.origins, write_ids=self.ctrl.write_ids, active_ids=self.ctrl.active, contact_ids=self.ctrl.contact_bodies, mesh_slots=self.ctrl.slots, mesh_names=self.runtime.mesh_names, bindings=self.bindings, shapes=self.raw_clearance.shapes, joint_names=self.joints, body_names=tuple((self.batch.sides[s].body_names for s in SIDES))))

    def snapshot(self, name, row):
        if self.probe_dir is not None:
            with (self.probe_dir / ('tensor_n2_' + name + '.pt')).open('xb') as stream:
                torch.save(probe_copy(row), stream)

    def rod_state(self):
        solver = self.runtime.solver
        (p, v) = (torch.as_tensor(x) for x in (solver.positions(), solver.velocities()))
        if any((x.device != self.device or x.shape != (self.n, 41, 3) for x in (p, v))):
            raise ValueError('actual same-device full rod position and velocity required')
        require_substep_guard(~(torch.isfinite(p) & torch.isfinite(v)).all((1, 2)), 'nonfinite rod')
        return (p, v)

    def distance(self, stamp, *, reset=False):
        (p, v) = self.rod_state()
        (self.nodes, self.velocity) = (p, v)
        if self.geometry is None:
            target = self.targets + p[self.row, self.stations].double() - self.runtime._initial[self.row, self.stations].double()
            return (self.state['base_link_pose_w'][..., :3].double() - target).norm(dim=-1).mean(-1)
        geometry = self.geometry.capture(self.state['link_pose_xyzw'], p, frame=stamp, radius=self.radii)
        accepted = geometry.source_complete if reset else geometry.geometry_passed
        require_substep_guard(~accepted, 'original source geometry or SDF coverage')
        return geometry.positive_pad_mean_m.mean(-1)

    def frame(self, stamp, distance):
        q = torch.stack([self.state['q'][:, s, self.ctrl.active[s]] for s in (0, 1)], dim=1)
        errors = measured_finger_errors(q, self.config.finger_goal)
        healthy = torch.full_like(distance, self.runtime.solver.last_error == 0, dtype=torch.bool)
        guard = self.ctrl.joint_failures(self.state)
        fields = dict(distance=distance, numerical_health=healthy, constraint_failure=guard, original_guard_joint_force_clearance_geometry=guard)
        fields.update(zip(ERROR_FIELDS, errors.unbind(-1), strict=True))
        return TensorRewardFrame(torch.full_like(distance, stamp.sdk_step, dtype=torch.long), torch.full_like(distance, stamp.solver_time_s, dtype=torch.float64), fields)

    def _reset_refs(self, ids):
        stamp = self.source._boundary()
        if self.radius != float(self.runtime.solver.parameter_snapshot()['contact_radius_m']):
            raise ValueError('Reset changed the verified contact radius')
        self.source_guards()
        self.clear = self.raw_clearance(self.state['link_pose_wxyz'])
        d = self.distance(stamp, reset=True)
        self.last = self.frame(stamp, d)
        self.source.reset(ids, d[list(ids)])
        selected = list(ids)
        for (field, seed_field) in (('q', 'seed_q'), ('qd', 'seed_qd'), ('root_pose_wxyz', 'seed_root_pose_w')):
            if not torch.allclose(self.state[field][selected], self.initial[seed_field][selected], atol=1e-06, rtol=0):
                raise ValueError('selected hand Reset readback differs from seed: ' + field)
        if self.state['root_velocity'][selected].count_nonzero() or self.velocity[selected].count_nonzero():
            raise ValueError('selected hand or rod Reset retained velocity')
        ds = 'NativePowerPoseApproach.fixed_material_station' if self.style == 'power' else 'PolicyOwned' + self.style.title() + 'Approach.fixed_material41'
        guardsource = 'original_joint_force_clearance_and_policy_source_geometry_guards'
        specs = (('distance', 'm', ds), ('numerical_health', 'dimensionless', 'RopeBatch.last_error.same_owner'), ('constraint_failure', 'dimensionless', guardsource), ('original_guard_joint_force_clearance_geometry', 'dimensionless', guardsource), *((name, 'rad', 'sdk_named_active_q') for name in ERROR_FIELDS))
        schemas = tuple((TensorLaneSchema(stamp.episode_uids[i], i, str(stamp.reset_ids[i]), stamp.rod_uids[i], 'unqualified_source', 'policy_owned_' + self.style + '_endpoint_v1', tuple(map(str, range(41))), specs) for i in ids))
        f = self.last
        selected = list(ids)
        reset = replace(f, physics_tick=f.physics_tick[selected], solver_time_s=f.solver_time_s[selected], fields={k: v[selected] for (k, v) in f.fields.items()})
        self.reward.reset_selected(ids, schemas, reset, previous_request=self.source.previous_request[selected], tokens=tuple((self.source.previous_request_tokens[i] for i in ids)), policy_uids=('controller_reset_zero',) * len(ids))
        self.ages[selected] = 0

    def _sdk_impl(self, actions, end):
        from dh116.physics.solver import step_thumb_physics
        phase_start = perf_counter()

        def mark(name):
            nonlocal phase_start
            now = perf_counter()
            self.timings[name] = self.timings.get(name, 0.0) + now - phase_start
            phase_start = now
        pre = self.source._boundary()
        trans = self.ctrl.prepare(actions, self.state, self.clear, self.state['base_link_pose_w'], stamp=pre)
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
        self.state = capture_regex_state(self.batch, body_com_pose_b_xyzw=self.initial['body_com_pose_b_xyzw'])
        mark('sdk_readback')
        self.clear = self.raw_clearance(self.state['link_pose_wxyz'])
        self.source_guards()
        self.bridge.submit(self.runtime.solver, self.state['link_pose_xyzw'])
        mark('raw_source_guard_mesh_submit')
        capture = self.runtime.advance(physics_step=sdk)
        mark('rod_flat_export_reduce')
        post = self.source._boundary()
        self.ctrl.finish(trans, self.state, stamp=post)
        if self.geometry is not None:
            projection = trans['projection']
            if projection['projection_source'] != SOURCE:
                raise ValueError('original wrist projection source differs')
            tracking = (self.state['base_link_pose_w'][..., :3].double() - projection['projected_root_poses'][..., :3]).norm(dim=-1)
            speed = self.state['base_link_com_velocity_w'][..., :3].double().norm(dim=-1)
            angular = torch.stack([self.state['q'][:, s, [self.joints[s].index(k) for k in WRIST_ANGULAR]] for s in (0, 1)], 1)
            wrench = capture['mesh_wrenches'][:, self.ctrl.slots[0] + self.ctrl.slots[1]]
            bad = ((tracking > WRIST['tracking_guard_m']) | (speed > WRIST['linear_speed_guard_m_s'])).any(1)
            bad |= (angular.double().abs() > math.pi / 2).any((1, 2)) | (wrench[..., :3].norm(dim=-1) > 30).any(1)
            bad |= ~torch.isfinite(self.state['body_com_velocity_w']).all((1, 2, 3)) | ~torch.isfinite(self.state['backend_targets']).all((1, 2))
            require_substep_guard(bad, 'original force or wrist guard')
        self.last = self.frame(post, self.distance(post))
        ready = self.reward.observe(self.last)
        mark('geometry_r0_and_post_guards')
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
        peers = tuple(sorted(set(self.ids) - set(ids)))

        def peer_epochs():
            return tuple(((self.runtime.episodes[i], self.runtime.resets[i], self.runtime.ticks[i], self.reward.schemas[i], self.source.previous_request_tokens[i]) for i in peers))
        epochs = peer_epochs()
        before = self.peer_state(peers) if peers else {}
        mesh = self.runtime.solver.mesh_state_snapshot() if peers else None
        sdk = self.runtime.physics_step
        self.state = restore_regex_seed(self.sim, self.batch, self.initial, env_ids=list(ids), active_effort_limit_nm=self.effort)
        self.runtime.reset_lanes(ids, episode_uids=tuple((uuid.uuid4().hex for _ in ids)))
        mesh_reset(self.runtime.solver, self.bridge, self.state, ids, before=mesh)
        self.ctrl.invalidate(list(ids))
        self._reset_refs(ids)
        if peers:
            after = self.peer_state(peers)
            if epochs != peer_epochs() or set(before) != set(after) or any((not torch.equal(before[k], after[k]) for k in before)):
                raise ValueError('selected Reset changed peer hand/rod/reaction/controller/source/reward state')
        self.snapshot('reset_' + str(sdk), dict(ids=ids, state=self.state, nodes=self.nodes, velocities=self.velocity, source=self.source._boundary(), references=self.last, schemas=tuple(self.reward.schemas)))
        if self.sim.current_time_step_index != sdk:
            raise ValueError('Reset unexpectedly advanced PhysX')
        self.reset_proofs.append(dict(sdk_step=sdk, selected=list(ids), peers=list(peers), peer_fields=list(before), native_mesh_verified=True, reset_ids=list(self.runtime.resets)))
