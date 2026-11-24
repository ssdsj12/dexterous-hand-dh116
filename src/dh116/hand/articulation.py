"""Two regex SDK Articulations, cached official reads and scene binding/setup."""
import hashlib as regex_artic__hashlib
from copy import deepcopy as regex_cache_readb__deepcopy
from copy import deepcopy as regex_task_setup__deepcopy
import importlib as regex_artic__importlib
import hashlib as regex_cache_readb__hashlib
import traceback
import math as regex_artic__math
import importlib.metadata
import importlib as regex_cache_readb__importlib
import importlib as regex_task_setup__importlib
import os
from pathlib import Path as regex_cache_readb__Path
import math as regex_task_setup__math
from pathlib import Path as regex_artic__Path
SDK_VERSION = '0.54.4'
import numpy as regex_task_setup__np
import re
SDK_SOURCE_SHA256 = {'assets/articulation/articulation_data.py': 'd69495c66c409fb20823e4e467f9d90e63163b2050b77d2ac8e433221d1d208a', 'assets/articulation/articulation.py': 'fb179aacdd848906367d01105020e1d36412bd0f1da7187be09f0781ff589309', 'utils/math.py': '881ab27758816475e26d6721bab22c934fd70d0c064392266a90b06f5bbe4149', 'utils/buffers/timestamped_buffer.py': '1020269322e63b6383c9782f8cf369cbe8b55f52e081bccd79d03ae52712bd59', 'utils/wrench_composer.py': '095ab1fcf374e4bce96b61430f810c83f3519b7225476ce6e0a6ceb403a5f6a8'}
import torch as regex_task_setup__torch
import stat
_GETTERS = ('joint_pos', 'joint_vel', 'root_link_pose_w', 'root_com_vel_w', 'body_link_pose_w', 'body_com_vel_w')
from dh116.env.contracts import JOINTS as regex_task_setup__JOINTS
import numpy as regex_artic__np
_INVALIDATE = ('_joint_pos', '_joint_vel', '_root_link_pose_w', '_root_com_vel_w', '_body_link_pose_w', '_body_com_vel_w', '_root_link_vel_w', '_root_com_pose_w', '_body_link_vel_w', '_body_com_pose_w', '_root_state_w', '_root_link_state_w', '_root_com_state_w', '_body_state_w', '_body_link_state_w', '_body_com_state_w')
regex_task_setup__SIDES = ('left', 'right')
import torch as regex_artic__torch

def _type_name(value):
    cls = type(value)
    return f'{cls.__module__}.{cls.__qualname__}'
DT = 1 / 480
from dh116.env.contracts import linea_coupl__ACTIVE as ACTIVE

def verify_sdk_cache_source(owner):
    """Fail closed on SDK version/source drift; report actual producer types.

    This only audits source bytes. A CPU spy executing the installed getters is
    still a CPU spy, and the metadata explicitly keeps native qualification false.
    No Isaac package is imported and no backend state is read during this gate.
    """
    version = regex_cache_readb__importlib.metadata.version('isaaclab')
    if version != SDK_VERSION:
        raise ValueError(f'unsupported ArticulationData SDK version: {version!r}')
    data = owner.hand.data
    source = regex_cache_readb__Path(type(data).joint_pos.fget.__code__.co_filename).resolve(strict=True)
    if source.parts[-3:] != ('assets', 'articulation', 'articulation_data.py'):
        raise ValueError('unsupported ArticulationData source path')
    sdk_root = source.parents[2]
    for name in _GETTERS:
        getter = getattr(type(data), name)
        if not isinstance(getter, property) or regex_cache_readb__Path(getter.fget.__code__.co_filename).resolve() != source:
            raise ValueError(f'unsupported ArticulationData source getter: {name}')
    if regex_cache_readb__Path(type(data).update.__code__.co_filename).resolve() != source:
        raise ValueError('unsupported ArticulationData update source')
    files = {}
    for (relative, expected) in SDK_SOURCE_SHA256.items():
        path = sdk_root / relative
        actual = regex_cache_readb__hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'installed SDK source SHA differs: {relative}')
        files[relative] = {'path': str(path), 'sha256': actual}
    return {'adapter': 'official_articulation_data_owned_readback_v1', 'sdk_version': version, 'source_files': files, 'actual_producer': {'hand_type': _type_name(owner.hand), 'data_type': _type_name(data), 'view_type': _type_name(owner.hand.root_physx_view), 'device': str(owner.device)}, 'native_equivalence_verified': False, 'qualification': 'installed_sdk_source_audit_only', 'pose_frames': {'root': 'root_link_pose_w', 'link': 'body_link_pose_w'}, 'velocity_frames': {'root': 'root_com_vel_w', 'link': 'body_com_vel_w'}, 'quaternion_conversion': 'component_permutation_only_no_normalization', 'backend_targets': 'raw_get_dof_position_targets', 'snapshot_ownership': 'index_select_owned_copy', 'external_write_contract': 'invalidate_after_writes_before_read'}
OWNED_NAMES = tuple((f'regex_{side}_{kind}' for side in regex_task_setup__SIDES for kind in ('hand', 'view', 'permanent', 'instantaneous')))
from dh116.env.contracts import JOINTS as regex_artic__JOINTS

class CachedSideReadback:
    """Reuse the original SDK cache, preserving the raw reader's owned fields."""

    def __init__(self, owner):
        self._metadata = verify_sdk_cache_source(owner)
        self.owner = owner
        self._hand = owner.hand
        self._data = self._hand.data
        self._view = self._hand.root_physx_view

    def _check_identity(self):
        if self.owner.hand is not self._hand or self._hand.data is not self._data or self._hand.root_physx_view is not self._view:
            raise ValueError('cached readback SDK owner/data/view identity changed')

    def metadata(self):
        """Return a detached startup source/producer record, never a native claim."""
        return regex_cache_readb__deepcopy(self._metadata)

    def read(self, env_ids=None):
        """Own q/qd/targets/root/link snapshots in the supplied environment order.

        Poses use actor/link frames. Velocities retain the raw SDK center-of-mass
        reference point. Both quaternion layouts use pure indexing, preserving
        sign, non-unit values and original component bit patterns.
        """
        self._check_identity()
        rows = self.owner.rows(env_ids)
        data = self._data
        state = {'q': data.joint_pos.index_select(0, rows), 'qd': data.joint_vel.index_select(0, rows), 'backend_targets': self._view.get_dof_position_targets().index_select(0, rows), 'root_pose_wxyz': data.root_link_pose_w.index_select(0, rows), 'root_velocity': data.root_com_vel_w.index_select(0, rows), 'link_pose_wxyz': data.body_link_pose_w.index_select(0, rows), 'link_velocity': data.body_com_vel_w.index_select(0, rows)}
        xyzw = [0, 1, 2, 4, 5, 6, 3]
        state['root_pose_xyzw'] = state['root_pose_wxyz'][..., xyzw]
        state['link_pose_xyzw'] = state['link_pose_wxyz'][..., xyzw]
        return state

    def invalidate(self, *, reason):
        """Force a fresh official acquisition after Reset/external state writes.

        SDK TimestampedBuffer objects are whole-view caches, so invalidation is
        whole-view even for a selected Reset. It does not modify physical peers,
        submitted targets, composed wrenches, cache clock, or derivative history.
        Official joint-velocity writers remain responsible for derivative history.
        """
        self._check_identity()
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('explicit external-write invalidation reason required')
        for name in _INVALIDATE:
            getattr(self._data, name).timestamp = float('-inf')
        for name in ('permanent_wrench_composer', 'instantaneous_wrench_composer'):
            getattr(self._hand, name)._link_poses_updated = False

def prepare_regex_task(sim, owners, trace, project, prepared, archive, lanes, seed_height=None):
    """Return batch, effective seed, original rod nodes, bindings and GPU state.

    bindings is ONE static mapping, keyed by the archived native mesh names.
    Each row retains its source/cooked/SDF identity and actual body_index;
    prim_path names env0 and prim_paths records all actual environment paths.
    There are no per-lane Articulation objects or wrappers.

    initial contains only same-device tensors in public environment order.
    Seed tensors are immutable-by-contract restore inputs, not measured state.
    COM velocities come from actual get_link_velocities (the installed SDK's
    body_com_vel_w source). World COM pose uses raw link quaternion and actual
    get_coms local transform. No root/COM/link frames are interchanged.

    seed_height is optional and explicit: delta_z_m/evidence_sha256 are passed
    to the original derive_height_seed. No grasp receives an automatic height
    change. Original physical authoring remains unchanged even for a candidate;
    only the later zero-step explicit seed restore uses the derived pose.
    """
    from types import SimpleNamespace
    import dh116.env.scene as assets
    import dh116.env.scene as supports
    import dh116.env.scene as support_material
    from dh116.env.scene import _seed
    from dh116.env.scene import derive_height_seed
    from dh116.env.scene import isolate_scenes
    _regex_task_setup_import__spawn_regex_articulations = spawn_regex_articulations
    from dh116.env.scene import spawn_parameters
    lanes = tuple(lanes)
    origins = regex_task_setup__np.asarray([origin for (_, origin) in lanes], dtype=regex_task_setup__np.float64)
    if sim.current_time_step_index != 0 or not 2 <= len(lanes) <= 128 or origins.shape != (len(lanes), 3) or (not regex_task_setup__np.isfinite(origins).all()) or regex_task_setup__np.any(origins[:, 2] != 0) or (archive.style not in ('pinch', 'tripod', 'power')):
        raise ValueError('fresh 480Hz Sim and 2..128 independently authored lanes required')
    (args, layout, sources) = spawn_parameters(archive.scene_inputs, project)
    if archive.style == 'pinch':
        seed = _seed(archive.pair['reset_seed'])
        rod_nodes = regex_task_setup__np.array(archive.pair['rod_seed']['proposed_nodes_m'], dtype=regex_task_setup__np.float64, copy=True)
    else:
        seed = _seed(archive.scene_inputs['authored_seed'])
        rod_nodes = regex_task_setup__np.array(archive.scene_inputs['authored_rod_nodes'], dtype=regex_task_setup__np.float64, copy=True)
    if rod_nodes.shape != (41, 3) or not regex_task_setup__np.isfinite(rod_nodes).all():
        raise ValueError('all 41 original finite material rod nodes required')
    original_seed_sha = seed['sha256']
    height_proof = None
    effective_seed = seed
    if seed_height is not None:
        if not isinstance(seed_height, dict) or set(seed_height) != {'delta_z_m', 'evidence_sha256'}:
            raise ValueError('explicit height delta and evidence SHA required')
        (effective_seed, height_proof) = derive_height_seed(seed, grasp=archive.style, **seed_height)
    effort = float(prepared['candidate_physics']['hand']['active_effort_limit_nm'])
    if not regex_task_setup__math.isfinite(effort) or effort <= 0:
        raise ValueError('original explicit positive candidate effort limit required')
    batch = owners.keep('regex_batch', _regex_task_setup_import__spawn_regex_articulations(sim, args, layout, sources, origins))
    for (env_id, scene) in enumerate(batch.scenes):
        scoped = dict(prepared, prefix=scene['prefix'], env_id=env_id, table=dict(prepared['table'], center_m=(regex_task_setup__np.asarray(prepared['table']['center_m']) + scene['origin']).tolist()))
        assets._bind_materials(sim.stage, scoped)
        if assets.inspect_source_stage(sim.stage, scoped)['original_colliders'] != 34:
            raise ValueError('all 34 original material-bound colliders per lane required')
        if archive.task is not None:
            support = regex_task_setup__deepcopy(archive.task['support_scene'])
            for box in support['supports']:
                box['center_m'] = (regex_task_setup__np.asarray(box['center_m']) + scene['origin']).tolist()
            paths = supports.scene_suppo__author_end_supports(sim.stage, scene['prefix'], support)
            material = prepared['candidate_physics']['table']
            support_material._bind_support_material(sim.stage, scene['prefix'], paths, material['mu_s'], material['mu_k'])
    isolate_scenes(sim.stage, batch.scenes)
    for (side, hand) in zip(regex_task_setup__SIDES, batch.hands, strict=True):
        trace.watch(f'regex_{side}_hand', hand, owner='regex_task_setup', release='owned scope then SDK STOP', callbacks=True)
    if sim.current_time_step_index != 0:
        raise ValueError('original USD authoring advanced physics')
    sim.reset()
    batch.bind()
    members = []
    for side in regex_task_setup__SIDES:
        hand = batch.sides[side].hand
        for (kind, value) in (('view', hand.root_physx_view), ('permanent', hand.permanent_wrench_composer), ('instantaneous', hand.instantaneous_wrench_composer)):
            trace.watch(f'regex_{side}_{kind}', value, owner='regex_task_setup', release='owned scope then SDK STOP', callbacks=True)
            if kind != 'view':
                members.append(value)
    owners.keep('composer', _Composers(members))
    assets.validate_simulation_settings(sim, prepared)
    if sim.current_time_step_index != 2:
        raise ValueError('one SDK reset with exactly two initialization ticks required')
    initial = _seed_tensors(batch, effective_seed, origins)
    initial.update(restore_regex_seed(sim, batch, initial, active_effort_limit_nm=effort))
    bindings = _actual_bindings(batch, archive.mesh_plan['bindings'])
    (positions, rotations) = _mesh_poses(archive, bindings, initial)
    (initial['mesh_positions_world_m'], initial['mesh_quaternions_xyzw']) = (positions, rotations)
    initial['rod_positions_w'] = regex_task_setup__torch.as_tensor(rod_nodes, device=initial['q'].device, dtype=regex_task_setup__torch.float32)[None] + initial['env_origins'].float()[:, None]
    batch.task_setup_metadata = dict(style=archive.style, source_seed_sha256=original_seed_sha, effective_seed_sha256=effective_seed['sha256'], height_derivation=height_proof, active_effort_limit_nm=effort, original_colliders_per_lane=34, sdk_initialization_steps=2, setup_physics_steps=0, native_setup_requalified=False, teacher_admission=False)
    return (batch, effective_seed, rod_nodes, bindings, initial)
from dh116.hand.kinematics import ANGULAR_JOINT_NAMES

def read_cached_side(owner, env_ids=None, *, reader):
    """Read with a startup-verified reader bound to this exact side owner."""
    if reader.owner is not owner:
        raise ValueError('cached readback reader belongs to a different owner')
    return reader.read(env_ids)

def restore_regex_seed(sim, batch, initial, *, env_ids=None, active_effort_limit_nm):
    """Zero-step original state restore; peers keep targets and both composers.

    initial is the prepare_regex_task tensor packet. Public IDs are its stable
    contiguous task row identities. This restores only hand mechanics: caller
    must reset rod, mesh poses, reaction ownership, controller and history in
    its same-boundary task transaction. No rollback is attempted on SDK faults.
    """
    n = len(batch.env_ids)
    if batch.env_ids != tuple(range(n)):
        raise ValueError('stable contiguous public task environment identities required')
    device = batch.sides['left'].device
    ids = initial['env_ids'] if env_ids is None else regex_task_setup__torch.as_tensor(env_ids, device=device)
    if ids.dtype not in (regex_task_setup__torch.int32, regex_task_setup__torch.int64):
        raise ValueError('integer selected environment identities required')
    ids = ids.to(dtype=regex_task_setup__torch.long)
    for side in regex_task_setup__SIDES:
        batch.sides[side].rows(ids)
    before = sim.current_time_step_index
    for (index, side) in enumerate(regex_task_setup__SIDES):
        batch.sides[side].reset(env_ids=ids, root_pose_wxyz=initial['seed_root_pose_w'].index_select(0, ids)[:, index], joint_pos=initial['seed_q'].index_select(0, ids)[:, index], joint_vel=initial['seed_qd'].index_select(0, ids)[:, index], active_effort_limit_nm=active_effort_limit_nm)
    batch.invalidate_readback(reason='original zero-step selected Reset state writes')
    sim.physics_sim_view.update_articulations_kinematic()
    batch.update(DT)
    if sim.current_time_step_index != before:
        raise ValueError('original hand restore unexpectedly advanced SDK physics')
    return capture_regex_state(batch, body_com_pose_b_xyzw=initial.get('body_com_pose_b_xyzw'))
from dh116.hand.kinematics import JOINT_NAMES

def capture_regex_state(batch, *, body_com_pose_b_xyzw=None):
    """Own current device state, reusing the immutable actual local COM cache.

    Omit the cache once at Startup to acquire get_coms. Pass that same verified
    cache for subsequent captures/Reset: the original task never changes COM.
    World COM always uses the freshly measured raw link quaternion and pose.
    """
    states = batch.read()
    result = {key: regex_task_setup__torch.stack([states[side][key] for side in regex_task_setup__SIDES], dim=1) for key in ('q', 'qd', 'backend_targets', 'root_pose_wxyz', 'root_pose_xyzw', 'root_velocity', 'link_pose_wxyz', 'link_pose_xyzw')}
    result['body_com_velocity_w'] = regex_task_setup__torch.stack([states[side]['link_velocity'] for side in regex_task_setup__SIDES], 1)
    if body_com_pose_b_xyzw is None:
        body_com_pose_b_xyzw = regex_task_setup__torch.stack([batch.sides[side].hand.root_physx_view.get_coms().to(device=batch.sides[side].device).index_select(0, batch.sides[side].row_indices) for side in regex_task_setup__SIDES], 1)
    if not isinstance(body_com_pose_b_xyzw, regex_task_setup__torch.Tensor) or body_com_pose_b_xyzw.shape != result['link_pose_xyzw'].shape or body_com_pose_b_xyzw.device != result['q'].device or (body_com_pose_b_xyzw.dtype != result['link_pose_xyzw'].dtype):
        raise ValueError('same-device actual original local COM pose cache required')
    result['body_com_pose_b_xyzw'] = body_com_pose_b_xyzw
    (link, local) = (result['link_pose_xyzw'].double(), result['body_com_pose_b_xyzw'].double())
    (xyz, w) = (link[..., 3:6], link[..., 6:7])
    offset = local[..., :3]
    cross = 2 * regex_task_setup__torch.linalg.cross(xyz, offset)
    position = link[..., :3] + offset + w * cross + regex_task_setup__torch.linalg.cross(xyz, cross)
    (local_xyz, local_w) = (local[..., 3:6], local[..., 6:7])
    quaternion_xyz = w * local_xyz + local_w * xyz + regex_task_setup__torch.linalg.cross(xyz, local_xyz)
    quaternion_w = w * local_w - (xyz * local_xyz).sum(dim=-1, keepdim=True)
    result['body_com_pose_w'] = regex_task_setup__torch.cat((position, quaternion_w, quaternion_xyz), dim=-1)
    result['base_link_pose_w'] = regex_task_setup__torch.stack([result['link_pose_wxyz'][:, index, batch.sides[side].body_names.index('base_link')] for (index, side) in enumerate(regex_task_setup__SIDES)], 1)
    result['base_link_com_velocity_w'] = regex_task_setup__torch.stack([result['body_com_velocity_w'][:, index, batch.sides[side].body_names.index('base_link')] for (index, side) in enumerate(regex_task_setup__SIDES)], 1)
    return result
regex_artic__SIDES = ('left', 'right')

def _seed_tensors(batch, seed, origins):
    device = batch.sides['left'].device
    n = len(batch.env_ids)
    roots = regex_task_setup__torch.tensor(seed['authored_root_seed_w'], device=device, dtype=regex_task_setup__torch.float64)[None].repeat(n, 1, 1)
    origin = regex_task_setup__torch.tensor(origins, device=device, dtype=regex_task_setup__torch.float64)
    roots[..., :3] += origin[:, None]
    reset_roots = regex_task_setup__torch.tensor(seed['reset_root_pose_w'], device=device, dtype=regex_task_setup__torch.float64)[None].repeat(n, 1, 1)
    reset_roots[..., :3] += origin[:, None]
    (q, qd) = ([], [])
    for (index, side) in enumerate(regex_task_setup__SIDES):
        names = batch.sides[side].joint_names
        values = regex_task_setup__torch.zeros((n, 17), device=device, dtype=regex_task_setup__torch.float32)
        velocity = regex_task_setup__torch.zeros_like(values)
        ids = [names.index(name) for name in regex_task_setup__JOINTS]
        values[:, ids] = regex_task_setup__torch.tensor(seed['joint_positions'][index], device=device, dtype=regex_task_setup__torch.float32)
        velocity[:, ids] = regex_task_setup__torch.tensor(seed['joint_velocities'][index], device=device, dtype=regex_task_setup__torch.float32)
        q.append(values)
        qd.append(velocity)
    return dict(env_ids=regex_task_setup__torch.arange(n, device=device), env_origins=origin, root_seed_w=roots, common_lift_offset_m=regex_task_setup__torch.full((n,), seed['initial_offset_m'], device=device, dtype=regex_task_setup__torch.float64), seed_root_pose_w=reset_roots.float(), seed_q=regex_task_setup__torch.stack(q, 1), seed_qd=regex_task_setup__torch.stack(qd, 1), palm_offsets_m=regex_task_setup__torch.zeros((n, 2, 3), device=device, dtype=regex_task_setup__torch.float64), angles_rad=regex_task_setup__torch.zeros((n, 2, 3), device=device, dtype=regex_task_setup__torch.float64))

def author_original_scene(sim, args, original_layout, sources, prefix, origin):
    """Author physical USD through the shared, policy-independent scene builder."""
    from dh116.env.scene import author_scene
    return author_scene(sim, args, original_layout, sources, prefix, origin)

def _actual_bindings(batch, archived):
    from dh116.env.scene import contact_links
    contact_links(archived)
    result = regex_task_setup__deepcopy(archived)
    for (key, row) in result.items():
        (side, name) = (row['side'], row['body_name'])
        owner = batch.sides[side]
        if key != side + '.' + name or owner.body_names.index(name) != row['body_index']:
            raise ValueError('actual original contact-link body ordering differs from source archive')
        rows = owner.row_indices.detach().cpu().tolist()
        paths = tuple((owner.hand.root_physx_view.link_paths[index][row['body_index']] for index in rows))
        expected = tuple((prefix + '/' + side + 'Hand/' + name for prefix in batch.prefixes))
        if paths != expected:
            raise ValueError('actual original contact link paths differ from authored task lanes')
        (row['prim_path'], row['prim_paths']) = (paths[0], paths)
    return result

def spawn_regex_articulations(sim, args, original_layout, sources, env_origins, *, env_ids=None, env_root='/World/envs'):
    """Author original physical scenes, then construct exactly two SDK assets.

    ``author_original_scene`` executes the unchanged protected spawn_scene code
    with a private constructor override. It authors the original physical USDs,
    mount/material/mimic and returns two configs in ``hands``; no protected
    signature or process-global SDK constructor is changed.

    Call ``sim.reset()`` once outside this function, then ``batch.bind()``.
    Reset seeds must be supplied explicitly: the single regex config's default
    root state is not a substitute for each authored environment's world pose.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from pxr import UsdPhysics
    origins = regex_artic__np.asarray(env_origins, dtype=float)
    if origins.ndim != 2 or origins.shape[1] != 3 or (not len(origins)) or (not regex_artic__np.isfinite(origins).all()):
        raise ValueError('finite nonempty environment origins [N,3] required')
    ids = tuple(range(len(origins))) if env_ids is None else tuple(env_ids)
    if len(ids) != len(origins) or len(set(ids)) != len(ids) or any((type(value) is not int or value < 0 for value in ids)):
        raise ValueError('unique nonnegative integer environment identities required')
    if not isinstance(env_root, str) or re.fullmatch('/(?:[A-Za-z_][A-Za-z_0-9]*/)*[A-Za-z_][A-Za-z_0-9]*', env_root) is None:
        raise ValueError('literal absolute USD environment root required')
    prefixes = tuple((f'{env_root}/env_{value}' for value in ids))
    scenes = []
    suffixes = {side: [] for side in regex_artic__SIDES}
    for (prefix, origin) in zip(prefixes, origins, strict=True):
        scene = author_original_scene(sim, args, original_layout, sources, prefix, origin)
        if len(scene['hands']) != 2 or scene['wire'] is not None or scene['sensors'] or (scene['prefix'] != prefix):
            raise ValueError('original bilateral author-only scene without duplicate wire/sensors required')
        for (side, cfg) in zip(regex_artic__SIDES, scene['hands'], strict=True):
            path = f'{prefix}/{side}Hand'
            if cfg.prim_path != path or cfg.spawn is None:
                raise ValueError('unchanged original authored hand configuration required')
            roots = sim_utils.get_all_matching_child_prims(path, predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI), stage=sim.stage, traverse_instance_prims=False)
            if len(roots) != 1:
                raise ValueError('exactly one actual USD ArticulationRootAPI per hand required')
            root_path = roots[0].GetPath().pathString
            if root_path != path and (not root_path.startswith(path + '/')):
                raise ValueError('actual articulation root must belong to its authored hand')
            suffixes[side].append(root_path[len(path):])
        scenes.append(scene)
    if any((len(set(suffixes[side])) != 1 for side in regex_artic__SIDES)):
        raise ValueError('one identical physical root layout per side required')
    hands = []
    for (index, side) in enumerate(regex_artic__SIDES):
        cfg = scenes[0]['hands'][index].copy()
        cfg.prim_path = f'{env_root}/env_.*/{side}Hand'
        cfg.spawn = None
        cfg.articulation_root_prim_path = suffixes[side][0] or None
        hands.append(Articulation(cfg))
    return RegexArticulationBatch(hands, ids, prefixes, tuple((suffixes[side][0] for side in regex_artic__SIDES)), scenes)

def _mesh_poses(archive, bindings, initial):
    entries = archive.mesh_plan['entries']
    names = tuple((entry['name'] for entry in entries))
    if len(set(names)) != len(names) or not set(bindings).issubset(names):
        raise ValueError('one original native mesh slot for each actual hand contact binding required')
    (device, n) = (initial['q'].device, len(initial['q']))
    position = regex_task_setup__torch.empty((n, len(entries), 3), device=device, dtype=regex_task_setup__torch.float32)
    rotation = regex_task_setup__torch.empty((n, len(entries), 4), device=device, dtype=regex_task_setup__torch.float32)
    pose_key = 'table_pose' if archive.style == 'pinch' else 'static_pose'
    for (slot, entry) in enumerate(entries):
        if entry['name'] in bindings:
            row = bindings[entry['name']]
            side = regex_task_setup__SIDES.index(row['side'])
            pose = initial['link_pose_xyzw'][:, side, row['body_index']]
            (position[:, slot], rotation[:, slot]) = (pose[:, :3], pose[:, 3:])
        else:
            (xyz, quat) = entry[pose_key]
            position[:, slot] = regex_task_setup__torch.tensor(xyz, device=device, dtype=regex_task_setup__torch.float32) + initial['env_origins'].float()
            rotation[:, slot] = regex_task_setup__torch.tensor(quat, device=device, dtype=regex_task_setup__torch.float32)
    if not bool(regex_task_setup__torch.isfinite(position).all()) or not bool(regex_task_setup__torch.isfinite(rotation).all()) or (not regex_task_setup__torch.allclose(rotation.norm(dim=-1), regex_task_setup__torch.ones_like(rotation[..., 0]), atol=2e-06, rtol=0)):
        raise ValueError('finite actual hand and unchanged static native mesh poses required')
    return (position, rotation)

class RegexArticulationSide:
    """One SDK Articulation, with public environment IDs mapped to view rows."""

    def __init__(self, hand, env_ids, prefixes, root_suffix):
        self.hand = hand
        self.env_ids = tuple(env_ids)
        self.prefixes = tuple(prefixes)
        self.device = regex_artic__torch.device(hand.device)
        self.joint_names = tuple(hand.joint_names)
        self.body_names = tuple(hand.body_names)
        if not hand.is_initialized or len(self.env_ids) != len(self.prefixes) or (not self.env_ids) or (len(set(self.env_ids)) != len(self.env_ids)) or (len(set(self.prefixes)) != len(self.prefixes)):
            raise ValueError('initialized view and unique environment identity layout required')
        expected_names = set(regex_artic__JOINTS + JOINT_NAMES + ANGULAR_JOINT_NAMES)
        if len(self.joint_names) != 17 or set(self.joint_names) != expected_names or len(self.body_names) != 23 or (len(set(self.body_names)) != 23):
            raise ValueError('original 17DOF/23-body mounted DH116 hand required')
        paths = tuple(hand.root_physx_view.prim_paths)
        pattern = re.compile('^(.+)/(left|right)Hand(?:' + re.escape(root_suffix) + ')?$')
        parsed = [pattern.fullmatch(path) for path in paths]
        if len(paths) != len(self.env_ids) or hand.num_instances != len(paths) or any((match is None for match in parsed)) or (len({match.group(2) for match in parsed}) != 1):
            raise ValueError('bijective actual articulation identity per environment required')
        prefix_to_id = dict(zip(self.prefixes, self.env_ids, strict=True))
        native_prefixes = tuple((match.group(1) for match in parsed))
        if len(set(native_prefixes)) != len(paths) or set(native_prefixes) != set(self.prefixes):
            raise ValueError('bijective actual articulation identity per environment required')
        self.side = parsed[0].group(2)
        self.row_env_ids = tuple((prefix_to_id[prefix] for prefix in native_prefixes))
        row_for_id = {value: row for (row, value) in enumerate(self.row_env_ids)}
        self.row_indices = regex_artic__torch.tensor([row_for_id[value] for value in self.env_ids], device=self.device, dtype=regex_artic__torch.long)
        ordered = sorted(row_for_id)
        self._sorted_env_ids = regex_artic__torch.tensor(ordered, device=self.device, dtype=regex_artic__torch.long)
        self._sorted_rows = regex_artic__torch.tensor([row_for_id[value] for value in ordered], device=self.device, dtype=regex_artic__torch.long)
        self.active_ids = [self.joint_names.index(name) for name in ACTIVE]
        self.mount_ids = [self.joint_names.index(name) for name in JOINT_NAMES + ANGULAR_JOINT_NAMES]

    def rows(self, env_ids=None):
        """Return actual PhysX rows in the caller's supplied environment order."""
        if env_ids is None:
            return self.row_indices
        values = regex_artic__torch.as_tensor(env_ids, device=self.device)
        if values.dtype not in (regex_artic__torch.int32, regex_artic__torch.int64):
            raise ValueError('integer environment identities required')
        values = values.to(dtype=regex_artic__torch.long)
        if values.ndim != 1 or not len(values):
            raise ValueError('nonempty environment identity vector required')
        if len(regex_artic__torch.unique(values)) != len(values):
            raise ValueError('unique environment identities required')
        index = regex_artic__torch.searchsorted(self._sorted_env_ids, values)
        safe = index.clamp_max(len(self._sorted_env_ids) - 1)
        if bool(((index >= len(self._sorted_env_ids)) | (self._sorted_env_ids[safe] != values)).any()):
            raise ValueError('unknown environment identity')
        return self._sorted_rows[index]

    def read(self, env_ids=None):
        """Own actual backend q/qd/root/link/targets; quaternions are explicit."""
        rows = self.rows(env_ids)
        view = self.hand.root_physx_view
        getters = dict(q='get_dof_positions', qd='get_dof_velocities', backend_targets='get_dof_position_targets', root_pose_xyzw='get_root_transforms', root_velocity='get_root_velocities', link_pose_xyzw='get_link_transforms', link_velocity='get_link_velocities')
        state = {key: getattr(view, method)().index_select(0, rows) for (key, method) in getters.items()}
        order = [0, 1, 2, 6, 3, 4, 5]
        state['root_pose_wxyz'] = state['root_pose_xyzw'][..., order]
        state['link_pose_wxyz'] = state['link_pose_xyzw'][..., order]
        return state

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        """Stage one native batched target; call batch.write_data_to_sim next."""
        rows = self.rows(env_ids)
        count = len(self.joint_names) if joint_ids is None else len(joint_ids)
        value = self._tensor(target, (len(rows), count), 'joint target')
        self.hand.set_joint_position_target(value, joint_ids=joint_ids, env_ids=rows)

    def set_wrenches(self, *, forces=None, torques=None, positions=None, body_ids=None, env_ids=None, is_global=False, composer='permanent', add=False):
        """One native composer call for a complete side, including selected envs.

        Positions and global/local frames have the original WrenchComposer
        semantics. This method does not negate, redistribute or clip rod loads.
        Incoming reaction forces already carry their physical sign.
        """
        if composer not in ('permanent', 'instantaneous'):
            raise ValueError('permanent or instantaneous composer required')
        rows = self.rows(env_ids)
        count = len(self.body_names) if body_ids is None else len(body_ids)
        shape = (len(rows), count, 3)
        values = {name: None if value is None else self._tensor(value, shape, name).contiguous() for (name, value) in (('forces', forces), ('torques', torques), ('positions', positions))}
        native = getattr(self.hand, composer + '_wrench_composer')
        method = native.add_forces_and_torques if add else native.set_forces_and_torques
        method(**values, body_ids=body_ids, env_ids=rows, is_global=is_global)

    def reset(self, *, env_ids, root_pose_wxyz, joint_pos, joint_vel, active_effort_limit_nm):
        """Restore selected original hand rows without stepping or global write.

        Full 17-DOF inputs must retain all eleven authored finger q/qd values and
        zero the six wrist coordinates. Effort and stiffness match the original
        Reset transaction. Controller/filter/contact/rod reset remains the
        caller's transaction; this method only owns the articulation rows.
        """
        rows = self.rows(env_ids)
        pose = self._tensor(root_pose_wxyz, (len(rows), 7), 'reset WXYZ root')
        q = self._tensor(joint_pos, (len(rows), 17), 'reset q')
        qd = self._tensor(joint_vel, (len(rows), 17), 'reset qd')
        if not all((bool(regex_artic__torch.isfinite(value).all()) for value in (pose, q, qd))) or not regex_artic__torch.allclose(pose[:, 3:].norm(dim=-1), regex_artic__torch.ones_like(pose[:, 0]), atol=1e-06, rtol=0) or bool((q[:, self.mount_ids] != 0).any()) or bool((qd[:, self.mount_ids] != 0).any()) or (not regex_artic__math.isfinite(active_effort_limit_nm)) or (active_effort_limit_nm <= 0):
            raise ValueError('finite original reset values, normalized roots and zero wrist state required')
        (hand, view) = (self.hand, self.hand.root_physx_view)
        hand.reset(rows)
        hand.write_joint_state_to_sim(q, qd, env_ids=rows)
        hand.write_root_pose_to_sim(pose, env_ids=rows)
        hand.write_root_velocity_to_sim(q.new_zeros((len(rows), 6)), env_ids=rows)
        hand.set_joint_position_target(q, env_ids=rows)
        stiffness = q.new_full((len(rows), len(self.active_ids)), 10.0)
        hand.write_joint_stiffness_to_sim(stiffness, joint_ids=self.active_ids, env_ids=rows)
        hand.actuators['active'].stiffness[rows] = 10.0
        hand.set_joint_velocity_target(regex_artic__torch.zeros_like(q), env_ids=rows)
        hand.set_joint_effort_target(regex_artic__torch.zeros_like(q), env_ids=rows)
        limits = q.new_full((len(rows), len(self.active_ids)), active_effort_limit_nm)
        hand.write_joint_effort_limit_to_sim(limits, joint_ids=self.active_ids, env_ids=rows)
        view.set_dof_position_targets(hand.data.joint_pos_target, indices=rows)
        view.set_dof_velocity_targets(hand.data.joint_vel_target, indices=rows)
        view.set_dof_actuation_forces(hand.data.joint_effort_target, indices=rows)
        zeros = q.new_zeros((hand.num_instances, len(self.body_names), 3))
        view.apply_forces_and_torques_at_position(force_data=zeros.view(-1, 3), torque_data=zeros.view(-1, 3), position_data=None, indices=rows, is_global=False)

    def _tensor(self, value, shape, label):
        tensor = regex_artic__torch.as_tensor(value, device=self.device, dtype=regex_artic__torch.float32)
        if tuple(tensor.shape) != shape:
            raise ValueError(f'{label} shape {shape} required')
        return tensor

class _Composers:
    """Existing run_owned_call resets this group before its CUDA synchronize."""

    def __init__(self, members):
        try:
            self.members = tuple(members)
        finally:
            self = members = None

    def reset(self):
        import traceback
        member = None
        failures = []
        try:
            for member in self.members:
                try:
                    member.reset()
                except BaseException as error:
                    failures.append(traceback.format_exc())
                    traceback.clear_frames(error.__traceback__)
                    error.__traceback__ = None
            if failures:
                raise RuntimeError(chr(10).join(failures)) from None
        finally:
            self = member = None

class RegexArticulationBatch:
    """Own the two original SDK callbacks and their startup-only scene records."""

    def __init__(self, hands, env_ids, prefixes, root_suffixes, scenes):
        self.hands = tuple(hands)
        if len(self.hands) != 2:
            raise ValueError('exactly two SDK Articulations required')
        self.env_ids = tuple(env_ids)
        self.prefixes = tuple(prefixes)
        self.root_suffixes = tuple(root_suffixes)
        self.scenes = tuple(scenes)
        self.sides = {}
        self.readback_mode = os.environ.get('DH116_NATIVE_READBACK_BACKEND', 'raw')
        if self.readback_mode not in ('raw', 'official_cache'):
            raise ValueError('explicit raw or official_cache readback backend required')
        self._cached_readers = {}
        self._readback_audit_enabled = os.environ.get('DH116_NATIVE_READBACK_AUDIT', '0') == '1'
        if self._readback_audit_enabled and (self.readback_mode != 'official_cache' or len(self.env_ids) != 2):
            raise ValueError('native cache oracle audit requires explicit official_cache N2 diagnostic')
        self._readback_audits = []

    def bind(self):
        """Resolve row identity once after the caller's single SDK reset."""
        if not self.sides:
            sides = {side: RegexArticulationSide(hand, self.env_ids, self.prefixes, suffix) for (side, hand, suffix) in zip(regex_artic__SIDES, self.hands, self.root_suffixes, strict=True)}
            if any((owner.side != side for (side, owner) in sides.items())):
                raise ValueError('actual bilateral hand identity differs')
            self.sides = sides
        return self

    def read(self, env_ids=None):
        if self.readback_mode == 'raw':
            return {side: self.sides[side].read(env_ids) for side in regex_artic__SIDES}
        if not self._cached_readers:
            _regex_artic_import__CachedSideReadback = CachedSideReadback
            self._cached_readers = {side: _regex_artic_import__CachedSideReadback(self.sides[side]) for side in regex_artic__SIDES}
        result = {}
        for side in regex_artic__SIDES:
            before = self.sides[side].read(env_ids) if self._readback_audit_enabled else None
            result[side] = self._cached_readers[side].read(env_ids)
            if self._readback_audit_enabled:
                after = self.sides[side].read(env_ids)
                mismatches = []
                for (key, value) in result[side].items():
                    if any((value.shape != raw[key].shape or value.dtype != raw[key].dtype or (not regex_artic__torch.equal(value.contiguous().view(regex_artic__torch.uint8), raw[key].contiguous().view(regex_artic__torch.uint8))) for raw in (before, after))):
                        mismatches.append(key)
                self._readback_audits.append(dict(side=side, fields=list(result[side]), mismatches=mismatches))
                if mismatches:
                    raise ValueError('official cache/raw SDK frame mismatch: ' + side + ':' + ','.join(mismatches))
        return result

    def readback_audit(self):
        return dict(enabled=self._readback_audit_enabled, diagnostic_only=True, side_snapshots=len(self._readback_audits), mismatches=[row for row in self._readback_audits if row['mismatches']], extra_raw_reads_per_side_snapshot=2 if self._readback_audit_enabled else 0)

    def invalidate_readback(self, *, reason):
        """Fresh official acquisition after this owner's zero-step state writes."""
        for reader in self._cached_readers.values():
            reader.invalidate(reason=reason)

    def readback_metadata(self):
        return dict(mode=self.readback_mode, sides={side: reader.metadata() for (side, reader) in self._cached_readers.items()})

    def write_data_to_sim(self):
        """One SDK write per side; called once before the shared physics step."""
        for hand in self.hands:
            hand.write_data_to_sim()

    def update(self, dt):
        """One SDK cache update per side; this never advances physics."""
        for hand in self.hands:
            hand.update(dt)

    def reset_composers(self, env_ids=None):
        """Clear both native composers per side with the same row mapping."""
        for side in regex_artic__SIDES:
            owner = self.sides[side]
            rows = None if env_ids is None else owner.rows(env_ids)
            owner.hand.permanent_wrench_composer.reset(rows)
            owner.hand.instantaneous_wrench_composer.reset(rows)
