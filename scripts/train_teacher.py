"""Train the native Teacher or replay a frozen checkpoint with offscreen cameras."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from functools import partial
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4

from dh116.env.contracts import (
    PROJECT_ROOT,
    ProjectPaths,
    action_sha256,
    checked_file,
    runti_io__digest as digest,
    runti_io__file_sha256 as file_sha256,
    source_file,
    write_new,
)
from dh116.rl.training import TEACHER_ENV_COUNTS, train_teacher_main

_EPISODE_WINDOWS = 600
_PREVIEW_LANES = 2
_CHECKPOINT_LANES = TEACHER_ENV_COUNTS
_ACTOR_MODULES = ('actor', 'actor_rod', 'actor_context', 'actor_head')
_CAMERA_CONFIG = PROJECT_ROOT / 'configs/cameras/student_stereo.json'
_GEOMETRY_BACKEND = 'cuda_cabi'
_READBACK_BACKEND = 'official_cache'


def freeze_replay_actor(model, policy, checkpoint_model):
    """Load only the inference modules, rebinding the physical lane count."""
    import torch

    from dh116.rl.student_data import tensor_state_sha256

    def unbatched_identity(identity, batch_size):
        fields = dict(identity)
        checkpoint_digest = fields.pop('model_identity_sha256', None)
        if digest(fields) != checkpoint_digest or fields.pop('parent_num_envs', None) != batch_size:
            raise ValueError('frozen Teacher model digest or native lane count differs')
        if fields.get('action_contract_sha256') != action_sha256():
            raise ValueError('frozen Teacher action contract differs')
        return fields

    if policy.get('_extra_state') != checkpoint_model:
        raise ValueError('checkpoint model identity and policy metadata differ')
    checkpoint_lanes = checkpoint_model.get('parent_num_envs')
    if type(checkpoint_lanes) is not int or checkpoint_lanes not in _CHECKPOINT_LANES:
        raise ValueError(f'frozen Teacher checkpoint must declare a supported size {_CHECKPOINT_LANES}')
    if unbatched_identity(checkpoint_model, checkpoint_lanes) != unbatched_identity(model.model_identity, _PREVIEW_LANES):
        raise ValueError('checkpoint and live Teacher model identities differ')
    prefixes = tuple(name + '.' for name in _ACTOR_MODULES)
    actor_weights = {name: tensor for name, tensor in policy.items() if name.startswith(prefixes)}
    expected = {name: tensor for name, tensor in model.state_dict().items() if name.startswith(prefixes)}
    if actor_weights.keys() != expected.keys() or any(
        not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
        or value.dtype != expected[name].dtype
        for name, value in actor_weights.items()
    ):
        raise ValueError('complete checkpoint Actor tensor topology required')
    weights_sha256 = tensor_state_sha256(actor_weights)
    for name in _ACTOR_MODULES:
        module = getattr(model, name)
        prefix = name + '.'
        module.load_state_dict({key[len(prefix):]: value for key, value in actor_weights.items()
                                if key.startswith(prefix)}, strict=True)
    model.requires_grad_(False).eval()
    return weights_sha256


def _checkpoint(args, scene_sha256, profile):
    import torch

    from dh116.rl.device import _KEYS, _SOURCES, tenso_check__VERSION
    from dh116.rl.training import _checkpoint_context, native_recipe

    packet = torch.load(io.BytesIO(checked_file(args.preview_checkpoint, args.preview_checkpoint_sha256)),
                        map_location='cpu', weights_only=True)
    if not isinstance(packet, dict) or set(packet) not in (_KEYS, _KEYS | {'mastery'}) or packet['version'] != tenso_check__VERSION:
        raise ValueError('current native Teacher checkpoint required')
    identity = packet['identity']
    update = packet['counters']['updates']
    checkpoint_lanes = identity['num_envs']
    if (type(update) is not int or update < 1 or packet['policy_version'] != update
            or type(checkpoint_lanes) is not int or checkpoint_lanes not in _CHECKPOINT_LANES
            or identity['model']['parent_num_envs'] != checkpoint_lanes
            or identity['receipt_sha256'] != scene_sha256 or identity['model']['grasp'] != args.grasp):
        raise ValueError(f'updated Teacher checkpoint with size {_CHECKPOINT_LANES} for this scene required')
    expected = {name: file_sha256(source_file(name)) for name in (*_SOURCES, 'rl/training.py')}
    if identity['source_sha256'] != expected or identity['backend_profile'] != asdict(profile):
        raise ValueError('frozen Teacher execution sources or backend differ from the current runtime')
    from dh116.rl.reward import load_config
    from dh116.rl.training import dof_action_indices
    active = identity.get('policy_action_indices')
    participating = load_config(
        PROJECT_ROOT / 'configs/training' / f'reward_{args.grasp}.json', full_grasp=True,
    ).finger_goal.participating
    dof_mode = 'full24' if active is None else ('grasp_lift' if active[-2:] == [14, 17] else 'finger_only')
    if active is not None and tuple(active) != dof_action_indices(dof_mode, participating):
        raise ValueError('checkpoint DoF indices differ from its grasp')
    variants = (False, True) if dof_mode == 'grasp_lift' else (False,)
    native_shas = [native_recipe(
        args.grasp, receipt_sha256=scene_sha256, num_envs=checkpoint_lanes,
        episode_windows=_EPISODE_WINDOWS, scene_build_mode='clone', full_grasp=True,
        dof_mode=dof_mode, gated_lift=gated,
    )[1] for gated in variants]
    declared = SimpleNamespace(
        model=SimpleNamespace(model_identity=identity['model']),
        rollout_steps=identity['rollout_steps'], num_envs=checkpoint_lanes,
    )
    for gated, native_sha in zip(variants, native_shas, strict=True):
        recipe = SimpleNamespace(
            episode_windows=_EPISODE_WINDOWS, native_recipe_sha256=native_sha,
            receipt_sha256=scene_sha256, policy_seed=identity['policy_seed'],
        )
        current = _checkpoint_context(declared, recipe, profile, {'sha256': identity['seed_sha256']})
        if identity['recipe_sha256'] == current['recipe_sha256']:
            if gated != ('mastery' in packet):
                raise ValueError('checkpoint mastery field differs from the replay recipe')
            return dict(packet, replay_dof_mode=dof_mode, replay_gated_lift=gated)
    raise ValueError('frozen Teacher checkpoint recipe differs from the current supported scene')


def _author_preview(stage, batch, supports):
    from pxr import Gf, Usd, UsdGeom, UsdLux

    prefix, peer = batch.prefixes
    UsdGeom.Imageable(stage.GetPrimAtPath(peer)).MakeInvisible()
    table = stage.GetPrimAtPath(prefix + '/Table')
    surfaces = [prim for prim in Usd.PrimRange(table) if prim.IsA(UsdGeom.Gprim)]
    if not surfaces or len(supports) != 2:
        raise ValueError('the original table and both supported-scene prims are required')
    for prim in surfaces:
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([(0.52, 0.52, 0.52)])
    for support in supports:
        prim = stage.GetPrimAtPath(prefix + '/EndSupports/' + support['name'])
        if not prim.IsA(UsdGeom.Cube):
            raise ValueError('original end support cube is missing')
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([(0.10, 0.10, 0.10)])

    UsdGeom.Xform.Define(stage, '/World/DH116Preview')
    dome = UsdLux.DomeLight.Define(stage, '/World/DH116Preview/Fill')
    dome.CreateIntensityAttr(850.0)
    key = UsdLux.DistantLight.Define(stage, '/World/DH116Preview/Key')
    key.CreateIntensityAttr(1850.0)
    key.CreateAngleAttr(12.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(35, -32, 20))

    wire = UsdGeom.Mesh.Define(stage, '/World/DH116Preview/RodSurface')
    wire.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    wire.CreateDisplayColorAttr([(0.72, 0.17, 0.018)])
    return wire


def _publish_wire(wire, nodes, radius):
    from pxr import Gf, Vt

    from dh116.env.cameras import rod_preview_mesh

    points, faces = rod_preview_mesh(nodes, radius)
    if not wire.GetFaceVertexCountsAttr().HasAuthoredValueOpinion():
        wire.CreateFaceVertexCountsAttr([len(face) for face in faces])
        wire.CreateFaceVertexIndicesAttr([vertex for face in faces for vertex in face])
    wire.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    wire.CreateExtentAttr().Set([Gf.Vec3f(*row.tolist()) for row in (points.min(0), points.max(0))])


def _snapshot(collector, calibration_sha256, *, reset_sdk_timestamp_s):
    from dh116.rl.student_data import tensor_state_sha256

    stamp = collector.source._boundary()
    measured = {name: collector.state[name][:1] for name in ('q', 'qd', 'base_link_pose_w')}
    measured.update(nodes=collector.nodes[:1], velocity=collector.velocity[:1],
                    previous_action=collector.source.previous_request[:1])
    rest_sha256 = tensor_state_sha256({'rest_lengths_m': collector.source.rest_lengths_m[:1]})
    return dict(run_uuid=stamp.run_uuid, env_id=0, episode_uid=stamp.episode_uids[0],
                reset_id=stamp.reset_ids[0], policy_tick=collector.controls,
                physics_tick_start=stamp.sdk_step, physics_tick_end=stamp.sdk_step,
                rod_step=collector.runtime.physics_step, rod_tick=stamp.rod_ticks[0],
                sdk_step=stamp.sdk_step, sdk_timestamp_s=stamp.sdk_time_s, solver_time_s=stamp.solver_time_s,
                material_reference_uid=f'{stamp.episode_uids[0]}:rest:{rest_sha256}',
                rod_uid=stamp.rod_uids[0], state_sha256=tensor_state_sha256(measured),
                simulation_timestamp=stamp.sdk_time_s - reset_sdk_timestamp_s,
                calibration_sha256=calibration_sha256)


def _export_frame(output, collector, wire, stereo, overview, snapshot):
    import torch
    from PIL import Image

    from dh116.env.cameras import brown_rod_mask, student_rgb_frame

    before = snapshot()
    state = collector.measurement()
    nodes = state.rod_positions_w[0].float().cpu().numpy()
    _publish_wire(wire, nodes, collector.radius)
    measured = state.state
    joints = [measured[field][:1, side, indices]
              for field in ('q', 'qd') for side, indices in enumerate(collector.ctrl.active)]
    proprio = torch.cat((*joints, measured['base_link_pose_w'][:1].reshape(1, 14),
                        collector.source.previous_request[:1]), dim=1).float()
    packet = stereo.capture()
    scene = overview.read()
    timing = packet['render_timing']
    if (packet['source_scope'] != 'native_tiled_reader' or scene is None
            or scene['native_evidence'] is not True or scene['raw_clock'] != timing['raw_left_clock']):
        raise RuntimeError('the SDK overview and both stereo cameras must share a fresh native render')
    if packet['identity'] != before or snapshot() != before:
        raise RuntimeError('preview rendering changed the measured SDK boundary')

    directory = output / f'window_{collector.controls:04d}'
    directory.mkdir()
    scores = {}
    for name, rgb in (('left', packet['left_rgb']), ('right', packet['right_rgb']), ('scene', scene['rgb'])):
        image = (rgb[0].permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
        with (directory / f'{name}.png').open('xb') as stream:
            Image.fromarray(image).save(stream, format='PNG')
        scores[name] = int(brown_rod_mask(image).sum())
    projected_nodes = {}
    for side in ('left', 'right'):
        transform = packet['T_world_' + side][0].cpu().double()
        intrinsic = packet['K_' + side][0].cpu().double()
        local = (torch.from_numpy(nodes).double() - transform[:3, 3]) @ transform[:3, :3]
        uvw = local @ intrinsic.T
        uv = uvw[:, :2] / uvw[:, 2:]
        height, width = packet[side + '_rgb'].shape[-2:]
        in_frame = ((local[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width)
                    & (uv[:, 1] >= 0) & (uv[:, 1] < height))
        projected_nodes[side] = int(in_frame.sum())
    raw = student_rgb_frame(packet, proprio)
    raw_path = directory / 'student_rgb.pt'
    with raw_path.open('xb') as stream:
        torch.save(raw, stream)
    write_new(directory / 'frame.json', dict(
        identity=before, render_timing=timing, render_passes=packet['render_passes'],
        scene_raw_clock=scene['raw_clock'], source_scope=packet['source_scope'],
        sdk_render_time_difference_s=timing['raw_left_clock']['simulation_time'] - timing['physical_timestamp_s'],
        raw_minus_absolute_sdk_timestamp_s=timing['raw_left_clock']['simulation_time'] - before['sdk_timestamp_s'],
        calibration_sha256=packet['calibration_sha256'],
        **{name: packet[name].cpu().tolist() for name in ('K_left', 'K_right', 'T_world_left', 'T_world_right')},
        scene_K=scene['K'].cpu().tolist(), scene_T_world=scene['T_world'].cpu().tolist(),
        brown_mask_pixels=scores, material_nodes_in_frame=projected_nodes,
        rod_midpoint_sag_m=float((nodes[0, 2] + nodes[-1, 2]) * 0.5 - nodes[20, 2]),
        palm_positions_w=measured['base_link_pose_w'][0, :, :3].cpu().tolist(),
        artifacts={path.name: file_sha256(path) for path in (*directory.glob('*.png'), raw_path)},
    ))
    return str(directory.relative_to(output))


def replay_detached_lift(reward, values, raw_lift):
    """Read the settled native lift only where the full-grasp contacts qualify."""
    eligible = reward._genuine_contact(values) & values['numerical_health'] & ~values['constraint_failure']
    eligible &= (values['soft_non_target_contact'] == 0) & (values['soft_reaction_force'] < 1.0)
    return raw_lift.masked_fill(~eligible, 0.0)


def _exercise_preview(sim, owners, trace, project, prepared, archive, lanes, output, gpu, end, *,
                      args, profile, reset, height, packet, calibration, readback_bytes, scene_center, horizontal_fov):
    import torch

    from dh116.env.cameras import FixedNativeCameraReader, TiledStereoCapture, look_at_optical
    from dh116.hand.articulation import prepare_regex_task
    from dh116.physics.solver import TensorRodOwner, make_solver
    from dh116.rl.student_data import tensor_state_sha256
    from dh116.rl.training import NativeTensorCollector, load_candidate

    center = torch.as_tensor(scene_center, dtype=torch.float64)
    authored_midpoint = torch.as_tensor(archive.scene_inputs['authored_rod_nodes'][20], dtype=torch.float64)
    if center.shape != (3,) or not torch.allclose(center, authored_midpoint, atol=2e-6, rtol=0.0):
        raise ValueError('fixed Student rig center differs from the authored rod midpoint')
    batch, seed, _, bindings, initial = prepare_regex_task(
        sim, owners, trace, project, prepared, archive, lanes, seed_height=height, scene_build_mode='clone',
    )
    names = tuple(row['name'] for row in archive.mesh_plan['entries'])
    if len(names) != 17 or packet['identity']['seed_sha256'] != seed['sha256']:
        raise ValueError('the checkpoint Reset and supported 17-mesh scene are required')
    solver = make_solver(load_candidate(profile), archive.mesh_plan, reset,
                         initial['env_origins'].float().cpu().numpy(), style=args.grasp,
                         capacity=41 * len(names) * 12, empty_lane=False)
    owners.keep('shared_rod_solver', solver)
    run_uuid = uuid4().hex
    rod = TensorRodOwner(solver, run_uuid=run_uuid, physics_step=sim.current_time_step_index,
                         episodes=tuple(f'{run_uuid}:{lane}:1' for lane in range(len(lanes))),
                         mesh_names=names, device='cuda:0')
    collector = NativeTensorCollector(
        project, sim, batch, prepared, archive, seed, initial, bindings, rod, profile,
        episode_windows=_EPISODE_WINDOWS, wall_cap_s=args.outer_timeout_s, full_grasp=True,
        dof_mode=packet.get('replay_dof_mode', 'full24'), gated_lift=packet.get('replay_gated_lift', False),
    )
    model = collector.model
    weights_sha256 = freeze_replay_actor(model, packet['policy'], packet['identity']['model'])
    stage = sim.stage
    wire = _author_preview(stage, batch, archive.task['support_scene']['supports'])

    origins = initial['env_origins'][:1]
    actual_midpoint = collector.nodes[0, 20].double().cpu()
    if bool(origins.count_nonzero()) or not torch.allclose(center, actual_midpoint, atol=2e-6, rtol=0.0):
        raise ValueError('Student rig requires the original env_0 origin and live rod Reset midpoint')
    center = center.numpy()
    overview_pose = look_at_optical(center + [-0.58, -0.20, 0.46], center + [0.05, 0.0, 0.04], [0, 0, 1])
    overview_cal = SimpleNamespace(
        width=800, height=600, K_left=torch.tensor([[640.0, 0.0, 400.0], [0.0, 640.0, 300.0], [0.0, 0.0, 1.0]]),
        T_world_left=torch.from_numpy(overview_pose),
    )
    initial_stamp = collector.source._boundary()
    snapshot = partial(_snapshot, collector, calibration.sha256, reset_sdk_timestamp_s=initial_stamp.sdk_time_s)
    summaries = output.parent
    peak_lift = peak_raw_lift = peak_hold = 0.0
    terminal = []
    checkpoint_update = packet['counters']['updates']
    policy_uid = f'frozen:{args.preview_checkpoint_sha256}'

    with ExitStack() as cleanup, torch.no_grad():
        cleanup.callback(stage.RemovePrim, '/World/DH116Preview')
        stereo = TiledStereoCapture.from_native(
            sim, origins, calibration, snapshot, width=calibration.width, height=calibration.height,
            device=collector.device, max_readback_bytes=readback_bytes, max_render_attempts=12,
            horizontal_fov_degrees=horizontal_fov, initial_reset_clock=True,
        )
        cleanup.callback(stereo.close)
        if stereo.calibration.sha256 != calibration.sha256:
            raise ValueError('native stereo camera calibration differs from the Student rig')
        overview = FixedNativeCameraReader(
            sim, origins, overview_cal, collector.device, rig_path='/World/DH116ReplayOverview',
        )
        cleanup.callback(overview.close)
        for camera in (*stereo.reader.cameras.values(), overview):
            cleanup.callback(camera.resources.detach)
        frames = [_export_frame(summaries, collector, wire, stereo, overview, snapshot)]
        for window in range(1, args.preview_windows + 1):
            collector.timings = {}
            action = model.deterministic(collector.observation())
            if collector.trainer.action_indices is not None:
                active = torch.zeros_like(action, dtype=torch.bool)
                active[:, collector.trainer.action_indices] = True
                action = action.masked_fill(~(active & collector.policy_action_mask()), 0)
            token = f'{policy_uid}:{window}'
            collector.begin_action_window(action, token=token, policy_uids=(policy_uid,) * collector.n, end=end)
            collector.advance_action_window(action, end)
            collector.record_window_request()
            result = collector.complete_action_window()
            values = collector.reward._last_values
            raw_lift = result.terms['lift']['raw']
            peak_raw_lift = max(peak_raw_lift, float(raw_lift[0]))
            detached_lift = replay_detached_lift(collector.reward, values, raw_lift)
            peak_lift = max(peak_lift, float(detached_lift[0]))
            peak_hold = max(peak_hold, float(result.post_context['hold_s'][0]))
            done = bool(result.done[0])
            if done or window == args.preview_windows or window % args.preview_every == 0:
                frames.append(_export_frame(summaries, collector, wire, stereo, overview, snapshot))
            if done:
                terminal = [name for name, mask in collector.reward._reasons.items() if bool(mask[0])]
                break
            if collector._reset_allowed:
                collector.reset(collector._reset_allowed)

    weights = {name: value for name, value in model.state_dict().items()
               if name.startswith(tuple(module + '.' for module in _ACTOR_MODULES))}
    updates = collector.trainer.counters['updates']
    if updates != 0 or tensor_state_sha256(weights) != weights_sha256:
        raise RuntimeError('preview changed frozen Teacher weights or performed PPO updates')
    return dict(sdk_step=rod.physics_step, controls=collector.controls, updates=updates,
                checkpoint_update=checkpoint_update, checkpoint_sha256=args.preview_checkpoint_sha256,
                actor_sha256=weights_sha256, frames=frames, terminal_reasons=terminal,
                peak_detached_lift_m=peak_lift, peak_raw_lift_m=peak_raw_lift, peak_hold_s=peak_hold,
                num_envs=collector.n, captured_env_id=0, active_mesh_names=names,
                scene_center_m=center.tolist(), captured_env_origin_m=origins[0].cpu().tolist(),
                checkpoint_num_envs=packet['identity']['num_envs'],
                checkpoint_recipe_sha256=packet['identity']['recipe_sha256'],
                camera_close_api='SimulationApp.close',
                calibration_sha256=calibration.sha256, backend_sha256=profile.binary_sha256,
                action_contract_sha256=action_sha256(), teacher_admission=False)


def _preview_worker(args, output, scene_sha256, task):
    from dh116.env.lifecycle import native_gpu_lock, run_native_session
    from dh116.env.scene import load_task_scene, task_scene_sha256
    from dh116.hand.articulation import OWNED_NAMES
    from dh116.physics.solver import TensorBackendProfile
    from dh116.physics.vision import StereoCalibration
    from dh116.rl.training import validate_native_options

    attempt = json.loads((output / 'attempt.json').read_text())
    if (attempt['preview_source_sha256'] != file_sha256(Path(__file__))
            or attempt['camera_config_sha256'] != file_sha256(_CAMERA_CONFIG)
            or attempt['checkpoint_sha256'] != args.preview_checkpoint_sha256
            or attempt['scene_sha256'] != scene_sha256):
        raise ValueError('preview script, camera configuration or checkpoint changed before startup')
    if (os.environ.get('DH116_TENSOR_GEOMETRY_BACKEND') != _GEOMETRY_BACKEND
            or os.environ.get('DH116_NATIVE_READBACK_BACKEND') != _READBACK_BACKEND):
        raise ValueError('preview requires the current native geometry and readback recipe')
    validate_native_options('clone')
    profile = TensorBackendProfile.load()
    packet = _checkpoint(args, scene_sha256, profile)
    near_wire = scene_sha256 != task_scene_sha256(task, full_grasp=True)
    archive, reset = load_task_scene(args.grasp, full_grasp=True, near_wire=near_wire)
    height = task.get('full_startup_height') if 'full_scene_input' in task else task.get('startup_height')
    if near_wire:
        height = None
    if height is not None:
        checked_file(PROJECT_ROOT / height['evidence_path'], height['evidence_sha256'])
        height = {key: height[key] for key in ('delta_z_m', 'evidence_sha256')}
    camera_config = json.loads(_CAMERA_CONFIG.read_text())
    calibration = StereoCalibration.from_dict(camera_config)
    exercise = partial(
        _exercise_preview, args=args, profile=profile, reset=reset, height=height, packet=packet,
        calibration=calibration, readback_bytes=camera_config['resource_limits']['max_readback_bytes'],
        scene_center=camera_config['scene_center_m'], horizontal_fov=camera_config['horizontal_fov_degrees'],
    )
    with native_gpu_lock(args.gpu):
        run_native_session(PROJECT_ROOT, archive, args.gpu, _PREVIEW_LANES, output / 'native',
                           max_sdk_step=2 + 8 * args.preview_windows, author_and_exercise=exercise,
                           owned_names=OWNED_NAMES, enable_cameras=True)
    return 0


def _preview_main(argv):
    parser = argparse.ArgumentParser(description='Replay a frozen native Teacher with fixed Student stereo RGB.')
    parser.add_argument('--preview-checkpoint', required=True, type=Path)
    parser.add_argument('--preview-checkpoint-sha256', required=True)
    parser.add_argument('--grasp', required=True, choices=('pinch', 'tripod', 'power'))
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--num-envs', type=int, choices=(_PREVIEW_LANES,), default=_PREVIEW_LANES)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--preview-windows', type=int, default=_EPISODE_WINDOWS)
    parser.add_argument('--preview-every', type=int, default=120)
    parser.add_argument('--outer-timeout-s', type=int, default=900)
    parser.add_argument('--preview-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not (1 <= args.preview_windows <= _EPISODE_WINDOWS and args.preview_every > 0 and args.outer_timeout_s > 0):
        parser.error('use 1–600 preview windows, a positive capture interval, and a positive timeout')
    output = ProjectPaths(PROJECT_ROOT).output(args.output)
    if output.relative_to(PROJECT_ROOT).parts[0] != 'runs':
        parser.error('preview output must be a new directory below runs/')
    task = json.loads((PROJECT_ROOT / f'configs/tasks/{args.grasp}.json').read_text())
    import torch
    from dh116.env.scene import task_scene_sha256

    raw = checked_file(args.preview_checkpoint, args.preview_checkpoint_sha256)
    identity = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)['identity']
    scene_sha256 = identity['receipt_sha256']
    known_scenes = {task_scene_sha256(task, full_grasp=True)}
    if 'near_wire_reset' in task:
        known_scenes.add(task_scene_sha256(task, full_grasp=True, near_wire=True))
    if scene_sha256 not in known_scenes:
        parser.error('checkpoint scene must match the current supported or near-wire Reset')
    if args.preview_worker:
        return _preview_worker(args, output, scene_sha256, task)

    from dh116.env.lifecycle import resolve_physical_gpu_uuid

    gpu_uuid = resolve_physical_gpu_uuid(args.gpu)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'native').mkdir()
    inventory = {str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
                 for path in (PROJECT_ROOT / 'src/dh116').rglob('*')
                 if path.is_file() and path.suffix in ('.py', '.cu')}
    write_new(output / 'attempt.json', dict(
        schema='dh116-frozen-native-stereo-preview-v1', grasp=args.grasp, scene_sha256=scene_sha256,
        num_envs=_PREVIEW_LANES, preview_windows=args.preview_windows, preview_every=args.preview_every,
        episode_windows=_EPISODE_WINDOWS, geometry_backend=_GEOMETRY_BACKEND, readback_backend=_READBACK_BACKEND,
        checkpoint_sha256=args.preview_checkpoint_sha256, camera_config_sha256=file_sha256(_CAMERA_CONFIG),
        preview_source_sha256=file_sha256(Path(__file__)), source_sha256=inventory, ppo_updates=0,
    ))
    command = [sys.executable, str(Path(__file__).resolve()), *argv, '--preview-worker']
    cache = PROJECT_ROOT / 'build' / 'native-cache' / gpu_uuid
    (cache / 'tmp').mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID', LIVESTREAM='0', XR='0',
               DH116_TENSOR_GEOMETRY_BACKEND=_GEOMETRY_BACKEND, DH116_NATIVE_READBACK_BACKEND=_READBACK_BACKEND,
               TMPDIR=str(cache / 'tmp'), CUDA_CACHE_PATH=str(cache / 'cuda'), XDG_CACHE_HOME=str(cache / 'xdg'))
    with (output / 'native.log').open('xb') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        write_new(output / 'pid.json', dict(pid=child.pid, command=command, gpu=args.gpu, gpu_uuid=gpu_uuid))
        try:
            code = child.wait(timeout=args.outer_timeout_s)
        except subprocess.TimeoutExpired:
            write_new(output / 'timeout.json', dict(pid=child.pid, timeout_s=args.outer_timeout_s,
                                                    natural_exit_verified=False, automatic_signal_sent=False))
            raise RuntimeError(f'preview worker exceeded its time budget, see {output / "native.log"}') from None
    write_new(output / 'process_exit.json', dict(returncode=code))
    if code:
        raise RuntimeError(f'preview worker exited {code}, see {output / "native.log"}')
    lifecycle = json.loads((output / 'native/result.json').read_text())
    close = json.loads((output / 'native/app_close_returned.json').read_text())
    actual = lifecycle['acquisition']
    if (lifecycle['failure'] is not None or lifecycle['close_mode'] != 'normal' or close['close_mode'] != 'normal'
            or lifecycle['lifetime_released_before_close'] is not True or close['app_close_returned'] is not True
            or actual['updates'] != 0 or len(actual['frames']) < 2
            or not 1 <= actual['controls'] <= args.preview_windows
            or not lifecycle['physics_steps'] == actual['sdk_step'] == 2 + actual['controls'] * 8):
        raise ValueError('complete frozen replay and verified normal process exit required')
    write_new(output / 'preview_result.json', dict(actual, close_mode='normal', process_exit_verified=True))
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if any(arg.split('=', 1)[0] == '--preview-checkpoint' for arg in argv):
        return _preview_main(argv)
    return train_teacher_main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
