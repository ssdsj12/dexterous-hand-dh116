"""Executable Stage6 file workflows; native trials remain separately recorded."""
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


VERSION = 'dh116-stage6-inputs-v1'


def load_stage6_config(path=None):
    """Consume the version-controlled main experiment, rejecting silent drift."""
    import yaml
    if path is None:
        path = Path(__file__).resolve().parents[4] / 'configs/dh116/stage6_pinch_sequence_bc.yaml'
    value = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Stage6 configuration must be an object')
    fixed = {
        ('experiment', 'base_commit'): '26f1bbcc5a0e596da55c5d89896befcb5622cda7',
        ('experiment', 'style'): 'pinch_thumb_index', ('experiment', 'default_control_path'): 'M0',
        ('sequence', 'length'): 16, ('sequence', 'batch_sequences'): 4,
        ('sequence', 'cross_episode'): False, ('sequence', 'cross_gap'): False,
        ('sequence', 'terminal_padding'): 'masked',
        ('sampling', 'uniform_fraction'): .4, ('sampling', 'event_fraction'): .6,
        ('sampling', 'source_balanced'): True,
        ('sampling', 'event_weights'): dict(START=.10, CONTACT=.35, LIFT_ONSET=.35, HOLD=.20),
        ('training', 'max_grad_norm'): .5, ('training', 'mixed_precision'): False,
        ('loss', 'action_mse'): 1., ('loss', 'command_weight'): .1,
        ('loss', 'command_warmup_start'): 250, ('loss', 'command_warmup_end'): 500,
        ('loss', 'delta_weight'): 0., ('loss', 'finger_scale_rad'): .01,
        ('loss', 'palm_scale_m'): .0005, ('loss', 'wrist_angle_scale_rad'): .01,
        ('model', 'architecture'): 'existing_four_frame_pointnet_mlp',
        ('model', 'point_activation'): 'relu',
        ('model', 'quaternion_preprocessing'): 'inherited',
        ('model', 'add_privileged_actor_inputs'): False,
        ('runtime', 'one_isaac_instance'): True, ('runtime', 'preserve_original_runs'): True,
        ('runtime', 'overwrite_checkpoint'): False,
        ('inputs', 'expected_actor_dim'): 3844, ('inputs', 'expected_critic_dim'): 371,
        ('inputs', 'expected_action_dim'): 24,
    }
    for (section, name), expected in fixed.items():
        if not isinstance(value.get(section), dict) or value[section].get(name) != expected:
            raise ValueError('Stage6 fixed configuration differs: ' + section + '.' + name)
    return value


def require_matched_exposures(report):
    if (report.get('sequence_length') != 16 or report.get('min_valid_steps') != 16
            or report.get('max_valid_steps') != 16 or report.get('padded_window_count') != 0):
        raise ValueError('matched B0/B1/B2 exposure experiment requires full16 windows; padding stays explicit')


def require_main_preprocessing(preprocessing):
    """The prescribed main arms retain ReLU/M0; ablations need another version."""
    if preprocessing.get('point_activation') != 'relu' or preprocessing.get('mirror') != 'm0_raw':
        raise ValueError('Stage6 main arms require relu activation and m0_raw; use a separate ablation contract')


def resolve_stage6_inputs(discovery_path, teacher_report, heldout_manifest, output):
    """Resolve actual files from pinned source/collection records, never filenames."""
    teacher_report = Path(teacher_report).resolve(strict=True)
    report = json.loads(teacher_report.read_text())
    if report.get('native_runtime_verified') is not True or report.get('successful_episodes', 0) < 4:
        raise ValueError('actual successful native Teacher collection is required before resolving inputs')
    discovery_path = Path(discovery_path).resolve(strict=True)
    discovery = json.loads(discovery_path.read_text())
    if discovery.get('version') != VERSION:
        raise ValueError('explicit Stage6 source-discovery manifest required')
    model_id = _checked_identity(discovery['source_checkpoint'], discovery_path.parent)
    import torch
    from .tabletop_bimanual_checkpoint_contract import (
        unwrap_physical_adapter_checkpoint, QUALIFICATION_KEY,
    )
    from .tabletop_vision_point_activation import point_activation_from_contract
    saved = torch.load(model_id['path'], map_location='cpu', weights_only=False)
    core = unwrap_physical_adapter_checkpoint(saved, allow_explicit=True)
    qualification = saved['contract'].get(QUALIFICATION_KEY)
    if not isinstance(qualification, dict):
        raise ValueError('Stage6 main input requires the explicitly qualified source Adapter model')
    launch_id = _checked_identity(discovery['source_launch'], discovery_path.parent)
    launch = json.loads(Path(launch_id['path']).read_text())
    command = launch['command']
    if '--depth-camera-config' not in command:
        raise ValueError('source launch must identify the real calibrated camera file')
    camera = file_identity(command[command.index('--depth-camera-config') + 1])
    value = deepcopy(discovery)
    value.update(source_checkpoint=model_id, dataset=_checked_identity(report['successful_dataset'], teacher_report.parent),
        sequence_context=_checked_identity(report['successful_sequence_context'], teacher_report.parent),
        teacher_collection_report=file_identity(teacher_report),
        assets={side: _checked_identity(item, teacher_report.parent)
                for side, item in qualification['mapping']['source_assets'].items()},
        camera=camera, model_version=saved['version'],
        preprocessing=dict(wrist_quaternion=core['contract'].get('actor_wrist_quaternion_preprocessing', 'raw'),
                           point_activation=point_activation_from_contract(core['contract']), mirror='m0_raw'),
        final_heldout_manifest=file_identity(heldout_manifest),
        source_discovery_manifest=file_identity(discovery_path),
        resolution='actual_pinned_source_launch_and_new_native_collection_report')
    _json_new(output, value)
    return load_stage6_inputs(output)


def file_identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def _json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, allow_nan=False)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(serialized + '\n')


def _checked_identity(value, parent):
    if not isinstance(value, dict) or not {'path', 'sha256'} <= value.keys():
        raise ValueError('input requires actual path and SHA256')
    path = Path(value['path'])
    if not path.is_absolute():
        path = parent / path
    actual = file_identity(path)
    if actual['sha256'] != value['sha256']:
        raise ValueError('Stage6 input SHA256 mismatch: ' + str(path))
    return actual


def load_stage6_inputs(path):
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('version') != VERSION:
        raise ValueError('explicit Stage6 inputs manifest required')
    if tuple(value.get(k) for k in ('actor_dim', 'critic_dim', 'action_dim')) != (3844, 371, 24):
        raise ValueError('Stage6 preserves 3844D/371D/24D')
    checked = deepcopy(value)
    checked['source_checkpoint'] = _checked_identity(value.get('source_checkpoint'), path.parent)
    for name, item in value.items():
        if name != 'source_checkpoint' and isinstance(item, dict) and {'path', 'sha256'} <= item.keys():
            checked[name] = _checked_identity(item, path.parent)
    required = {'dataset', 'sequence_context', 'teacher_collection_report', 'assets', 'camera',
                'preprocessing', 'model_version', 'final_heldout_manifest'}
    if not required <= value.keys():
        raise ValueError('complete bound Stage6 input manifest required: ' + str(sorted(required - value.keys())))
    import torch
    from .tabletop_bimanual_checkpoint_contract import unwrap_physical_adapter_checkpoint, ADAPTER_KEY
    from .tabletop_vision_point_activation import point_activation_from_contract
    saved = torch.load(checked['source_checkpoint']['path'], map_location='cpu', weights_only=False)
    core = unwrap_physical_adapter_checkpoint(saved, allow_explicit=True)
    if checked['model_version'] != saved['version']:
        raise ValueError('source model version differs from input manifest')
    expected_preprocessing = dict(wrist_quaternion=core['contract'].get('actor_wrist_quaternion_preprocessing', 'raw'),
                                  point_activation=point_activation_from_contract(core['contract']), mirror='m0_raw')
    if checked['preprocessing'] != expected_preprocessing:
        raise ValueError('source preprocessing differs from bound Stage6 manifest')
    require_main_preprocessing(checked['preprocessing'])
    assets = value['assets']
    if not isinstance(assets, dict) or set(assets) != {'left', 'right'}:
        raise ValueError('both actual hand asset bindings are required')
    checked['assets'] = {side: _checked_identity(assets[side], path.parent) for side in assets}
    expected_assets = saved['contract'].get(ADAPTER_KEY, {}).get('source_asset_sha256',
                         core['contract'].get('privileged_critic_contract', {}).get('source_sha256'))
    if {side: item['sha256'] for side, item in checked['assets'].items()} != expected_assets:
        raise ValueError('source hand assets differ from bound input files')
    if checked['camera']['sha256'] != core['contract']['camera_config_sha256']:
        raise ValueError('source camera differs from bound input calibration')
    checked['input_manifest'] = file_identity(path)
    return checked


def require_native_shadow_parity(parity):
    if not isinstance(parity, dict) or parity.get('passed') is not True:
        raise ValueError('recorded native Teacher shadow parity must pass')
    readback = parity.get('native_sdk_readback', {})
    if (parity.get('native_sdk_readback_required') is not True
            or any(readback.get(source, {}).get('passed') is not True for source in ('backend', 'command_cache'))):
        raise ValueError('separate actual native backend/cache readback parity required')
    return deepcopy(parity)


def verify_command_windows(sequence, controller_contract, *, batch_windows=64):
    """Check accumulated 16-policy-step commands against recorded real targets."""
    import torch
    from .tabletop_vision_command_shadow import shadow_sequence
    maxima = dict(finger=0., palm=0., wrist_angle=0.)
    valid_samples = 0
    with torch.no_grad():
        for start in range(0, len(sequence), batch_windows):
            batch = sequence.batch(list(range(start, min(start + batch_windows, len(sequence)))))
            initial = {name: value[:, 0] for name, value in batch['controller_start'].items()}
            predicted = shadow_sequence(initial, batch['actions'], batch['physics_context'],
                                        controller_contract, physics_valid=batch['physics_valid'])
            mask = batch['physics_valid']
            valid_samples += int(mask.sum())
            for name in maxima:
                error = (predicted[name] - batch['teacher_targets'][name]).abs()
                selected = error[mask]
                if selected.numel():
                    maxima[name] = max(maxima[name], float(selected.max()))
    passed = valid_samples > 0 and max(maxima.values()) == 0.
    if not passed:
        raise ValueError('full-window Teacher command parity failed: ' + str(maxima))
    return dict(passed=True, sequence_length=16, windows=len(sequence),
                valid_physics_samples=valid_samples, maximum_float64_error=maxima,
                scope='recorded_feedback_conditioned_commands_not_future_physics')


def _drop_options(command, names):
    result = []
    i = 0
    while i < len(command):
        token = command[i]
        name = token.split('=', 1)[0]
        if name not in names:
            result.append(token)
            i += 1
            continue
        if names[name] and '=' not in token:
            if i + 1 >= len(command):
                raise ValueError('missing native option value: ' + name)
            i += 2
        else:
            i += 1
    return result


def _set_option(command, name, value):
    command[:] = _drop_options(command, {name: True})
    command.extend((name, str(value)))


def require_native_mode(mode, config):
    if mode not in ('source', 'branch', 'heldout', 'teacher-replay') or config.get('mode') != mode:
        raise ValueError('requested native mode differs from the pinned configuration mode')


def build_stage6_native_command(template, source, output, *, mode, native_config=None):
    """Replace code paths only; data paths remain pinned to their actual files."""
    if mode not in ('teacher-context', 'source', 'branch', 'heldout', 'teacher-replay'):
        raise ValueError('unknown Stage6 native mode')
    source, output = Path(source), Path(output)
    command = list(template['command'])
    if len(command) < 3 or not command[1].endswith('/scripts/python.sh'):
        raise ValueError('verified native Python/script template required')
    command[1] = str(source / 'scripts/python.sh')
    command[2] = str(source / 'scripts/train_dh116_tabletop_gpu_ppo.py')
    _set_option(command, '--output', output)
    wrappers = {'--bimanual-teacher-physics-audit': False,
                '--bimanual-tripod-teacher-coordination-proof': True}
    command = _drop_options(command, wrappers)
    if mode == 'teacher-context':
        if '--vision-teacher-source' not in command:
            raise ValueError('Teacher context requires a real Teacher template')
        if '--vision-sequence-context-sidecar' not in command:
            command.append('--vision-sequence-context-sidecar')
        return command
    command = _drop_options(command, {
        '--vision-teacher-source': True, '--vision-teacher-mpc': False,
        '--vision-teacher-perturbation-manifest': True,
        '--vision-teacher-mpc-profile': True, '--vision-teacher-max-control-steps': True,
        '--vision-teacher-hold-tail-control-steps': True, '--vision-teacher-raw-chunk-frames': True,
        '--vision-sequence-context-sidecar': False, '--bimanual-explicit-sdk-adapter': False,
        '--vision-evaluation-checkpoint': True, '--vision-evaluation-manifest': True,
        '--vision-evaluation-physics-audit': False, '--bimanual-checkpoint-adapter': False,
        '--vision-evaluation-max-control-steps': True, '--paired-bias-arm': True,
        '--vision-m2-replay': True, '--resume': True,
    })
    _set_option(command, '--vision-mirror-preprocessing', 'm0_raw')
    _set_option(command, '--vision-action-distribution', 'tanh_gaussian')
    if native_config is None:
        raise ValueError('isolated Stage6 native configuration required')
    if mode == 'heldout':
        cfg = json.loads(Path(native_config).read_text())
        _set_option(command, '--vision-evaluation-checkpoint', cfg['checkpoint']['path'])
        _set_option(command, '--vision-evaluation-manifest', cfg['manifest']['path'])
        _set_option(command, '--vision-evaluation-max-control-steps', 1800)
        _set_option(command, '--num-envs', cfg['num_envs'])
        _set_option(command, '--seed', cfg['seed'])
        if '--bimanual-checkpoint-adapter' not in command:
            command.append('--bimanual-checkpoint-adapter')
    else:
        _set_option(command, '--vision-stage6-native-config', native_config)
        _set_option(command, '--max-iterations', 1)
    return command


def freeze_source(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source in target.parents:
        raise ValueError('frozen target must be outside source tree')
    target.mkdir(parents=True, exist_ok=False)
    excluded = {'.git', 'artifacts', 'runs', 'archive', '__pycache__', '.pytest_cache', '.venv'}
    identities = {}
    for base, dirs, files in os.walk(source):
        dirs[:] = [d for d in dirs if d not in excluded]
        relative = Path(base).relative_to(source)
        destination = target / relative
        destination.mkdir(parents=True, exist_ok=True)
        for name in files:
            if name.endswith(('.pyc', '.pyo')):
                continue
            new = destination / name
            shutil.copyfile(Path(base) / name, new)
            identities[str(new.relative_to(target))] = file_identity(new)['sha256']
    return identities


def launch_native(template_path, source, output, *, mode, native_config=None):
    """One real Isaac instance, exclusive evidence and immutable source copy."""
    import fcntl
    template_path, output = Path(template_path).resolve(strict=True), Path(output).absolute()
    source_path, output_path = Path(source).resolve(strict=True), output.resolve()
    if source_path == output_path or source_path in output_path.parents:
        raise ValueError('native output must be outside source tree')
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if mode != 'teacher-context':
        if native_config is None:
            raise ValueError('explicit native configuration mode required')
        require_native_mode(mode, json.loads(Path(native_config).read_text()))
    frozen = output.with_name(output.name + '_source')
    hashes = freeze_source(source, frozen)
    cfg = json.loads(template_path.read_text())
    command = build_stage6_native_command(cfg, frozen, output, mode=mode, native_config=native_config)
    bound_inputs = {}
    if native_config is not None:
        import torch
        from .tabletop_bimanual_checkpoint_contract import unwrap_physical_adapter_checkpoint
        native_data = json.loads(Path(native_config).read_text())
        model_id = _checked_identity(native_data['checkpoint'], Path(native_config).parent)
        saved = torch.load(model_id['path'], map_location='cpu', weights_only=False)
        core = unwrap_physical_adapter_checkpoint(saved, allow_explicit=True)
        _set_option(command, '--vision-wrist-quaternion-preprocessing',
                    core['contract'].get('actor_wrist_quaternion_preprocessing', 'raw'))
        bound_inputs['checkpoint'] = model_id
        bound_inputs['native_config'] = file_identity(native_config)
    meta = dict(mode=mode, command=command, cwd=str(frozen), source_sha256=hashes,
                template=file_identity(template_path), env=cfg.get('env', {}), bound_inputs=bound_inputs)
    _json_new(output.with_name(output.name + '_launch.json'), meta)
    with open('/tmp/xk-dh116-stage4-native.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        busy = subprocess.run(['pgrep', '-af', '[s]cripts/train_dh116_tabletop_gpu_ppo.py'],
                              text=True, capture_output=True)
        if busy.returncode not in (0, 1) or busy.stdout.strip():
            raise RuntimeError('another Isaac training/evaluation process is active')
        if file_identity(template_path) != meta['template']:
            raise RuntimeError('native template changed after freeze')
        for identity in bound_inputs.values():
            _checked_identity(identity, output.parent)
        with output.with_name(output.name + '.log').open('x') as log:
            result = subprocess.run(command, cwd=frozen, env=dict(os.environ, **cfg.get('env', {})),
                                    stdout=log, stderr=subprocess.STDOUT)
        _json_new(output.with_name(output.name + '_exit.json'), dict(exit_code=result.returncode))
        if result.returncode:
            raise RuntimeError('native Stage6 job failed; frozen output/log retained')
    return output


def prepare_sequences(inputs_path, teacher_report, output):
    """Build an identity-joined index from newly collected native context."""
    import torch
    from .tabletop_vision_sequence_dataset import SequenceDataset
    from .tabletop_vision_supervision_dataset import validate_dataset
    inputs = load_stage6_inputs(inputs_path)
    teacher_report = Path(teacher_report).resolve(strict=True)
    if file_identity(teacher_report) != inputs['teacher_collection_report']:
        raise ValueError('Teacher report differs from the complete input manifest binding')
    report = json.loads(teacher_report.read_text())
    if (report.get('native_runtime_verified') is not True or report.get('successful_episodes', 0) < 4
            or report.get('ppo_updates') != 0 or report.get('teacher_success_is_student_success') is not False):
        raise ValueError('four genuinely successful native pinch Teacher episodes required')
    data_id = _checked_identity(report.get('successful_dataset'), teacher_report.parent)
    context_id = _checked_identity(report.get('successful_sequence_context'), teacher_report.parent)
    if data_id != inputs['dataset'] or context_id != inputs['sequence_context']:
        raise ValueError('Teacher dataset/context differ from complete input manifest bindings')
    data = torch.load(data_id['path'], map_location='cpu', weights_only=False)
    context = torch.load(context_id['path'], map_location='cpu', weights_only=False)
    validate_dataset(data)
    reported = {str(key): episode for key, episode in report.get('episodes', {}).items()
                if episode.get('success') is True and episode.get('first_episode') is True
                and episode.get('native_physics') is True and episode.get('recovered') is False
                and episode.get('grasp_style') == 'pinch_thumb_index'}
    if len(reported) < 4 or set(data['episodes']) != set(reported):
        raise ValueError('native Teacher report and dataset must contain the same four-or-more pinch episodes')
    for key, episode in data['episodes'].items():
        if json.dumps(episode, sort_keys=True) != json.dumps(reported[key], sort_keys=True):
            raise ValueError('dataset episode/terminal differs from bound Teacher report')
    if report['successful_dataset'].get('samples') != len(data['observations']):
        raise ValueError('successful dataset sample count differs from Teacher report')
    sequence = SequenceDataset(data, context, dataset_sha256=data_id['sha256'], sampling_mode='event')
    if any(style != 'pinch_thumb_index' for style in data['grasp_styles']):
        raise ValueError('this Stage6 expert pool is pinch-only')
    parity = report['successful_sequence_context'].get('teacher_shadow_parity', {})
    if not parity:
        parity = context.get('teacher_shadow_parity', {})
    require_native_shadow_parity(parity)
    full_window_parity = verify_command_windows(sequence, context['controller_contract'])
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    index_path = output / 'sequence_index.pt'
    with index_path.open('xb') as stream:
        torch.save(dict(version='dh116-stage6-sequence-index-v1', windows=sequence.windows,
                        probabilities=sequence.sampling_probabilities, report=sequence.report), stream)
    prepared = dict(version='dh116-stage6-prepared-sequences-v1', inputs=inputs,
                    teacher_report=file_identity(teacher_report), dataset=data_id, context=context_id,
                    index=file_identity(index_path), sequence_report=sequence.report,
                    controller_contract=context.get('controller_contract'), teacher_shadow_parity=parity,
                    accumulated_window_parity=full_window_parity,
                    heldout_source_count=0, fit_only=sequence.report['fit_only'],
                    actor_input='original_saved_3844D_only', native_success_claimed=False)
    _json_new(output / 'prepared.json', prepared)
    return prepared


def train_sequences(prepared_path, output, *, profile='B2', updates=3000, seed=116601,
                    device='cpu', checkpoint_interval=250, learning_rate=1e-4):
    """Fresh independent BC optimizer; preserve source PPO state in every save."""
    import torch
    from .tabletop_bimanual_checkpoint_contract import unwrap_physical_adapter_checkpoint
    from .tabletop_vision_point_activation import point_activation_from_contract
    from .tabletop_vision_sequence_dataset import SequenceDataset
    from .tabletop_vision_supervised_bc import fit_actor_sequences, _same_state
    prepared_path = Path(prepared_path).resolve(strict=True)
    prepared = json.loads(prepared_path.read_text())
    if prepared.get('version') != 'dh116-stage6-prepared-sequences-v1':
        raise ValueError('prepared Stage6 sequence manifest required')
    input_id = _checked_identity(prepared['inputs']['input_manifest'], prepared_path.parent)
    inputs = load_stage6_inputs(input_id['path'])
    if (inputs != prepared['inputs'] or inputs['dataset'] != prepared['dataset']
            or inputs['sequence_context'] != prepared['context']
            or inputs['teacher_collection_report'] != prepared['teacher_report']):
        raise ValueError('prepared sequences differ from their pinned full input manifest')
    identities = [prepared[k] for k in ('dataset', 'context', 'teacher_report', 'index')]
    identities.append(prepared['inputs']['source_checkpoint'])
    for item in identities:
        _checked_identity(item, prepared_path.parent)
    data = torch.load(prepared['dataset']['path'], map_location='cpu', weights_only=False)
    context = torch.load(prepared['context']['path'], map_location='cpu', weights_only=False)
    source = torch.load(prepared['inputs']['source_checkpoint']['path'], map_location='cpu', weights_only=False)
    core = unwrap_physical_adapter_checkpoint(source, allow_explicit=True)
    if profile not in ('B0', 'B1', 'B2', 'B3'):
        raise ValueError('unknown Stage6 training arm')
    if profile == 'B3':
        raise ValueError('B3 requires a separately verified B2-vs-B1 improvement gate')
    if profile == 'B2':
        require_native_shadow_parity(prepared.get('teacher_shadow_parity'))
        if prepared.get('accumulated_window_parity', {}).get('passed') is not True:
            raise ValueError('B2 requires full accumulated-window command parity')
    sequence = SequenceDataset(data, context, dataset_sha256=prepared['dataset']['sha256'],
                               sampling_mode='uniform' if profile == 'B0' else 'event', seed=seed)
    require_matched_exposures(sequence.report)
    from .tabletop_vision_supervision_dataset import _canonical_visual_contract
    sensor_contract, actor_contract = deepcopy(data['visual_contract']), deepcopy(core['contract'])
    for value in (sensor_contract, actor_contract):
        value.pop('actor_wrist_quaternion_preprocessing', None)
        value.pop('actor_pointnet_activation', None)
    if _canonical_visual_contract(sensor_contract) != _canonical_visual_contract(actor_contract):
        raise ValueError('sequence data changes source sensor/physics/action contract')
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    snapshots = []
    preserved = {k: deepcopy(v) for k, v in source.items() if k not in ('policy',)}
    implementation = {name: file_identity(Path(__file__).with_name(name + '.py')) for name in (
        'tabletop_vision_stage6_workflow', 'tabletop_vision_supervised_bc',
        'tabletop_vision_sequence_dataset', 'tabletop_vision_sequence_losses',
        'tabletop_vision_command_shadow', 'tabletop_command_kernels')}

    def save_snapshot(payload):
        checkpoint = deepcopy(source)
        for name, value in payload['actor'].items():
            checkpoint['policy']['actor.' + name] = value.detach().cpu().clone()
        for name, value in source['policy'].items():
            if not name.startswith('actor.') and not _same_state(checkpoint['policy'][name], value):
                raise RuntimeError('BC modified privileged Critic or exploration state')
        step = int(payload['step'])
        checkpoint['stage6_sequence_bc'] = dict(version='dh116-stage6-sequence-bc-v1',
            profile=profile, source_checkpoint=prepared['inputs']['source_checkpoint'],
            dataset=prepared['dataset'], context=prepared['context'], updates=step,
            seed=seed, status=payload['status'], exact_ppo_resume=False, ppo_updates=0,
            training_implementation=implementation,
            stage6_fit_only=sequence.report['fit_only'], actor_point_activation=point_activation_from_contract(core['contract']))
        checkpoint['stage6_bc_optimizer'] = deepcopy(payload['optimizer'])
        checkpoint['stage6_bc_sampling_rng'] = payload['sampling_rng'].clone()
        checkpoint['stage6_pointnet_probe'] = deepcopy(payload['probes'])
        if 'input' in payload:
            checkpoint['stage6_diagnostic_input'] = deepcopy(payload['input'])
        unwrap_physical_adapter_checkpoint(checkpoint, allow_explicit=True)
        target = output / f'{profile}_update_{step:05d}_{payload["status"]}.pt'
        with target.open('xb') as stream:
            torch.save(checkpoint, stream)
        snapshots.append(dict(step=step, status=payload['status'], **file_identity(target)))
        _json_new(target.with_suffix('.json'), snapshots[-1])

    actor_state = {name[6:]: value for name, value in core['policy'].items() if name.startswith('actor.')}
    fitted = fit_actor_sequences(actor_state, sequence, profile=profile, steps=updates,
        batch_sequences=4, learning_rate=learning_rate, seed=seed, device=device,
        wrist_quaternion_preprocessing=core['contract'].get('actor_wrist_quaternion_preprocessing', 'raw'),
        point_activation=point_activation_from_contract(core['contract']),
        controller_contract=context.get('controller_contract'), checkpoint_interval=checkpoint_interval,
        snapshot_callback=save_snapshot)
    for name, value in preserved.items():
        if not _same_state(source[name], value):
            raise RuntimeError('in-memory source PPO state was modified')
    for item in identities:
        _checked_identity(item, prepared_path.parent)
    for item in implementation.values():
        _checked_identity(item, prepared_path.parent)
    report = {k: v for k, v in fitted.items() if k not in (
        'actor', 'actor_state', 'optimizer', 'optimizer_state', 'sampling_rng', 'sampling_rng_state')}
    report.update(profile=profile, prepared=file_identity(prepared_path), checkpoints=snapshots,
                  source_checkpoint=prepared['inputs']['source_checkpoint'],
                  training_implementation=implementation,
                  native_success_claimed=False, ppo_updates=0, checkpoint_selection='prespecified_final_update_not_train_MSE')
    _json_new(output / 'sequence_training_report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare-data')
    prepare.add_argument('--inputs', type=Path)
    prepare.add_argument('--discovery-inputs', type=Path)
    prepare.add_argument('--heldout-manifest', type=Path)
    prepare.add_argument('--teacher-report', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--config', type=Path)
    train = sub.add_parser('train-bc')
    train.add_argument('--prepared', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--profile', choices=('B0', 'B1', 'B2', 'B3'))
    train.add_argument('--updates', type=int)
    train.add_argument('--seed', type=int)
    train.add_argument('--device', default='cpu')
    train.add_argument('--checkpoint-interval', type=int)
    train.add_argument('--config', type=Path)
    for command in ('diagnose-branch', 'evaluate-source', 'evaluate-heldout'):
        native = sub.add_parser(command)
        native.add_argument('--template', type=Path, required=True)
        native.add_argument('--native-config', type=Path, required=True)
        native.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare-data':
        load_stage6_config(args.config)
        inputs = args.inputs
        if args.discovery_inputs:
            if inputs is not None or args.heldout_manifest is None:
                parser.error('source discovery requires --heldout-manifest and excludes --inputs')
            inputs = args.output.with_name(args.output.name + '_inputs.json')
            resolve_stage6_inputs(args.discovery_inputs, args.teacher_report, args.heldout_manifest, inputs)
        if inputs is None:
            parser.error('prepare-data requires full --inputs or explicit --discovery-inputs')
        result = prepare_sequences(inputs, args.teacher_report, args.output)
    elif args.command == 'train-bc':
        config = load_stage6_config(args.config)
        options = config['training']
        result = train_sequences(args.prepared, args.output, profile=args.profile or options['profile'],
                                 updates=args.updates if args.updates is not None else options['updates'],
                                 seed=args.seed if args.seed is not None else options['seed'], device=args.device,
                                 learning_rate=options['learning_rate'],
                                 checkpoint_interval=args.checkpoint_interval if args.checkpoint_interval is not None
                                 else options['checkpoint_interval'])
    else:
        modes = {'diagnose-branch': 'branch', 'evaluate-source': 'source', 'evaluate-heldout': 'heldout'}
        output = launch_native(args.template, Path(__file__).resolve().parents[4], args.output,
                               mode=modes[args.command], native_config=args.native_config)
        result = dict(output=str(output), native_process_completed=True)
    print(json.dumps(result, allow_nan=False, default=str))
