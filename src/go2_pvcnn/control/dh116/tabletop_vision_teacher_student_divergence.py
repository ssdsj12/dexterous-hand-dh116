"""Read-only Stage7 Teacher/Student diagnostics; never a controller or trainer.

Saved Teacher observations are forwarded without adding privileged inputs.
Conditional command differences reuse actual saved feedback, not a prediction
of future physics. Numerical departures never establish failure causality.
"""
import argparse
from copy import deepcopy
import hashlib
import gzip
import json
import os
from pathlib import Path
import sys

import torch
import numpy as np


VERSION = 'dh116-stage7-teacher-student-divergence-v1'
SOURCE_SHA256 = 'c58f9cf0718d36cd9feec181c9180a9657e97b54d6fa8e99ce1b3f16244f6d97'
EVENTS = ('START', 'CONTACT', 'LIFT_ONSET', 'HOLD')
ACTION_GROUPS = dict(all=list(range(24)), left_thumb=[0, 1], left_index=[2],
    right_thumb=[6, 7], right_index=[8], left_fingers=list(range(6)),
    right_fingers=list(range(6, 12)), left_palm_xyz=[12, 13, 14],
    right_palm_xyz=[15, 16, 17], left_wrist_rotation=[18, 19, 20],
    right_wrist_rotation=[21, 22, 23])
COMMAND_SCALES = dict(finger=.01, palm=.0005, wrist_angle=.01)
COMMAND_UNITS = dict(finger='rad', palm='m', wrist_angle='rad')


def _absolute_statistics(error):
    value = error.detach().cpu().double().abs().reshape(-1)
    return dict(elements=value.numel(), mae=float(value.mean()) if value.numel() else None,
        p95=float(torch.quantile(value, .95)) if value.numel() else None,
        max=float(value.max()) if value.numel() else None)


def _action_reduction(predicted, teacher):
    def reduce(a, b):
        nonzero = b != 0
        nonzero_n = int(nonzero.sum()); zero_n = int((~nonzero).sum())
        return dict(**_absolute_statistics(a-b), direction=dict(
            nonzero_teacher_elements=nonzero_n,
            nonzero_teacher_sign_agreement=float((a.sign()[nonzero] == b.sign()[nonzero]).double().mean()) if nonzero_n else None,
            zero_teacher_elements=zero_n,
            zero_teacher_predicted_nonzero_elements=int(((a != 0) & ~nonzero).sum()),
            zero_teacher_predicted_zero_elements=int(((a == 0) & ~nonzero).sum())))
    return dict(rows=len(predicted), groups={name: reduce(predicted[:, ids], teacher[:, ids]) for name, ids in ACTION_GROUPS.items()},
        by_dimension=[reduce(predicted[:, i], teacher[:, i]) for i in range(24)])


def action_statistics(predicted, teacher, event_labels):
    """Uniform original-row statistics, exact signs and explicit zero labels."""
    if (not isinstance(predicted, torch.Tensor) or not isinstance(teacher, torch.Tensor)
            or predicted.ndim != 2 or predicted.shape != teacher.shape or predicted.shape[1] != 24
            or len(predicted) == 0 or not predicted.is_floating_point() or not teacher.is_floating_point()
            or not torch.isfinite(predicted).all() or not torch.isfinite(teacher).all()):
        raise ValueError('matching finite original N24 action rows required')
    predicted, teacher = predicted.detach().cpu().double(), teacher.detach().cpu().double()
    masks = {}
    for name in EVENTS:
        value = event_labels.get(name)
        if not isinstance(value, torch.Tensor) or value.shape != (len(teacher),) or value.dtype != torch.bool:
            raise ValueError('original event label boolean N masks required')
        masks[name] = value.detach().cpu()
    return dict(all=_action_reduction(predicted, teacher),
        event_strata={name: _action_reduction(predicted[mask], teacher[mask]) for name, mask in masks.items()},
        action_group_indices=deepcopy(ACTION_GROUPS), event_labels_may_overlap=True,
        event_scope='SequenceDataset.event_labels: overlapping event neighborhoods;not online exclusive phases',
        direction_scope='exact torch.sign equality on nonzero Teacher labels;zero labels counted separately;no epsilon',
        percentile_scope='float64 linear quantile over absolute scalar errors;uniform original rows',
        cause_claimed=False)


def _full_windows(sequence):
    if sequence.joined is None or sequence.context is None: raise ValueError('full actual joined L16 feedback required')
    indices = []
    for i, window in enumerate(sequence.windows):
        rows = window['row_indices']
        if len(rows) != 16: continue
        if len(set(rows)) != 16: raise ValueError('continuous L16 window has duplicate rows')
        metadata = [sequence.data['sample_metadata'][row] for row in rows]
        first = metadata[0]
        for j, row in enumerate(metadata):
            if (row['tick'] != first['tick'] + j or any(row[k] != first[k] for k in ('run_uuid', 'env_id', 'episode_uid'))
                    or (j and (row.get('reset') is True or row.get('reset_mask') is True))
                    or (j < 15 and row.get('terminal') is True)
                    or (j and abs(row['policy_timestamp_s']-metadata[j-1]['policy_timestamp_s']-1/60) > 1e-6)
                    or (j and row['first_physics_step']-metadata[j-1]['first_physics_step'] != 8)):
                raise ValueError('continuous same-episode L16 window required;no gap/reset/terminal crossing')
        joined = sequence.joined['context_indices'][rows]
        live = sequence.context['physics_context']['live'][joined]
        source_valid = sequence.context['physics_context']['source_valid'][joined]
        last_count = int(live[-1].sum())
        if (not bool(live[:-1].all()) or last_count < 1 or not torch.equal(live[-1], torch.arange(8) < last_count)
                or not bool(source_valid[live].all())):
            raise ValueError('all16 actions require actual active contiguous physics samples;no missing feedback')
        indices.append(i)
    if not indices: raise ValueError('at least one full real16-action window required')
    return indices


@torch.inference_mode()
def compare_l16_commands(sequence, predictions, controller_contract, *, batch_windows=32):
    """Each rollout retains its own command state over all16 actual actions."""
    from .tabletop_vision_stage6_workflow import verify_command_windows
    from .tabletop_vision_command_shadow import shadow_sequence
    if type(batch_windows) is not int or batch_windows < 1 or not predictions:
        raise ValueError('positive window batch and explicit model predictions required')
    windows = _full_windows(sequence)
    for value in predictions.values():
        if (not isinstance(value, torch.Tensor) or value.shape != sequence.data['actions'].shape
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError('finite matching original-row model actions required')
    # Recompute original parity before any Student projection; a saved boolean
    # alone cannot validate the present inputs or implementation.
    parity = verify_command_windows(sequence, controller_contract, batch_windows=batch_windows)
    differences = {model: {name: [] for name in COMMAND_SCALES} for model in predictions}
    records = []
    for start in range(0, len(windows), batch_windows):
        selected = windows[start:start+batch_windows]; batch = sequence.batch(selected)
        if not bool(batch['valid'].all()): raise ValueError('full16 action mask required')
        mask = batch['physics_valid']
        if not bool(mask[:, -1].any(-1).all()): raise ValueError('real last-action physical substep required')
        initial = {name: value[:, 0].clone() for name, value in batch['controller_start'].items()}
        teacher = shadow_sequence(initial, batch['actions'], batch['physics_context'], controller_contract, physics_valid=mask)
        endpoints = mask[:, -1].long().sum(-1)-1
        # Validity must be a contiguous prefix in a terminal action.
        expected = torch.arange(8)[None, :] <= endpoints[:, None]
        if not torch.equal(mask[:, -1], expected): raise ValueError('last-action valid substeps must form an actual contiguous prefix')
        bindex = torch.arange(len(selected))
        for model, action in predictions.items():
            student = shadow_sequence({k: v.clone() for k, v in initial.items()}, action[batch['row_indices']],
                batch['physics_context'], controller_contract, physics_valid=mask)
            for name in COMMAND_SCALES:
                differences[model][name].append((student[name][bindex, -1, endpoints]-teacher[name][bindex, -1, endpoints]).clone())
        for k, index in enumerate(selected):
            window = sequence.windows[index]; rows = window['row_indices']; first = sequence.data['sample_metadata'][rows[0]]
            records.append(dict(window_index=index, row_indices=list(rows), run_uuid=first['run_uuid'], env_id=first['env_id'],
                source_episode_uid=first['episode_uid'], first_policy_step=first['tick'], last_policy_step=first['tick']+15,
                last_valid_substep=int(endpoints[k])+1,
                event_neighborhoods=[name for name in EVENTS if bool(sequence.event_labels[name][rows].any())]))
    results = {}
    for model, groups in differences.items():
        values = {name: torch.cat(parts) for name, parts in groups.items()}
        statistics = {}
        for name, value in values.items():
            def stats(rows): return dict(raw=_absolute_statistics(rows), normalized=_absolute_statistics(rows/COMMAND_SCALES[name]))
            statistics[name] = dict(**stats(value), unit=COMMAND_UNITS[name], normalization_scale=COMMAND_SCALES[name],
                by_dimension=[stats(value.flatten(1)[:, j]) for j in range(value[0].numel())],
                event_strata={event: stats(value[torch.tensor([event in row['event_neighborhoods'] for row in records])]) for event in EVENTS})
        results[model] = dict(signed_endpoint_differences=values, groups=statistics)
    return dict(teacher_exact_parity=parity, full_window_count=len(records), skipped_non16_windows=len(sequence.windows)-len(records),
        windows=records, models=results, normalization_scales=COMMAND_SCALES.copy(), units=COMMAND_UNITS.copy(),
        endpoint='after last actual valid physics substep of policy action16;Student minus Teacher accumulated commands',
        feedback_scope='same saved feedback;each model and Teacher maintains its own cumulative commands;not future physics',
        event_scope='window intersects overlapping SequenceDataset event neighborhoods;not online phases',
        sampling_scope='every legal full rolling L16 window once;overlapping rows are repeated in command-window statistics',
        native_jobs_executed=0, control_modified=False, cause_claimed=False)


def file_identity(path):
    path = Path(path).resolve(strict=True); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def _checked(identity, parent=None):
    if (not isinstance(identity, dict) or not isinstance(identity.get('path'), str)
            or not isinstance(identity.get('sha256'), str) or len(identity['sha256']) != 64):
        raise ValueError('explicit actual file path/SHA256 required')
    path = Path(identity['path'])
    if not path.is_absolute(): path = Path(parent or '.')/path
    actual = file_identity(path)
    if actual['sha256'] != identity['sha256'].lower(): raise ValueError('immutable input SHA256 mismatch: '+str(path))
    return actual


def completed_model(report_path, model, prepared, source):
    report_id = file_identity(report_path); report = json.loads(Path(report_id['path']).read_text())
    if (model not in ('B0', 'B1', 'B2') or report.get('profile') != model or report.get('status') != 'completed'
            or report.get('steps') != 3000 or report.get('requested_steps') != 3000
            or report.get('valid_exposures') != 192000 or report.get('source_immutable') is not True
            or report.get('ppo_updates') != 0 or report.get('checkpoint_selection') != 'prespecified_final_update_not_train_MSE'
            or report.get('prepared') != prepared or report.get('source_checkpoint') != source):
        raise ValueError('bound completed3000 report required;no interim model or MSE selection')
    final = [row for row in report.get('checkpoints', []) if row.get('status') == 'completed' and row.get('step') == 3000]
    if len(final) != 1: raise ValueError('unique completed update3000 checkpoint required')
    return dict(checkpoint=_checked(final[0], Path(report_id['path']).parent), training_report=report_id,
        checkpoint_selection='unique_completed_update3000;not_MSE', report=report)


def audit_source_preservation(source, candidate):
    from .tabletop_vision_supervised_bc import _same_state
    allowed = {'stage6_sequence_bc', 'stage6_bc_optimizer', 'stage6_bc_sampling_rng', 'stage6_pointnet_probe', 'stage6_diagnostic_input'}
    if set(source)-set(candidate) or set(candidate)-set(source)-allowed: raise ValueError('original non-Actor keys changed or unknown appended state')
    required = ('optimizer', 'learning_rate', 'iteration', 'timesteps', 'episodes', 'successes', 'torch_rng', 'cuda_rng', 'contract')
    if any(key not in source for key in required): raise ValueError('explicit original PPO/counters/RNG/contract required')
    if set(source['policy']) != set(candidate['policy']): raise ValueError('original policy layout changed')
    names = [key for key in source['policy'] if key.startswith('actor.')]
    if not names or not any(key.startswith('critic.') for key in source['policy']) or not any('log_std' in key for key in source['policy']):
        raise ValueError('original Actor/Critic/noise state required')
    if any(not _same_state(source[key], candidate[key]) for key in source if key != 'policy'):
        raise ValueError('original non-Actor PPO/LR/counters/RNG/contract changed')
    if any(not _same_state(value, candidate['policy'][key]) for key, value in source['policy'].items() if not key.startswith('actor.')):
        raise ValueError('original non-Actor Critic/noise state changed')
    square = 0.
    for key in names:
        left, right = source['policy'][key], candidate['policy'][key]
        if left.shape != right.shape or left.dtype != right.dtype or not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise ValueError('original Actor parameter layout or finiteness changed')
        square += float((left.double()-right.double()).square().sum())
    return dict(all_original_non_actor_state_unchanged=True, actor_parameter_l2_distance=square**.5,
        exact_checked=['all original top-level non-policy state', 'Critic', 'log_std', 'PPO Adam', 'LR', 'counters', 'CPU/CUDA RNG', 'original contract'],
        independent_added_bc_fields=sorted(set(candidate)-set(source)))


def _load_actor(saved):
    from .tabletop_bimanual_checkpoint_contract import unwrap_physical_adapter_checkpoint, VERSION as ADAPTER_VERSION
    from .tabletop_vision_point_activation import point_activation_from_contract
    from .tabletop_vision_quaternion import HEMISPHERE_WRIST_QUATERNIONS
    from .tabletop_vision_squashed_policy import UnboundedVisionStudentActor
    core = unwrap_physical_adapter_checkpoint(saved, allow_explicit=True); contract = core['contract']
    if (saved.get('version') != ADAPTER_VERSION or point_activation_from_contract(contract) != 'relu'
            or contract.get('actor_wrist_quaternion_preprocessing') != HEMISPHERE_WRIST_QUATERNIONS
            or contract.get('network', {}).get('actor_output') != 'unbounded_latent_mean'):
        raise ValueError('actual V4 ReLU/hemisphere unbounded Actor required')
    with torch.random.fork_rng(devices=[]):
        actor = UnboundedVisionStudentActor(wrist_quaternion_preprocessing=HEMISPHERE_WRIST_QUATERNIONS, point_activation='relu')
        actor.load_state_dict({key[6:]: value for key, value in saved['policy'].items() if key.startswith('actor.')}, strict=True)
    if not isinstance(actor.action_net[-1], torch.nn.Identity): raise ValueError('unbounded Actor must not contain final tanh')
    return actor.eval().requires_grad_(False)


@torch.inference_mode()
def bounded_predictions(actor, observations, *, batch_size=128):
    if (observations.ndim != 2 or observations.shape[1] != 3844 or not torch.isfinite(observations).all()
            or type(batch_size) is not int or batch_size < 1): raise ValueError('original finite N3844 observations required')
    results = []
    for start in range(0, len(observations), batch_size):
        output = actor(observations[start:start+batch_size].to(device='cpu', dtype=torch.float32)).tanh()
        if output.shape != (min(batch_size, len(observations)-start), 24) or not torch.isfinite(output).all():
            raise ValueError('finite deterministic bounded action24 required')
        results.append(output.cpu())
    return torch.cat(results)


def _dependencies():
    # The pure command kernel imports this CPU module lazily on its first
    # safe-z projection. Load that actual closure before taking the snapshot;
    # a newly imported module is not evidence that its bytes changed.
    from .tabletop_vision_command_shadow import shadow_sequence
    from .tabletop_tensor_world import constrain_palm_z_movement
    base = Path(__file__).resolve().parent
    result = {name: file_identity(module.__file__) for name, module in sorted(sys.modules.items())
        if name.startswith('go2_pvcnn') and getattr(module, '__file__', None) and base in Path(module.__file__).resolve().parents}
    result['analysis_implementation'] = file_identity(__file__)
    return result


def _json_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream: json.dump(value, stream, indent=2, allow_nan=False)


def analyze_offline(prepared_path, training_reports, output, *, batch_size=128, batch_windows=32):
    """Re-forward actual final four models, recheck source and command parity."""
    from . import tabletop_vision_stage6_workflow as workflow
    from .tabletop_vision_sequence_dataset import SequenceDataset
    from .tabletop_vision_command_shadow import shadow_sequence
    from .tabletop_bimanual_checkpoint_contract import unwrap_physical_adapter_checkpoint
    from .tabletop_vision_supervision_dataset import _canonical_visual_contract
    if set(training_reports) != {'B0', 'B1', 'B2'}: raise ValueError('all three actual completed B0/B1/B2 reports required')
    target = Path(output).resolve()
    if target.exists(): raise FileExistsError('analysis output already exists: '+str(target))
    if target == Path(__file__).resolve().parents[4] or Path(__file__).resolve().parents[4] in target.parents:
        raise ValueError('private output must be outside source workspace')
    prepared_id = file_identity(prepared_path); prepared = json.loads(Path(prepared_id['path']).read_text())
    if prepared.get('version') != 'dh116-stage6-prepared-sequences-v1': raise ValueError('actual prepared02 manifest required')
    input_id = _checked(prepared['inputs']['input_manifest'], Path(prepared_id['path']).parent)
    inputs = workflow.load_stage6_inputs(input_id['path'])
    if inputs != prepared['inputs']: raise ValueError('prepared full input bindings changed')
    identities = [prepared_id, input_id]
    for key, source_key in [('dataset', 'dataset'), ('context', 'sequence_context'), ('teacher_report', 'teacher_collection_report')]:
        if prepared[key] != inputs[source_key]: raise ValueError('prepared input differs: '+key)
        identities.append(_checked(prepared[key], Path(prepared_id['path']).parent))
    identities.append(_checked(prepared['index'], Path(prepared_id['path']).parent))
    source_id = _checked(inputs['source_checkpoint'], Path(prepared_id['path']).parent); identities.append(source_id)
    if source_id['sha256'] != SOURCE_SHA256: raise ValueError('exact original c58 source SHA256 required')
    workflow.require_native_shadow_parity(prepared.get('teacher_shadow_parity'))
    if prepared.get('accumulated_window_parity', {}).get('passed') is not True: raise ValueError('qualified prepared accumulated parity required')
    models = {'E0': dict(checkpoint=source_id, checkpoint_selection='actual original c58;not_MSE')}
    for name in ('B0', 'B1', 'B2'):
        models[name] = completed_model(training_reports[name], name, prepared_id, source_id)
        identities.extend([models[name]['checkpoint'], models[name]['training_report']])
    data = torch.load(prepared['dataset']['path'], map_location='cpu', weights_only=False)
    context = torch.load(prepared['context']['path'], map_location='cpu', weights_only=False)
    if data['observations'].shape != (4698, 3844) or data['actions'].shape != (4698, 24):
        raise ValueError('exact same actual4698 Teacher observation rows required')
    sequence = SequenceDataset(data, context, dataset_sha256=prepared['dataset']['sha256'], sampling_mode='event')
    if sequence.report['covered_original_rows'] != 4698 or sequence.report['excluded_rows'] != 0: raise ValueError('all actual rows required;no silent filtering')
    source = torch.load(source_id['path'], map_location='cpu', weights_only=False)
    sensor, actor_contract = deepcopy(data['visual_contract']), deepcopy(unwrap_physical_adapter_checkpoint(source, allow_explicit=True)['contract'])
    for contract in (sensor, actor_contract):
        contract.pop('actor_wrist_quaternion_preprocessing', None); contract.pop('actor_pointnet_activation', None)
    if _canonical_visual_contract(sensor) != _canonical_visual_contract(actor_contract): raise ValueError('source sensor/physics/action contract differs from saved observations')
    actors, preservation = {}, {}
    for name, model in models.items():
        saved = source if name == 'E0' else torch.load(model['checkpoint']['path'], map_location='cpu', weights_only=False)
        if name != 'E0':
            meta = saved.get('stage6_sequence_bc', {})
            if (meta.get('profile') != name or meta.get('status') != 'completed' or meta.get('updates') != 3000
                    or meta.get('source_checkpoint') != source_id or meta.get('dataset') != prepared['dataset'] or meta.get('context') != prepared['context']):
                raise ValueError('actual final model metadata does not bind its report: '+name)
        preservation[name] = audit_source_preservation(source, saved); actors[name] = _load_actor(saved)
    dependencies = _dependencies(); rng = torch.get_rng_state().clone()
    observations_before, actions_before = data['observations'].clone(), data['actions'].clone()
    print('STAGE7_OFFLINE: four actual final models bound; recomputing full L16 Teacher parity', flush=True)
    # Hard parity before even the independent same-observation forward.
    parity = workflow.verify_command_windows(sequence, context['controller_contract'], batch_windows=batch_windows)
    predicted, statistics = {}, {}
    for name, actor in actors.items():
        parameters = {key: value.clone() for key, value in actor.state_dict().items()}
        predicted[name] = bounded_predictions(actor, data['observations'], batch_size=batch_size)
        if any(not torch.equal(value, actor.state_dict()[key]) for key, value in parameters.items()): raise RuntimeError('forward changed model parameters')
        statistics[name] = action_statistics(predicted[name], data['actions'], sequence.event_labels)
        print('STAGE7_OFFLINE: same-observation forward completed '+name, flush=True)
    print('STAGE7_OFFLINE: independently accumulating every full L16 window for four models', flush=True)
    commands = compare_l16_commands(sequence, predicted, context['controller_contract'], batch_windows=batch_windows)
    if not torch.equal(rng, torch.get_rng_state()) or not torch.equal(observations_before, data['observations']) or not torch.equal(actions_before, data['actions']):
        raise RuntimeError('offline analysis mutated RNG or saved input tensors')
    before = {item['path']: item for item in identities}
    after = {path: file_identity(path) for path in before}
    code_after = _dependencies()
    if before != after or dependencies != code_after:
        changes = dict(input_paths=[path for path in before if before[path] != after[path]],
            implementation_modules=[name for name in sorted(set(dependencies)|set(code_after)) if dependencies.get(name) != code_after.get(name)])
        raise RuntimeError('immutable inputs or implementation changed during analysis: '+json.dumps(changes, sort_keys=True))
    target.mkdir(parents=True, exist_ok=False)
    command_tensors = {name: result.pop('signed_endpoint_differences') for name, result in commands['models'].items()}
    tensor_path = target/'conditional_l16_signed_endpoint_differences.pt'
    with tensor_path.open('xb') as stream:
        torch.save(dict(version=VERSION, windows=commands['windows'], units=COMMAND_UNITS, normalization_scales=COMMAND_SCALES,
            Student_minus_Teacher=command_tensors, source_dataset=prepared['dataset'], context=prepared['context']), stream)
    report = dict(version=VERSION, status='completed_offline_only', rows=4698, actor_dim=3844, critic_dim=371, action_dim=24,
        prepared=prepared_id, models={name: {key: value for key, value in model.items() if key != 'report'} for name, model in models.items()},
        action_statistics=statistics, conditional_l16_commands=commands, command_endpoint_tensors=file_identity(tensor_path),
        original_source_state_preservation=preservation, teacher_exact_parity_before_forward=parity,
        immutable_inputs_before=before, immutable_inputs_after=after, source_dependencies=dependencies,
        all_input_file_sha256_unchanged=True, saved_observations_actions_and_rng_unchanged=True,
        forward_contract='V4 unbounded Actor;inherited ReLU/hemisphere preprocessing;exactly one final tanh;float32 saved3844D sensor input only',
        event_labels_scope='overlapping original SequenceDataset event neighborhoods;not online phase partitions',
        training_anchor_loss_is_overall_pool_metric=False, model_selection_by_metric=False, causality_claimed=False,
        physical_success_claimed=False, heldout_generalization_claimed=False, native_jobs_executed=0,
        training_updates=0, ppo_updates=0, torch_threads=torch.get_num_threads(), cuda_used=False)
    _json_new(target/'teacher_student_divergence.json', report)
    _json_new(target/'command.json', dict(command=[sys.executable, *sys.argv], implementation=file_identity(__file__),
        environment={key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}))
    return report


def require_native_event_scope(report, checkpoint):
    """A bounded branch is censored; a source must own all four first terminals."""
    if (report.get('native_runtime_verified') is not True or report.get('error') is not None or report.get('ppo_updates') != 0
            or report.get('inputs', {}).get('checkpoint') != checkpoint
            or report.get('mode') not in ('source', 'branch')):
        raise ValueError('actual source/branch execution bound to the selected final model required')
    if report['mode'] == 'source':
        rows = report.get('source_rows', [])
        if (report.get('status') != 'completed' or report.get('completed_count') != 4
                or len(rows) != 4 or {row.get('env_id') for row in rows} != set(range(4))
                or any(not isinstance(row.get('terminal'), dict) or row['terminal'].get('action_owner') != 'student'
                    or row.get('teacher_prefix_policy_steps_executed') != 0 or row.get('teacher_replay_policy_steps_executed') != 0 for row in rows)):
            raise ValueError('complete pure Student source with four owned first terminals required;partial source refused')
    elif report.get('status') not in ('completed', 'bounded_branch_completed'):
        raise ValueError('completed finite branch execution required;bounded horizon remains censored')


def _event_with_pointers(row, **extra):
    from .tabletop_vision_physical_divergence import _event
    result = _event(row, raw_pointer=deepcopy(row.get('raw_pointer')), diagnostic_pointer=deepcopy(row.get('diagnostic_pointer')), **extra)
    if type(row.get('sdk_physics_step')) is int:
        sdk = row['sdk_physics_step']; derived = False
        source = ('actual_clock_before_action.simulation_step_index' if row.get('phase') == 'actual_pre_action_policy_boundary'
            else 'actual_recorded_Teacher_post_feedback_and_backend_getter_pair')
    elif type(row.get('derived_sdk_physics_step')) is int:
        sdk = row['derived_sdk_physics_step']; derived = True
        source = 'derived_from_actual_policy_World_SDK_anchor_offset'
    else: sdk, derived, source = None, None, 'unavailable_no_recorded_or_validated_derived_SDK_step'
    result.update(sdk_physics_step=sdk, sdk_physics_step_is_derived=derived, sdk_step_source=source)
    result.setdefault('phase', row.get('phase'))
    return result


def _reference_info(run_uuid, row):
    return dict(_event_with_pointers(row), execution_run_uuid=run_uuid,
        reference_kind=row.get('action_owner'), owner_scope='actual_comparison_reference;not_Student_action_or_causality')


def _pad_sample(row):
    pad = row.get('pad')
    if row.get('measurement_valid') is not True or not isinstance(pad, dict): return None
    load, accepted = np.asarray(pad.get('load_n')), np.asarray(pad.get('accepted'))
    if load.shape != (2, 5) or accepted.shape != (2, 5) or accepted.dtype != np.bool_ or not np.isfinite(load).all() or (load < 0).any():
        raise ValueError('healthy native five-pad loads/accepted booleans required')
    if not np.array_equal(load > 0., accepted): raise ValueError('accepted contact must equal healthy positive native pad load')
    return load, accepted, accepted & (load > .05)


def _tracking_pair(row):
    if row.get('tracking_pair_available') is not True: return None
    q, target = row.get('q_rad'), row.get('backend_target_rad')
    if q is None or target is None: return None
    q, target = np.asarray(q), np.asarray(target)
    return target-q if q.shape == (12,) and target.shape == (12,) and np.isfinite(q).all() and np.isfinite(target).all() else None


def _named_tracking(row):
    if row.get('named_tracking') is not None: return stage_values(row['named_tracking'])
    value = _tracking_pair(row)
    if value is None: return {}
    from .linear_coupling import ACTIVE
    return {f'{side}.{joint}': dict(value=float(value[h*6+j]), unit='rad')
        for h, side in enumerate(('left', 'right')) for j, joint in enumerate(ACTIVE)}


def _joint_control_role(joint):
    from .linear_coupling import ACTIVE, MIMIC
    from .tabletop_joint_target_readback import WRIST_LINEAR, WRIST_ANGULAR
    if joint in ACTIVE: return 'ACTIVE_controlled_finger'
    if joint in WRIST_LINEAR+WRIST_ANGULAR: return 'driven_wrist'
    if joint in {name for name, *_ in MIMIC}: return 'passive_mimic_backend_diagnostic'
    return 'unknown_control_role'


def _stages(row):
    from .linear_coupling import ACTIVE
    result = {name: stage_values(packet) for name, packet in (row.get('target_stages') or {}).items()}
    target = row.get('backend_target_rad')
    if target is not None:
        result['physx_backend_active_fingers'] = {f'{side}.{name}': dict(value=float(target[hi*6+j]), unit='rad')
            for hi, side in enumerate(('left', 'right')) for j, name in enumerate(ACTIVE)}
    return result


def stage_values(packet):
    """Compact file packets keep native named units; no mixed-unit maximum."""
    if set(packet) == {'values', 'names', 'units'}:
        value = np.asarray(packet['values']).reshape(-1)
        if len(value) != len(packet['names']) or len(value) != len(packet['units']) or len(set(packet['names'])) != len(value):
            raise ValueError('actual unique named target units required')
        return {name: dict(value=float(v), unit=unit) for name, v, unit in zip(packet['names'], value, packet['units'])}
    return packet


def _jsonl_pointer(identity):
    with gzip.open(identity['path'], 'rt', encoding='utf-8') as stream:
        for line, text in enumerate(stream, 1):
            value = json.loads(text)
            value['raw_pointer'] = dict(**identity, line=line)
            yield value


def load_native_event_fields(run):
    """Enrich only samples already bound by the published physical loader.

    Raw body IDs, raw contact impulses and independent getter clocks are not
    reconstructed from finite force vectors or a callback timestamp.
    """
    from . import tabletop_vision_physical_divergence as physical
    for identity in run['input_identities']: _checked(identity)
    provenance = run['provenance']; result = dict(run, samples=[dict(row) for row in run['samples']])
    samples = {(row['env_id'], row['world_physics_step'], row['timestamp_s']): row for row in result['samples']}
    policies = list(_jsonl_pointer(provenance['policy_trace'])); result['policy_rows'] = policies
    seen = set()
    for raw in _jsonl_pointer(provenance['raw_physics']):
        key = (raw['env_id'], raw['physics_step'], raw['timestamp_s'])
        if key not in samples: raise ValueError('extra raw fields outside the already bound actual execution')
        row = samples[key]; payload = raw['payload']; seen.add(key)
        row.update(raw_pointer=raw['raw_pointer'], measurement_valid=payload.get('measurement_valid') is True,
            prior_anchor_latched=payload.get('prior_anchor_latched'), slip_applicable=payload.get('slip_applicable'),
            slip_reason_code=payload.get('slip_reason_code'), pad=None, target_stages={}, unavailable_target_stages=[],
            unavailable_contact_fields=dict(native_body_id=None, raw_normal=None, raw_impulse=None, independent_getter_bracket=None))
    if seen != set(samples): raise ValueError('bound physical sample has no exact extra raw source')
    audit = json.loads(Path(provenance['audit_summary']['path']).read_text())
    statuses = {side: sorted([e for e in audit['source_events'] if e.get('kind') == 'target_source_status' and e.get('hand') == side],
        key=lambda e: e['physics_step']) for side in ('left', 'right')}
    xyz_fields = ('requested_palm_target_m', 'pre_projection_palm_target_m', 'projected_palm_target_m',
        'final_sdk_palm_target_m', 'control_frame_sdk_target_m')
    diagnostic_seen = set()
    for identity in audit['files']:
        current = _checked(identity)
        if Path(current['path']).suffix != '.npz': continue
        with np.load(current['path'], allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        for batch, slot in np.ndindex(arrays['env_id'].shape):
            key = (int(arrays['env_id'][batch, slot]), int(arrays['physics_step'][batch, slot]), float(arrays['timestamp_s'][batch, slot]))
            if key not in samples: continue
            if key in diagnostic_seen: raise ValueError('duplicate extra native diagnostic identity')
            diagnostic_seen.add(key); row = samples[key]; select = batch, slot
            row['diagnostic_pointer'] = dict(**current, batch=batch, env_index=slot)
            def take(name):
                if name not in arrays or any(flag not in arrays or not bool(arrays[flag][select])
                    for flag in (name+'__available', name+'__finite', name+'__source_valid')): return None
                return arrays[name][select].copy()
            load, accepted = take('effective_pad_loads_n'), take('effective_pad_valid')
            if load is not None and accepted is not None:
                row['pad'] = dict(load_n=load, accepted=accepted, force_vector_w_n=take('effective_pad_vectors_w'),
                    weighted_contact_position_w_m=take('effective_pad_points_w'), depth_m=take('effective_pad_depth_m'))
            for field in xyz_fields:
                value = take(field)
                if value is None: row['unavailable_target_stages'].append(field); continue
                row['target_stages'][field] = dict(values=value.reshape(-1),
                    names=tuple(f'{side}.legacy_base_link_offset_chart.{axis}' for side in ('left', 'right') for axis in 'xyz'), units=('m',)*6)
            value = take('root_target_w_submission')
            if value is not None:
                row['target_stages']['root_target_w_submission'] = dict(values=value.reshape(-1),
                    names=tuple(f'{side}.base_link_target_w.{axis}' for side in ('left', 'right') for axis in ('x','y','z','qw','qx','qy','qz')),
                    units=('m','m','m','unitless','unitless','unitless','unitless')*2)
            else: row['unavailable_target_stages'].append('root_target_w_submission')
            for side in ('left', 'right'):
                eligible = [e for e in statuses[side] if e['physics_step'] <= row['world_physics_step'] and e.get('reset_epoch') == 0]
                for quantity, stage in (('backend_target', side+'_physx_backend'), ('command_cache', side+'_command_cache')):
                    value = take(side+'_'+quantity)
                    evidence = eligible[-1]['evidence'] if eligible else {}
                    names, units = evidence.get('joint_names', []), evidence.get('units_by_name', {})
                    if value is None or len(value) != len(names) or any(name not in units for name in names):
                        row['unavailable_target_stages'].append(stage); continue
                    row['target_stages'][stage] = dict(values=value, names=tuple(names), units=tuple(units[name] for name in names))
            tracking_values, tracking_names, tracking_units = [], [], []
            for side in ('left', 'right'):
                q, backend = take(side+'_q'), take(side+'_backend_target')
                packet = row['target_stages'].get(side+'_physx_backend')
                if q is None or backend is None or take(side+'_dof_types') is None or packet is None: continue
                tracking_values.extend((backend.astype(np.float64)-q.astype(np.float64)).tolist())
                tracking_names.extend(side+'.'+name for name in packet['names']); tracking_units.extend(packet['units'])
            row['named_tracking'] = (dict(values=np.asarray(tracking_values), names=tuple(tracking_names), units=tuple(tracking_units))
                if tracking_values else None)
    if diagnostic_seen != set(samples): raise ValueError('bound sample has no exact extra native diagnostic packet')
    for identity in run['input_identities']: _checked(identity)
    result['extra_source_provenance'] = dict(raw_measurement_health='original payload.measurement_valid=finite_physics and adapter_fault_bits==0',
        pad_source='native object-only filtered real body-local accepted contact manifold;not FK prediction',
        native_contact_normal_available=False, native_contact_impulse_available=False, native_contact_body_id_available=False,
        vector_scope='sum of accepted native force scalar times normal ON HAND;not a unit normal',
        independent_getter_clock_bracket_available=False, named_palm_link_available=False)
    return result


def bind_policy_teacher_actions(policy_rows, data, report, dataset_identity):
    """Time-aligned references must be actual labels from the exact source key."""
    cases = {row['env_id']: row for row in report['source_case_definitions']}
    index = {}
    for i, row in enumerate(data['sample_metadata']):
        key = (row['run_uuid'], row['env_id'], row['episode_uid'], row['tick'])
        if key in index: raise ValueError('duplicate original Teacher action identity')
        index[key] = i
    result = []
    for raw in policy_rows:
        row = dict(raw); case = cases[row['env_id']]
        if row.get('recorded_teacher_episode_uid') != case['source_episode_uid']:
            raise ValueError('time-aligned Teacher reference source UID changed')
        key = case['source_run_uuid'], row['env_id'], case['source_episode_uid'], row['policy_step']
        label = row.get('recorded_teacher_action')
        if key not in index:
            if label is not None: raise ValueError('Teacher action claimed beyond original source tape')
        else:
            i = index[key]
            if label is None or not np.array_equal(np.asarray(label, dtype=np.float64), data['actions'][i].double().numpy()):
                raise ValueError('recorded Teacher action reference differs from actual four-key source label')
            row['source_teacher_pointer'] = dict(**dataset_identity, row_index=i, sample_key=list(key))
        result.append(row)
    return result


def attach_original_teacher_post_pointers(original, context, context_identity):
    """Locate independent post-q and backend records by their exact source key."""
    from .tabletop_vision_teacher_tracking_calibration import _identity
    tables = {}
    for name in ('post_physics_feedback', 'native_backend_targets'):
        tables[name] = {}
        for i, record in enumerate(context[name]):
            key = _identity(record['sample_key'])
            if key in tables[name]: raise ValueError('duplicate original Teacher post/backend pointer identity')
            tables[name][key] = i
    result = dict(original, samples=[dict(row) for row in original['samples']])
    for row in result['samples']:
        key = original['execution_run_uuid'], row['env_id'], row['source_episode_uid'], row['policy_step']
        for table, field in (('post_physics_feedback', 'raw_pointer'), ('native_backend_targets', 'diagnostic_pointer')):
            if key not in tables[table]: raise ValueError('actual original Teacher post/backend sample key missing')
            index = tables[table][key]; sub = row['substep']-1
            packet = context[table][index]['physics_substeps'][sub]
            if (packet.get('sdk_physics_step'), packet.get('sdk_timestamp_s')) != (row['sdk_physics_step'], row['timestamp_s']):
                raise ValueError('original Teacher pointer does not match actual post/backend SDK clock')
            row[field] = dict(**context_identity, table=table, entry_index=index, sample_key=list(key), physics_substep_index=sub)
    return result


def save_native_six_first(config_path, output):
    """File entry: original physical binding first, then six independent events.

    A source config cannot ingest an in-progress B2 report. Branch horizons
    remain explicitly censored and never become pure-Student source outcomes.
    """
    from . import tabletop_vision_physical_divergence as physical
    from . import tabletop_vision_stage6_workflow as workflow
    config_id = file_identity(config_path); config = json.loads(Path(config_id['path']).read_text())
    implementations = [file_identity(__file__), file_identity(physical.__file__)]
    if config.get('version') != 'dh116-stage7-six-first-inputs-v1' or config.get('model') not in ('E0', 'B0', 'B1', 'B2'):
        raise ValueError('explicit Stage7 six-first config/model required')
    target = Path(output).resolve()
    if target.exists(): raise FileExistsError(target)
    if Path(__file__).resolve().parents[4] in target.parents: raise ValueError('private analysis output must be outside source')
    prepared_id = _checked(config['prepared']); prepared = json.loads(Path(prepared_id['path']).read_text())
    if prepared.get('version') != 'dh116-stage6-prepared-sequences-v1': raise ValueError('actual prepared02 input required')
    inputs = workflow.load_stage6_inputs(_checked(prepared['inputs']['input_manifest'])['path'])
    if (inputs != prepared['inputs'] or inputs['source_checkpoint']['sha256'] != SOURCE_SHA256
            or any(prepared[key] != inputs[name] for key, name in (
                ('dataset', 'dataset'), ('context', 'sequence_context'), ('teacher_report', 'teacher_collection_report')))):
        raise ValueError('original c58/full prepared input binding changed')
    source_id = _checked(inputs['source_checkpoint']); model = config['model']; identities = [config_id, prepared_id, source_id]
    if model == 'E0': checkpoint = source_id
    else:
        report_id = _checked(config['training_report']); identities.append(report_id)
        selection = completed_model(report_id['path'], model, prepared_id, source_id); checkpoint = selection['checkpoint']
    identities.append(checkpoint)
    source = torch.load(source_id['path'], map_location='cpu', weights_only=False)
    candidate_saved = source if model == 'E0' else torch.load(checkpoint['path'], map_location='cpu', weights_only=False)
    preservation = audit_source_preservation(source, candidate_saved)
    if model != 'E0':
        meta = candidate_saved.get('stage6_sequence_bc', {})
        if (meta.get('profile') != model or meta.get('status') != 'completed' or meta.get('updates') != 3000
                or meta.get('source_checkpoint') != source_id or meta.get('dataset') != prepared['dataset'] or meta.get('context') != prepared['context']):
            raise ValueError('actual completed checkpoint metadata differs from prepared source')
    physical_id = _checked(config['physical_config']); identities.append(physical_id)
    native_config = json.loads(Path(physical_id['path']).read_text())
    if native_config['dataset'] != prepared['dataset'] or native_config['context'] != prepared['context']:
        raise ValueError('native divergence belongs to a different saved Teacher source pool')
    report_id = _checked(native_config['candidate']['report']); native_report = json.loads(Path(report_id['path']).read_text())
    require_native_event_scope(native_report, checkpoint)
    # Include optional new event_reference diagnostics without changing the
    # original physical loader's checkpoint/job/UID or four-source bindings.
    for item in native_report['inputs'].values():
        if isinstance(item, dict) and {'path', 'sha256'} <= set(item): identities.append(_checked(item))
    target.mkdir(parents=True, exist_ok=False)
    physical_path = target/'physical_repeat_analysis.json'
    verify_tail = native_report['mode'] == 'branch'
    existing = physical.save_physical_divergence(physical_id['path'], physical_path, verify_branch_complete_tail=verify_tail)
    identities.extend(existing['input_identities'])
    calibration = json.loads(Path(native_config['calibration']['path']).read_text())
    data = torch.load(native_config['dataset']['path'], map_location='cpu', weights_only=False)
    context = torch.load(native_config['context']['path'], map_location='cpu', weights_only=False)
    candidate = load_native_event_fields(physical.load_native_run(native_config['candidate'],
        dataset_id=native_config['dataset'], context_id=native_config['context'], calibration=calibration, verify_branch_complete_tail=verify_tail))
    reference_files = native_config.get('native_teacher_repeats', [native_config.get('teacher_repeat')])
    references = [load_native_event_fields(physical.load_native_run(files, dataset_id=native_config['dataset'],
        context_id=native_config['context'], calibration=calibration)) for files in reference_files]
    original = attach_original_teacher_post_pointers(physical.load_teacher_context(data, context, calibration,
        provenance=dict(dataset=native_config['dataset'], context=native_config['context'])), context, native_config['context'])
    policies = bind_policy_teacher_actions(candidate['policy_rows'], data, native_report, native_config['dataset'])
    reduced = summarize_six_first(candidate, [original, *references], policies, calibration['trigger_rad'])
    authoritative = {row['env_id']: row for row in existing['analysis']['per_env']}
    for env in reduced['per_env']:
        # The significant event is recomputed by the published analyzer, with
        # its full actual SDK/source qualification, never a local new threshold.
        local, proven = env['first_significant_tracking_departure'], authoritative[env['env_id']]['first_significant_tracking_departure']
        if (local is None) != (proven is None) or local is not None and (local['policy_step'], local['substep']) != (proven['policy_step'], proven['substep']):
            raise ValueError('six-event tracking disagrees with the published qualified physical analyzer')
    identities.extend(implementations)
    unique = {row['path']: row for row in identities}
    for identity in unique.values(): _checked(identity)
    result = dict(version=VERSION, status='completed_read_only_six_first_analysis', model=model, mode=native_report['mode'],
        input_config=config_id, selected_checkpoint=checkpoint, original_source_state_preservation=preservation,
        execution_run_uuid=candidate['execution_run_uuid'], candidate_provenance=candidate['provenance'],
        extra_source_provenance=candidate['extra_source_provenance'], events=reduced,
        branch_horizon_censored=native_report['mode'] == 'branch',
        physical_repeat_analysis=file_identity(physical_path), inputs_sha256_unchanged=True,
        input_identities=list(unique.values()), implementation=implementations,
        native_jobs_executed=0, new_independent_initial_states=0, physical_success_certificate_issued=False,
        Student_failure_causality_claimed=False, controller_modified=False)
    _json_new(target/'six_first_events.json', result)
    _json_new(target/'command.json', dict(command=[sys.executable, *sys.argv], implementation=implementations,
        environment={name: os.environ.get(name) for name in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}))
    return result


def summarize_six_first(candidate, references, policy_rows, trigger_rad):
    """Separate first events on already bound rows; this pure API certifies no run."""
    from . import tabletop_vision_physical_divergence as physical
    from .linear_coupling import ACTIVE
    samples, excluded = physical._index_run(candidate)
    if len({run['execution_run_uuid'] for run in references}) != len(references):
        raise ValueError('copied reference UUIDs cannot add repeat samples')
    ref_maps = [physical._index_run(run)[0] for run in references]
    if not samples: raise ValueError('actual owned physical samples required')
    envs = {}
    fields = ('first_action_difference', 'first_control_target_difference', 'first_joint_tracking_difference',
        'first_control_target_repeat_range_departure', 'first_joint_tracking_repeat_range_departure', 'first_actual_bilateral_opposed_contact',
        'first_significant_tracking_departure', 'first_finger_pad_contact_loss', 'first_bearing_pad_loss',
        'first_repeat_pad_contact_departure', 'first_object_slip', 'first_nonzero_valid_fixed_anchor_slip', 'first_failure', 'first_terminal')
    for row in samples.values():
        envs.setdefault(row['env_id'], dict(env_id=row['env_id'], source_episode_uid=row['source_episode_uid'],
            execution_episode_uid=row['execution_episode_uid'], **{name: None for name in fields},
            first_any_owned_action_difference=None, first_control_target_difference_by_stage={}, first_control_target_repeat_range_departure_by_stage={}, pad_transitions={},
            observed_counts=dict(valid_fixed_anchor_slip_samples=0, healthy_pad_samples=0, contiguous_pad_transitions=0)))
    unavailable = dict(candidate_pad_samples=0, repeat_pad_samples_below_two=0, target_stage_reference_below_two=0,
        candidate_tracking_pair=0, tracking_reference_below_two=0, qualified_fixed_anchor_slip_samples=0)
    for policy in sorted(policy_rows, key=lambda r: (r['policy_step'], r['env_id'])):
        env = envs.get(policy['env_id'])
        if env is None: continue
        if policy.get('episode_uid') != env['execution_episode_uid'] or policy.get('recorded_teacher_episode_uid') != env['source_episode_uid']:
            raise ValueError('policy action source/execution UID differs from physical rows')
        if policy.get('recorded_teacher_action') is None: continue
        issued, teacher = np.asarray(policy['issued_action'], dtype=np.float64), np.asarray(policy['recorded_teacher_action'], dtype=np.float64)
        if issued.shape != (24,) or teacher.shape != (24,) or not np.isfinite(issued).all() or not np.isfinite(teacher).all():
            raise ValueError('actual finite issued and recorded Teacher action24 required')
        if not (issued != teacher).any(): continue
        first = next(row for row in samples.values() if row['env_id'] == policy['env_id'])
        clock = policy['clock_before_action']; action_row = dict(first, action_owner=policy['owner'], policy_step=policy['policy_step'],
            substep=None, world_physics_step=clock['physics_step'], timestamp_s=clock['timestamp_s'], raw_pointer=policy.get('raw_pointer'), diagnostic_pointer=None,
            phase='actual_pre_action_policy_boundary', clock_quality='actual_saved_pre_action_World_SDK_policy_anchor',
            independent_getter_clock_pair_available=False, sdk_physics_step=clock['simulation_step_index'], derived_sdk_physics_step=None)
        event = _event_with_pointers(action_row, phase='actual_pre_action_policy_boundary', issued_action24=issued.tolist(),
            recorded_teacher_action24=teacher.tolist(), signed_difference24=(issued-teacher).tolist(),
            source_teacher_pointer=deepcopy(policy.get('source_teacher_pointer')),
            clock_before_action=deepcopy(clock), reference_scope='time-aligned recorded actually-issued Teacher action;not a current-state expert query')
        if env['first_any_owned_action_difference'] is None: env['first_any_owned_action_difference'] = event
        if policy['owner'] == 'student' and env['first_action_difference'] is None: env['first_action_difference'] = event
    if len(references) >= 2:
        existing = physical.compare_to_repeat(candidate, physical.build_repeat_envelope(references), trigger_rad)
        for entry in existing['per_env']:
            env = envs[entry['env_id']]; event = entry['first_significant_tracking_departure']
            if event is not None:
                source = next(row for row in samples.values() if row['env_id'] == entry['env_id'] and row['policy_step'] == event['policy_step'] and row['substep'] == event['substep'])
                meta = _event_with_pointers(source)
                for name in ('raw_pointer', 'diagnostic_pointer', 'sdk_physics_step', 'sdk_physics_step_is_derived', 'sdk_step_source', 'phase'):
                    event[name] = meta[name]
                env['first_significant_tracking_departure'] = event
    previous = {}
    digits = ('thumb', 'index', 'middle', 'ring', 'little')
    for key, row in samples.items():
        env = envs[row['env_id']]
        reference_rows = [(run['execution_run_uuid'], mapping[key]) for run, mapping in zip(references, ref_maps) if key in mapping]
        reference_rows.sort(key=lambda entry: entry[1].get('action_owner') != 'original_Teacher02')
        refs = [other for _, other in reference_rows]
        if (row.get('measurement_valid') is True and row.get('contact_valid') is True and row.get('bilateral_opposed_contact') is True
                and env['first_actual_bilateral_opposed_contact'] is None):
            env['first_actual_bilateral_opposed_contact'] = _event_with_pointers(row,
                predicate='actual same-postphysics healthy valid bilateral_opposed_contact;not inferred from individual pad touches')
        reference_stages = [(uid, other, _stages(other)) for uid, other in reference_rows]
        for stage, packet in _stages(row).items():
            entries = [(uid, other, stages[stage]) for uid, other, stages in reference_stages if stage in stages]
            packets = [value for _, _, value in entries]
            if len(packets) < 2: unavailable['target_stage_reference_below_two'] += 1
            if not packets: continue
            if any(set(value) != set(packet) for value in packets): raise ValueError('named control target stage layouts differ')
            differences, range_departures = [], []
            for name, value in packet.items():
                if any(other[name]['unit'] != value['unit'] for other in packets): raise ValueError('control target units differ')
                values = [float(other[name]['value']) for other in packets]; actual = float(value['value'])
                if not np.isfinite([actual, *values]).all(): raise ValueError('finite observed control targets required')
                detail = dict(name=name, unit=value['unit'], actual=actual, reference_value=values[0], reference_min=min(values), reference_max=max(values))
                if actual != values[0]: differences.append(detail)
                if len(values) >= 2 and (actual < min(values) or actual > max(values)): range_departures.append(detail)
            if differences and stage not in env['first_control_target_difference_by_stage']:
                event = _event_with_pointers(row, stage=stage, differences=differences, observed_reference_samples=len(packets),
                    repeat_range_eligible=len(packets) >= 2,
                    comparison_reference=_reference_info(entries[0][0], entries[0][1]),
                    reference_scope='exact original Teacher field when available;otherwise first listed available actual native Teacher execution',
                    classification='numeric_command_stage_difference_from_explicit_Teacher_reference;not_physical_causality;cache_is_not_backend')
                env['first_control_target_difference_by_stage'][stage] = event
                if 'cache' not in stage and env['first_control_target_difference'] is None: env['first_control_target_difference'] = event
            if range_departures and stage not in env['first_control_target_repeat_range_departure_by_stage']:
                event = _event_with_pointers(row, stage=stage, differences=range_departures, observed_reference_samples=len(packets),
                    comparison_references=[_reference_info(uid, other) for uid, other, _ in entries],
                    classification='outside_observed_Teacher_repeat_range;not_confidence_interval_or_causality')
                env['first_control_target_repeat_range_departure_by_stage'][stage] = event
                if 'cache' not in stage and env['first_control_target_repeat_range_departure'] is None: env['first_control_target_repeat_range_departure'] = event
        tracking = _named_tracking(row); reference_tracking = [(uid, other, _named_tracking(other)) for uid, other in reference_rows]
        if not tracking: unavailable['candidate_tracking_pair'] += 1
        elif env['first_joint_tracking_difference'] is None or env['first_joint_tracking_repeat_range_departure'] is None:
            changed, range_departures = [], []
            for name, value in tracking.items():
                entries = [(uid, other, packet[name]['value']) for uid, other, packet in reference_tracking if name in packet and packet[name]['unit'] == value['unit']]
                values = [actual for _, _, actual in entries]
                if len(values) < 2: unavailable['tracking_reference_below_two'] += 1
                if values:
                    side, joint = name.split('.', 1)
                    detail = dict(hand=side, joint=joint, unit=value['unit'], signed_backend_minus_postq=value['value'],
                        original_control_role=_joint_control_role(joint), qualified_ACTIVE_trigger_applicable=joint in ACTIVE,
                        reference_value=values[0], reference_min=min(values), reference_max=max(values), observed_reference_samples=len(values), repeat_range_eligible=len(values)>=2)
                    if value['value'] != values[0]:
                        changed.append(dict(detail, comparison_reference=_reference_info(entries[0][0], entries[0][1])))
                    if len(values) >= 2 and (value['value'] < min(values) or value['value'] > max(values)):
                        range_departures.append(dict(detail, comparison_references=[_reference_info(uid, other) for uid, other, _ in entries]))
            if changed and env['first_joint_tracking_difference'] is None: env['first_joint_tracking_difference'] = _event_with_pointers(row,
                classification='numeric_actual_backend_minus_postq_difference_from_explicit_Teacher_reference;not_significance_or_causality',
                joints=changed, scope='named full native DOFs when recorded;meters and radians kept separate;qualified trigger remains ACTIVE12 only')
            if range_departures and env['first_joint_tracking_repeat_range_departure'] is None:
                env['first_joint_tracking_repeat_range_departure'] = _event_with_pointers(row,
                    joints=range_departures, classification='actual_tracking_outside_observed_Teacher_repeat_range;not_confidence_interval_or_causality')
        pad = _pad_sample(row)
        ref_pad = [value for other in refs if other.get('contact_valid') is True and other.get('bilateral_opposed_contact') is True
            and (value := _pad_sample(other)) is not None]
        old = previous.get(row['env_id'])
        contiguous = old is not None and row.get('world_physics_step') == old[0].get('world_physics_step', -2)+1
        if pad is None:
            unavailable['candidate_pad_samples'] += 1; previous[row['env_id']] = None
        else:
            load, accepted, bearing = pad
            env['observed_counts']['healthy_pad_samples'] += 1
            env['observed_counts']['contiguous_pad_transitions'] += int(contiguous)
            if len(ref_pad) < 2: unavailable['repeat_pad_samples_below_two'] += 1
            for h, side in enumerate(('left', 'right')):
                for d, digit in enumerate(digits):
                    name = side+'.'+digit
                    state = env['pad_transitions'].setdefault(name, dict(first_contact_lost=None, first_contact_reacquired=None,
                        first_bearing_lost=None, first_bearing_reacquired=None))
                    for index, kind in ((1, 'contact'), (2, 'bearing')):
                        if not contiguous: continue
                        was, now = bool(old[1][index][h, d]), bool(pad[index][h, d])
                        if was and not now:
                            event = _event_with_pointers(row, hand=side, digit=digit, accepted_load_before_n=float(old[1][0][h, d]),
                                accepted_load_now_n=float(load[h, d]), original_threshold_n=0. if kind == 'contact' else .05,
                                previous_raw_pointer=deepcopy(old[0].get('raw_pointer')), previous_diagnostic_pointer=deepcopy(old[0].get('diagnostic_pointer')),
                                aggregate_opposed_at_sample=row.get('bilateral_opposed_contact'),
                                aggregate_contact_valid_at_sample=row.get('contact_valid'),
                                had_observed_bilateral_contact_before_loss=(env['first_actual_bilateral_opposed_contact'] is not None
                                    and env['first_actual_bilateral_opposed_contact']['world_physics_step'] < row['world_physics_step']),
                                was_original_bearing_before_loss=bool(old[1][2][h,d]),
                                predicate='healthy native positive accepted pad load' if kind == 'contact' else 'original healthy accepted pad load >0.05N')
                            if state['first_'+kind+'_lost'] is None: state['first_'+kind+'_lost'] = event
                            field = 'first_finger_pad_contact_loss' if kind == 'contact' else 'first_bearing_pad_loss'
                            if d < 2 and env[field] is None: env[field] = event
                        if not was and now and state['first_'+kind+'_lost'] is not None and state['first_'+kind+'_reacquired'] is None:
                            state['first_'+kind+'_reacquired'] = _event_with_pointers(row, hand=side, digit=digit, accepted_load_n=float(load[h, d]))
                    if (d < 2 and len(ref_pad) >= 2 and all(value[1][h, d] for value in ref_pad) and not accepted[h, d]
                            and env['first_repeat_pad_contact_departure'] is None):
                        env['first_repeat_pad_contact_departure'] = _event_with_pointers(row, hand=side, digit=digit,
                            observed_reference_samples=len(ref_pad), classification='accepted_pad_absent_vs_two_healthy_postphysical_repeats;not_causality')
            previous[row['env_id']] = (row, pad)
        valid_slip = (row.get('measurement_valid') is True and row.get('slip_valid') is True
            and row.get('prior_anchor_latched') is True and row.get('slip_applicable') is True
            and type(row.get('slip_m')) in (int, float) and np.isfinite(row['slip_m']))
        if valid_slip:
            unavailable['qualified_fixed_anchor_slip_samples'] += 1
            env['observed_counts']['valid_fixed_anchor_slip_samples'] += 1
            for threshold, name in ((0., 'first_nonzero_valid_fixed_anchor_slip'), (.0005, 'first_object_slip')):
                if row['slip_m'] > threshold and env[name] is None:
                    env[name] = _event_with_pointers(row, slip_m=row['slip_m'], original_threshold_m=threshold,
                        measurement_valid=True, slip_valid=True, prior_anchor_latched=True, slip_applicable=True,
                        slip_reason_code=row.get('slip_reason_code'),
                        scope='actual original fixed-anchor slip with valid latched applicable anchor;not moving-anchor reconstruction')
        failure = row.get('failure_code')
        if type(failure) is int and failure != 0 and env['first_failure'] is None:
            env['first_failure'] = _event_with_pointers(row, failure_code=failure, classification='actual original failure;terminal success is separate')
        if row.get('terminal') is True and env['first_terminal'] is None: env['first_terminal'] = _event_with_pointers(row, failure_code=failure)
    for env in envs.values():
        slip_count = env['observed_counts']['valid_fixed_anchor_slip_samples']
        env['availability'] = dict(fixed_anchor_slip=dict(valid_samples=slip_count,
            status='unavailable_no_valid_fixed_anchor_samples' if slip_count == 0 else
                'observed_original_threshold_crossing' if env['first_object_slip'] is not None else 'observed_no_original_threshold_crossing'),
            accepted_pad_loss=dict(healthy_samples=env['observed_counts']['healthy_pad_samples'],
                contiguous_transitions=env['observed_counts']['contiguous_pad_transitions'],
                status='unavailable_no_contiguous_healthy_pad_samples' if env['observed_counts']['contiguous_pad_transitions'] == 0 else
                    'observed_contact_loss' if env['first_finger_pad_contact_loss'] is not None else 'observed_no_required_pad_contact_loss'))
    return dict(version=VERSION, per_env=[envs[i] for i in sorted(envs)], unavailable_counts=unavailable,
        contact_repeat_minimum_reference_samples=2, first_pad_loss_scope='original pinch thumb/index;all five pads per hand retained in pad_transitions',
        accepted_contact_threshold_n=0., bearing_threshold_n=.05, fixed_anchor_slip_task_threshold_m=.0005,
        postterminal_samples_excluded=excluded, validation_scope='pure reducer over caller-bound rows;not a native certificate',
        missing_raw_contact_fields=['per-contact body IDs', 'raw per-contact normal/impulse', 'effective_pad_bad_env as independent raw column'],
        independent_native_getter_clock_bracket_available=False, cause_claimed=False, native_jobs_executed=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--training-report', action='append', default=[], metavar='B0_OR_B1_OR_B2=ACTUAL_REPORT')
    parser.add_argument('--physical-config', type=Path, help='explicit Stage7 six-first input JSON;independent from offline forward')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--batch-windows', type=int, default=32)
    args = parser.parse_args(argv); reports = {}
    if args.physical_config is not None:
        if args.prepared is not None or args.training_report: parser.error('six-first physical-config is independent of offline forward options')
        torch.set_num_threads(2); torch.set_num_interop_threads(2)
        save_native_six_first(args.physical_config, args.output); return
    if args.prepared is None: parser.error('offline forward requires explicit prepared+three final training reports')
    for item in args.training_report:
        name, separator, path = item.partition('=')
        if not separator or name not in ('B0', 'B1', 'B2') or name in reports: parser.error('three unique explicit actual B0/B1/B2 reports required')
        reports[name] = path
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    analyze_offline(args.prepared, reports, args.output, batch_size=args.batch_size, batch_windows=args.batch_windows)


if __name__ == '__main__': main()
