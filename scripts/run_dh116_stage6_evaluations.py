"""Prepare/check a finite Stage6 evaluation batch; launch only with --execute.

No model filename is inferred. E0 comes from prepared.inputs; B0/B1/B2 come
only from an explicit training report's unique completed update3000 checkpoint.
All native jobs run sequentially through the existing workflow.launch_native.
Missing prepared/model reports fail immediately; this script never waits for
training, retries native jobs, starts PPO, or treats neutral rows as trials.
"""
import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys


VERSION = 'dh116-stage6-finite-native-batch-v1'
EVENTS = ('CONTACT', 'LIFT_ONSET', 'HOLD')
HORIZONS = (1, 4, 16, 60)
PHASES = ('e0-replay', 'e0-source', 'b0-source', 'b1-source', 'b2-source',
          'trained-sources', 'sources', 'e0-branches', 'b2-branches', 'branches', 'all')


def _workflow():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_stage6_workflow')


def job_specs(phase='all'):
    if phase not in PHASES:
        raise ValueError('unknown finite batch phase')
    jobs = []
    if phase in ('e0-replay', 'all'):
        jobs.append(dict(job_id='teacher_replay_E0', arm='E0', mode='teacher-replay'))
    arms = {'e0-source':('E0',), 'b0-source':('B0',), 'b1-source':('B1',), 'b2-source':('B2',),
            'trained-sources':('B0','B1','B2'), 'sources':('E0','B0','B1','B2'),
            'all':('E0','B0','B1','B2')}.get(phase, ())
    for arm in arms: jobs.append(dict(job_id='source_' + arm, arm=arm, mode='source'))
    if phase in ('e0-branches', 'b2-branches', 'branches', 'all'):
        for arm in (('E0',) if phase == 'e0-branches' else ('B2',) if phase == 'b2-branches' else ('E0','B2')):
            for event in EVENTS:
                for steps in HORIZONS:
                    jobs.append(dict(job_id=f'branch_{arm}_{event}_{steps:02d}', arm=arm,
                        mode='branch', event=event, branch_steps=steps))
    return jobs


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def select_completed_checkpoint(report_path, *, profile, prepared_identity, source_identity):
    wf = _workflow(); report_path = Path(report_path).resolve(strict=True)
    wf._checked_identity(prepared_identity, report_path.parent)
    wf._checked_identity(source_identity, report_path.parent)
    report = _read_json(report_path)
    if (report.get('profile') != profile or report.get('status') != 'completed'
            or report.get('steps') != 3000 or report.get('requested_steps') != 3000
            or report.get('valid_exposures') != 192000 or report.get('source_immutable') is not True
            or report.get('ppo_updates') != 0
            or report.get('checkpoint_selection') != 'prespecified_final_update_not_train_MSE'):
        raise ValueError(f'{profile} requires completed update3000,192000-valid-exposure independent BC report')
    if report.get('prepared') != prepared_identity or report.get('source_checkpoint') != source_identity:
        raise ValueError(f'{profile} training report belongs to another prepared dataset or source model')
    wf.require_matched_exposures(report.get('sequence_report', {}))
    final = [item for item in report.get('checkpoints', [])
             if item.get('status') == 'completed' and item.get('step') == 3000]
    if len(final) != 1: raise ValueError(f'{profile} must name exactly one completed update3000 checkpoint')
    return dict(checkpoint=wf._checked_identity(final[0], report_path.parent),
        training_report=wf.file_identity(report_path), selected_update=3000,
        selection='prespecified_final_update3000;no_MSE_or_peak_selection')


def _check_initial_state(path):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('seed') != 42 or state.get('source') != 'actual_native_cold_reset_before_camera':
        raise ValueError('explicit actual native source cold-reset state with seed42 required')
    for name, shape in (('command', (8, 12)), ('roots', (8, 2, 7)), ('object_position', (8, 3)),
                        ('object_quaternion', (8, 4)), ('reference_bottom', (8,))):
        value = state.get(name)
        if not isinstance(value, torch.Tensor) or value.shape != shape or not bool(torch.isfinite(value).all()):
            raise ValueError('actual source initial-state shape/nonfinite mismatch: ' + name)
    for name in ('q', 'qd'):
        value = state.get(name)
        if (not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[:2] != (8, 2)
                or not bool(torch.isfinite(value).all()) or value.shape[-1] < 6):
            raise ValueError('actual complete two-hand initial q/qd required')
    if state['q'].shape != state['qd'].shape: raise ValueError('initial q/qd shape mismatch')


def _load_bound_prepared(path):
    wf = _workflow(); path = Path(path).resolve(strict=True); value = _read_json(path)
    if value.get('version') != 'dh116-stage6-prepared-sequences-v1':
        raise ValueError('completed native prepared sequence02 manifest required')
    inputs = wf.load_stage6_inputs(value['inputs']['input_manifest']['path'])
    if inputs != value['inputs']:
        raise ValueError('prepared inputs differ from the complete pinned inputs manifest')
    for left, right in (('dataset', 'dataset'), ('context', 'sequence_context'), ('teacher_report', 'teacher_collection_report')):
        if value[left] != inputs[right]: raise ValueError('prepared native source binding differs: ' + left)
        wf._checked_identity(value[left], path.parent)
    wf._checked_identity(value['index'], path.parent)
    wf.require_native_shadow_parity(value.get('teacher_shadow_parity'))
    if value.get('accumulated_window_parity', {}).get('passed') is not True:
        raise ValueError('prepared source must have full accumulated16-step native Teacher target parity')
    return value, wf.file_identity(path)


def _validate_model(path, source, *, arm, prepared):
    import torch
    from go2_pvcnn.control.dh116.tabletop_bimanual_checkpoint_contract import unwrap_physical_adapter_checkpoint
    from go2_pvcnn.control.dh116.tabletop_vision_point_activation import point_activation_from_contract
    from go2_pvcnn.control.dh116.tabletop_vision_supervised_bc import _same_state
    saved = torch.load(path, map_location='cpu', weights_only=False)
    core = unwrap_physical_adapter_checkpoint(saved, allow_explicit=True)
    if (point_activation_from_contract(core['contract']) != 'relu'
            or core['contract'].get('network', {}).get('actor_output') != 'unbounded_latent_mean'):
        raise ValueError('main evaluation requires ReLU and unbounded source followed by one Tanh')
    if arm != 'E0':
        meta = saved.get('stage6_sequence_bc', {})
        if (meta.get('profile') != arm or meta.get('updates') != 3000 or meta.get('status') != 'completed'
                or meta.get('source_checkpoint') != prepared['inputs']['source_checkpoint']
                or meta.get('dataset') != prepared['dataset'] or meta.get('context') != prepared['context']):
            raise ValueError('actual trained checkpoint provenance differs from its final training report')
        if saved['policy'].keys() != source['policy'].keys():
            raise ValueError('trained checkpoint changed the source policy parameter layout')
        for name, value in source['policy'].items():
            if not name.startswith('actor.') and not _same_state(saved['policy'][name], value):
                raise ValueError('trained checkpoint changed Critic or latent exploration state')
        for name, value in source.items():
            if name != 'policy' and not _same_state(saved.get(name), value):
                raise ValueError('trained checkpoint changed original PPO/contract/counter/RNG state: ' + name)
    return saved


def _source_api(source):
    source = Path(source).resolve(strict=True)
    sys.path.insert(0, str(source / 'src'))
    wf = _workflow()
    if Path(wf.__file__).resolve().parents[4] != source:
        raise ValueError('Python imported a workflow outside the explicitly selected current source')
    branch = importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_branching_eval')
    if not callable(getattr(branch.SourceTelemetryCollector, 'record_issued_difference', None)):
        raise ValueError('source lacks actual-issued-only numeric divergence recording; update before evaluation')
    return source, wf, branch


def _single_option(command, name):
    values = []
    for i, item in enumerate(command):
        if item == name and i + 1 < len(command): values.append(command[i + 1])
        elif item.startswith(name + '='): values.append(item.split('=', 1)[1])
    if len(values) != 1: raise ValueError('native template requires one explicit ' + name)
    return values[0]


def native_template_environment(template):
    """Use the recorded GPU3 job environment; expose only an explicit whitelist."""
    env = template.get('env', {})
    if (not isinstance(env, dict) or env.get('CUDA_VISIBLE_DEVICES') != '3'
            or not isinstance(env.get('DH116_PYTHON'), str) or not env['DH116_PYTHON']):
        raise ValueError('actual 02 native template must bind CUDA_VISIBLE_DEVICES=3 and DH116_PYTHON')
    if not any(re.search(r'(?:^|\s)--/renderer/activeGpu=3(?:\s|$)', arg)
               for arg in template.get('command', [])):
        raise ValueError('actual 02 Kit renderer activeGpu3 argument required')
    return {name: env.get(name) for name in ('CUDA_VISIBLE_DEVICES','DH116_PYTHON','DH116_NUM_THREADS',
                                            'OMP_NUM_THREADS','PYTHONUNBUFFERED')}


def validate_comparison_caches(caches, models, common_inputs, implementation_sha256):
    """Only the offline three-input/code match is reused; native jobs stay new."""
    from go2_pvcnn.control.dh116.tabletop_vision_comparison_cache import load_comparison_cache
    wf = _workflow(); checked = {}
    for arm, value in caches.items():
        if arm not in ('E0', 'B2') or arm not in models:
            raise ValueError('comparison caches must name a selected E0/B2 model')
        if isinstance(value, dict):
            identity = wf._checked_identity(value, Path.cwd())
        else:
            path = Path(value)
            if path.is_dir(): path = path / 'cache.json'
            identity = wf.file_identity(path)
        load_comparison_cache(identity, dict(common_inputs, checkpoint=models[arm]['checkpoint']),
            implementation_sha256=implementation_sha256)
        checked[arm] = identity
    return checked


def effective_native_config(job, available_caches):
    """Publish a new runtime config if an earlier job produced its offline cache."""
    if job['arm'] not in available_caches: return job['config']
    wf = _workflow(); original = _read_json(job['config']['path'])
    cache = available_caches[job['arm']]
    if 'same_state_comparison_cache' in original:
        if original['same_state_comparison_cache'] != cache:
            raise ValueError('prepared native config names another comparison cache')
        return job['config']
    runtime = dict(original, same_state_comparison_cache=cache)
    path = Path(job['config']['path']).with_name(job['job_id'] + '_with_comparison_cache.json')
    wf._json_new(path, runtime)
    return wf.file_identity(path)


def prepare_batch(source, prepared_path, initial_state_path, output, *, phase='all', training_reports=None,
                  bc_root=None, replay_report=None, comparison_caches=None, native_template=None):
    """Prepare only. No native function is called and no pending model is awaited."""
    source, wf, branch = _source_api(source)
    prepared, prepared_id = _load_bound_prepared(prepared_path)
    initial_state_path = Path(initial_state_path).resolve(strict=True)
    if initial_state_path.parent != Path(prepared['teacher_report']['path']).parent:
        raise ValueError('initial state must be an explicitly supplied artifact of the same native collection directory')
    _check_initial_state(initial_state_path)
    jobs = job_specs(phase); arms = {job['arm'] for job in jobs}
    reports = {} if training_reports is None else dict(training_reports)
    if bc_root is not None:
        # This is the public workflow's report artifact name, never a model
        # filename. The model itself is resolved exclusively from that report.
        for arm in ('B0', 'B1', 'B2'):
            reports.setdefault(arm, Path(bc_root) / arm / 'sequence_training_report.json')
    missing = sorted(arms - {'E0'} - reports.keys())
    if missing: raise ValueError('training reports not ready; no waiting or model-name guessing: ' + ','.join(missing))
    source_model = prepared['inputs']['source_checkpoint']
    source_saved = _validate_model(source_model['path'], None, arm='E0', prepared=prepared)
    models = dict(E0=dict(checkpoint=source_model, selection='actual_source_checkpoint_from_bound_inputs'))
    for arm in sorted(arms - {'E0'}):
        models[arm] = select_completed_checkpoint(reports[arm], profile=arm,
            prepared_identity=prepared_id, source_identity=source_model)
        _validate_model(models[arm]['checkpoint']['path'], source_saved, arm=arm, prepared=prepared)
    if any(job['mode'] == 'branch' for job in jobs):
        import torch
        data = torch.load(prepared['dataset']['path'], map_location='cpu', weights_only=False)
        context = torch.load(prepared['context']['path'], map_location='cpu', weights_only=False)
        cases = branch.source_cases(data, context, _read_json(prepared['teacher_report']['path']),
            dataset_sha256=prepared['dataset']['sha256'])
        if any(case['event_ticks'][event] is None for case in cases for event in EVENTS):
            raise ValueError('each actual source requires recorded CONTACT/LIFT_ONSET/HOLD entries; no fabricated branches')
    if native_template is None:
        raise ValueError('explicit actual eight-environment Teacher --native-template required; model discovery launch is provenance only')
    template = wf.file_identity(native_template)
    template_data = _read_json(template['path'])
    native_env = native_template_environment(template_data)
    command = template_data['command']
    for name, wanted in (('--seed', '42'), ('--num-envs', '8'), ('--physics-hz', '480'),
                         ('--vision-width', '224'), ('--vision-height', '96')):
        if _single_option(command, name) != wanted:
            raise ValueError('original source cold-reset template differs: ' + name)
    if Path(_single_option(command, '--output')).resolve() != Path(prepared['teacher_report']['path']).parent:
        raise ValueError('native template output is not the actual bound Teacher collection directory')
    implementation = {name: wf.file_identity(source / 'src/go2_pvcnn/control/dh116' / (name + '.py'))
        for name in ('tabletop_vision_stage6_workflow', 'tabletop_vision_branching_eval', 'tabletop_vision_inference',
                     'tabletop_vision_comparison_cache')}
    output = Path(output).absolute()
    if output == source or source in output.parents:
        raise ValueError('native batch artifacts/output must be outside the selected source tree')
    output.mkdir(parents=True, exist_ok=False); (output / 'configs').mkdir()
    common = dict(teacher_dataset=prepared['dataset'], sequence_context=prepared['context'],
        source_initial_state=wf.file_identity(initial_state_path), source_collection_report=prepared['teacher_report'])
    selected_caches = {arm: value for arm, value in (comparison_caches or {}).items() if arm in arms}
    cached = validate_comparison_caches(selected_caches, models, common,
        implementation['tabletop_vision_branching_eval']['sha256'])
    replay_identity = replay_reset_identity = replay_baseline = None
    if replay_report is not None:
        replay_identity = wf.file_identity(replay_report)
        replay_reset_identity = wf.file_identity(Path(replay_report).parent / 'source_reset_comparison.json')
        replay_baseline = require_teacher_replay_baseline(_read_json(replay_report), source_identity=source_model,
            common_inputs=common, reset_comparison=_read_json(replay_reset_identity['path']))
    ready = []
    for spec in jobs:
        config = dict(version=branch.VERSION, mode=spec['mode'], checkpoint=models[spec['arm']]['checkpoint'],
            initial_state_tolerance=0., **common)
        if spec['arm'] in cached: config['same_state_comparison_cache'] = cached[spec['arm']]
        if spec['mode'] == 'branch': config.update(event=spec['event'], branch_steps=spec['branch_steps'])
        config_path = output / 'configs' / (spec['job_id'] + '.json')
        wf._json_new(config_path, config)
        target = output / 'jobs' / spec['job_id']
        projected = wf.build_stage6_native_command(_read_json(template['path']), source, target,
            mode=spec['mode'], native_config=config_path)
        wf.require_native_mode(spec['mode'], config)
        ready.append(dict(**spec, config=wf.file_identity(config_path), output=str(target),
            projected_native_command=projected, cold_reset=True,
            evaluated_env_ids=[0, 1, 2, 3], neutral_env_ids=[4, 5, 6, 7]))
    plan = dict(version=VERSION, phase=phase, prepared=prepared_id, source=str(source),
        template=template, models=models, common_inputs=common, implementation=implementation,
        model_discovery_launch=prepared['inputs'].get('source_launch'),
        native_template_scope='actual_bound_eight_environment_Teacher_collection;not_model_discovery_launch',
        native_environment_whitelist=native_env,
        comparison_caches=cached, automatic_offline_cache_builds=['E0_after_teacher_replay','B2_after_source'],
        cache_scope='saved_observation_comparisons_only;native_trials_reset_physics_terminals_never_reused',
        driver=wf.file_identity(Path(__file__)), jobs=ready, native_jobs_executed=0,
        bounded_job_count=len(ready), maximum_job_count=29, concurrent_native_instances=1,
        evaluation_denominator_per_job=4, neutral_env_ids=[4, 5, 6, 7],
        ppo_updates=0, automatic_PPO=False, automatic_retry=False, pending_input_wait=False,
        checkpoint_selection='explicit_completed_update3000_from_training_report;never_MSE',
        teacher_prefix_is_pure_student_success=False, heldout_generalization_claimed=False,
        replay_report=replay_identity, replay_reset_comparison=replay_reset_identity, replay_baseline=replay_baseline,
        student_execution_gate=('internal_replay_native_execution_and_exact_reset_baseline_first' if any(j['mode'] == 'teacher-replay' for j in ready)
            else 'bound_completed_replay_baseline;physical_failures_retained' if replay_identity is not None else 'replay_report_pending;prepare_only'),
        source_revision_requirement='record_issued_difference_actual_student_owner_only',
        schema_check='bound_prepared_inputs_actual_bytes_models_and_finite_native_command_schema')
    wf._json_new(output / 'batch_plan.json', plan)
    return plan


def summarize_native_report(report):
    """Four evaluated cases only; retain prefix/Student counts separately."""
    if (report.get('native_runtime_verified') is not True or report.get('ppo_updates') != 0
            or report.get('evaluated_source_count') != 4
            or sorted(row['env_id'] for row in report.get('source_rows', [])) != [0, 1, 2, 3]):
        raise ValueError('actual native four-source-only report required')
    neutral = report.get('unevaluated_environments', [])
    if (sorted(row['env_id'] for row in neutral) != [4, 5, 6, 7]
            or any(row.get('evaluated') is not False or row.get('action') != 'neutral_zero24' for row in neutral)):
        raise ValueError('neutral rows4..7 cannot be counted as successes or failures')
    if report['mode'] != 'source' and report.get('pure_student_native_success_count') != 0:
        raise ValueError('Teacher-prefix branches cannot claim pure Student source success')
    return dict(mode=report['mode'], status=report['status'], evaluated_case_count=4,
        pure_student_native_success_count=report.get('pure_student_native_success_count', 0),
        teacher_prefix_native_success_count=report.get('teacher_prefix_native_success_count', 0),
        student_segment_native_success_count=report.get('student_segment_native_success_count', 0),
        branch_switch_reached_count=report.get('branch_switch_reached_count', 0),
        completed_first_episode_count=report.get('completed_count', 0),
        source_rows=deepcopy(report['source_rows']), neutral_env_ids=[4, 5, 6, 7],
        teacher_prefix_is_pure_student_success=False, new_independent_initial_states=0)


def require_teacher_replay_baseline(report, *, source_identity, common_inputs, reset_comparison):
    """Require a complete native baseline, retaining its physical failures.

    Tape replay is not a state-query expert. Its physical success rate is a
    diagnostic result, not an additional Student-evaluation eligibility gate.
    Input/runtime/reset failures must be fixed before any later interpretation.
    """
    from go2_pvcnn.control.dh116.tabletop_vision_branching_eval import SOURCE_STATE_FIELDS
    if (report.get('mode') != 'teacher-replay' or report.get('status') != 'completed'
            or report.get('error') is not None or report.get('native_runtime_verified') is not True
            or report.get('ppo_updates') != 0 or report.get('completed_count') != 4
            or report.get('pure_student_native_success_count') != 0
            or sorted(row['env_id'] for row in report.get('source_rows', [])) != [0, 1, 2, 3]):
        raise ValueError('completed native Teacher-replay execution with four first terminals required')
    for row in report['source_rows']:
        terminal = row.get('terminal')
        if (not isinstance(terminal, dict) or terminal.get('action_owner') != 'teacher_replay'
                or type(row.get('native_success')) is not bool
                or row.get('student_policy_steps_executed') != 0
                or row.get('teacher_prefix_policy_steps_executed') != 0
                or type(row.get('teacher_replay_policy_steps_executed')) is not int
                or row['teacher_replay_policy_steps_executed'] < 1):
            raise ValueError('actual Teacher-replay-owned first terminal and policy-step evidence required')
    if report.get('inputs') != dict(checkpoint=source_identity, **common_inputs):
        raise ValueError('Teacher-replay sanity report belongs to other model/source inputs')
    if (not isinstance(reset_comparison, dict) or reset_comparison.get('matched') is not True
            or reset_comparison.get('tolerance') != 0.
            or any(reset_comparison.get('fields', {}).get(name, {}).get(flag) is not True
                for name in SOURCE_STATE_FIELDS for flag in ('matched', 'dtype_equal', 'shape_equal', 'finite'))):
        raise ValueError('actual complete source reset comparison must match exactly before Student evaluation')
    succeeded = sum(row['native_success'] for row in report['source_rows'])
    if report.get('native_success_count') != succeeded:
        raise ValueError('replay success count differs from its four actual first-episode outcomes')
    return dict(native_execution_completed=True, exact_source_reset=True, evaluated_case_count=4,
        native_success_count=succeeded, failed_baseline_env_ids=[row['env_id'] for row in report['source_rows'] if not row['native_success']],
        source_rows=deepcopy(report['source_rows']), physical_success_is_eligibility_gate=False,
        failed_baseline_is_student_failure_claimed=False, qualified_teacher_prefix_claimed=False)


def execute_batch(plan_path):
    """Explicit bounded sequential execution. Never called by default preparation."""
    wf = _workflow(); plan_path = Path(plan_path).resolve(strict=True); plan = _read_json(plan_path)
    if plan.get('version') != VERSION or not 1 <= len(plan.get('jobs', [])) <= 29:
        raise ValueError('finite prepared batch plan required')
    source, wf, branch = _source_api(plan['source'])
    expected = job_specs(plan['phase'])
    if [row['job_id'] for row in plan['jobs']] != [row['job_id'] for row in expected]:
        raise ValueError('batch jobs differ from the prespecified finite schedule')
    for identity in [plan['prepared'], plan['template'], plan['driver'], *plan['implementation'].values(), *plan['common_inputs'].values()]:
        wf._checked_identity(identity, plan_path.parent)
    for model in plan['models'].values():
        wf._checked_identity(model['checkpoint'], plan_path.parent)
        if 'training_report' in model: wf._checked_identity(model['training_report'], plan_path.parent)
    available_caches = validate_comparison_caches(plan.get('comparison_caches', {}), plan['models'], plan['common_inputs'],
        plan['implementation']['tabletop_vision_branching_eval']['sha256'])
    replay_ready = False; replay_baseline = None
    if plan.get('replay_report') is not None:
        wf._checked_identity(plan['replay_report'], plan_path.parent)
        wf._checked_identity(plan['replay_reset_comparison'], plan_path.parent)
        replay_baseline = require_teacher_replay_baseline(_read_json(plan['replay_report']['path']),
            source_identity=plan['models']['E0']['checkpoint'], common_inputs=plan['common_inputs'],
            reset_comparison=_read_json(plan['replay_reset_comparison']['path']))
        replay_ready = True
    if any(job['mode'] != 'teacher-replay' for job in plan['jobs']) and not replay_ready:
        if plan['jobs'][0]['mode'] != 'teacher-replay':
            raise ValueError('Teacher-replay sanity report not ready; no waiting and no Student launch')
    for job, spec in zip(plan['jobs'], expected):
        if any(job.get(name) != value for name, value in spec.items()):
            raise ValueError('job mode/event/horizon differs from the prespecified schedule')
        if Path(job['output']).absolute() != plan_path.parent / 'jobs' / spec['job_id']:
            raise ValueError('native job output escaped the batch jobs directory')
        wf._checked_identity(job['config'], plan_path.parent)
        cfg = dict(version=branch.VERSION, mode=spec['mode'],
            checkpoint=plan['models'][spec['arm']]['checkpoint'], initial_state_tolerance=0., **plan['common_inputs'])
        if spec['arm'] in plan.get('comparison_caches', {}):
            cfg['same_state_comparison_cache'] = plan['comparison_caches'][spec['arm']]
        if spec['mode'] == 'branch': cfg.update(event=spec['event'], branch_steps=spec['branch_steps'])
        if _read_json(job['config']['path']) != cfg:
            raise ValueError('native job configuration differs from its bound model/source/horizon')
    wf._json_new(plan_path.parent / 'execution_started.json', dict(plan=wf.file_identity(plan_path),
        bounded_job_count=len(expected), no_retry=True, sequential=True))
    results = []
    for job in plan['jobs']:
        if job['mode'] != 'teacher-replay' and not replay_ready:
            raise ValueError('native replay/reset baseline is incomplete; refusing unsupported Student interpretation')
        busy = subprocess.run(['pgrep', '-af', '[s]cripts/train_dh116_tabletop_gpu_ppo.py'], capture_output=True, text=True)
        if busy.returncode not in (0, 1) or busy.stdout.strip():
            raise RuntimeError('another native owner is active; refusing to wait or launch a second Isaac')
        wf._checked_identity(job['config'], plan_path.parent)
        runtime_config = effective_native_config(job, available_caches)
        print(json.dumps(dict(event='starting', job_id=job['job_id'], output=job['output'])), flush=True)
        output = wf.launch_native(plan['template']['path'], source, job['output'],
            mode=job['mode'], native_config=runtime_config['path'])
        report_path = output / 'stage6_native_report.json'; report = _read_json(report_path)
        value = dict(job_id=job['job_id'], arm=job['arm'], report=wf.file_identity(report_path),
            effective_native_config=runtime_config, offline_comparison_cache=available_caches.get(job['arm']),
            summary=summarize_native_report(report))
        if report['mode'] != job['mode']: raise ValueError('native output mode differs from its requested job')
        if job['mode'] == 'teacher-replay':
            reset_id = wf.file_identity(output / 'source_reset_comparison.json')
            replay_baseline = require_teacher_replay_baseline(report,
                source_identity=plan['models']['E0']['checkpoint'], common_inputs=plan['common_inputs'],
                reset_comparison=_read_json(reset_id['path']))
            value['reset_comparison'] = reset_id
            replay_ready = True
        if ((job['arm'] == 'E0' and job['mode'] == 'teacher-replay')
                or (job['arm'] == 'B2' and job['mode'] == 'source')) and job['arm'] not in available_caches:
            from go2_pvcnn.control.dh116.tabletop_vision_comparison_cache import build_comparison_cache
            new_cache = build_comparison_cache(report_path, plan_path.parent / 'comparison_caches' / job['arm'])
            available_caches[job['arm']] = new_cache
            value['created_offline_comparison_cache'] = new_cache
            wf._json_new(plan_path.parent / ('comparison_cache_' + job['arm'] + '.json'), new_cache)
        value['replay_baseline'] = deepcopy(replay_baseline)
        value['summary']['qualified_teacher_prefix_claimed'] = False
        wf._json_new(plan_path.parent / (job['job_id'] + '_outcome.json'), value)
        results.append(value); print(json.dumps(dict(event='finished', **value)), flush=True)
    summary = dict(version=VERSION, plan=wf.file_identity(plan_path), results=results,
        comparison_caches=available_caches, native_trials_reused=0,
        executed_jobs=len(results), neutral_rows_excluded=True, ppo_updates=0,
        teacher_prefix_is_pure_student_success=False, heldout_generalization_claimed=False)
    wf._json_new(plan_path.parent / 'batch_execution_report.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--source-initial-state', type=Path)
    parser.add_argument('--native-template', type=Path,
        help='actual 02 Teacher eight-env seed42 launch; the model discovery launch is never a runtime template')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--phase', choices=PHASES, default='all')
    parser.add_argument('--training-report', action='append', default=[], metavar='ARM=PATH')
    parser.add_argument('--bc-root', type=Path, help='actual B0/B1/B2 report directories; model names are never inferred')
    parser.add_argument('--replay-report', type=Path, help='completed native prior E0 replay with exact reset and four first terminals; physical failures retained')
    parser.add_argument('--comparison-cache', action='append', default=[], metavar='E0_OR_B2=PATH',
        help='optional exact model/data/context/code bound offline cache; native physics is never reused')
    parser.add_argument('--list-only', action='store_true')
    parser.add_argument('--execute', type=Path, metavar='BATCH_PLAN_JSON')
    args = parser.parse_args(argv)
    if args.list_only:
        print(json.dumps(dict(jobs=job_specs(args.phase), native_jobs_executed=0), indent=2)); return 0
    if args.source is None: parser.error('--source is required for prepared or executed batches')
    _source_api(args.source)
    if args.execute is not None:
        result = execute_batch(args.execute)
    else:
        if any(value is None for value in (args.prepared, args.source_initial_state, args.output, args.native_template)):
            parser.error('--prepared --source-initial-state --native-template --output are required; preparation does not launch Isaac')
        reports = {}
        for value in args.training_report:
            arm, sep, path = value.partition('=')
            if not sep or arm not in ('B0', 'B1', 'B2') or arm in reports:
                parser.error('--training-report must name one unique B0/B1/B2=actual_report_path')
            reports[arm] = Path(path)
        caches = {}
        for value in args.comparison_cache:
            arm, sep, path = value.partition('=')
            if not sep or arm not in ('E0','B2') or arm in caches:
                parser.error('--comparison-cache requires unique E0/B2=actual_cache_manifest_path')
            caches[arm] = Path(path)
        result = prepare_batch(args.source, args.prepared, args.source_initial_state, args.output,
            phase=args.phase, training_reports=reports, bc_root=args.bc_root, replay_report=args.replay_report,
            comparison_caches=caches, native_template=args.native_template)
    print(json.dumps(result, indent=2, allow_nan=False)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
