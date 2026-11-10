"""CPU checks for the finite batch driver; no native process is started."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


def driver():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_dh116_stage6_evaluations.py'
    spec = importlib.util.spec_from_file_location('stage6_finite_batch', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_prespecified_batch_has_teacher_replay_then_four_source_jobs_and_24_branches_only_for_e0_b2():
    jobs = driver().job_specs('all')
    assert len(jobs) == 29 and len({j['job_id'] for j in jobs}) == 29
    assert jobs[0]['arm'] == 'E0' and jobs[0]['mode'] == 'teacher-replay'
    assert [(j['arm'], j['mode']) for j in jobs[1:5]] == [(arm, 'source') for arm in ('E0', 'B0', 'B1', 'B2')]
    branches = jobs[5:]
    assert len(branches) == 24
    assert {j['arm'] for j in branches} == {'E0', 'B2'}
    assert {j['event'] for j in branches} == {'CONTACT', 'LIFT_ONSET', 'HOLD'}
    assert {j['branch_steps'] for j in branches} == {1, 4, 16, 60}
    assert len(driver().job_specs('e0-source')) == 1
    assert len(driver().job_specs('e0-replay')) == 1
    assert len(driver().job_specs('trained-sources')) == 3


@pytest.mark.parametrize('phase,arm,mode,count', [('b0-source','B0','source',1),
    ('b1-source','B1','source',1), ('b2-source','B2','source',1),
    ('e0-branches','E0','branch',12), ('b2-branches','B2','branch',12)])
def test_ready_subsets_select_only_the_requested_arm_without_waiting_for_other_models(phase, arm, mode, count):
    jobs = driver().job_specs(phase)
    assert len(jobs) == count and {row['arm'] for row in jobs} == {arm}
    assert {row['mode'] for row in jobs} == {mode}


def final_report(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_vision_stage6_workflow import file_identity
    identities = {}
    for name in ('prepared', 'source', 'dataset', 'context', 'final-model'):
        path = tmp_path / (name + '.bin'); path.write_bytes(name.encode())
        identities[name] = file_identity(path)
    report = dict(profile='B2', status='completed', steps=3000, requested_steps=3000,
        valid_exposures=192000, source_immutable=True, ppo_updates=0,
        prepared=identities['prepared'], source_checkpoint=identities['source'],
        sequence_report=dict(sequence_length=16, min_valid_steps=16, max_valid_steps=16, padded_window_count=0),
        checkpoint_selection='prespecified_final_update_not_train_MSE',
        checkpoints=[dict(step=250, status='checkpoint', **identities['final-model']),
                     dict(step=3000, status='completed', **identities['final-model'])])
    path = tmp_path / 'report.json'; path.write_text(json.dumps(report))
    return path, report, identities


def test_final_checkpoint_is_resolved_from_completed_step3000_report_not_filename_or_mse(tmp_path):
    path, report, ids = final_report(tmp_path)
    selected = driver().select_completed_checkpoint(path, profile='B2', prepared_identity=ids['prepared'],
        source_identity=ids['source'])
    assert selected['checkpoint'] == ids['final-model']
    assert selected['training_report']['path'] == str(path.resolve())
    report['loss_after'] = -9999
    report['checkpoints'][0]['step'] = 2999
    path.write_text(json.dumps(report))
    assert driver().select_completed_checkpoint(path, profile='B2', prepared_identity=ids['prepared'],
        source_identity=ids['source'])['checkpoint'] == ids['final-model']


@pytest.mark.parametrize('bad', ['stopped', 'missing_final', 'duplicate_final', 'wrong_profile', 'unmatched_budget', 'wrong_source', 'changed_model'])
def test_incomplete_or_mismatched_model_reports_cannot_enter_batch(tmp_path, bad):
    path, report, ids = final_report(tmp_path)
    if bad == 'stopped': report['status'] = 'stopped_dead_pointnet'
    elif bad == 'missing_final': report['checkpoints'].pop()
    elif bad == 'duplicate_final': report['checkpoints'].append(copy.deepcopy(report['checkpoints'][-1]))
    elif bad == 'wrong_profile': report['profile'] = 'B1'
    elif bad == 'unmatched_budget': report['valid_exposures'] = 191999
    elif bad == 'wrong_source': report['source_checkpoint']['sha256'] = '0' * 64
    elif bad == 'changed_model': Path(ids['final-model']['path']).write_bytes(b'changed')
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        driver().select_completed_checkpoint(path, profile='B2', prepared_identity=ids['prepared'], source_identity=ids['source'])


def test_native_summary_counts_only_four_evaluated_rows_and_separates_prefix_success():
    raw = dict(mode='branch', native_runtime_verified=True, ppo_updates=0, status='bounded_branch_completed',
        evaluated_source_count=4, pure_student_native_success_count=0,
        branch_or_teacher_prefix_native_success_count=3, branch_switch_reached_count=2,
        teacher_prefix_native_success_count=1, student_segment_native_success_count=2,
        source_rows=[dict(env_id=i, native_success=i < 3, first_numeric_difference=None) for i in range(4)],
        unevaluated_environments=[dict(env_id=i, action='neutral_zero24', evaluated=False, expert=False) for i in range(4, 8)])
    result = driver().summarize_native_report(raw)
    assert result['evaluated_case_count'] == 4
    assert result['pure_student_native_success_count'] == 0
    assert result['teacher_prefix_native_success_count'] == 1
    raw['evaluated_source_count'] = 8
    with pytest.raises(ValueError): driver().summarize_native_report(raw)


def test_default_e0_preparation_requires_no_trained_reports_and_never_launches_isaac(tmp_path, monkeypatch):
    import torch
    from test_stage6_workflow_integration import _bound_inputs, _prepare
    from go2_pvcnn.control.dh116 import tabletop_vision_stage6_workflow as workflow
    fixture = _bound_inputs(tmp_path)
    prepared = _prepare(fixture, tmp_path / 'prepared')
    inputs = json.loads(prepared.read_text())['inputs']
    template = tmp_path / 'launch.json'
    template.write_text(json.dumps(dict(command=['bash', '/original/scripts/python.sh',
        '/original/scripts/train_dh116_tabletop_gpu_ppo.py', '--seed', '42', '--num-envs', '8',
        '--physics-hz', '480', '--vision-width', '224', '--vision-height', '96',
        '--output',str(fixture.report.parent), '--kit_args=--allow-root --/renderer/activeGpu=3'],
        env=dict(CUDA_VISIBLE_DEVICES='3', DH116_PYTHON='/actual/native/python', DH116_NUM_THREADS='2'))))
    discovery_launch = tmp_path / 'old_model_discovery_launch.json'
    discovery_launch.write_text(json.dumps(dict(command=['bash','/old/scripts/python.sh','old.py',
        '--num-envs','200','--seed','116507'])))
    full_inputs = json.loads(fixture.inputs.read_text())
    full_inputs['source_launch'] = workflow.file_identity(discovery_launch)
    fixture.inputs.write_text(json.dumps(full_inputs))
    prepared_data = json.loads(prepared.read_text())
    prepared_data['inputs'] = workflow.load_stage6_inputs(fixture.inputs)
    prepared.write_text(json.dumps(prepared_data))
    initial = tmp_path / 'actual_initial_state.pt'
    roots = torch.zeros(8, 2, 7); roots[..., 3] = 1.
    quat = torch.zeros(8, 4); quat[:, 0] = 1.
    torch.save(dict(seed=42, source='actual_native_cold_reset_before_camera',
        q=torch.zeros(8, 2, 11), qd=torch.zeros(8, 2, 11), command=torch.zeros(8, 12),
        roots=roots, object_position=torch.zeros(8, 3), object_quaternion=quat,
        reference_bottom=torch.zeros(8)), initial)
    def forbidden(*args, **kwargs): pytest.fail('prepare/check must not launch native code')
    monkeypatch.setattr(workflow, 'launch_native', forbidden)
    source = Path(workflow.__file__).resolve().parents[4]
    out = tmp_path / 'batch'
    plan = driver().prepare_batch(source, prepared, initial, out, phase='e0-source', bc_root=tmp_path / 'not-yet-trained',
        native_template=template)
    assert len(plan['jobs']) == 1 and plan['native_jobs_executed'] == 0
    config = json.loads(Path(plan['jobs'][0]['config']['path']).read_text())
    assert config['checkpoint'] == inputs['source_checkpoint']
    assert config['mode'] == 'source' and config['initial_state_tolerance'] == 0.
    assert config['source_initial_state'] == workflow.file_identity(initial)
    from go2_pvcnn.control.dh116.tabletop_vision_branching_eval import load_native_config
    assert load_native_config(plan['jobs'][0]['config']['path'])['mode'] == 'source'
    branch_config = dict(config, mode='branch', event='CONTACT', branch_steps=60)
    branch_path = tmp_path / 'schema_only_branch.json'; branch_path.write_text(json.dumps(branch_config))
    assert load_native_config(branch_path)['branch_steps'] == 60
    assert plan['evaluation_denominator_per_job'] == 4
    assert plan['neutral_env_ids'] == [4, 5, 6, 7]
    assert plan['template'] == workflow.file_identity(template)
    assert plan['model_discovery_launch'] == workflow.file_identity(discovery_launch)
    with pytest.raises(ValueError, match='replay|Replay'):
        driver().execute_batch(out / 'batch_plan.json')
    replay = driver().prepare_batch(source, prepared, initial, tmp_path / 'replay_only', phase='e0-replay', native_template=template)
    assert len(replay['jobs']) == 1 and replay['jobs'][0]['mode'] == 'teacher-replay'
    assert load_native_config(replay['jobs'][0]['config']['path'])['mode'] == 'teacher-replay'
    wrong = json.loads(template.read_text()); wrong['command'][wrong['command'].index('--output') + 1] = '/another/collection'
    bad_template = tmp_path / 'wrong_template.json'; bad_template.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match='output|collection'):
        driver().prepare_batch(source, prepared, initial, tmp_path / 'rejected_template', phase='e0-replay', native_template=bad_template)


def test_teacher_replay_sanity_requires_complete_exact_execution_but_does_not_gate_on_physical_success():
    ids = {name: dict(path='/' + name, sha256='a' * 64) for name in
        ('teacher_dataset', 'sequence_context', 'source_initial_state', 'source_collection_report')}
    model = dict(path='/actual-e0', sha256='b' * 64)
    report = dict(mode='teacher-replay', status='completed', native_runtime_verified=True, ppo_updates=0,
        completed_count=4, native_success_count=4, pure_student_native_success_count=0,
        source_rows=[dict(env_id=i, native_success=True, terminal=dict(action_owner='teacher_replay', failure_code=0),
            student_policy_steps_executed=0, teacher_prefix_policy_steps_executed=0,
            teacher_replay_policy_steps_executed=8) for i in range(4)],
        inputs=copy.deepcopy(dict(checkpoint=model, **ids)))
    from go2_pvcnn.control.dh116.tabletop_vision_branching_eval import SOURCE_STATE_FIELDS
    reset = dict(matched=True, tolerance=0., fields={name: dict(matched=True, dtype_equal=True,
        shape_equal=True, finite=True) for name in SOURCE_STATE_FIELDS})
    driver().require_teacher_replay_baseline(report, source_identity=model, common_inputs=ids, reset_comparison=reset)
    report['source_rows'][1]['native_success'] = False
    report['source_rows'][1]['terminal']['failure_code'] = 3
    report['native_success_count'] = 3
    result = driver().require_teacher_replay_baseline(report, source_identity=model, common_inputs=ids, reset_comparison=reset)
    assert result['native_success_count'] == 3 and result['failed_baseline_env_ids'] == [1]
    assert result['physical_success_is_eligibility_gate'] is False
    for bad in ('reset', 'native_error', 'missing_terminal', 'wrong_owner', 'wrong_inputs', 'incomplete'):
        changed, changed_reset = copy.deepcopy(report), copy.deepcopy(reset)
        if bad == 'reset': changed_reset['fields']['q']['matched'] = False
        elif bad == 'native_error': changed['error'] = 'SDK input/runtime failure'
        elif bad == 'missing_terminal': changed['source_rows'][0]['terminal'] = None
        elif bad == 'wrong_owner': changed['source_rows'][0]['terminal']['action_owner'] = 'student'
        elif bad == 'wrong_inputs': changed['inputs']['teacher_dataset']['sha256'] = 'f' * 64
        elif bad == 'incomplete': changed['completed_count'] = 3
        with pytest.raises(ValueError):
            driver().require_teacher_replay_baseline(changed, source_identity=model, common_inputs=ids, reset_comparison=changed_reset)


def test_actual_model_body_must_preserve_source_critic_noise_optimizer_and_contract(tmp_path):
    import torch
    from test_stage6_workflow_integration import _bound_inputs, _prepare
    fixture = _bound_inputs(tmp_path)
    prepared = json.loads(_prepare(fixture, tmp_path / 'prepared').read_text())
    source = torch.load(fixture.source_path, map_location='cpu', weights_only=False)
    candidate = copy.deepcopy(source)
    candidate['stage6_sequence_bc'] = dict(profile='B2', updates=3000, status='completed',
        source_checkpoint=prepared['inputs']['source_checkpoint'], dataset=prepared['dataset'], context=prepared['context'])
    actor_key = next(name for name in candidate['policy'] if name.startswith('actor.'))
    candidate['policy'][actor_key].add_(.0001)
    path = tmp_path / 'arbitrary_actual_candidate.pt'; torch.save(candidate, path)
    driver()._validate_model(path, source, arm='B2', prepared=prepared)
    candidate['policy']['log_std'][0] += .01; torch.save(candidate, path)
    with pytest.raises(ValueError, match='Critic|exploration'):
        driver()._validate_model(path, source, arm='B2', prepared=prepared)


def test_batch_accepts_only_model_data_context_code_matched_offline_cache(tmp_path):
    from test_stage6_comparison_cache import fixture
    from go2_pvcnn.control.dh116.tabletop_vision_comparison_cache import build_comparison_cache
    source, _, config, implementation, _ = fixture(tmp_path)
    cache = build_comparison_cache(source, tmp_path / 'cache')
    models = {'E0':dict(checkpoint=config['checkpoint'])}
    common = {name: config[name] for name in ('teacher_dataset', 'sequence_context')}
    result = driver().validate_comparison_caches({'E0':cache['path']}, models, common, implementation)
    assert result == {'E0':cache}
    with pytest.raises(ValueError):
        driver().validate_comparison_caches({'E0':cache['path']}, models, common, '0' * 64)
    from go2_pvcnn.control.dh116.tabletop_vision_stage6_workflow import file_identity
    config_path = tmp_path / 'original_config.json'
    original = dict(version='dh116-stage6-native-v1', mode='branch', event='HOLD', branch_steps=4,
        checkpoint=config['checkpoint'], **common)
    config_path.write_text(json.dumps(original))
    original_identity = file_identity(config_path)
    job = dict(job_id='branch_E0_HOLD_04', arm='E0', config=original_identity)
    assert driver().effective_native_config(job, {}) == original_identity
    effective = driver().effective_native_config(job, result)
    assert effective != original_identity and file_identity(config_path) == original_identity
    assert json.loads(Path(effective['path']).read_text()) == dict(original, same_state_comparison_cache=cache)


def test_native_template_uses_actual_gpu3_environment_and_prints_only_whitelist():
    template = dict(command=['bash','python.sh','train.py','--kit_args=--allow-root --/renderer/activeGpu=3'],
        env=dict(CUDA_VISIBLE_DEVICES='3', DH116_PYTHON='/actual/python', DH116_NUM_THREADS='2',
                 PRIVATE_EXTRA='not-for-output'))
    result = driver().native_template_environment(template)
    assert result['CUDA_VISIBLE_DEVICES'] == '3' and 'PRIVATE_EXTRA' not in result
    for bad in ('cpu_mask', 'missing_python', 'renderer'):
        changed = copy.deepcopy(template)
        if bad == 'cpu_mask': changed['env']['CUDA_VISIBLE_DEVICES'] = ''
        elif bad == 'missing_python': changed['env'].pop('DH116_PYTHON')
        else: changed['command'][-1] = '--kit_args=--/renderer/activeGpu=2'
        with pytest.raises(ValueError): driver().native_template_environment(changed)
