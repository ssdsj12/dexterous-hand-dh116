"""CPU fixtures test scheduling/evidence, and never certify native success."""
import copy
import gzip
import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT


def fixture_data(lengths, groups):
    from go2_pvcnn.control.dh116.tabletop_vision_supervision_dataset import build_dataset
    rows, episodes, observations, actions = [], {}, [], []
    for env_id, length in enumerate(lengths):
        episodes[str(env_id)] = dict(env_id=env_id, first_episode=True, success=True, recovered=False,
            native_physics=True, grasp_style='pinch_thumb_index', source_trajectory_id=groups[env_id],
            source_episode_id=env_id, proof_path='cpu_fixture_not_native',
            terminal=dict(success=True, recovered=False, failure=0, control_steps=length))
        previous = torch.zeros(24)
        for tick in range(length):
            action = torch.full((24,), tick / 1000.)
            observation = torch.zeros(3844)
            observation[3 * 961 + 934:3 * 961 + 958] = previous
            observations.append(observation); actions.append(action)
            rows.append(dict(env_id=env_id, episode_id=env_id, run_uuid='fixture', episode_uid=f'fixture-{env_id}',
                tick=tick, frame_id=tick + 1, simulation_time_s=tick/60, policy_timestamp_s=tick/60,
                physics_step=tick * 8, first_physics_step=tick * 8, terminal=tick == length - 1,
                previous_action=previous.tolist(), grasp_style='pinch_thumb_index',
                source_trajectory_id=groups[env_id], source_episode_id=env_id))
            previous = action
    contract = dict(actor_observation_size=3844, critic_observation_size=371, actor_uses_privileged_state=False,
        action_contract=copy.deepcopy(ACTION_CONTRACT), camera_config_sha256='1' * 64,
        primary_reset_sha256='2' * 64, secondary_reset_sha256='3' * 64)
    return build_dataset(torch.stack(observations), torch.stack(actions), torch.ones(len(actions), 24),
                         rows, episodes, visual_contract=contract, teacher_source_sha256='a' * 64)


def fixture_context(data):
    import hashlib
    n = len(data['actions']); rows = data['sample_metadata']
    samples = {key: [row[key] for row in rows] for key in
        ('run_uuid', 'env_id', 'episode_uid', 'tick', 'policy_timestamp_s', 'first_physics_step', 'terminal')}
    samples.update(valid=torch.ones(n, dtype=torch.bool), event_id=['recorded'] * n,
        actions=data['actions'].clone(), previous_action=torch.tensor([r['previous_action'] for r in rows]))
    return dict(version='dh116-sequence-context-v1', dataset_sha256='d' * 64,
        controller_source_sha256='c' * 64,
        action_contract_sha256=hashlib.sha256(json.dumps(ACTION_CONTRACT, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest(), samples=samples,
        controller_start=dict(joint_command_rad=torch.zeros(n, 12),
            palm_command_offset_m=torch.zeros(n, 2, 3), wrist_command_angle_rad=torch.zeros(n, 2, 3)),
        physics_context=dict(dt_s=torch.full((n, 8), 1/480, dtype=torch.float64),
            measured_q_rad=torch.zeros(n, 8, 12), measured_qd_rad_s=torch.zeros(n, 8, 12),
            measured_root_pose=torch.zeros(n, 8, 2, 7), clearance_m=torch.zeros(n, 8, 2),
            task_offset_m=torch.zeros(n, 8), live=torch.ones(n, 8, dtype=torch.bool),
            allowed=torch.ones(n, 8, dtype=torch.bool), joint_lower=torch.full((n, 8, 12), -1.),
            joint_upper=torch.ones(n, 8, 12), active_controller_flags=torch.zeros(n, 8, dtype=torch.long),
            source_valid=torch.ones(n, 8, dtype=torch.bool)),
        teacher_targets=dict(finger=torch.zeros(n, 8, 12), palm=torch.zeros(n, 8, 2, 3),
            wrist_angle=torch.zeros(n, 8, 2, 3)), events=dict(opposed_contact=torch.zeros(n, dtype=torch.bool),
            measured_palm_velocity_m_s=torch.zeros(n, 2, 3), native_hold_qualified=torch.zeros(n, dtype=torch.bool)))


def module():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_branching_eval')


def source_fixture(length=8):
    data = fixture_data((length,) * 4, ['original-template'] * 4)
    context = fixture_context(data)
    for env_id in range(4):
        start = env_id * length
        context['events']['opposed_contact'][start + 2:start + length] = True
        context['events']['measured_palm_velocity_m_s'][start + 3:start + 6, :, 2] = .001
        context['events']['native_hold_qualified'][start + 6:start + length] = True
    report = dict(native_runtime_verified=True, successful_episodes=4,
                  episodes=copy.deepcopy(data['episodes']))
    return data, context, report


def test_only_actual_four_successful_sources_are_evaluated():
    runner = module()
    data, context, report = source_fixture()
    cases = runner.source_cases(data, context, report, dataset_sha256='d' * 64)
    assert [case['env_id'] for case in cases] == [0, 1, 2, 3]
    assert cases[0]['event_ticks'] == dict(CONTACT=2, LIFT_ONSET=3, HOLD=6)
    assert torch.equal(cases[0]['actions'], data['actions'][:8])
    assert not torch.equal(cases[0]['actions'], torch.ones(8, 24))
    report['episodes']['3']['success'] = False
    with pytest.raises(ValueError, match='source|qualified'):
        module().source_cases(data, context, report, dataset_sha256='d' * 64)


def test_source_tape_cannot_be_reinterpreted_at_a_different_physics_frequency():
    data, context, report = source_fixture()
    context['physics_context']['dt_s'] *= 2
    with pytest.raises(ValueError, match='480|60|clock'):
        module().source_cases(data, context, report, dataset_sha256='d' * 64)


@pytest.mark.parametrize('event,switch', [('CONTACT', 2), ('LIFT_ONSET', 3), ('HOLD', 6)])
@pytest.mark.parametrize('horizon', [1, 4, 16, 60])
def test_each_branch_replays_exact_prefix_then_bounded_student_without_teacher_fallback(event, switch, horizon):
    data, context, report = source_fixture()
    cases = module().source_cases(data, context, report, dataset_sha256='d' * 64)
    config = dict(mode='branch', event=event, branch_steps=horizon)
    student = torch.full((8, 24), .73)
    ended = torch.zeros(8, dtype=torch.bool)
    for tick in range(switch + horizon + 2):
        action, owners = module().scheduled_actions(cases, config, tick, student, ended)
        assert torch.equal(action[4:], torch.zeros(4, 24))
        if tick < switch:
            assert torch.equal(action[0], cases[0]['actions'][tick]) and owners[0] == 'teacher_prefix'
        elif tick < switch + horizon:
            assert torch.equal(action[0], student[0]) and owners[0] == 'student'
        else:
            assert torch.equal(action[0], torch.zeros(24)) and owners[0] == 'branch_complete'
    ended[0] = True
    action, owners = module().scheduled_actions(cases, config, switch, student, ended)
    assert owners[0] == 'first_terminal_closed' and not action[0].any()


def test_source_always_issues_student_and_replay_never_repeats_last_tape_action():
    data, context, report = source_fixture()
    cases = module().source_cases(data, context, report, dataset_sha256='d' * 64)
    student = torch.full((8, 24), -.81)
    ended = torch.zeros(8, dtype=torch.bool)
    action, owners = module().scheduled_actions(cases, dict(mode='source'), 500, student, ended)
    assert torch.equal(action[:4], student[:4]) and owners[:4] == ['student'] * 4
    action, owners = module().scheduled_actions(cases, dict(mode='teacher-replay'), 8, student, ended)
    assert not action.any() and owners[0] == 'tape_exhausted'


def test_event_entries_use_pre_action_predicates_not_sampler_windows_or_phase():
    events = dict(opposed_contact=torch.tensor([False, False, True, True, True, True, True]),
        native_hold_qualified=torch.tensor([False] * 6 + [True]),
        measured_palm_velocity_m_s=torch.zeros(7, 2, 3),
        phase=torch.full((7,), 999))
    events['measured_palm_velocity_m_s'][3:6, :, 2] = .001
    assert module().event_entry_ticks(events) == dict(CONTACT=2, LIFT_ONSET=3, HOLD=6)
    events['measured_palm_velocity_m_s'][4, :, 2] = 0.
    assert module().event_entry_ticks(events)['LIFT_ONSET'] is None


def test_missing_velocity_is_derived_only_from_recorded_pre_action_actual_palm_positions():
    data, context, report = source_fixture()
    del context['events']['measured_palm_velocity_m_s']
    for case in range(4):
        start = case * 8
        # At pre-action tick3, measured upward motion over interval2→3
        # begins. Action requests themselves deliberately remain irrelevant.
        for tick in range(8):
            context['physics_context']['measured_root_pose'][start + tick, :, :, 2] = max(0, tick - 2) * .001 / 60
    cases = module().source_cases(data, context, report, dataset_sha256='d' * 64)
    assert cases[0]['event_ticks']['LIFT_ONSET'] == 3
    assert cases[0]['event_sources']['velocity'] == 'backward_difference_recorded_pre_action_measured_root_pose'


def test_missing_contact_event_uses_only_identity_joined_raw_precontrol_pad_measurements():
    data, context, report = source_fixture()
    del context['events']['opposed_contact']
    context['raw_measured_packets'] = []
    for meta in data['sample_metadata']:
        loads = torch.zeros(1, 2, 5)
        normals = torch.zeros(1, 2, 5, 3)
        if meta['tick'] >= 2:
            loads[..., :2] = 1.; normals[..., 0, 0] = 1.; normals[..., 1, 0] = -1.
        raw = dict(effective_pad_loads_n=loads, effective_pad_vectors_w=normals,
            effective_pad_valid=torch.ones(1, 2, 5, dtype=torch.bool))
        context['raw_measured_packets'].append(dict(sample_key=tuple(meta[key] for key in
            ('run_uuid', 'env_id', 'episode_uid', 'tick')), physics_substeps=[raw] * 8))
    cases = module().source_cases(data, context, report, dataset_sha256='d' * 64)
    assert cases[0]['event_ticks']['CONTACT'] == 2
    assert cases[0]['event_sources']['contact'] == 'identity_joined_raw_precontrol_effective_pad_measurements'


def test_complete_initial_source_readback_is_exact_by_default_and_keeps_mismatch_details():
    names = module().SOURCE_STATE_FIELDS
    state = {key: torch.zeros(8, 2, 7) for key in names}
    changed = copy.deepcopy(state)
    changed['object_position'][0, 0, 0] = 1e-7
    details = module().compare_source_reset(state, changed)
    assert not details['matched']
    assert details['tolerance'] == 0.
    assert details['fields']['object_position']['max_abs_error'] > 0
    assert details['fields']['q']['dtype_equal'] and details['fields']['q']['shape_equal']
    with pytest.raises(ValueError, match='complete|reference_bottom'):
        module().compare_source_reset({k: v for k, v in state.items() if k != 'reference_bottom'}, state)


def test_action_groups_preserve_original_sdk24_and_first_numeric_difference_is_not_failure():
    error = module().action_group_errors(torch.arange(24.) / 24, torch.zeros(24))
    assert error['left_thumb']['max_abs'] == pytest.approx(1 / 24)
    assert error['right_index']['max_abs'] == pytest.approx(8 / 24)
    assert error['left_palm_xyz']['max_abs'] == pytest.approx(14 / 24)
    assert error['right_wrist_rotation']['max_abs'] == pytest.approx(23 / 24)


def test_first_physical_event_and_first_terminal_are_owned_pre_reset_and_never_overwritten():
    runner = module()
    rows = []
    collector = runner.SourceTelemetryCollector([dict(env_id=0, episode_id='original')], 8,
        native=False, sink=rows.append)
    collector.begin(reset_clock=dict(physics_step=0, timestamp_s=0.), reset_episode_ids=torch.zeros(8, dtype=torch.long))
    def payload(contact, terminal=False, success=False):
        return dict(dt_s=1/480, live=torch.ones(8, dtype=torch.bool), done=torch.tensor([terminal] + [False] * 7),
            native_success=torch.tensor([success] + [False] * 7), measurement_valid=torch.ones(8, dtype=torch.bool),
            contact_valid=torch.ones(8, dtype=torch.bool), bilateral_opposed_contact=torch.full((8,), contact),
            lift_m=torch.full((8,), .031), qualified_hold_s=torch.full((8,), 5.),
            slip_valid=torch.ones(8, dtype=torch.bool), slip_m=torch.zeros(8))
    collector.set_policy_step(0, ['student'] * 4 + ['unevaluated'] * 4)
    collector.observe(payload(True), physics_step=1, timestamp_s=1/480, failure=torch.zeros(8, dtype=torch.long))
    collector.observe(payload(False, True, True), physics_step=2, timestamp_s=2/480,
        failure=torch.tensor([3] + [0] * 7))
    collector.observe(payload(True, True, False), physics_step=3, timestamp_s=3/480,
        failure=torch.tensor([6] + [0] * 7))
    result = collector.summary()['source_rows'][0]
    assert len(rows) == 2 and result['terminal']['failure_code'] == 3
    assert result['terminal']['physics_step'] == 2
    assert result['first_task_related_physical_event']['kind'] == 'joint_limit'
    assert result['native_success'] is False and result['observed_native_success_flag'] is True


def test_execution_uids_are_unique_and_teacher_prefix_termination_is_separate_from_student_execution():
    runner = module()
    first = runner.SourceTelemetryCollector([dict(env_id=0, episode_id='a')], 8, native=False, sink=lambda row: None)
    second = runner.SourceTelemetryCollector([dict(env_id=0, episode_id='a')], 8, native=False, sink=lambda row: None)
    assert first.execution_run_uuid != second.execution_run_uuid


def write_config(tmp_path, mode='source'):
    runner = module()
    identities = {}
    for name in runner.INPUT_NAMES:
        path = tmp_path / (name + '.bin')
        path.write_bytes(name.encode())
        identities[name] = dict(path=str(path), sha256=runner.file_sha256(path))
    config = dict(version='dh116-stage6-native-v1', mode=mode, **identities)
    path = tmp_path / 'native.json'
    path.write_text(json.dumps(config))
    return path, config


def test_native_config_pins_actual_paths_and_rechecks_immutable_inputs(tmp_path):
    path, config = write_config(tmp_path)
    loaded = module().load_native_config(path)
    assert loaded['mode'] == 'source'
    checkpoint = tmp_path / 'checkpoint.bin'
    checkpoint.write_bytes(b'modified')
    with pytest.raises(ValueError, match='SHA|sha256|changed'):
        module().verify_inputs(loaded)


@pytest.mark.parametrize('changes', [dict(mode='ppo'), dict(mode='branch', event='CONTACT', branch_steps=2),
    dict(mode='branch', event='phase', branch_steps=4), dict(initial_state_tolerance=2e-5)])
def test_native_config_rejects_wrong_modes_horizons_and_silent_reset_tolerance(tmp_path, changes):
    path, config = write_config(tmp_path)
    config.update(changes)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        module().load_native_config(path)


def runner_fixture(tmp_path, monkeypatch, *, mode='source', event='CONTACT', steps=1, terminal_tick=2):
    runner = module()
    from go2_pvcnn.control.dh116 import tabletop_vision_training as training
    from go2_pvcnn.control.dh116 import tabletop_vision_inference as inference
    from go2_pvcnn.control.dh116 import tabletop_vision_student as student_module
    data, context, collection = source_fixture()
    native = SimpleNamespace(ids=[list(range(6)), list(range(6))],
        sim=SimpleNamespace(current_time=0., current_time_step_index=0))
    device = SimpleNamespace(n=8, w=native, _physics_steps=0,
        q=torch.zeros(8, 2, 6), qd=torch.zeros(8, 2, 6),
        roots=torch.zeros(8, 2, 7), wp=torch.zeros(8, 3), wq=torch.zeros(8, 4),
        origins=torch.zeros(8, 3))
    device.roots[..., 3] = 1.; device.wq[:, 0] = 1.
    task = SimpleNamespace(command=torch.zeros(8, 12), surface_reference_bottom_m=torch.zeros(8),
        failure=torch.zeros(8, dtype=torch.long))
    expected = runner._reset_snapshot(device, task)
    expected.update(seed=42, source='actual_native_cold_reset_before_camera')
    identities = {}
    for name, value in [('checkpoint', b'fixture-model'), ('teacher_dataset', data),
        ('source_initial_state', expected), ('source_collection_report', collection)]:
        path = tmp_path / (name + ('.json' if name == 'source_collection_report' else '.pt'))
        if isinstance(value, bytes): path.write_bytes(value)
        elif name == 'source_collection_report': path.write_text(json.dumps(value))
        else: torch.save(value, path)
        identities[name] = dict(path=str(path), sha256=runner.file_sha256(path))
    context['dataset_sha256'] = identities['teacher_dataset']['sha256']
    path = tmp_path / 'sequence_context.pt'; torch.save(context, path)
    identities['sequence_context'] = dict(path=str(path), sha256=runner.file_sha256(path))
    config = dict(version=runner.VERSION, mode=mode, event=event, branch_steps=steps, **identities)
    config_path = tmp_path / 'native.json'; config_path.write_text(json.dumps(config))
    issued = []
    class FakeEnv:
        evaluation_telemetry = None
        diagnostic_episode_ids = torch.zeros(8, dtype=torch.long)
        def enable_evaluation_telemetry(self, callback):
            self.evaluation_telemetry = SimpleNamespace(callback=callback)
        def reset(self):
            self.tick = 0
            return torch.full((8, 371), 11.)
        def step(self, action):
            issued.append(action.clone())
            done = torch.zeros(8, dtype=torch.bool)
            if self.tick == terminal_tick: done[:4] = True
            for substep in range(8):
                device._physics_steps += 1
                native.sim.current_time_step_index += 1
                native.sim.current_time += 1/480
                terminal = done & (substep == 7)
                payload = dict(dt_s=1/480, live=torch.ones(8, dtype=torch.bool), done=terminal,
                    native_success=terminal, measurement_valid=torch.ones(8, dtype=torch.bool),
                    contact_valid=torch.ones(8, dtype=torch.bool), bilateral_opposed_contact=torch.ones(8, dtype=torch.bool),
                    lift_m=torch.full((8,), .031), qualified_hold_s=torch.full((8,), 5.),
                    slip_valid=torch.ones(8, dtype=torch.bool), slip_m=torch.zeros(8), action=action.clone())
                self.evaluation_telemetry.callback(payload)
            self.tick += 1
            self.diagnostic_episode_ids += done.long()
            return torch.full((8, 371), -999.), torch.zeros(8), done, dict(recovered=torch.zeros(8, dtype=torch.bool))
    class FakeCamera:
        def sample(self):
            return dict(points=torch.zeros(8, 2, 128, 3), point_valid=torch.ones(8, 128, dtype=torch.bool),
                image_valid=torch.ones(8, dtype=torch.bool), object_valid=torch.ones(8, dtype=torch.bool))
    class FakePolicy:
        checkpoint_sha256 = identities['checkpoint']['sha256']
        provenance = dict(source_contract=data['visual_contract'])
        def act(self, obs):
            observation = obs if isinstance(obs, torch.Tensor) else obs['policy']
            assert observation.shape[1] == 3844
            return torch.full((len(observation), 24), .73)
    monkeypatch.setattr(inference, 'load_vision_inference', lambda *a, **kw: FakePolicy())
    monkeypatch.setattr(training, '_contract', lambda *a: data['visual_contract'])
    def no_ppo(*args, **kwargs): raise AssertionError('PPO construction is forbidden')
    monkeypatch.setattr(student_module, 'make_vision_algorithm', no_ppo)
    args = SimpleNamespace(num_envs=8, seed=42, device='cpu', vision_stage6_native_config=str(config_path),
        vision_mirror_preprocessing='m0_raw', vision_camera_mode='fixed', vision_action_distribution='tanh_gaussian',
        output=str(tmp_path / 'output'))
    return args, native, device, task, FakeEnv(), FakeCamera(), issued


@pytest.mark.parametrize('mode,horizon,expected_steps', [('source', 1, 3), ('branch', 1, 3),
    ('branch', 4, 6), ('teacher-replay', 1, 8)])
def test_runner_uses_actual_history_and_first_episode_bounded_scheduling_without_ppo(
        tmp_path, monkeypatch, mode, horizon, expected_steps):
    args, native, device, task, env, camera, issued = runner_fixture(tmp_path, monkeypatch,
        mode=mode, steps=horizon, terminal_tick=2 if mode == 'source' else 999)
    report = module().run_stage6_native(native, device, task, env, args, {}, camera=camera)
    assert report['control_steps'] == expected_steps and len(issued) == expected_steps
    assert report['native_runtime_verified'] is False and report['native_success_count'] == 0
    assert report['pure_student_native_success_count'] == 0 and report['ppo_updates'] == 0
    assert report['production_evaluation_certificate_issued'] is False
    assert len(report['source_rows']) == 4
    if mode == 'source': assert report['completed_count'] == 4
    assert all(torch.equal(action[4:], torch.zeros(4, 24)) for action in issued)
    with gzip.open(tmp_path / 'output' / 'live_policy_60hz.jsonl.gz', 'rt') as stream:
        live = [json.loads(line) for line in stream]
    row = next(row for row in live if row['policy_step'] == 1 and row['env_id'] == 0)
    assert row['actual_previous_issued_action'] == issued[0][0].tolist()
    assert row['episode_uid'] == report['source_rows'][0]['execution_episode_uid']
    assert row['episode_uid'].startswith(report['execution_run_uuid'] + ':0:')
    if mode == 'branch':
        assert issued[0][0].sum() == 0. and issued[1][0, 0] == pytest.approx(.001)
        assert torch.equal(issued[2][0], torch.full((24,), .73))
    assert not (tmp_path / 'output' / 'vision_evaluation_certificate.json').exists()


def test_injected_camera_cannot_bypass_native_type_checks_without_explicit_fixture_path(tmp_path, monkeypatch):
    args, native, device, task, env, _, _ = runner_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='native|fixtures'):
        module().run_stage6_native(native, device, task, env, args, {}, camera=None)


def test_runner_retains_reset_mismatch_and_writes_no_success(tmp_path, monkeypatch):
    args, native, device, task, env, camera, issued = runner_fixture(tmp_path, monkeypatch)
    device.wp[0, 0] = 1e-7
    with pytest.raises(RuntimeError, match='initial state mismatch'):
        module().run_stage6_native(native, device, task, env, args, {}, camera=camera)
    assert not issued
    report = json.loads((tmp_path / 'output' / 'stage6_native_report.json').read_text())
    assert report['status'] == 'error' and report['native_success_count'] == 0
    detail = json.loads((tmp_path / 'output' / 'source_reset_comparison.json').read_text())
    assert not detail['matched'] and detail['fields']['object_position']['max_abs_error'] > 0


def test_runner_rechecks_source_inputs_after_the_last_actual_step(tmp_path, monkeypatch):
    args, native, device, task, env, camera, _ = runner_fixture(tmp_path, monkeypatch)
    original = env.step
    def tampered_step(action):
        result = original(action)
        (tmp_path / 'checkpoint.pt').write_bytes(b'mutated_checkpoint')
        return result
    env.step = tampered_step
    with pytest.raises(ValueError, match='immutable.*SHA256.*checkpoint'):
        module().run_stage6_native(native, device, task, env, args, {}, camera=camera)


def test_conditional_comparison_retains_cumulative_student_targets_over_actual_eight_substeps(tmp_path):
    from test_vision_command_shadow import fixture
    from go2_pvcnn.control.dh116 import tabletop_vision_command_shadow as shadow
    task, world = fixture()
    state = shadow.command_state(task, world)
    one = shadow.measured_context(task, world, torch.ones(3, dtype=torch.bool))
    one['allowed'][:] = True
    data, context, _ = source_fixture()
    rows = list(range(8))
    context['controller_contract'] = shadow.controller_contract(task, world)
    context['controller_start'] = {k: v[:1].expand(32, *v.shape[1:]).clone() for k, v in state.items()}
    context['physics_context'] = {k: v[:1, None].expand(32, 8, *v.shape[1:]).clone() for k, v in one.items()}
    teacher = shadow.shadow_sequence({k: v[:1] for k, v in state.items()}, data['actions'][rows][None],
        {k: v[rows][None] for k, v in context['physics_context'].items()}, context['controller_contract'])
    context['teacher_targets'] = {k: value[0].repeat(4, *([1] * (value.ndim - 2))) for k, value in teacher.items()}
    case = dict(rows=rows, context_indices=torch.arange(8), env_id=0, episode_uid='fixture',
        actions=data['actions'][rows], event_ticks=dict(CONTACT=2, LIFT_ONSET=3, HOLD=6))
    class Policy:
        def act(self, obs): return torch.full((len(obs), 24), .1)
    result = module()._saved_teacher_comparisons(Policy(), data, context, [case], tmp_path)
    with gzip.open(result['path'], 'rt') as stream: samples = [json.loads(row) for row in stream]
    assert len(samples) == 8
    first = samples[0]['conditional_cumulative_targets']['wrist_angle']['max_abs']
    last = samples[-1]['conditional_cumulative_targets']['wrist_angle']['max_abs']
    assert last > first > 0.
    assert result['first_numeric_difference'][0]['caused_failure_claimed'] is False


def test_branch_terminal_during_teacher_prefix_never_claims_a_student_branch(tmp_path, monkeypatch):
    args, native, device, task, env, camera, issued = runner_fixture(tmp_path, monkeypatch,
        mode='branch', event='CONTACT', steps=16, terminal_tick=1)
    report = module().run_stage6_native(native, device, task, env, args, {}, camera=camera)
    assert report['branch_switch_reached_count'] == 0
    assert report['teacher_prefix_terminal_count'] == 4
    assert all(row['student_policy_steps_executed'] == 0 for row in report['source_rows'])
    assert len(issued) == 2


def test_reused_native_physics_writer_labels_actual_mixed_ownership_instead_of_calling_teacher_student(tmp_path):
    from test_vision_evaluation_physics_audit import fixture, record
    env, world, manifest = fixture(tmp_path)
    owners = ['teacher_prefix', 'student']
    writer = module().install_stage6_physics_diagnostics(env, world, manifest, tmp_path,
        ownership=lambda: owners, checkpoint_sha256='a' * 64)
    writer.submit(torch.zeros(2, 24), policy_step=0)
    record(writer)
    result = writer.finish()
    with gzip.open(tmp_path / 'evaluation_action_diagnostics' / 'policy_requests.jsonl.gz', 'rt') as stream:
        row = json.loads(next(stream))
    assert row['source'] == 'actual_issued_SDK24_action'
    assert row['action_owner'] == owners
    assert result['actor_source'] == 'explicit_per_environment_action_owner;Teacher_prefix_or_Student'
    assert result['native_trials_certified_by_writer'] == 0


def test_neutral_four_environments_are_ineligible_even_in_reused_physics_diagnostics(tmp_path):
    env = SimpleNamespace(physics_action_diagnostics=None, _diagnostic_policy_step=0,
        diagnostic_episode_ids=torch.zeros(8, dtype=torch.long))
    world = SimpleNamespace(n=8)
    manifest = dict(trials=[dict(trial_id=str(i), grasp_style='pinch_thumb_index') for i in range(8)])
    writer = module().install_stage6_physics_diagnostics(env, world, manifest, tmp_path,
        ownership=lambda: ['student'] * 4 + ['unevaluated'] * 4)
    writer.submit(torch.zeros(8, 24), policy_step=0)
    summary = writer.finish()
    with gzip.open(tmp_path / 'evaluation_action_diagnostics' / 'policy_requests.jsonl.gz', 'rt') as stream:
        row = json.loads(next(stream))
    assert row['first_trial_eligible'] == [True] * 4 + [False] * 4
    assert summary['evaluated_source_identity_count'] == 4


def test_first_issued_numeric_difference_excludes_unexecuted_student_prefix_prediction():
    collector = module().SourceTelemetryCollector([{'env_id':0,'episode_id':'source'}],8,
                                                native=False,sink=lambda row:None)
    actual = torch.ones(24)
    reference = torch.zeros(24)
    collector.record_issued_difference(0,policy_step=1,owner='teacher_prefix',
        issued_action=actual,recorded_teacher_action=reference,clock={'physics_step':8,'timestamp_s':1/60})
    assert collector.rows[0]['first_numeric_difference'] is None
    collector.record_issued_difference(0,policy_step=2,owner='student',
        issued_action=actual,recorded_teacher_action=reference,clock={'physics_step':16,'timestamp_s':2/60})
    collector.record_issued_difference(0,policy_step=3,owner='student',
        issued_action=actual*2,recorded_teacher_action=reference,clock={'physics_step':24,'timestamp_s':3/60})
    first = collector.rows[0]['first_numeric_difference']
    assert first['policy_step'] == 2 and first['max_absolute_action_difference'] == 1.
    assert first['caused_failure_claimed'] is False
    assert collector.rows[0]['first_task_related_physical_event'] is None
