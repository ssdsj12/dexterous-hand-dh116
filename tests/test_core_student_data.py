"""Current six-key Teacher queries and independent offline Student updates."""
import copy
import importlib
import importlib.util
import pytest
import torch
from test_core_student import history
from test_core_student import NETWORK
from test_core_student import PREFIX
from dh116.rl.student_data import PreActionState
from dh116.rl.student_data import StepResult
from dh116.rl.student_data import collect_step
from dh116.rl.teacher import tenso_teach_schem__INPUT_KEYS as INPUT_KEYS
from dh116.rl.teacher import common_selection as common_selection

class Runtime:
    evidence_scope = 'cpu_fixture'

    def __init__(self):
        self.tick = 0
        self.h = history()
        self.actions = []

    def snapshot(self):
        ident = dict(run_uuid='fixture', env_id=0, episode_uid='episode-0', reset_id=0, policy_tick=self.tick, physics_tick_start=self.tick * 8, physics_tick_end=self.tick * 8, simulation_timestamp=self.tick / 60, frame_id=self.tick, calibration_sha256='c' * 64, physical_initial_condition_sha256='a' * 64, source_episode_group='fixture-0')
        valid = torch.zeros(1, 371, dtype=torch.bool)
        valid[:, common_selection()] = True
        raw = torch.full((1, 371), float('nan'))
        raw[valid] = 0.0
        return PreActionState(self.h.clone(), raw, valid, [ident], extra_labels={'geometry': torch.full((1, 128, 3), 0.2)}, label_validity={'geometry': torch.ones(1, 128, 3, dtype=torch.bool)})

    def set_action_owners(self, owners):
        self.owners = owners

    def step(self, actions):
        before = self.tick
        self.actions.append(actions.clone())
        self.tick += 1
        self.h.proprio[:, -1, 38:] = actions
        info = dict(evidence_scope='cpu_fixture', execution=[dict(physics_tick_start=before * 8, physics_tick_end=self.tick * 8, simulation_timestamp_start=before / 60, simulation_timestamp_end=self.tick / 60)])
        return StepResult(self.snapshot(), torch.ones(1), torch.zeros(1, dtype=torch.bool), info, actions.clone())

class Teacher(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.calls = []

    def deterministic(self, value):
        assert set(value) == set(INPUT_KEYS)
        assert torch.isnan(value['common371'][~value['common_valid']]).all()
        self.calls.append(copy.deepcopy(value))
        return torch.full((1, 24), 0.2)

class Student(torch.nn.Module):

    def deterministic(self, value):
        assert type(value).__name__ == 'StereoHistoryBatch'
        return torch.full((len(value), 24), 0.1)

def teacher_inputs(before):
    return dict(inputs=dict(common371=before.privileged.clone(), common_valid=before.privileged_valid.clone(), rod_nodes=torch.ones(1, 41, 13), rod_mask=torch.ones(1, 41, dtype=torch.bool), task_context=torch.full((1, 50), float('nan')), task_context_valid=torch.zeros(1, 50, dtype=torch.bool)), identities=copy.deepcopy(before.identities))

def record(runtime=None):
    return collect_step(Runtime() if runtime is None else runtime, Student(), Teacher(), teacher_probability=0.0, schema_only=True, teacher_inputs=teacher_inputs)

def dataset():
    from dh116.rl.student_data import build_dataset
    return build_dataset([record()], source_sha256='d' * 64, schema_only=True)

def test_same_preaction_six_key_query_and_only_student_advances_runtime():
    runtime = Runtime()
    teacher = Teacher()
    result = collect_step(runtime, Student(), teacher, teacher_probability=0.0, schema_only=True, teacher_inputs=teacher_inputs)
    assert len(teacher.calls) == len(runtime.actions) == 1 and runtime.tick == 1
    assert runtime.owners == result['action_owner'] == ['student']
    assert result['identities'][0]['policy_tick'] == 0
    assert (result['teacher_mean'] == 0.2).all() and (runtime.actions[0] == 0.1).all()
    assert torch.isnan(result['privileged'][~result['privileged_valid']]).all()
    assert not result['native_qualified']

def test_teacher_input_identity_or_label_mutation_stops_before_execution():
    runtime = Runtime()
    teacher = Teacher()

    def wrong(before):
        result = teacher_inputs(before)
        result['identities'][0]['frame_id'] += 1
        return result
    with pytest.raises(ValueError, match='aligned'):
        collect_step(runtime, Student(), teacher, teacher_probability=0.0, schema_only=True, teacher_inputs=wrong)
    assert not runtime.actions and (not teacher.calls)

    def fabricated(before):
        result = teacher_inputs(before)
        result['inputs']['common371'] = torch.nan_to_num(result['inputs']['common371'])
        return result
    with pytest.raises(ValueError, match='common|NaN|snapshot'):
        collect_step(runtime, Student(), teacher, teacher_probability=0.0, schema_only=True, teacher_inputs=fabricated)
    assert not runtime.actions

def test_group_loss_matches_original_three_group_arithmetic_and_masked_nan():
    loss_module = importlib.import_module(PREFIX + 'rl.student_data')
    assert hasattr(loss_module, 'masked_group_normalized_loss'), 'pure stereo action loss is missing'
    p = torch.arange(48, dtype=torch.float32).reshape(2, 24) / 48
    p.requires_grad_()
    mask = torch.zeros(2, 24, dtype=torch.bool)
    mask[0, :12] = True
    mask[0, 12:15] = True
    mask[1, 18:] = True
    t = torch.full_like(p, float('nan'))
    t[mask] = 0.25
    expected = ((p[0, :12] - 0.25).square().mean() + (p[0, 12:15] - 0.25).square().mean()) / 4 + (p[1, 18:] - 0.25).square().mean() / 2
    loss = loss_module.masked_group_normalized_loss(p, t, mask)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(p.grad).all() and (p.grad[~mask] == 0).all()

def test_dataset_expert_admission_and_batch_binding_remain_fail_closed():
    from dh116.rl.student_data import validate_dataset as validate_dataset
    from dh116.rl.student_data import batch_dataset as batch_dataset
    from dh116.rl.student_data import StereoBCTrainer as StereoBCTrainer
    from dh116.rl.student_model import make_stereo_actor
    from dh116.rl.student_data import require_teacher as require_teacher
    data = dataset()
    with pytest.raises(ValueError, match='schema-only'):
        validate_dataset(data)
    with pytest.raises(ValueError):
        require_teacher({'qualified': True}, 'd' * 64)
    batch = batch_dataset(data, [0])
    batch['teacher_actions'] += 0.01
    with pytest.raises(ValueError, match='differs'):
        StereoBCTrainer(make_stereo_actor(NETWORK), schema_only=True).step(batch)

def test_current_offline_bc_state_fit_and_shared_adapter(tmp_path):
    assert importlib.util.find_spec(PREFIX + 'rl.training'), 'pure current Student API is missing'
    api = importlib.import_module(PREFIX + 'rl.training')
    data = dataset()
    report = api.train_bc(data, NETWORK, {}, output=tmp_path / 'bc', steps=1, batch_size=1, schema_only=True)
    assert report['steps'] == 1 and report['schema_only'] and (report['native_task_qualified'] is False)
    state = api.fit_state(data, NETWORK, output=tmp_path / 'state', steps=1, batch_size=1, schema_only=True)
    assert state['steps'] == 1 and state['native_task_qualified'] is False
    assert state['updates'][0]['losses']['state'] > 0
    assert state['updates'][0]['gradient_norms']['state'] > 0
    runtime = Runtime()
    adapter = api.SharedStudentCollector(runtime, sensor_history=lambda before: before.sensor_history, teacher_inputs=teacher_inputs)
    sample = adapter.collect(Student(), Teacher(), steps=1, schema_only=True)
    assert runtime.tick == 1 and sample[0]['action_owner'] == ['student']
    with pytest.raises(ValueError):
        api.train_bc(data, NETWORK, {}, output=tmp_path / 'formal', steps=1)
    assert not (tmp_path / 'formal').exists()

def test_current_production_ppo_and_collection_require_existing_qualifications():
    api = importlib.import_module(PREFIX + 'rl.training')
    with pytest.raises(ValueError, match='qualified Teacher'):
        api.make_student_trainer(dict(network=NETWORK), num_envs=1)
    runtime = Runtime()
    adapter = api.SharedStudentCollector(runtime, sensor_history=lambda before: before.sensor_history, teacher_inputs=teacher_inputs)
    with pytest.raises(ValueError, match='qualified Teacher'):
        adapter.collect(Student(), Teacher(), steps=1)
    assert not runtime.actions

def test_state_fit_rejects_completely_unobserved_labels(tmp_path):
    api = importlib.import_module(PREFIX + 'rl.training')
    data = dataset()
    data['rows'][0]['label_validity']['geometry'].zero_()
    with pytest.raises(ValueError, match='observed label'):
        api.fit_state(data, NETWORK, output=tmp_path / 'empty', steps=1, schema_only=True)
    assert not (tmp_path / 'empty').exists()

def test_state_fit_selects_masked_heldout_state_loss_without_reading_action_head(tmp_path, monkeypatch):
    api = importlib.import_module(PREFIX + 'rl.training')
    from dh116.rl.student_model import StereoStudentActor as StereoStudentActor
    from dh116.rl.student_data import batch_dataset as batch_dataset
    from dh116.rl.student_data import StereoEstimationLoss as StereoEstimationLoss
    data = dataset()
    heldout = copy.deepcopy(data['rows'][0])
    for name in ('identity', 'privileged_identity'):
        heldout[name].update(env_id=1, physical_initial_condition_sha256='b' * 64, source_episode_group='heldout-b')
    heldout['extra_labels']['geometry'].fill_(0.8)
    data['rows'].append(heldout)

    def forbidden(*args, **kwargs):
        raise AssertionError('state-only validation must not read untrained action head')
    monkeypatch.setattr(StereoStudentActor, 'deterministic', forbidden)
    result = api.fit_state(data, NETWORK, output=tmp_path / 'heldout', steps=2, batch_size=1, heldout_initial_sha256=['b' * 64], schema_only=True)
    assert result['selection'] == 'lowest_heldout_state_loss'
    scores = [row['validation_state_loss'] for row in result['updates']]
    assert all((score is not None and score > 0 for score in scores))
    actor = api.load_deployment_actor(tmp_path / 'heldout/stereo_bc_actor.pt')
    batch = batch_dataset(data, [1])
    with torch.no_grad():
        actual = StereoEstimationLoss()(actor.estimator(batch['history']), batch['targets'], batch['validity'])['total']
    assert float(actual) == pytest.approx(min(scores), abs=1e-07)

def test_second_snapshot_cannot_drop_a_valid_environment_before_execution():
    from dh116.rl.student_model import StereoHistoryBatch as StereoHistoryBatch
    runtime = Runtime()
    one = runtime.snapshot()
    two = one.clone()
    two.sensor_history = StereoHistoryBatch.cat([one.sensor_history, one.sensor_history])
    two.privileged = one.privileged.repeat(2, 1)
    two.privileged_valid = one.privileged_valid.repeat(2, 1)
    two.identities.append(dict(one.identities[0], env_id=1))
    two.validate()
    states = iter([two, one])
    runtime.snapshot = lambda : next(states)

    def forbidden(*args):
        raise AssertionError('SDK step ran after the current batch lost a lane')
    runtime.step = forbidden

    def inputs(before):
        n = len(before.sensor_history)
        value = teacher_inputs(one)
        value['inputs'] = {key: part.repeat(n, *[1] * (part.ndim - 1)) for (key, part) in value['inputs'].items()}
        value['identities'] = copy.deepcopy(before.identities)
        return value
    teacher = Teacher()
    teacher.deterministic = lambda value: torch.zeros(len(value['common371']), 24)
    with pytest.raises(ValueError, match='batch|environment|count'):
        collect_step(runtime, Student(), teacher, teacher_probability=0.0, schema_only=True, teacher_inputs=inputs)

def test_teacher_checkpoint_binding_refuses_different_or_changed_weights(tmp_path):
    api = importlib.import_module(PREFIX + 'rl.training')
    assert hasattr(api, 'bind_frozen_teacher'), 'actual checkpoint/model binding required'
    from dh116.rl.teacher import TensorNativeTeacher
    from dh116.rl.teacher import initialize_native_network
    from dh116.env.contracts import stere_contr__file_sha256 as file_sha256
    teacher = TensorNativeTeacher.__new__(TensorNativeTeacher)
    initialize_native_network(teacher, {'test_scope': 'CPU_architecture_only'}, __import__('types').SimpleNamespace(binary_sha256='0' * 64), 'approach_wiring_smoke_v1', 0.1)
    path = tmp_path / 'teacher.pt'
    torch.save(dict(version='dh116-tensor-native-checkpoint-v2', identity={'model': teacher.model_identity}, policy=copy.deepcopy(teacher.state_dict())), path)
    sha = file_sha256(path)
    binding = api.bind_frozen_teacher(teacher, path, expected_sha256=sha)
    binding.verify(teacher, sha)
    assert not teacher.training and (not any((parameter.requires_grad for parameter in teacher.parameters())))
    wrong = copy.deepcopy(teacher)
    with pytest.raises(ValueError, match='model|Teacher'):
        binding.verify(wrong, sha)
    next(teacher.parameters()).data.add_(0.01)
    with pytest.raises(ValueError, match='weight|state|changed|version'):
        binding.verify(teacher, sha)
    with pytest.raises(ValueError, match='state|weight|checkpoint'):
        api.bind_frozen_teacher(teacher, path, expected_sha256=sha)

@pytest.mark.parametrize('when', ['query', 'step'])
def test_bound_teacher_changes_during_collection_do_not_publish_labels(tmp_path, when):
    api = importlib.import_module(PREFIX + 'rl.training')
    from dh116.rl.teacher import TensorNativeTeacher
    from dh116.rl.teacher import initialize_native_network
    from dh116.env.contracts import stere_contr__file_sha256 as file_sha256
    teacher = TensorNativeTeacher.__new__(TensorNativeTeacher)
    initialize_native_network(teacher, {'test_scope': 'CPU_architecture_only'}, __import__('types').SimpleNamespace(binary_sha256='0' * 64), 'approach_wiring_smoke_v1', 0.1)
    path = tmp_path / 'teacher.pt'
    torch.save(dict(version='dh116-tensor-native-checkpoint-v2', identity={'model': teacher.model_identity}, policy=copy.deepcopy(teacher.state_dict())), path)
    sha = file_sha256(path)
    binding = api.bind_frozen_teacher(teacher, path, expected_sha256=sha)
    teacher.deterministic = lambda inputs: torch.full((len(inputs['common371']), 24), 0.2)
    teacher.latent_mean = lambda inputs: torch.zeros(len(inputs['common371']), 24)
    runtime = Runtime()

    def inputs(before):
        if when == 'query':
            next(teacher.parameters()).data.add_(0.01)
        return teacher_inputs(before)
    step = runtime.step

    def advancing(actions):
        result = step(actions)
        if when == 'step':
            next(teacher.parameters()).data.add_(0.01)
        return result
    runtime.step = advancing
    with pytest.raises(ValueError, match='weight|state|changed'):
        collect_step(runtime, Student(), teacher, teacher_probability=0.0, schema_only=True, teacher_inputs=inputs, teacher_binding=binding, source_sha256=sha)
    assert len(runtime.actions) == (0 if when == 'query' else 1)
