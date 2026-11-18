"""CPU source joins with only LOWER native qualification mocked; no native pass."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from test_dh116_teacher_expert_assembly import artifacts, CPU_only_gate
from test_dh116_privileged_cli import CPUFixtureBridge, CONFIG
from go2_pvcnn.control.dh116 import tabletop_privileged_cli as cli
from go2_pvcnn.control.dh116 import tabletop_privileged_dataset as data
from go2_pvcnn.control.dh116 import tabletop_privileged_expert_assembly as build
from go2_pvcnn.control.dh116 import tabletop_privileged_warm_start as warm
from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256


@pytest.fixture
def assembled(tmp_path, monkeypatch):
    identity, manifest = artifacts(tmp_path)
    CPU_only_gate(monkeypatch)
    build.assemble_experts(manifest, identity, tmp_path / 'assembled')
    path = tmp_path / 'assembled/privileged_native_expert.pt'
    return identity, torch.load(path, weights_only=True), path


def damage_rows(payload, damage):
    if damage == 'observation':
        payload['observations'][0, 0] = 123.
    elif damage == 'action':
        payload['actions'][0, 0] = .9
    elif damage == 'mask':
        payload['valid_mask'][0, 0] = False
    elif damage == 'order':
        payload['actions'] = payload['actions'].flip(0)
    elif damage == 'dtype':
        payload['observations'] = payload['observations'].double()
    elif damage == 'action_dtype':
        payload['actions'] = payload['actions'].double()
    else:
        raise AssertionError(damage)


@pytest.mark.parametrize('damage', ['observation', 'action', 'mask', 'order', 'dtype', 'action_dtype'])
def test_same_proofs_cannot_bless_changed_training_rows(assembled, damage):
    identity, payload, _ = assembled
    damage_rows(payload, damage)
    with pytest.raises(ValueError, match='source|issued|mask|dtype'):
        data.validate_expert_dataset(payload, expected_identity=identity)


def test_intact_source_and_tuple_bindings_are_preserved(assembled):
    identity, payload, path = assembled
    assert data.validate_expert_dataset(payload, expected_identity=identity) is payload
    loaded, bindings = warm.read_expert(path, file_sha256(path), identity)
    assert torch.equal(loaded['actions'], payload['actions'])
    assert bindings == [dict(episode_id='cpu-fixture', source_sha256=payload['episodes'][0]['source_sha256'],
        rows=2, first_policy_tick=0, last_policy_tick=1)]


def test_each_admission_replays_once_and_does_not_cache_mutable_payload(assembled, monkeypatch):
    identity, payload, path = assembled
    replay = warm.verify_supervised_rows
    calls = []
    def counted(value):
        calls.append(value)
        return replay(value)
    monkeypatch.setattr(warm, 'verify_supervised_rows', counted)
    warm.read_expert(path, file_sha256(path), identity)
    assert len(calls) == 1
    warm.read_expert(path, file_sha256(path), identity)
    assert len(calls) == 2
    data.validate_expert_dataset(payload, expected_identity=identity)
    assert len(calls) == 3
    damage_rows(payload, 'mask')
    with pytest.raises(ValueError, match='mask|source'):
        data.validate_expert_dataset(payload, expected_identity=identity)
    assert len(calls) == 4


def test_same_uid_different_source_file_is_not_owned_by_qualified_process(assembled, tmp_path):
    identity, payload, _ = assembled
    episode = payload['episodes'][0]
    other = tmp_path / 'foreign_process_source.pt'
    other.write_bytes(Path(episode['source_path']).read_bytes())
    episode['source_path'] = str(other)
    with pytest.raises(ValueError, match='process completion'):
        data.validate_expert_dataset(payload, expected_identity=identity)


def test_same_uid_other_process_completion_cannot_bless_source(assembled, tmp_path):
    identity, payload, _ = assembled
    qualification = json.loads(Path(payload['episodes'][0]['qualification_evidence']).read_text())
    foreign = tmp_path / 'other_process_completion.json'
    foreign.write_text(json.dumps(dict(produced_artifacts=[qualification['raw_artifact']])))
    qualification['completion_artifact'] = dict(path=str(foreign), sha256=file_sha256(foreign))
    payload['episodes'][0]['qualification_evidence'] = qualification
    with pytest.raises(ValueError, match='process completion'):
        data.validate_expert_dataset(payload, expected_identity=identity)


@pytest.mark.parametrize('field', ['physics_step', 'policy_tick', 'actor_issued_action', 'terminal'])
def test_raw_action_and_clock_join_is_replayed_on_admission(assembled, tmp_path, field):
    identity, payload, _ = assembled
    qualification = json.loads(Path(payload['episodes'][0]['qualification_evidence']).read_text())
    physical = Path(qualification['raw_artifact']['path'])
    rows = [json.loads(line) for line in physical.read_text().splitlines()]
    if field == 'actor_issued_action':
        rows[1][field][0] = .9
    elif field == 'terminal':
        rows[1][field] = True
    else:
        rows[1][field] += 1
    altered = tmp_path / 'altered_physics.jsonl'
    altered.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    qualification['raw_artifact'] = dict(path=str(altered), sha256=file_sha256(altered))
    payload['episodes'][0]['qualification_evidence'] = qualification
    with pytest.raises(ValueError, match='source|clock|issued'):
        data.validate_expert_dataset(payload, expected_identity=identity)


def test_legacy_3844_observations_are_rejected_without_conversion(assembled):
    identity, payload, _ = assembled
    payload['observations'] = torch.zeros(2, 3844)
    with pytest.raises(ValueError, match='371'):
        data.validate_expert_dataset(payload, expected_identity=identity)
    assert payload['observations'].shape == (2, 3844)


@pytest.mark.parametrize('damage', ['observation', 'action', 'mask', 'order', 'dtype'])
def test_direct_actor_bc_uses_bound_admission_before_actor_update(assembled, damage):
    _, payload, _ = assembled
    damage_rows(payload, damage)
    model = cli.PrivilegedTeacher()
    bc = data.ActorBCTrainer(model)
    before = deepcopy(model.state_dict())
    with pytest.raises(ValueError, match='source|issued|mask|dtype'):
        bc.step(payload['observations'], payload['actions'], payload['valid_mask'], dataset=payload)
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert not bc.optimizer.state and bc.steps == 0


def test_direct_actor_bc_revalidates_each_independent_step(assembled, monkeypatch):
    _, payload, _ = assembled
    bc = data.ActorBCTrainer(cli.PrivilegedTeacher())
    replay = warm.verify_supervised_rows
    calls = []
    def counted(value):
        calls.append(1)
        return replay(value)
    monkeypatch.setattr(warm, 'verify_supervised_rows', counted)
    result = bc.step(payload['observations'], payload['actions'], payload['valid_mask'], dataset=payload)
    assert result['steps'] == 1 and len(calls) == 1
    damage_rows(payload, 'mask')
    with pytest.raises(ValueError, match='mask|source'):
        bc.step(payload['observations'], payload['actions'], payload['valid_mask'], dataset=payload)
    assert len(calls) == 2 and bc.steps == 1


@pytest.mark.parametrize('damage', ['observation', 'action', 'mask', 'order', 'dtype'])
def test_cli_expert_loader_rejects_changed_rows_before_model_construction(assembled, monkeypatch, tmp_path, damage):
    identity, payload, path = assembled
    damage_rows(payload, damage)
    torch.save(payload, path)
    monkeypatch.setattr(cli, 'training_identity', lambda *args: identity)
    def forbidden(*args, **kwargs):
        pytest.fail('model must not be constructed before bound expert admission')
    monkeypatch.setattr(cli, 'PrivilegedTeacher', forbidden)
    config, ppo = cli.load_config(CONFIG)
    with pytest.raises(ValueError, match='source|issued|mask|dtype'):
        cli.train_teacher(CPUFixtureBridge(), config, ppo, output=tmp_path, updates=1,
            expert_dataset=path, expert_dataset_sha256=file_sha256(path), bc_steps=1)


def test_cli_expert_training_keeps_read_expert_tuple_and_bc_admissions(monkeypatch, tmp_path):
    config, _ = cli.load_config(CONFIG)
    config['rollout_steps'] = 2
    ppo = cli.PPOConfig(epochs=1, minibatches=1)
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    _, manifest = artifacts(inputs)
    identity = cli.training_identity(ppo, CPUFixtureBridge().contract)
    CPU_only_gate(monkeypatch)
    build.assemble_experts(manifest, identity, inputs / 'assembled')
    path = inputs / 'assembled/privileged_native_expert.pt'
    replay = warm.verify_supervised_rows
    calls = []
    def counted(value):
        calls.append(1)
        return replay(value)
    monkeypatch.setattr(warm, 'verify_supervised_rows', counted)
    output = tmp_path / 'trained'; output.mkdir()
    result = cli.train_teacher(CPUFixtureBridge(), config, ppo, output=output, updates=1,
        expert_dataset=path, expert_dataset_sha256=file_sha256(path), bc_steps=2)
    assert result['updates'] == 1 and not result['native_grasp_qualification']
    assert len(calls) == 3  # one load and two independent BC steps
    assert (output / 'expert_bc_optimizer.pt').is_file()


def test_bound_validator_returns_payload_and_bindings_with_one_replay(assembled, monkeypatch):
    identity, payload, _ = assembled
    replay = warm.verify_supervised_rows
    calls = []
    def counted(value):
        calls.append(value)
        return replay(value)
    monkeypatch.setattr(warm, 'verify_supervised_rows', counted)
    admitted, bindings = data.validate_bound_expert_dataset(payload, expected_identity=identity)
    assert admitted is payload and calls == [payload]
    assert bindings[0]['rows'] == 2
    assert bindings[0]['source_sha256'] == payload['episodes'][0]['source_sha256']


@pytest.mark.parametrize('fault', ['legacy_3844', 'noncanonical_actions'])
def test_source_contract_is_reopened_instead_of_trusting_matching_payload(assembled, fault):
    identity, payload, _ = assembled
    episode = payload['episodes'][0]
    source_path = Path(episode['source_path'])
    source = torch.load(source_path, weights_only=True)
    if fault == 'legacy_3844':
        source['observations'] = torch.zeros(2, 3844)
        source['privileged_valid'] = torch.ones(2, 3844, dtype=torch.bool)
    else:
        source['action_contract'] = dict(forged_24d=True)
    torch.save(source, source_path)
    episode['source_sha256'] = file_sha256(source_path)
    receipt_path = Path(episode['proof_path'])
    receipt = json.loads(receipt_path.read_text())
    receipt['source_sha256'] = episode['source_sha256']
    receipt_path.write_text(json.dumps(receipt))
    episode['proof_sha256'] = file_sha256(receipt_path)
    qualification_path = Path(episode['qualification_evidence'])
    qualification = json.loads(qualification_path.read_text())
    completion_path = Path(qualification['completion_artifact']['path'])
    completion = json.loads(completion_path.read_text())
    completion['produced_artifacts'][0]['sha256'] = episode['source_sha256']
    completion_path.write_text(json.dumps(completion))
    qualification['completion_artifact']['sha256'] = file_sha256(completion_path)
    qualification_path.write_text(json.dumps(qualification))
    with pytest.raises(ValueError, match='source|371|canonical'):
        data.validate_expert_dataset(payload, expected_identity=identity)
    assert payload['observations'].shape == (2, 371)
