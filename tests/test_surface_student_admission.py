"""A low BC loss or a teacher replay must not admit a long student run."""
import copy
import hashlib
import json

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_surface_imitation as bc
from go2_pvcnn.control.dh116 import tabletop_surface_admission as admission


def fixture(tmp_path):
    contract = dict(observation_size=351, action_size=24, policy_command_size=24,
        observation_version=bc.OBSERVATION_VERSION, num_envs=2, physics_hz=480)
    actor = tmp_path / 'actor.pt'
    torch.save(dict(version=bc.ACTOR_VERSION, actor=bc.make_actor().state_dict(), contract=contract), actor)
    sha = hashlib.sha256(actor.read_bytes()).hexdigest()
    report = dict(status='surface_student_evaluation_complete', execution_mode='deterministic_student',
        actor_source_file_sha256=sha, contract=contract, actor_unchanged=True,
        contract_sha256=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        policy_weights_before_sha256='a'*64,policy_weights_after_sha256='a'*64,policy_weights_hash_equal=True,
        optimization_performed=False, mpc_enabled=False,
        success_source='ordinary_native_task;not_model_prediction', successes=[1, 1], episodes=[1, 1])
    return actor, report, contract


def test_native_student_report_is_bound_to_exact_actor_and_physics(tmp_path):
    actor, report, contract = fixture(tmp_path)
    path = tmp_path / 'native.json';path.write_text(json.dumps(report))
    evidence = bc.validate_student_admission(actor, [path], dict(contract, num_envs=4096))
    assert evidence['native_student_successes'] == 2
    assert evidence['actor_sha256'] == report['actor_source_file_sha256']


@pytest.mark.parametrize('field,value', [
    ('status', 'teacher_replay_validated'), ('execution_mode', 'teacher_replay'),
    ('actor_source_file_sha256', '0' * 64), ('actor_unchanged', False),
    ('optimization_performed', True), ('mpc_enabled', True),
    ('success_source', 'offline_mse'), ('successes', [0, 0]),
    ('policy_weights_hash_equal',False),('policy_weights_after_sha256','b'*64),
    ('contract_sha256','0'*64)])
def test_teacher_loss_stale_or_modified_model_cannot_pass_admission(tmp_path, field, value):
    actor, report, contract = fixture(tmp_path)
    report[field] = value
    path = tmp_path / 'invalid.json';path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        bc.validate_student_admission(actor, [path], contract)


def test_changed_physics_or_no_native_reports_is_rejected(tmp_path):
    actor, report, contract = fixture(tmp_path)
    with pytest.raises(ValueError):
        bc.validate_student_admission(actor, [], contract)
    report['contract']['physics_hz'] = 240
    path = tmp_path / 'wrong_physics.json';path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='contract'):
        bc.validate_student_admission(actor, [path], dict(contract, physics_hz=480))


def contract_id(contract):
    return hashlib.sha256(bc._canonical_contract(contract).encode('utf-8')).hexdigest()


def family_fixture(tmp_path):
    actor, report, contract = fixture(tmp_path)
    first = dict(contract, close_seed_sha256='a' * 64,
                 grasp_initialization_sha256='b' * 64)
    second = dict(first, close_seed_sha256='c' * 64,
                  grasp_initialization_sha256='d' * 64)
    family = [dict(label=label, contract=value, proof_path=f'{label}.json',
                   source_dataset_sha256=letter * 64)
              for label, value, letter in [('pinch', first, 'e'), ('tripod', second, 'f')]]
    # A valid single-style actor is intentionally paired with a larger training family.
    torch.save(dict(version=bc.ACTOR_VERSION, actor=bc.make_actor().state_dict(),
                    contract=first), actor)
    identity = hashlib.sha256(actor.read_bytes()).hexdigest()
    paths = []
    for index, value in enumerate((first, second)):
        row = copy.deepcopy(report)
        row.update(actor_source_file_sha256=identity, contract=value,
                   contract_sha256=hashlib.sha256(json.dumps(value, sort_keys=True,
                       separators=(',', ':')).encode()).hexdigest())
        path = tmp_path / f'native_{index}.json'
        path.write_text(json.dumps(row))
        paths.append(path)
    return actor, paths, first, second, family


def test_actual_training_family_requires_both_styles_of_single_style_actor(tmp_path):
    actor, paths, first, _, family = family_fixture(tmp_path)
    with pytest.raises(ValueError, match='every|cover'):
        admission.validate_student_admission(actor, paths[:1], first, training_family=family)


def test_training_family_proof_records_exact_actor_and_canonical_coverage(tmp_path):
    actor, paths, first, second, family = family_fixture(tmp_path)
    proof = admission.validate_student_admission(actor, paths, first, training_family=family)
    assert proof['actor_sha256'] == hashlib.sha256(actor.read_bytes()).hexdigest()
    assert set(proof['covered_contract_ids']) == {contract_id(first), contract_id(second)}
    assert {row['contract_id'] for row in proof['reports']} == set(proof['covered_contract_ids'])
    assert proof['native_student_successes'] == 4
    before = copy.deepcopy(proof)
    assert admission.validate_saved_admission(proof, dict(second, num_envs=4096),
                                             training_family=family) == before
    assert proof == before


def test_actor_family_is_still_required_when_training_family_is_not_supplied(tmp_path):
    actor, paths, first, _, family = family_fixture(tmp_path)
    payload = torch.load(actor, weights_only=True)
    payload['reset_family'] = family
    torch.save(payload, actor)
    identity = hashlib.sha256(actor.read_bytes()).hexdigest()
    for path in paths:
        row = json.loads(path.read_text())
        row['actor_source_file_sha256'] = identity
        path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='every|cover'):
        admission.validate_student_admission(actor, paths[:1], first)
    assert len(admission.validate_student_admission(actor, paths, first)['covered_contract_ids']) == 2


def saved_fixture(tmp_path):
    actor, report, contract = fixture(tmp_path)
    path = tmp_path / 'native.json'
    path.write_text(json.dumps(report))
    return admission.validate_student_admission(actor, [path], contract), contract


@pytest.mark.parametrize('field,value', [
    ('version', 'teacher_replay_validated'), ('actor_sha256', 'missing'),
    ('native_student_successes', 0), ('native_student_successes', True),
    ('native_student_successes', 99), ('reports', []), ('reports', True),
    ('covered_contract_ids', []), ('covered_contract_ids', ['f' * 64]),
    ('all_declared_reset_styles_validated', False),
])
def test_saved_proof_rejects_invalid_structure_and_false_coverage(tmp_path, field, value):
    proof, contract = saved_fixture(tmp_path)
    proof[field] = value
    with pytest.raises(ValueError):
        admission.validate_saved_admission(proof, contract)


@pytest.mark.parametrize('proof', [None, True, {}, {'all_declared_reset_styles_validated': True}])
def test_saved_proof_never_accepts_a_truthy_admission_marker(proof):
    contract = dict(observation_size=351, action_size=24,
                    observation_version=bc.OBSERVATION_VERSION)
    with pytest.raises(ValueError):
        admission.validate_saved_admission(proof, contract)


@pytest.mark.parametrize('field,value', [
    ('path', ''), ('sha256', 'bad'), ('successes', [0, 0]),
    ('successes', [True, 1]), ('successes', [2, 1]), ('episodes', [1]),
    ('episodes', [-1, 1]), ('contract_id', 'f' * 64),
])
def test_saved_proof_checks_report_evidence_not_only_top_level_marker(tmp_path, field, value):
    proof, contract = saved_fixture(tmp_path)
    proof['reports'][0][field] = value
    with pytest.raises(ValueError):
        admission.validate_saved_admission(proof, contract)


def test_saved_proof_cannot_claim_an_unreported_second_style(tmp_path):
    actor, paths, first, second, family = family_fixture(tmp_path)
    proof = admission.validate_student_admission(actor, paths[:1], first)
    with pytest.raises(ValueError):
        admission.validate_saved_admission(proof, first, training_family=family)
    proof['covered_contract_ids'].append(contract_id(second))
    with pytest.raises(ValueError):
        admission.validate_saved_admission(proof, first, training_family=family)


def test_saved_proof_rejects_other_physics_and_current_contract_outside_family(tmp_path):
    actor, paths, first, second, family = family_fixture(tmp_path)
    proof = admission.validate_student_admission(actor, paths, first, training_family=family)
    with pytest.raises(ValueError):
        admission.validate_saved_admission(proof, dict(first, physics_hz=240))
    with pytest.raises(ValueError):
        admission.validate_saved_admission(proof, dict(second, close_seed_sha256='0' * 64),
                                          training_family=family)


def test_explicit_empty_training_family_cannot_disable_family_validation(tmp_path):
    actor, paths, first, _, _ = family_fixture(tmp_path)
    with pytest.raises(ValueError):
        admission.validate_student_admission(actor, paths, first, training_family=[])
