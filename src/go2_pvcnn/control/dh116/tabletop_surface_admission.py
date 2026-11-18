"""Bind long student training to native success by the exact initialized actor."""
import hashlib
import io
import json
from pathlib import Path

import torch


VERSION = 'native_surface_student_admission_v1'


def _contract_id(contract):
    from .tabletop_surface_imitation import _canonical_contract
    return hashlib.sha256(_canonical_contract(contract).encode('utf-8')).hexdigest()


def _required_contract_ids(current_contract, family):
    from .tabletop_surface_imitation import validate_reset_family
    required = {_contract_id(current_contract)}
    if family is not None:
        validate_reset_family(family, current_contract, current_contract)
        required.update(_contract_id(member['contract']) for member in family)
    return required


def _successful_episode_count(counts, episodes):
    if (not isinstance(counts, list) or not counts or not isinstance(episodes, list)
            or len(counts) != len(episodes)
            or any(type(x) is not int or x < 0 for x in counts + episodes)
            or any(x > y for x, y in zip(counts, episodes)) or sum(counts) < 1):
        raise ValueError('actual completed native successful episodes required')
    return sum(counts)


def validate_saved_admission(proof, current_contract, training_family=None):
    """Validate portable checkpoint evidence without changing it or reading files.

    Contract IDs bind coverage to successful report entries. Paths and file
    hashes are provenance references; native report qualification happens in
    validate_student_admission before this summary is created.
    """
    from .tabletop_surface_imitation import _sha256_reference
    if (not isinstance(proof, dict) or proof.get('version') != VERSION
            or not _sha256_reference(proof.get('actor_sha256'))
            or not isinstance(proof.get('actor_path'), str) or not proof['actor_path'].strip()
            or type(proof.get('native_student_successes')) is not int
            or proof['native_student_successes'] < 1
            or proof.get('all_declared_reset_styles_validated') is not True):
        raise ValueError('complete native student admission proof required')
    covered = proof.get('covered_contract_ids')
    if (not isinstance(covered, list) or not covered
            or any(not _sha256_reference(value) for value in covered)):
        raise ValueError('native admission requires covered canonical contract IDs')
    covered_ids = {value.lower() for value in covered}
    if len(covered_ids) != len(covered):
        raise ValueError('native admission contract IDs must be unique')
    reports = proof.get('reports')
    if not isinstance(reports, list) or not reports:
        raise ValueError('native admission requires successful report evidence')
    reported_ids, total = set(), 0
    for row in reports:
        if (not isinstance(row, dict)
                or not isinstance(row.get('path'), str) or not row['path'].strip()
                or not _sha256_reference(row.get('sha256'))
                or not _sha256_reference(row.get('contract_id'))):
            raise ValueError('native admission report path, SHA256 and contract ID required')
        total += _successful_episode_count(row.get('successes'), row.get('episodes'))
        reported_ids.add(row['contract_id'].lower())
    if total != proof['native_student_successes']:
        raise ValueError('native admission success count disagrees with reports')
    if reported_ids != covered_ids:
        raise ValueError('native admission coverage must match successful report contracts')
    if not _required_contract_ids(current_contract, training_family) <= covered_ids:
        raise ValueError('native student success is required for every actual training contract')
    return proof


def validate_student_admission(actor_path, reports, current_contract, training_family=None):
    """Require exact-actor native success across the actual training reset family."""
    from .tabletop_surface_imitation import (ACTOR_VERSION, _check_contract,
        _sha256_reference, validate_reset_family)
    source = Path(actor_path).resolve()
    raw = source.read_bytes()
    identity = hashlib.sha256(raw).hexdigest()
    actor = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    if not isinstance(actor, dict) or actor.get('version') != ACTOR_VERSION:
        raise ValueError('current surface actor required for student admission')
    if not isinstance(reports, (list, tuple)) or not reports:
        raise ValueError('native deterministic student success reports are required')
    family = training_family if training_family is not None else actor.get('reset_family')

    def check(contract):
        if family is not None:
            validate_reset_family(family, actor['contract'], contract)
        else:
            _check_contract(actor['contract'], contract)

    check(current_contract)
    required = _required_contract_ids(current_contract, family)
    covered, evidence, total = set(), [], 0
    for filename in reports:
        path = Path(filename).resolve()
        content = path.read_bytes()
        row = json.loads(content)
        if not isinstance(row, dict):
            raise ValueError('native student report must be a JSON object')
        if (row.get('status') != 'surface_student_evaluation_complete'
                or row.get('execution_mode') != 'deterministic_student'
                or row.get('actor_source_file_sha256') != identity
                or row.get('actor_unchanged') is not True
                or row.get('optimization_performed') is not False
                or row.get('mpc_enabled') is not False
                or row.get('policy_weights_hash_equal') is not True
                or not _sha256_reference(row.get('policy_weights_before_sha256'))
                or row.get('policy_weights_before_sha256') != row.get('policy_weights_after_sha256')
                or row.get('success_source') != 'ordinary_native_task;not_model_prediction'):
            raise ValueError('native student report does not certify this unchanged pure student')
        contract_hash=hashlib.sha256(json.dumps(row.get('contract'),sort_keys=True,
            separators=(',',':')).encode()).hexdigest()
        if row.get('contract_sha256') != contract_hash:
            raise ValueError('native student report contract hash mismatch')
        check(row.get('contract'))
        counts, episodes = row.get('successes'), row.get('episodes')
        total += _successful_episode_count(counts, episodes)
        contract_id = _contract_id(row['contract'])
        covered.add(contract_id)
        evidence.append(dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(),
                             successes=counts, episodes=episodes, contract_id=contract_id))
    if not required <= covered:
        raise ValueError('native student success is required for every declared reset family member')
    proof = dict(version=VERSION, actor_path=str(source),
                actor_sha256=identity, native_student_successes=total, reports=evidence,
                covered_contract_ids=sorted(covered),
                all_declared_reset_styles_validated=True)
    return validate_saved_admission(proof, current_contract, training_family=family)
