"""Contract-checked DAgger labels for states actually visited by Student.

This module only assembles labels. The native evaluator owns execution and the
teacher callback must return a safe 24D action for each recorded failure state;
no teacher controller is installed in the Student environment.
"""
import hashlib
import json
from pathlib import Path

import torch

from .tabletop_surface_contract import validate_actions, validate_contract


DATASET_VERSION = 'dh116-surface-dagger-failure-relabel-v1'


def contract_sha(contract):
    encoded=json.dumps(contract,sort_keys=True,separators=(',',':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def failure_rows(evaluation):
    observations=evaluation.get('observations')
    failures=evaluation.get('failure')
    if not isinstance(observations,torch.Tensor) or not isinstance(failures,torch.Tensor):
        raise ValueError('Student evaluation must contain observations and failure tensors')
    if observations.ndim!=3 or failures.shape[:2]!=observations.shape[:2]:
        raise ValueError('Student evaluation tensors must be [time,env,...] aligned')
    return failures.ne(0)


def build_failure_relabels(evaluation, teacher_actions, contract, *, source_sha256,
                           output=None):
    """Build a BC auxiliary pool from Student failure states only.

    ``teacher_actions`` are labels evaluated at the exact observations selected
    by :func:`failure_rows`; they are never executed by this function.
    """
    validate_contract(contract)
    observations=evaluation['observations']
    failures=evaluation['failure']
    mask=failure_rows(evaluation)
    selected_obs=observations[mask]
    selected_failure=failures[mask]
    labels=torch.as_tensor(teacher_actions,dtype=torch.float32)
    if labels.shape!=(selected_obs.shape[0],24):
        raise ValueError('DAgger labels must be one 24D action per visited failure state')
    validate_actions(labels)
    student_actions=evaluation.get('nominal_actions')
    if student_actions is None or student_actions.shape[:2]!=observations.shape[:2]:
        raise ValueError('Student nominal actions must be aligned with observations')
    student_actions=student_actions[mask].to(torch.float32)
    payload=dict(version=DATASET_VERSION,execution_owner='student',teacher_servo_used=False,
        source='actual_student_failure_states',source_sha256=source_sha256,
        contract_sha256=contract_sha(contract),contract=contract,
        observations=selected_obs.to(torch.float32),student_actions=student_actions,
        teacher_actions=labels,failure_codes=selected_failure.to(torch.int64),
        sample_count=int(selected_obs.shape[0]))
    if output is not None:
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        torch.save(payload,output)
    return payload


def load_evaluation(path):
    value=torch.load(Path(path),map_location='cpu',weights_only=False)
    if not isinstance(value,dict):raise ValueError('Student evaluation must be a mapping')
    return value
