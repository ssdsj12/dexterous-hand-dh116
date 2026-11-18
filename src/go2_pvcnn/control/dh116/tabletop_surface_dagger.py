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
    if observations.ndim!=3 or observations.shape[-1]!=371 or failures.shape!=observations.shape[:2]:
        raise ValueError('Student evaluation tensors must be [time,env,...] aligned')
    if not observations.is_floating_point() or not torch.isfinite(observations).all():
        raise ValueError('finite real 371D Student observations required')
    if failures.dtype not in (torch.int64,torch.int32) or (failures<0).any():
        raise ValueError('integer native failure codes required')
    return failures.ne(0)


def build_failure_relabels(evaluation, teacher_actions, contract, *, source_sha256,
                           output=None):
    """Build a BC auxiliary pool from Student failure states only.

    ``teacher_actions`` are labels evaluated at the exact observations selected
    by :func:`failure_rows`; they are never executed by this function.
    """
    validate_contract(contract)
    if (not isinstance(source_sha256,str) or len(source_sha256)!=64
            or any(c not in '0123456789abcdef' for c in source_sha256)):
        raise ValueError('actual evaluation file SHA256 required')
    # A bare action matrix cannot establish state alignment or native safety.
    proof=teacher_actions
    if not isinstance(proof,dict) or proof.get('version')!='dh116-native-failure-correction-candidates-v1':
        raise ValueError('native state-bound correction artifact required; bare labels rejected')
    if proof.get('source_evaluation_sha256')!=source_sha256 or proof.get('contract_sha256')!=contract_sha(contract):
        raise ValueError('correction source or contract SHA mismatch')
    if proof.get('native_correction_qualified') is not True:
        raise ValueError('native correction validation required before BC admission')
    observations=evaluation['observations']
    failures=evaluation['failure']
    mask=failure_rows(evaluation)
    failure_rows(evaluation)
    rows=proof.get('rows',[])
    if not rows:raise ValueError('nonempty state-bound native labels required')
    indices=[]
    for row in rows:
        i,j=row['tick'],row['env']
        if not (type(i) is int and type(j) is int and 0<=i<len(observations)
                and 0<=j<observations.shape[1]):raise ValueError('invalid label tick/env index')
        if not torch.equal(row['observation'],observations[i,j]):
            raise ValueError('teacher label observation is not the exact visited Student state')
        if not bool(mask[i:min(i+30,len(mask)),j].any()):
            raise ValueError('label is outside Student failure prewindow')
        indices.append((i,j))
    if len(set(indices))!=len(indices):raise ValueError('duplicate state labels')
    selected_obs=torch.stack([observations[i,j] for i,j in indices])
    selected_failure=torch.stack([failures[i:min(i+30,len(failures)),j].max() for i,j in indices])
    labels=torch.stack([row['action'] for row in rows]).to(torch.float32)
    if labels.shape!=(selected_obs.shape[0],24):
        raise ValueError('DAgger labels must be one 24D action per visited failure state')
    validate_actions(labels)
    student_actions=evaluation.get('nominal_actions')
    if student_actions is None or student_actions.shape[:2]!=observations.shape[:2]:
        raise ValueError('Student nominal actions must be aligned with observations')
    validate_actions(student_actions.reshape(-1,24))
    student_actions=torch.stack([student_actions[i,j] for i,j in indices]).to(torch.float32)
    payload=dict(version=DATASET_VERSION,execution_owner='student',teacher_servo_used=False,
        native_correction_qualified=True,
        source='actual_student_failure_states',source_sha256=source_sha256,
        contract_sha256=contract_sha(contract),contract=contract,
        observations=selected_obs.to(torch.float32),student_actions=student_actions,
        teacher_actions=labels,failure_codes=selected_failure.to(torch.int64),
        sample_count=int(selected_obs.shape[0]))
    if output is not None:
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('xb') as stream:torch.save(payload,stream)
    return payload


def load_evaluation(path):
    value=torch.load(Path(path),map_location='cpu',weights_only=False)
    if not isinstance(value,dict):raise ValueError('Student evaluation must be a mapping')
    return value


class CorrectiveRegularizer:
    """Auxiliary-only pool; shares the retained BC Adam, never PPO storage."""
    def __init__(self,data,contract,device):
        from .tabletop_surface_imitation import _check_contract
        if (data.get('version')!=DATASET_VERSION
                or data.get('native_correction_qualified') is not True
                or data.get('execution_owner')!='student'
                or data.get('teacher_servo_used') is not False):
            raise ValueError('native validated Student-state corrections required')
        _check_contract(data['contract'],contract)
        if data.get('contract_sha256')!=contract_sha(data['contract']):
            raise ValueError('corrective contract SHA mismatch')
        x=data['observations'];y=data['teacher_actions']
        validate_actions(y)
        if x.shape!=(len(y),371) or not len(x) or not torch.isfinite(x).all():
            raise ValueError('nonempty finite real 371D corrective observations required')
        self.x=x.to(device=device,dtype=torch.float32).clone()
        self.y=y.to(device=device,dtype=torch.float32).clone()

    def update(self,algorithm,demonstrations,batch_size=32):
        from .tabletop_surface_imitation import action_fit_loss
        if not demonstrations.independent_optimizer:
            raise ValueError('corrections require the retained independent BC Adam')
        ids=torch.randint(len(self.x),(batch_size,),device=self.x.device)
        opt=demonstrations.optimizer_for(algorithm)
        opt.zero_grad(set_to_none=True)
        loss=action_fit_loss(algorithm.policy.actor(self.x[ids]),self.y[ids])
        if not torch.isfinite(loss):raise RuntimeError('nonfinite corrective BC loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(algorithm.policy.actor.parameters(),.5)
        opt.step()
        return dict(loss=float(loss.detach()),samples=batch_size,source='native_validated_student_failure_states',ppo_storage_used=False)
