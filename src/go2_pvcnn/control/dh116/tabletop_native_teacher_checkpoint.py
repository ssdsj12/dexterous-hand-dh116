"""Independent CPU train-state format; no physical solver snapshot or migration.

Reuses measured/deformable checkpoint validation helpers and its exclusive
fsync/link storage algorithm. Historical writers and their globals are untouched.
"""
import copy
from dataclasses import asdict
import hashlib
import io
import math
import os
from pathlib import Path
import pickle
import random
from tempfile import NamedTemporaryFile
import numpy as np
import torch
from .tabletop_deformable_privileged_ppo import (DeformablePPOTrainer, BASE_PPO_SHA256,
    VERSION as PPO_VERSION, verify_base_ppo_source)
from .tabletop_deformable_privileged_checkpoint import _rng_state, _decode_rng, _same_adam_setting
from .tabletop_privileged_checkpoint import _clean, _cpu_tree, _finite_tree, _validate_adam, _sha
from .tabletop_native_teacher_model import NativeContextTeacher, verify_native_teacher_identity
from .tabletop_stereo_contract import digest, file_sha256
from .tabletop_surface_contract import ACTION_CONTRACT

VERSION="dh116-native-context-teacher-checkpoint-cpu-v1"
CONTRACT_VERSION="dh116-native-context-training-contract-v1"


def stop_update_target(loaded_updates, *, additional_updates):
    if type(loaded_updates) is not int or loaded_updates < 0:
        raise ValueError("nonnegative loaded updates required")
    if type(additional_updates) is not int or additional_updates < 1:
        raise ValueError("positive additional updates required")
    return loaded_updates + additional_updates


def _trainer(trainer):
    if type(trainer) is not DeformablePPOTrainer or type(trainer.model) is not NativeContextTeacher:
        raise ValueError("independent native context CPU Teacher and original trainer required")
    if str(trainer.device)!="cpu":raise ValueError("CPU trainer/RNG contract required")
    verify_base_ppo_source()
    identity=trainer.model.model_identity
    verify_native_teacher_identity(identity,identity)
    return identity


def build_training_contract(trainer, *, scope, reward_sha256, target_sha256, reset_sha256,
                            source_sha256, config_sha256):
    model=_trainer(trainer)
    if scope not in ("cpu_fixture","owner_source_unqualified"):
        raise ValueError("explicit CPU fixture or unqualified owner source scope required")
    if (any(not _sha(v) for v in (reward_sha256,target_sha256,reset_sha256,config_sha256))
            or not isinstance(source_sha256,dict) or not source_sha256
            or any(not isinstance(k,str) or not k or not _sha(v) for k,v in source_sha256.items())):
        raise ValueError("complete reward/target/reset/source/config SHA contracts required")
    return dict(schema=CONTRACT_VERSION,scope=scope,reward_profile=model['context_mode'],
        reward_sha256=reward_sha256,target_sha256=target_sha256,reset_sha256=reset_sha256,
        source_sha256=copy.deepcopy(source_sha256),config_sha256=config_sha256,
        action_sha256=digest(ACTION_CONTRACT),normalization=None,
        observation_scaling="fixed_model_input_schema_"+model['input_schema_sha256'])


def checkpoint_identity(trainer,contract):
    model=_trainer(trainer)
    if not isinstance(contract,dict):raise ValueError("training contract required")
    try:
        expected=build_training_contract(trainer,**{k:contract[k] for k in (
            'scope','reward_sha256','target_sha256','reset_sha256','source_sha256','config_sha256')})
    except KeyError as error:raise ValueError("complete training contract required") from error
    if contract!=expected:raise ValueError("training reward/context/action/normalization contract differs")
    names=('tabletop_flexible_measured_checkpoint.py','tabletop_deformable_privileged_checkpoint.py',
        'tabletop_privileged_checkpoint.py','tabletop_deformable_privileged_ppo.py',
        'tabletop_native_teacher_training.py')
    return dict(model=model,training_contract=copy.deepcopy(contract),ppo_version=PPO_VERSION,
        ppo_config=asdict(trainer.config),base_ppo_source_sha256=BASE_PPO_SHA256,
        checkpoint_source_sha256=file_sha256(__file__),
        dependencies_sha256={n:file_sha256(Path(__file__).with_name(n)) for n in names},
        native_ready=False,training_reward_qualified=False,device="cpu")


def _origin(value):
    if not isinstance(value,dict):raise ValueError("explicit training origin required")
    if value==dict(kind="fresh"):return
    if (set(value)=={'kind','checkpoint_sha256'} and value['kind']=='resume'
            and _sha(value['checkpoint_sha256'])):return
    raise ValueError("unsupported origin; smoke/full migration requires independent version")


def _validate_payload(payload,trainer,identity):
    if not isinstance(payload,dict) or payload.get('version')!=VERSION:
        raise ValueError("independent native checkpoint version required; historical transfer rejected")
    fields={'version','identity','policy','optimizer','policy_version','counters','rollout_steps','num_envs',
        'history_reset','rng','normalization','accepted_minibatch_updates','origin',
        'physical_state_saved','restart_requires_verified_reset'}
    if set(payload)!=fields or payload['identity']!=identity:raise ValueError("checkpoint identity/fields differ")
    if (payload['history_reset']!='clear' or payload['rollout_steps']!=trainer.rollout_steps
            or payload['num_envs']!=trainer.num_envs or payload['normalization'] is not None
            or payload['physical_state_saved'] is not False or payload['restart_requires_verified_reset'] is not True):
        raise ValueError("checkpoint rollout/normalization/fresh verified Reset contract differs")
    counters=payload['counters'];version=payload['policy_version'];accepted=payload['accepted_minibatch_updates']
    if (not isinstance(counters,dict) or set(counters)!={'environment_steps','updates'}
            or any(type(v) is not int or v<0 for v in counters.values())
            or type(version) is not int or version!=counters['updates'] or version<1
            or type(accepted) is not int or accepted<version
            or accepted>version*trainer.config.epochs*trainer.config.minibatches):
        raise ValueError("complete nonnegative updates/accepted minibatch counters required")
    samples=trainer.num_envs*trainer.rollout_steps
    if counters['environment_steps']%samples or counters['environment_steps']<version*samples:
        raise ValueError("environment steps must cover complete original rollouts")
    _origin(payload['origin']);_finite_tree(payload)
    current=trainer.model.state_dict();saved=payload['policy']
    if not isinstance(saved,dict) or saved.keys()!=current.keys():raise ValueError("checkpoint topology differs")
    for key,value in current.items():
        if isinstance(value,torch.Tensor):
            item=saved[key]
            if not isinstance(item,torch.Tensor) or item.shape!=value.shape or item.dtype!=value.dtype:
                raise ValueError("checkpoint policy topology/dtype differs")
        elif saved[key]!=value:raise ValueError("checkpoint model extra identity differs")
    noise=saved['log_std']
    if bool(((noise<math.log(trainer.config.min_std))|(noise>math.log(trainer.config.max_std))).any()):
        raise ValueError("checkpoint noise outside policy bounds")
    optimizer=payload['optimizer']
    if not isinstance(optimizer,dict) or not isinstance(optimizer.get('state'),dict):raise ValueError("complete Adam required")
    _validate_adam(optimizer,trainer.model,trainer.config,version)
    for entry in optimizer['state'].values():
        step=entry['step']
        if step.dtype!=torch.float32 or step.ndim!=0 or step.item()!=accepted:
            raise ValueError("Adam scalar float32 steps differ from accepted minibatch total")
        for name in ('exp_avg_sq','max_exp_avg_sq'):
            if name in entry and bool((entry[name]<0).any()):
                raise ValueError("Adam second moments must be nonnegative")
    rng=_decode_rng(payload['rng'])
    candidate=copy.deepcopy(trainer.model);candidate.load_state_dict(saved,strict=True)
    candidate_optimizer=torch.optim.Adam(candidate.parameters(),lr=trainer.config.lr)
    saved_group,=optimizer['param_groups'];expected_group,=candidate_optimizer.param_groups
    if saved_group.keys()!=expected_group.keys() or any(not _same_adam_setting(saved_group[k],v)
            for k,v in expected_group.items() if k!='params'):
        raise ValueError("Adam parameter-group settings differ")
    candidate_optimizer.load_state_dict(optimizer)
    return rng


def audit_checkpoint_bytes(raw,trainer,contract,*,expected_origin=None):
    """Validate immutable CPU checkpoint bytes without restoring live state.

    Only the existing source-identity checks read files. No checkpoint path is
    accepted or opened; returned metadata owns every nested value.
    """
    if type(raw) is not bytes:raise TypeError("exact immutable checkpoint bytes required")
    if expected_origin is not None:_origin(expected_origin)
    identity=checkpoint_identity(trainer,contract)
    try:payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    except (pickle.UnpicklingError,EOFError) as error:raise ValueError("unsupported safe checkpoint format") from error
    _validate_payload(payload,trainer,identity)
    if expected_origin is not None and payload['origin']!=expected_origin:raise ValueError("checkpoint origin differs from expected origin")
    from .tabletop_deformable_r0_replay import _digest
    return copy.deepcopy(dict(version=payload['version'],identity=payload['identity'],
        counters=payload['counters'],policy_version=payload['policy_version'],
        accepted_minibatch_updates=payload['accepted_minibatch_updates'],
        physical_state_saved=payload['physical_state_saved'],
        restart_requires_verified_reset=payload['restart_requires_verified_reset'],
        checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
        policy_weights_sha256=_digest(payload['policy'])))


def save_checkpoint(path,trainer,contract):
    identity=checkpoint_identity(trainer,contract);_clean(trainer,'clear')
    if (getattr(trainer,'_native_training_failed',False)
            or getattr(trainer,'_native_completed_update',None)!=trainer.counters['updates']
            or trainer.counters['updates']<1):
        raise RuntimeError("checkpoint requires verified completed accepted optimizer update")
    payload=dict(version=VERSION,identity=identity,policy=_cpu_tree(trainer.model.state_dict()),
        optimizer=_cpu_tree(trainer.optimizer.state_dict()),policy_version=trainer.policy_version,
        counters=copy.deepcopy(trainer.counters),rollout_steps=trainer.rollout_steps,num_envs=trainer.num_envs,
        history_reset='clear',rng=_rng_state(),normalization=None,
        accepted_minibatch_updates=trainer._native_accepted_minibatches,
        origin=copy.deepcopy(getattr(trainer,'_native_training_origin',dict(kind='fresh'))),
        physical_state_saved=False,restart_requires_verified_reset=True)
    _validate_payload(payload,trainer,identity)
    # Same exclusive fsync/link algorithm as measured-common storage.
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with NamedTemporaryFile(dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name)
        try:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno());os.link(temporary,path)
        finally:temporary.unlink()
    return dict(path=str(path),version=VERSION,sha256=file_sha256(path),identity=identity)


def load_checkpoint(path,trainer,contract):
    """Restore one immutable byte snapshot; later path changes are irrelevant.

    Digest, deserialization and validation all refer to the same bytes. Every
    path read and restoration metadata preparation finishes before live commit.
    """
    identity=checkpoint_identity(trainer,contract);_clean(trainer,'clear')
    raw=Path(path).read_bytes()
    checkpoint_sha256=hashlib.sha256(raw).hexdigest()
    try:payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    except (pickle.UnpicklingError,EOFError) as error:raise ValueError("unsupported safe checkpoint format") from error
    rng=_validate_payload(payload,trainer,identity)
    counters=copy.deepcopy(payload['counters'])
    policy_version=payload['policy_version'];accepted=payload['accepted_minibatch_updates']
    origin=dict(kind='resume',checkpoint_sha256=checkpoint_sha256)
    result=dict(version=VERSION,identity=identity,policy_version=policy_version,
        counters=copy.deepcopy(counters),accepted_minibatch_updates=accepted,
        restart_requires_verified_reset=True)
    # Disposable model/Adam/RNG validation and all fallible path I/O are done.
    # No path is reopened after this point, even if it was deleted/replaced.
    trainer.model.load_state_dict(payload['policy'],strict=True)
    trainer.optimizer.load_state_dict(payload['optimizer'])
    trainer.policy_version=policy_version;trainer.counters=counters;trainer._new_rollout()
    trainer._native_accepted_minibatches=accepted
    trainer._native_completed_update=counters['updates'];trainer._native_training_failed=False
    trainer._native_training_origin=origin
    torch.set_rng_state(rng['torch']);random.setstate(rng['python']);np.random.set_state(rng['numpy'])
    return result


def checkpoint_trainer_factory(factory,path,contract_factory):
    """Restore during C04's trainer factory, after fresh verified Reset/live.

    The first collector freezes the loaded version N directly. No already-active
    collector is retagged, no policy epoch is fabricated and no history restored.
    The caller supplies the exact independently bound training contract.
    """
    if not callable(factory) or not callable(contract_factory):
        raise TypeError("explicit original trainer and contract factories required")
    def restored(*args,**kwargs):
        trainer=factory(*args,**kwargs)
        load_checkpoint(path,trainer,contract_factory(trainer))
        return trainer
    return restored
