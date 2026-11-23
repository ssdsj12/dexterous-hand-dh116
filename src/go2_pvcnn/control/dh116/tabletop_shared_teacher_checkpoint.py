"""Exclusive shared-family checkpoints; reuse original Adam/RNG validators."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
import os
import random

import numpy as np
import torch

from .tabletop_deformable_privileged_ppo import DeformablePPOTrainer, verify_base_ppo_source
from .tabletop_shared_native_teacher import SharedNativeTeacher
from .tabletop_privileged_checkpoint import _clean, _cpu_tree, _finite_tree, _validate_adam, _validate_rng, _sha
from .tabletop_deformable_privileged_checkpoint import _rng_state, _decode_rng, _same_adam_setting
from .tabletop_stereo_contract import file_sha256

VERSION="dh116-shared-native-teacher-checkpoint-v1"


def identity(trainer,recipe_sha256):
    if (type(trainer) is not DeformablePPOTrainer or type(trainer.model) is not SharedNativeTeacher
            or not _sha(recipe_sha256)):
        raise ValueError("actual shared Teacher/original trainer and full recipe SHA required")
    verify_base_ppo_source()
    return dict(version=VERSION,model=trainer.model.model_identity,recipe_sha256=recipe_sha256,
        ppo_config=asdict(trainer.config),num_envs=trainer.num_envs,rollout_steps=trainer.rollout_steps,
        device_type=trainer.device.type,source_family="shared_batched_solver_v1",
        physical_state_saved=False,restart_requires_verified_reset=True,teacher_admission=False,
        torch_version=str(torch.__version__))


def save_checkpoint(path,trainer,*,recipe_sha256):
    _clean(trainer,"clear")
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    rng=_rng_state()
    rng["cuda"]=torch.cuda.get_rng_state_all() if trainer.device.type=="cuda" else []
    packet=dict(version=VERSION,identity=identity(trainer,recipe_sha256),
        policy=_cpu_tree(trainer.model.state_dict()),optimizer=_cpu_tree(trainer.optimizer.state_dict()),
        policy_version=trainer.policy_version,counters=deepcopy(trainer.counters),rng=rng)
    _finite_tree(packet)
    path.parent.mkdir(parents=True,exist_ok=True)
    with NamedTemporaryFile(dir=path.parent,delete=False) as stream:
        temp=Path(stream.name)
        try:
            torch.save(packet,stream);stream.flush();os.fsync(stream.fileno())
            os.link(temp,path)
        finally:
            temp.unlink()
    return dict(path=str(path),sha256=file_sha256(path),identity=packet["identity"])


def load_checkpoint(path,trainer,*,recipe_sha256,expected_sha256):
    _clean(trainer,"clear")
    if not _sha(expected_sha256) or file_sha256(path)!=expected_sha256:
        raise ValueError("actual complete shared checkpoint SHA differs")
    packet=torch.load(path,map_location="cpu",weights_only=True)
    if (not isinstance(packet,dict) or set(packet)!={"version","identity","policy","optimizer","policy_version","counters","rng"}
            or packet["version"]!=VERSION or packet["identity"]!=identity(trainer,recipe_sha256)):
        raise ValueError("shared checkpoint source/recipe/model identity differs")
    _finite_tree(packet)
    current=trainer.model.state_dict();saved=packet["policy"]
    if not isinstance(saved,dict) or saved.keys()!=current.keys():
        raise ValueError("complete shared policy topology required")
    for key,value in current.items():
        if isinstance(value,torch.Tensor):
            if not isinstance(saved[key],torch.Tensor) or saved[key].shape!=value.shape or saved[key].dtype!=value.dtype:
                raise ValueError("shared policy topology differs")
        elif saved[key]!=value:
            raise ValueError("shared model metadata differs")
    counts=packet["counters"];version=packet["policy_version"]
    if (not isinstance(counts,dict) or set(counts)!={"updates","environment_steps"}
            or any(type(v) is not int or v<0 for v in counts.values())
            or type(version) is not int or version!=counts["updates"]
            or counts["environment_steps"]% (trainer.num_envs*trainer.rollout_steps)):
        raise ValueError("complete shared PPO counters required")
    noise=saved["log_std"]
    if not bool(((noise>=np.log(trainer.config.min_std))&(noise<=np.log(trainer.config.max_std))).all()):
        raise ValueError("shared policy noise outside contract")
    _validate_adam(packet["optimizer"],trainer.model,trainer.config,counts["updates"])
    rng=_decode_rng(dict(packet["rng"],cuda=[]))
    rng["cuda"]=packet["rng"]["cuda"]
    _validate_rng(rng)
    for i,state in enumerate(rng["cuda"]):
        torch.Generator(device="cuda:"+str(i)).set_state(state)
    model=deepcopy(trainer.model)
    model.load_state_dict(saved)
    optimizer=torch.optim.Adam(model.parameters(),lr=trainer.config.lr)
    stored_group,=packet["optimizer"]["param_groups"]
    expected_group,=optimizer.param_groups
    if stored_group.keys()!=expected_group.keys() or any(not _same_adam_setting(stored_group[k],v)
            for k,v in expected_group.items() if k!="params"):
        raise ValueError("shared Adam settings differ")
    optimizer.load_state_dict(packet["optimizer"])
    trainer.model.load_state_dict(saved);trainer.optimizer.load_state_dict(packet["optimizer"])
    trainer.counters=deepcopy(counts);trainer.policy_version=version;trainer._new_rollout()
    torch.set_rng_state(rng["torch"]);random.setstate(rng["python"]);np.random.set_state(rng["numpy"])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return dict(counters=deepcopy(counts),policy_version=version,restart_requires_verified_reset=True)


def verify_roundtrip(original,restored):
    """Compare actual tensor trees, preserving Adam's integer parameter keys."""
    def equal(a,b):
        if isinstance(a,torch.Tensor):
            return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(
                a.detach().cpu(),b.detach().cpu())
        if isinstance(a,dict):
            return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
        if isinstance(a,(tuple,list)):
            return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return type(a) is type(b) and a==b
    for label,a,b in (
            ("policy",original.model.state_dict(),restored.model.state_dict()),
            ("optimizer",original.optimizer.state_dict(),restored.optimizer.state_dict()),
            ("counters",original.counters,restored.counters),
            ("policy_version",original.policy_version,restored.policy_version)):
        if not equal(a,b):
            raise ValueError("shared checkpoint roundtrip differs: "+label)
