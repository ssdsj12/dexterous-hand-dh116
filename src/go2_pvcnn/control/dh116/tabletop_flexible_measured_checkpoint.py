"""Independent checkpoint format for the measured-common CPU Teacher."""
import copy
from dataclasses import asdict
import math
import os
from pathlib import Path
import pickle
import random
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from .tabletop_deformable_privileged_checkpoint import _decode_rng, _rng_state, _same_adam_setting
from .tabletop_privileged_checkpoint import _clean, _cpu_tree, _finite_tree, _validate_adam
from .tabletop_deformable_privileged_ppo import (
    BASE_PPO_SHA256, VERSION as PPO_VERSION, DeformablePPOTrainer, verify_base_ppo_source,
)
from .tabletop_flexible_measured_common_teacher import MeasuredCommonTeacher, VERSION as MODEL_VERSION
from .tabletop_stereo_contract import digest, file_sha256


VERSION = "dh116-flexible-measured-common-checkpoint-cpu-v1"


def checkpoint_identity(trainer):
    if (not isinstance(trainer, DeformablePPOTrainer)
            or type(trainer.model) is not MeasuredCommonTeacher):
        raise ValueError("independent measured-common CPU Teacher trainer required")
    verify_base_ppo_source()
    model = trainer.model.model_identity
    identity_sha = digest({k:v for k,v in model.items() if k != "model_identity_sha256"})
    if (model["policy_version"] != MODEL_VERSION or model["model_identity_sha256"] != identity_sha
            or model["native_ready"] is not False or model["training_reward_qualified"] is not False):
        raise ValueError("exact measured-common model identity required")
    dependencies = ("tabletop_deformable_privileged_checkpoint.py",
                    "tabletop_privileged_checkpoint.py", "tabletop_deformable_privileged_ppo.py")
    parent = Path(__file__).parent
    return dict(model=model, ppo_version=PPO_VERSION, ppo_config=asdict(trainer.config),
        base_ppo_source_sha256=BASE_PPO_SHA256, checkpoint_source_sha256=file_sha256(__file__),
        dependencies_sha256={name:file_sha256(parent/name) for name in dependencies},
        native_ready=False, training_reward_qualified=False)


def save_checkpoint(path, trainer):
    """Write an exclusive checkpoint at a clean measured-policy boundary."""
    identity = checkpoint_identity(trainer)
    _clean(trainer, "clear")
    payload = dict(version=VERSION, identity=identity, policy=_cpu_tree(trainer.model.state_dict()),
                   optimizer=_cpu_tree(trainer.optimizer.state_dict()), policy_version=trainer.policy_version,
                   counters=copy.deepcopy(trainer.counters), rollout_steps=trainer.rollout_steps,
                   num_envs=trainer.num_envs, history_reset="clear", rng=_rng_state())
    _finite_tree(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink()
    return dict(path=str(path), version=VERSION, sha256=file_sha256(path), identity=identity)


def load_checkpoint(path, trainer):
    """Validate the complete payload on disposable state before restoration."""
    identity = checkpoint_identity(trainer)
    _clean(trainer, "clear")
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError("unsupported structured flexible checkpoint version or format") from error
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("unsupported structured flexible checkpoint version")
    if payload.get("identity") != identity:
        raise ValueError("structured flexible checkpoint identity differs")
    if (payload.get("history_reset"), payload.get("rollout_steps"), payload.get("num_envs")) != (
            "clear", trainer.rollout_steps, trainer.num_envs):
        raise ValueError("structured checkpoint rollout/history contract differs")
    counters = payload.get("counters")
    policy_version = payload.get("policy_version")
    if (not isinstance(counters, dict) or set(counters) != {"environment_steps", "updates"}
            or any(type(value) is not int or value < 0 for value in counters.values())
            or type(policy_version) is not int or policy_version != counters["updates"]):
        raise ValueError("complete nonnegative structured checkpoint counters required")
    rollout_samples = trainer.num_envs * trainer.rollout_steps
    collected = counters["environment_steps"]
    if collected % rollout_samples or collected < counters["updates"] * rollout_samples:
        raise ValueError("structured checkpoint environment steps must cover complete rollouts")
    _finite_tree(payload)
    current = trainer.model.state_dict()
    saved = payload.get("policy")
    if (not isinstance(saved, dict) or saved.keys() != current.keys()
            or any(not isinstance(saved[key], torch.Tensor) or saved[key].shape != value.shape
                   or saved[key].dtype != value.dtype for key, value in current.items())):
        raise ValueError("structured checkpoint policy topology differs")
    noise = saved["log_std"]
    if bool(((noise < math.log(trainer.config.min_std)) | (noise > math.log(trainer.config.max_std))).any()):
        raise ValueError("structured checkpoint noise outside policy bounds")
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict):
        raise ValueError("complete structured Adam checkpoint required")
    _validate_adam(optimizer, trainer.model, trainer.config, counters["updates"])
    rng = _decode_rng(payload.get("rng"))
    candidate = copy.deepcopy(trainer.model)
    candidate.load_state_dict(saved, strict=True)
    candidate_optimizer = torch.optim.Adam(candidate.parameters(), lr=trainer.config.lr)
    saved_group, = optimizer["param_groups"]
    expected_group, = candidate_optimizer.param_groups
    if saved_group.keys() != expected_group.keys() or any(
            not _same_adam_setting(saved_group[key], value)
            for key, value in expected_group.items() if key != "params"):
        raise ValueError("structured checkpoint Adam parameter-group settings differ")
    candidate_optimizer.load_state_dict(optimizer)
    trainer.model.load_state_dict(saved, strict=True)
    trainer.optimizer.load_state_dict(optimizer)
    trainer.policy_version = policy_version
    trainer.counters = copy.deepcopy(counters)
    trainer._new_rollout()
    torch.set_rng_state(rng["torch"])
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    return dict(version=VERSION, identity=identity, policy_version=policy_version, counters=copy.deepcopy(counters))
