"""Exclusive checkpoints for the CPU-only structured flexible Teacher."""

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

from .tabletop_deformable_privileged_ppo import (
    BASE_PPO_SHA256, VERSION as PPO_VERSION, DeformablePPOTrainer, verify_base_ppo_source,
)
from .tabletop_deformable_privileged_teacher import verify_deformable_teacher_identity
from .tabletop_privileged_checkpoint import (
    _clean, _cpu_tree, _finite_tree, _validate_adam, _validate_rng,
)
from .tabletop_stereo_contract import file_sha256


VERSION = "dh116-deformable-structured-critic-checkpoint-v1"


def checkpoint_identity(trainer):
    if not isinstance(trainer, DeformablePPOTrainer):
        raise ValueError("structured flexible PPO trainer required")
    verify_base_ppo_source()
    model = trainer.model.model_identity
    verify_deformable_teacher_identity(model, model)
    return dict(model=model, ppo_version=PPO_VERSION, ppo_config=asdict(trainer.config),
                base_ppo_source_sha256=BASE_PPO_SHA256,
                base_checkpoint_validation_source_sha256=file_sha256(
                    Path(__file__).with_name("tabletop_privileged_checkpoint.py")),
                structured_ppo_source_sha256=file_sha256(Path(__file__).with_name("tabletop_deformable_privileged_ppo.py")),
                checkpoint_source_sha256=file_sha256(Path(__file__)), native_ready=False)


def _rng_state():
    algorithm, keys, position, has_gaussian, cached = np.random.get_state()
    return dict(torch=torch.get_rng_state(), python=random.getstate(),
                numpy=(algorithm, torch.as_tensor(keys.astype(np.int64)), position, has_gaussian, cached), cuda=[])


def _decode_rng(value):
    if not isinstance(value, dict) or set(value) != {"torch", "python", "numpy", "cuda"} or value["cuda"] != []:
        raise ValueError("complete CPU RNG checkpoint required")
    numpy = value["numpy"]
    if not isinstance(numpy, (tuple, list)) or len(numpy) != 5:
        raise ValueError("complete NumPy RNG state required")
    algorithm, keys, position, has_gaussian, cached = numpy
    if (algorithm != "MT19937" or type(position) is not int or not 0 <= position <= 624
            or type(has_gaussian) is not int or has_gaussian not in (0, 1)
            or type(cached) not in (int, float) or not math.isfinite(cached)
            or not isinstance(keys, torch.Tensor) or keys.dtype != torch.int64
            or keys.shape != (624,) or bool(((keys < 0) | (keys > 2**32 - 1)).any())):
        raise ValueError("invalid NumPy RNG key array")
    rng = dict(value, numpy=(algorithm, keys.numpy().astype(np.uint32), position, has_gaussian, cached))
    _validate_rng(rng)
    return rng


def _same_adam_setting(value, expected):
    if type(value) is not type(expected):
        return False
    if isinstance(expected, tuple):
        return len(value) == len(expected) and all(
            _same_adam_setting(left, right) for left, right in zip(value, expected)
        )
    return value == expected


def save_checkpoint(path, trainer):
    """Save at a clean update boundary. Existing paths are never replaced."""
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
    """Verify flexible identity, topology and complete optimizer/RNG before restore."""
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
