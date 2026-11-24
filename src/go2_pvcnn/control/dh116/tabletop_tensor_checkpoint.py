"""Exclusive tensor Teacher restarts at clean PPO boundaries.

Only policy/Adam, counters and RNG are resumed. The caller must rebuild and
verify the physical Reset and all source/reward histories before collecting.
This module never stores native state or grants Teacher/grasp admission.
"""
from copy import deepcopy
from dataclasses import asdict
import math
import os
from pathlib import Path
import random
from tempfile import NamedTemporaryFile
import uuid

import numpy as np
import torch

from .tabletop_deformable_privileged_checkpoint import _rng_state, _decode_rng, _same_adam_setting
from .tabletop_deformable_privileged_ppo import verify_base_ppo_source
from .tabletop_device_ppo import DeviceDeformablePPOTrainer
from .tabletop_native_teacher_profile import ROOT, digest
from .tabletop_privileged_checkpoint import _clean, _cpu_tree, _finite_tree, _sha, _validate_adam, _validate_rng
from .tabletop_stereo_contract import file_sha256
from .tabletop_tensor_backend_profile import TensorBackendProfile
from .tabletop_tensor_native_teacher import TensorNativeTeacher

VERSION = "dh116-tensor-native-checkpoint-v2"
_KEYS = {"version", "identity", "policy", "optimizer", "policy_version", "counters", "rng"}
_SOURCES = (
    "tabletop_tensor_checkpoint.py", "tabletop_tensor_native_teacher.py", "tabletop_native_teacher_model.py",
    "tabletop_tensor_native_inputs.py", "tabletop_tensor_backend_profile.py", "tabletop_tensor_collector.py",
    "tabletop_tensor_source_owner.py", "tabletop_tensor_training_reward.py", "tabletop_training_reward.py",
    "tabletop_device_ppo.py", "tabletop_device_rollout.py", "tabletop_deformable_privileged_ppo.py",
    "tabletop_privileged_ppo.py", "tabletop_squashed_gaussian.py", "tabletop_privileged_checkpoint.py",
    "tabletop_deformable_privileged_checkpoint.py",
    "tabletop_tensor_control.py", "tabletop_tensor_geometry.py",
    "tabletop_regex_articulation.py", "tabletop_regex_task_setup.py",
    "tabletop_tensor_mesh_bridge.py", "tabletop_tensor_pair_capture.py",
    "tabletop_tensor_rod_owner.py", "tabletop_tensor_tripod_guard.py",
    "tabletop_tensor_wrist_projection.py", "tabletop_regex_cached_readback.py",
    "tabletop_runtime_execution.py", "tabletop_runtime_lifecycle.py", "tabletop_runtime_source_audit.py",
    "tabletop_tensor_physics_runtime.py",
    "tabletop_cuda_geometry.py", "cuda_measurements/__init__.py",
    "cuda_measurements/hull_support.py", "cuda_measurements/pad_support.py",
    "cuda_measurements/hull_support.cu", "cuda_measurements/pad_support.cu",
)


def identity(trainer, *, recipe_sha256, receipt_sha256, backend_profile, seed_sha256, policy_seed):
    """Bind the exact tensor namespace, execution recipe and current source bytes."""
    if type(trainer) is not DeviceDeformablePPOTrainer or type(trainer.model) is not TensorNativeTeacher:
        raise ValueError("actual tensor Teacher and device PPO trainer required")
    if (any(not _sha(value) for value in (recipe_sha256, receipt_sha256, seed_sha256))
            or type(policy_seed) is not int or not 0 <= policy_seed < 2**64):
        raise ValueError("complete recipe/receipt/seed SHA and policy seed required")
    if type(backend_profile) is not TensorBackendProfile or trainer.model._backend_profile != backend_profile:
        raise ValueError("same actual flat backend profile required")
    backend_profile.verify()
    verify_base_ppo_source()
    model = trainer.model.model_identity
    grasp = model.get("grasp")
    if (grasp not in ("pinch", "tripod", "power")
            or model.get("checkpoint_namespace") != "tensor_native_teacher_"+grasp+"_v1"
            or model.get("policy_version") != "dh116-tensor-native-context-teacher-v1"
            or model.get("model_identity_sha256") != digest({k:v for k,v in model.items() if k != "model_identity_sha256"})
            or model.get("parent_num_envs") != trainer.num_envs
            or model.get("backend_profile_sha256") != backend_profile.digest
            or model.get("backend_sha256") != backend_profile.binary_sha256
            or any(model.get(key) is not False for key in ("native_ready", "teacher_admission", "grasp_qualified"))):
        raise ValueError("tensor model namespace/profile identity differs")
    for name, sha in model["dependency_source_sha256"].items():
        if file_sha256(Path(__file__).with_name(name)) != sha:
            raise ValueError("tensor model dependency source SHA differs: "+name)
    sources = {name:file_sha256((Path(__file__).parent/name)) for name in _SOURCES}
    sources["train_dh116_tensor_native_teacher.py"] = file_sha256(ROOT/"scripts/train_dh116_tensor_native_teacher.py")
    return dict(version=VERSION, model=model, recipe_sha256=recipe_sha256, receipt_sha256=receipt_sha256,
        seed_sha256=seed_sha256, policy_seed=policy_seed, backend_profile=asdict(backend_profile),
        ppo_config=asdict(trainer.config), rollout_steps=trainer.rollout_steps, num_envs=trainer.num_envs,
        device_type=trainer.device.type, cuda_rng_devices=torch.cuda.device_count() if torch.cuda.is_initialized() else 0,
        source_sha256=sources, torch_version=str(torch.__version__), history_reset="clear",
        physical_state_saved=False, restart_requires_verified_reset=True, teacher_admission=False, grasp_qualified=False)


def _same_tree(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left, right))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same_tree(v, right[k]) for k,v in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(_same_tree(a,b) for a,b in zip(left,right))
    return type(left) is type(right) and left == right


def _validated_rng(encoded, expected_devices):
    if not isinstance(encoded, dict) or set(encoded) != {"torch", "python", "numpy", "cuda"}:
        raise ValueError("complete tensor checkpoint RNG required")
    cuda = encoded["cuda"]
    if not isinstance(cuda, list) or len(cuda) != expected_devices:
        raise ValueError("tensor checkpoint CUDA RNG device contract differs")
    rng = _decode_rng(dict(encoded, cuda=[]))
    rng["cuda"] = cuda
    _validate_rng(rng)
    # Validate complete CUDA generator bytes without touching default streams.
    for index, state in enumerate(cuda):
        try:
            torch.Generator(device="cuda:"+str(index)).set_state(state)
        except (ValueError, RuntimeError, TypeError) as error:
            raise ValueError("invalid tensor checkpoint CUDA RNG state") from error
    return rng


def _validate(packet, trainer, expected_identity):
    if (not isinstance(packet, dict) or set(packet) != _KEYS or packet["version"] != VERSION
            or packet["identity"] != expected_identity):
        raise ValueError("tensor checkpoint source/recipe/model namespace identity differs")
    _finite_tree(packet)
    counts, version = packet["counters"], packet["policy_version"]
    samples = trainer.num_envs*trainer.rollout_steps
    if (not isinstance(counts, dict) or set(counts) != {"updates", "environment_steps"}
            or any(type(v) is not int or v < 0 for v in counts.values())
            or type(version) is not int or version != counts["updates"]
            or counts["environment_steps"] % samples or counts["environment_steps"] < counts["updates"]*samples):
        raise ValueError("complete consistent tensor PPO counters required")
    saved, current = packet["policy"], trainer.model.state_dict()
    if not isinstance(saved, dict) or saved.keys() != current.keys():
        raise ValueError("complete tensor policy topology required")
    for key, value in current.items():
        if isinstance(value, torch.Tensor):
            if not isinstance(saved[key], torch.Tensor) or saved[key].shape != value.shape or saved[key].dtype != value.dtype:
                raise ValueError("tensor policy topology differs")
        elif saved[key] != value:
            raise ValueError("tensor policy model metadata differs")
    noise = saved["log_std"]
    if bool(((noise < math.log(trainer.config.min_std)) | (noise > math.log(trainer.config.max_std))).any()):
        raise ValueError("tensor policy noise outside declared bounds")
    adam = packet["optimizer"]
    if not isinstance(adam, dict) or set(adam) != {"state", "param_groups"} or not isinstance(adam["state"], dict):
        raise ValueError("complete tensor Adam checkpoint required")
    _validate_adam(adam, trainer.model, trainer.config, counts["updates"])
    for entry in adam["state"].values():
        if (set(entry) != {"step", "exp_avg", "exp_avg_sq"} or entry["step"].ndim != 0
                or entry["step"].item() % 1 or entry["step"].item() < counts["updates"]
                or bool((entry["exp_avg_sq"] < 0).any())):
            raise ValueError("complete consistent tensor Adam moments/steps required")
    rng = _validated_rng(packet["rng"], expected_identity["cuda_rng_devices"])
    # Loading on disposable copies validates PyTorch's full policy/Adam paths
    # before any live model, optimizer, session or default RNG is touched.
    model = deepcopy(trainer.model)
    model.load_state_dict(saved, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=trainer.config.lr)
    stored_group, = adam["param_groups"]
    expected_group, = optimizer.param_groups
    if stored_group.keys() != expected_group.keys() or any(not _same_adam_setting(stored_group[k], v)
            for k,v in expected_group.items() if k != "params"):
        raise ValueError("tensor Adam parameter-group settings differ")
    optimizer.load_state_dict(adam)
    return rng


def _report(path, packet):
    return dict(path=str(path), sha256=file_sha256(path), version=VERSION, identity=deepcopy(packet["identity"]),
        counters=deepcopy(packet["counters"]), policy_version=packet["policy_version"],
        physical_state_saved=False, restart_requires_verified_reset=True, teacher_admission=False, grasp_qualified=False)


def save_checkpoint(path, trainer, **context):
    """Publish a verified exclusive snapshot without changing live state or RNG."""
    _clean(trainer, "clear")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    expected = identity(trainer, **context)
    if (trainer.rollout.session_id != trainer.session_id or trainer.rollout.policy_version != trainer.policy_version
            or trainer.rollout._capability is not trainer._capability):
        raise ValueError("active tensor rollout session/policy identity differs")
    rng = _rng_state()
    rng["cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    packet = dict(version=VERSION, identity=expected, policy=_cpu_tree(trainer.model.state_dict()),
        optimizer=_cpu_tree(trainer.optimizer.state_dict()), policy_version=trainer.policy_version,
        counters=deepcopy(trainer.counters), rng=rng)
    _validate(packet, trainer, expected)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(packet, stream)
            stream.flush()
            os.fsync(stream.fileno())
            serialized = torch.load(temporary, map_location="cpu", weights_only=True)
            if not _same_tree(packet, serialized):
                raise ValueError("serialized tensor checkpoint changed policy/Adam/counters/RNG")
            _validate(serialized, trainer, expected)
            os.link(temporary, path)
        finally:
            temporary.unlink()
    return _report(path, packet)


def load_checkpoint(path, trainer, *, expected_sha256, **context):
    """Restore validated policy/Adam/counters and all RNG; require physical Reset.

    A new session, append capability and empty rollout invalidate old sample
    identities. Native/source/reward histories are intentionally not restored.
    """
    _clean(trainer, "clear")
    path = Path(path)
    if not _sha(expected_sha256) or file_sha256(path) != expected_sha256:
        raise ValueError("actual complete tensor checkpoint SHA differs")
    expected = identity(trainer, **context)
    packet = torch.load(path, map_location="cpu", weights_only=True)
    rng = _validate(packet, trainer, expected)
    trainer.model.load_state_dict(packet["policy"], strict=True)
    trainer.optimizer.load_state_dict(packet["optimizer"])
    trainer.counters = deepcopy(packet["counters"])
    trainer.policy_version = packet["policy_version"]
    trainer.session_id, trainer._capability = uuid.uuid4().hex, object()
    trainer._pending = None
    trainer._new_rollout()
    torch.set_rng_state(rng["torch"])
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return dict(_report(path, packet), session_id=trainer.session_id)
