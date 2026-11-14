"""Independent actual-backend checkpoint identity with unchanged storage logic.

The frozen measured-common serializer/validator function code is reused with a
private globals mapping. No module globals, old writer or algorithm are patched.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import FunctionType

from . import tabletop_flexible_measured_checkpoint as original
from .tabletop_actual_backend_teacher import ActualBackendMeasuredTeacher, VERSION as MODEL_VERSION, _check_binding
from .tabletop_deformable_privileged_ppo import (
    DeformablePPOTrainer, verify_base_ppo_source, VERSION as PPO_VERSION, BASE_PPO_SHA256,
)
from .tabletop_stereo_contract import digest, file_sha256

VERSION = "dh116-actual-backend-measured-checkpoint-cpu-v1"


def checkpoint_identity(trainer):
    if (type(trainer) is not DeformablePPOTrainer
            or type(trainer.model) is not ActualBackendMeasuredTeacher):
        raise ValueError("explicit actual-backend measured Teacher required")
    verify_base_ppo_source()
    model = trainer.model.model_identity
    _check_binding(model["adapter"])
    if (trainer.model.backend_sha256 != model["backend_sha256"]
            or trainer.model._actual_binding != model["adapter"]):
        raise ValueError("runtime backend or source differs from checkpoint identity")
    if (model["policy_version"] != MODEL_VERSION
            or model["model_identity_sha256"] != digest({
                k:v for k,v in model.items() if k != "model_identity_sha256"})
            or any(model.get(key) is not False for key in (
                "native_ready", "training_reward_qualified", "reset_mesh_pose_qualified", "teacher_admission"))):
        raise ValueError("actual backend checkpoint model identity differs")
    parent = Path(__file__).parent
    dependencies = ("tabletop_flexible_measured_checkpoint.py", "tabletop_actual_backend_teacher.py",
                    "tabletop_deformable_privileged_checkpoint.py", "tabletop_privileged_checkpoint.py",
                    "tabletop_deformable_privileged_ppo.py")
    return dict(model=deepcopy(model), ppo_version=PPO_VERSION, ppo_config=asdict(trainer.config),
        base_ppo_source_sha256=BASE_PPO_SHA256, checkpoint_source_sha256=file_sha256(__file__),
        dependencies_sha256={name:file_sha256(parent/name) for name in dependencies},
        native_ready=False, training_reward_qualified=False, reset_mesh_pose_qualified=False)


def _delegate(name):
    function = getattr(original, name)
    namespace = dict(function.__globals__, VERSION=VERSION, checkpoint_identity=checkpoint_identity)
    delegated = FunctionType(function.__code__, namespace, name=name,
                             argdefs=function.__defaults__, closure=function.__closure__)
    delegated.__kwdefaults__ = deepcopy(function.__kwdefaults__)
    delegated.__annotations__ = dict(function.__annotations__)
    return delegated


def save_checkpoint(path, trainer):
    return _delegate("save_checkpoint")(path, trainer)


def load_checkpoint(path, trainer):
    return _delegate("load_checkpoint")(path, trainer)
