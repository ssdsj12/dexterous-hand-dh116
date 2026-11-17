"""Strict checkpoint state for the fixed-cohort WireSeg diagnostic."""

import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch


FORMAT = "wireseg-fixed-cohort-checkpoint-v2"
_RNG_KEYS = {"python", "numpy", "torch_cpu", "torch_cuda"}
_REQUIRED = {
    "model", "optimizer", "adam_state_ids", "global_step", "steps", "sampler", "scheduler", "scaler",
    "dataset_manifest", "dataset_manifest_sha256", "training_config", "training_config_sha256",
}


def canonical_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_checkpoint(path, expected_sha256=None):
    if expected_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("checkpoint SHA256 must be 64 lowercase hexadecimal characters")
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        observed = digest.hexdigest()
        if expected_sha256 is not None and expected_sha256 != observed:
            raise ValueError("checkpoint SHA256 differs from the expected bytes")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("checkpoint must contain a WireSeg state dictionary")
    return state, observed


def missing_fields(state):
    rng = state.get("rng_states")
    present_rng = rng.keys() if isinstance(rng, dict) else set()
    return sorted((_REQUIRED - state.keys()) | {f"rng_states.{name}" for name in _RNG_KEYS - present_rng})


def require_v2_state(state):
    missing = missing_fields(state)
    if state.get("checkpoint_format") != FORMAT:
        raise ValueError(f"legacy or unsupported WireSeg checkpoint cannot strict resume; missing {missing}")
    if missing:
        raise ValueError(f"strict WireSeg checkpoint missing {missing}")


def audit_checkpoint(path):
    state, sha256 = read_checkpoint(path)
    missing = missing_fields(state)
    legacy = state.get("checkpoint_format") != FORMAT
    updates = state.get("steps", [])
    last = updates[-1] if updates else {}
    contract = state.get("training_config", state.get("contract", {}))
    return {
        "status": ("legacy_strict_resume_blocked" if legacy else
                   "v2_incomplete" if missing else "v2_fields_present"),
        "sha256": sha256,
        "global_step": state.get("global_step"),
        "mode": contract.get("mode"),
        "last_recorded_visit": last.get("visit"),
        "legacy_torch_cpu_rng_present": state.get("rng") is not None if legacy else None,
        "legacy_torch_cuda_rng_present": state.get("cuda_rng") is not None if legacy else None,
        "optimizer_present": "optimizer" in state,
        "dataset_manifest_sha256": state.get("dataset_manifest_sha256"),
        "training_config_sha256": state.get("training_config_sha256"),
        "missing": missing,
    }


def write_checkpoint(path, state):
    """Publish a complete file at a new path while preserving existing bytes."""
    with NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def capture_rng(device):
    algorithm, keys, index, has_gaussian, cached_gaussian = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (algorithm, keys.tolist(), index, has_gaussian, cached_gaussian),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _validated_rng(state, device):
    if state.keys() != _RNG_KEYS:
        raise ValueError("WireSeg checkpoint RNG fields differ")
    try:
        algorithm, keys, index, has_gaussian, cached_gaussian = state["numpy"]
        if (algorithm != "MT19937" or not isinstance(keys, list) or len(keys) != 624
                or any(type(key) is not int or not 0 <= key <= 2**32 - 1 for key in keys)
                or type(index) is not int or not 0 <= index <= 624
                or type(has_gaussian) is not int or has_gaussian not in (0, 1)
                or not isinstance(cached_gaussian, float) or not math.isfinite(cached_gaussian)):
            raise ValueError("invalid NumPy MT19937 state")
        numpy = (algorithm, np.asarray(keys, dtype=np.uint32), index, has_gaussian, cached_gaussian)
        random.Random(0).setstate(state["python"])
        np.random.RandomState(0).set_state(numpy)
        torch.Generator(device="cpu").set_state(state["torch_cpu"])
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("WireSeg checkpoint has invalid CPU RNG state") from error
    cuda = state["torch_cuda"]
    if device.type == "cuda":
        if not isinstance(cuda, torch.Tensor) or cuda.dtype != torch.uint8 or cuda.ndim != 1 or not cuda.numel():
            raise ValueError("WireSeg checkpoint is missing CUDA RNG state for the selected device")
    elif cuda is not None:
        raise ValueError("WireSeg checkpoint CUDA RNG disagrees with the CPU device contract")
    return numpy


def restore_rng(state, device, *, validated_numpy=None):
    numpy = _validated_rng(state, device) if validated_numpy is None else validated_numpy
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)
    torch.set_rng_state(state["torch_cpu"])
    random.setstate(state["python"])
    np.random.set_state(numpy)


def _sampler_state(visit_history, updates, cohort_size):
    visits = len(visit_history)
    return {
        "kind": "fixed_order_cyclic_v1",
        "cohort_size": cohort_size,
        "visits_completed": visits,
        "next_index": visits % cohort_size,
        "skipped_visits": visits - updates,
        "visit_history": visit_history,
    }


def _validated_history(state, cohort, supervised):
    sampler = state["sampler"]
    steps = state["steps"]
    if (not isinstance(sampler, dict) or not isinstance(steps, list)
            or type(state["global_step"]) is not int or state["global_step"] != len(steps)
            or not isinstance(sampler.get("visit_history"), list)):
        raise ValueError("WireSeg sampler or update count is incomplete")
    history = sampler["visit_history"]
    expected = _sampler_state(history, len(steps), len(cohort))
    if (sampler != expected or any(type(sampler[name]) is not int
                                  for name in ("cohort_size", "visits_completed", "next_index", "skipped_visits"))
            or expected["skipped_visits"] < 0 or len(supervised) != len(cohort)):
        raise ValueError("WireSeg sampler cursor or skip count differs from its visit history")
    update = 0
    for visit, entry in enumerate(history, 1):
        index = (visit - 1) % len(cohort)
        sample = cohort[index]
        identity = dict(visit=visit, file_name=sample["file_name"], pair_sha256=sample["pair_sha256"],
                        updated=supervised[index])
        if (not isinstance(entry, dict) or entry != identity
                or type(entry["visit"]) is not int or type(entry["updated"]) is not bool):
            raise ValueError("WireSeg sampler visit does not match the fixed cohort and supervision")
        if identity["updated"]:
            update += 1
            if update > len(steps) or not isinstance(steps[update - 1], dict):
                raise ValueError("WireSeg sampler visit lacks an optimizer update record")
            step = steps[update - 1]
            if (type(step.get("update")) is not int or step["update"] != update
                    or any(type(step.get(name)) is not type(value) or step[name] != value
                           for name, value in identity.items())):
                raise ValueError("WireSeg sampler and successful optimizer update histories differ")
    if update != len(steps):
        raise ValueError("WireSeg sampler and global step disagree")
    return list(steps), list(history)


def save_diagnostic_checkpoint(path, model, optimizer, device, contract, dataset_manifest,
                               training_config, steps, visit_history):
    adam = optimizer.state_dict()
    state = {
        "checkpoint_format": FORMAT,
        "model": model.state_dict(), "optimizer": adam, "adam_state_ids": sorted(adam["state"]),
        "rng_states": capture_rng(device), "scheduler": None, "scaler": None,
        "global_step": len(steps), "steps": steps, "counts": {"rgb_updates": len(steps)},
        "sampler": _sampler_state(visit_history, len(steps), len(dataset_manifest["training_cohort"])),
        "dataset_manifest": dataset_manifest, "dataset_manifest_sha256": canonical_sha256(dataset_manifest),
        "training_config": training_config, "training_config_sha256": canonical_sha256(training_config),
        "contract": contract, "backbone_transfer": model.transfer_state(),
        "stereo_correspondence_supervised": False, "dh116_policy_supervised": False,
    }
    write_checkpoint(path, state)


def _validate_adam_state(adam, recorded_ids, optimizer, updates):
    current_groups = optimizer.state_dict()["param_groups"]
    if (not isinstance(adam, dict) or adam.keys() != {"param_groups", "state"}
            or adam["param_groups"] != current_groups or not isinstance(adam["state"], dict)
            or bool(updates) != bool(adam["state"]) or any(type(index) is not int for index in adam["state"])
            or recorded_ids != sorted(adam["state"])):
        raise ValueError("WireSeg checkpoint lacks compatible AdamW optimizer state")
    parameters = {index: (parameter, group["amsgrad"])
                  for spec, group in zip(current_groups, optimizer.param_groups)
                  for index, parameter in zip(spec["params"], group["params"])}
    for index, slot in adam["state"].items():
        if index not in parameters or not isinstance(slot, dict):
            raise ValueError("WireSeg checkpoint AdamW parameter slots differ")
        parameter, amsgrad = parameters[index]
        moments = {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"} if amsgrad else {"exp_avg", "exp_avg_sq"}
        step = slot.get("step")
        if (slot.keys() != moments | {"step"} or not isinstance(step, torch.Tensor)
                or step.numel() != 1 or not step.is_floating_point()):
            raise ValueError("WireSeg checkpoint AdamW moments or update step are incomplete")
        count = float(step.item())
        if (not count.is_integer() or not 0 < count <= updates
                or any(not isinstance(slot[name], torch.Tensor) or slot[name].shape != parameter.shape
                       or slot[name].dtype != parameter.dtype or not torch.isfinite(slot[name]).all()
                       for name in moments)):
            raise ValueError("WireSeg checkpoint AdamW moments or update step are invalid")


def restore_diagnostic_checkpoint(state, model, optimizer, device, dataset_manifest, training_config, supervised):
    require_v2_state(state)
    for name, expected in (("dataset_manifest", dataset_manifest), ("training_config", training_config)):
        if state[name] != expected or state[f"{name}_sha256"] != canonical_sha256(expected):
            raise ValueError(f"WireSeg strict resume {name.replace('_', ' ')} differs")
    if state["scheduler"] is not None or state["scaler"] is not None:
        raise ValueError("WireSeg diagnostic requires explicit absent scheduler and scaler")
    steps, history = _validated_history(state, dataset_manifest["training_cohort"], supervised)
    weights = state["model"]
    current = model.state_dict()
    if (not isinstance(weights, dict) or weights.keys() != current.keys()
            or any(not isinstance(weights[name], torch.Tensor) or weights[name].shape != tensor.shape
                   or weights[name].dtype != tensor.dtype for name, tensor in current.items())):
        raise ValueError("WireSeg checkpoint model tensors differ from the training configuration")
    adam = state["optimizer"]
    _validate_adam_state(adam, state["adam_state_ids"], optimizer, len(steps))
    rng = state["rng_states"]
    numpy = _validated_rng(rng, device)
    model.load_state_dict(weights, strict=True)
    optimizer.load_state_dict(adam)
    restore_rng(rng, device, validated_numpy=numpy)
    return steps, history
