"""Explicit, auditable migration of the 4096-environment visual checkpoint.

This module does not load/save files, reset scenes, or change a PPO policy. A
migration retains each style's first rows; its absolute PPO update/timestep
counters and all model, Adam, learning-rate and RNG values remain unchanged.
"""
from copy import deepcopy
import math

import torch


VERSION = "dh116-vision-student-ppo-v2-bounded"
_LABELS = ("pinch_thumb_index", "tripod_thumb_index_middle")
_MISSING = object()


def _same(before, after):
    """Exact recursive values, including scalar types; no physical tolerance."""
    if type(before) is not type(after):
        return False
    if isinstance(before, dict):
        return before.keys() == after.keys() and all(
            _same(value, after[key]) for key, value in before.items())
    if isinstance(before, (list, tuple)):
        return len(before) == len(after) and all(
            _same(old, new) for old, new in zip(before, after))
    return before == after


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"visual resume requires a nonnegative integer {name}")
    return value


def _validate_contract(contract):
    if not isinstance(contract, dict):
        raise ValueError("visual resume requires a visual contract dictionary")
    required = dict(version=VERSION, actor_observation_size=3844,
        critic_observation_size=371, actor_uses_privileged_state=False,
        point_cloud_shape=[2, 128, 3], history_frames=4,
        point_frame="actual_palm_body_link", position_unit="m")
    for name, expected in required.items():
        if not _same(contract.get(name, _MISSING), expected):
            raise ValueError(f"visual resume incompatible contract field: {name}")
    n = _integer(contract.get("num_envs"), "contract.num_envs")
    if n == 0 or n % 2:
        raise ValueError("visual resume requires an even positive environment count")
    return n


def _validate_saved(saved):
    if not isinstance(saved, dict) or saved.get("version") != VERSION:
        raise ValueError("visual resume requires the same visual checkpoint version")
    n = _validate_contract(saved.get("contract"))
    for name in ("policy", "optimizer"):
        if not isinstance(saved.get(name), dict):
            raise ValueError(f"visual resume checkpoint is missing {name}")
    for name in ("iteration", "timesteps"):
        _integer(saved.get(name), name)
    rate = saved.get("learning_rate")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("visual resume requires a finite positive learning_rate")
    if not isinstance(saved.get("torch_rng"), torch.Tensor) or "cuda_rng" not in saved:
        raise ValueError("visual resume checkpoint is missing RNG state")
    for name in ("episodes", "successes"):
        value = saved.get(name)
        if (not isinstance(value, torch.Tensor) or value.shape != (n,)
                or value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
                or bool((value < 0).any())):
            raise ValueError(f"visual resume requires nonnegative integer {name}[{n}]")
    if bool((saved["successes"] > saved["episodes"]).any()):
        raise ValueError("visual resume successes cannot exceed episodes")
    return n


def _differences(before, after, path=""):
    if type(before) is dict and type(after) is dict:
        for key in sorted(before.keys() | after.keys()):
            name = f"{path}.{key}" if path else str(key)
            old, new = before.get(key, _MISSING), after.get(key, _MISSING)
            if old is _MISSING or new is _MISSING:
                yield dict(path=name, operation="added" if old is _MISSING else "removed",
                    **{"from": None if old is _MISSING else deepcopy(old),
                       "to": None if new is _MISSING else deepcopy(new)})
            else:
                yield from _differences(old, new, name)
    elif (type(before) is list and type(after) is list
            and len(before) == len(after)
            and all(isinstance(x, dict) for x in before + after)):
        for i, (old, new) in enumerate(zip(before, after)):
            yield from _differences(old, new, f"{path}[{i}]")
    elif not _same(before, after):
        yield dict(path=path, operation="changed", **{"from": deepcopy(before), "to": deepcopy(after)})


def _project_reference(critic, path, old_n, indices):
    parent = critic
    for key in path[:-1]:
        parent = parent.get(key) if isinstance(parent, dict) else None
    value = parent.get(path[-1]) if isinstance(parent, dict) else None
    if not isinstance(value, list) or len(value) != old_n:
        raise ValueError("visual resume requires the source per-environment reference: " + ".".join(path))
    parent[path[-1]] = [value[i] for i in indices]


def prepare_visual_resume(saved, current_contract, *, allow_downsize=False):
    """Return an independent checkpoint copy compatible with ``current_contract``.

    The default requires exact contracts. Explicit permission permits only the
    old bounded 4096 layout to fixed 2048/1024, preserving resolution or changing
    320x240 to 160x120 or 224x96. Other contract fields match exactly, including
    projected physical references. The caller remains responsible for loading
    the returned model/Adam/RNG and persisting ``resume_migration`` in new saves.
    """
    old_n = _validate_saved(saved)
    new_n = _validate_contract(current_contract)
    old = saved["contract"]
    if _same(old, current_contract):
        return deepcopy(saved)
    if not allow_downsize:
        raise ValueError("visual resume requires the same contract unless explicit downsize is enabled")
    if old_n != 4096 or new_n not in (2048, 1024):
        raise ValueError("explicit visual downsize only permits 4096 to 2048 or 1024 environments")
    if ("camera_mode" in old
            or old.get("camera_batching") != "same_frozen_physics_state_sequential_real_RGBD"
            or old.get("visibility_mode") != "all"
            or old.get("pinned_source_environment", _MISSING) is not None):
        raise ValueError("explicit visual downsize requires the original bounded all-visible camera contract")
    old_half, new_half = old_n // 2, new_n // 2
    groups = old.get("groups")
    if (not isinstance(groups, list) or len(groups) != 2
            or any(not isinstance(g, dict) or g.get("label") != label
                or not _same(g.get("num_envs"), old_half) for g, label in zip(groups, _LABELS))):
        raise ValueError("visual resume requires contiguous primary-then-secondary equal groups")
    if (not _same(old.get("camera_resolution_wh"), [320, 240])
            or not any(_same(current_contract.get("camera_resolution_wh"), resolution)
                for resolution in ([320, 240], [160, 120], [224, 96]))):
        raise ValueError("explicit visual downsize only permits retaining 320x240 or reducing to 160x120 or 224x96")
    indices = list(range(new_half)) + list(range(old_half, old_half + new_half))
    expected = deepcopy(old)
    expected["num_envs"] = new_n
    for group in expected["groups"]:
        group["num_envs"] = new_half
    critic = expected.get("privileged_critic_contract")
    if not isinstance(critic, dict) or not _same(critic.get("num_envs"), old_n):
        raise ValueError("visual resume requires a matching source critic environment count")
    critic["num_envs"] = new_n
    _project_reference(critic, ("success_bottom_m",), old_n, indices)
    _project_reference(critic, ("surface_force_sharing", "lift_reference_bottom_m"), old_n, indices)
    expected.update(camera_mode="fixed_per_env", camera_batch_size=new_n,
        camera_batching="fixed_per_environment_real_RGBD", visibility_mode="all",
        pinned_source_environment=None,
        camera_resolution_wh=deepcopy(current_contract["camera_resolution_wh"]))
    unexpected = list(_differences(expected, current_contract))
    if unexpected:
        raise ValueError("visual resume has unapproved contract changes: " + ", ".join(x["path"] for x in unexpected))
    result = deepcopy(saved)
    result["contract"] = deepcopy(current_contract)
    for name in ("episodes", "successes"):
        index = torch.tensor(indices, dtype=torch.long, device=saved[name].device)
        result[name] = saved[name].index_select(0, index)
    totals = lambda values: {name: int(values[name].sum().item()) for name in ("episodes", "successes")}
    source_totals, retained_totals = totals(saved), totals(result)
    metadata = dict(version="dh116-visual-downsize-resume-v1", source_contract=deepcopy(old),
        source_iteration=saved["iteration"], source_timesteps=saved["timesteps"],
        source_num_envs=old_n, target_num_envs=new_n,
        retained_ranges=[[0, new_half], [old_half, old_half + new_half]],
        discarded_ranges=[[new_half, old_half], [old_half + new_half, old_n]],
        source_totals=source_totals, retained_totals=retained_totals,
        discarded_totals={name: source_totals[name] - retained_totals[name] for name in source_totals},
        groups={}, changes=list(_differences(old, current_contract)))
    for i, label in enumerate(_LABELS):
        start = i * old_half
        selected = lambda lo, hi: {name: int(saved[name][lo:hi].sum().item())
            for name in ("episodes", "successes")}
        metadata["groups"][label] = dict(source_num_envs=old_half,
            retained_num_envs=new_half, discarded_num_envs=old_half-new_half,
            source=selected(start, start+old_half),
            retained=selected(start, start+new_half),
            discarded=selected(start+new_half, start+old_half))
    result["resume_migration"] = metadata
    return result
