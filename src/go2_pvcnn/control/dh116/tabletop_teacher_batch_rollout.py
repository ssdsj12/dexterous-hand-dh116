"""Bind complete physical lane windows to the pending structured Teacher PPO sample."""

import math
from collections.abc import Mapping

import torch

from .tabletop_deformable_privileged_ppo import DeformablePPOTrainer
from .tabletop_deformable_rollout_admission import gae_boundary
from .tabletop_training_reward import PHYSICS_HZ, RewardPacket, SCHEMA_SHA256


OWNER_FIELDS = ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid")
CLOCK_FIELDS = ("physics_step", "rod_tick", "timestamp_s", "solver_time_s")
GAE_FIELDS = (*OWNER_FIELDS[1:], *CLOCK_FIELDS[:3])
CONTROL_SUBSTEPS = 8
SOLVER_SUBSTEPS = 12


def _identity(clock, fields):
    return tuple(clock[key] for key in fields)


def _check_physical_rows(rows, env_id, action):
    if len(rows) != CONTROL_SUBSTEPS:
        raise ValueError("eight physical source rows per environment required")
    fields = (*OWNER_FIELDS, *CLOCK_FIELDS)
    prior = rows[0]["action_readback"]["pre"]["identity"][0]
    start = prior
    if start["env_id"] != env_id:
        raise ValueError("physical rows belong to another environment")
    for substep, row in enumerate(rows):
        readback = row["action_readback"]
        clock = row["rod"]
        before, after = readback["pre"]["identity"], readback["post"]["identity"]
        if (len(before) != 1 or len(after) != 1
                or _identity(before[0], fields) != _identity(prior, fields)
                or _identity(after[0], fields) != _identity(clock, fields)
                or _identity(clock, OWNER_FIELDS) != _identity(start, OWNER_FIELDS)
                or row["substep"] != substep or row["physics_step"] != clock["physics_step"]
                or clock["physics_step"] != prior["physics_step"] + 1
                or clock["rod_tick"] != prior["rod_tick"] + SOLVER_SUBSTEPS
                or clock.get("clock_valid") is not True
                or row["guard_failure"] is not None or "native_teacher_source_error" in row):
            raise ValueError("uninterrupted guarded SDK and rod readbacks required")
        issued = readback["request"]["action24"]
        if not isinstance(issued, torch.Tensor) or not torch.equal(issued.cpu().double(), action):
            raise ValueError("issued command differs from the sampled policy action")
        prior = clock
    return start, prior


def _reward_binding(stamp, clock):
    return (stamp.env_id, stamp.episode_uid, stamp.reset_uid, stamp.rod_uid,
            stamp.physics_tick, stamp.solver_time_s) == (
        clock["env_id"], clock["episode_uid"], str(clock["reset_id"]), clock["rod_uid"],
        clock["physics_step"], clock["solver_time_s"])


def record_group_window(trainer, sample, report, rows, settlements, bootstrap, *, policy_uid, reward_config):
    """Write one PPO step after verifying every lane; bootstrap inputs come from the owner's critic."""
    if not isinstance(trainer, DeformablePPOTrainer):
        raise ValueError("structured deformable Teacher PPO trainer required")
    env_ids = report["env_ids"]
    terminal = report["terminal_reasons"]
    if (report["aborted"] is not False or report["physical_samples_per_env"] != CONTROL_SUBSTEPS
            or len(env_ids) != trainer.num_envs or list(env_ids) != sorted(set(env_ids))
            or any(not isinstance(item, Mapping) for item in (rows, settlements, bootstrap, terminal))
            or set(env_ids) != set(rows) or set(env_ids) != set(settlements)
            or set(terminal) - set(env_ids) or set(bootstrap) - set(env_ids)):
        raise ValueError("complete ordered non-aborted group with a settlement for every lane required")
    token = sample["collection_token"]
    actions = sample["actions"].detach().cpu().double()
    if actions.shape != (trainer.num_envs, 24):
        raise ValueError("sampled policy actions must have shape [N,24]")

    boundaries = []
    rewards = []
    values = []
    shared_clock = None
    for env_id, action in zip(env_ids, actions, strict=True):
        start, end = _check_physical_rows(rows[env_id], env_id, action[None])
        lane_clock = (start["run_uuid"], start["physics_step"], end["physics_step"])
        if shared_clock is not None and lane_clock != shared_clock:
            raise ValueError("all lanes must share the same SDK window")
        shared_clock = lane_clock
        packet = settlements[env_id]
        if (not isinstance(packet, RewardPacket) or packet.token != token
                or packet.policy_uid != policy_uid or packet.config_sha256 != reward_config.config_sha256
                or packet.schema_sha256 != SCHEMA_SHA256 or packet.profile != reward_config.profile
                or packet.gamma != trainer.config.gamma
                or len(packet.substep_terms) != CONTROL_SUBSTEPS
                or packet.duration_s != CONTROL_SUBSTEPS / PHYSICS_HZ or not math.isfinite(packet.reward)):
            raise ValueError("complete same-policy physical reward packet required")
        if (not _reward_binding(packet.pre_endpoint.stamp, start)
                or not _reward_binding(packet.endpoint.stamp, end)):
            raise ValueError("reward endpoints differ from the issued lane clock")
        if (packet.done != (env_id in terminal) or packet.done != (packet.terminated or packet.truncated)
                or packet.truncated != (terminal.get(env_id) == "time_limit")):
            raise ValueError("lane reward and physical terminal boundary differ")
        identity = {key: end[key] for key in GAE_FIELDS}
        identity["phase"] = "post_action_pre_reset"
        if packet.terminated:
            if env_id in bootstrap:
                raise ValueError("true terminal cannot bootstrap")
            value, next_identity = 0., None
        else:
            if env_id not in bootstrap:
                raise ValueError("critic value must come from the pre-reset endpoint")
            value, next_identity = bootstrap[env_id]
        boundaries.append(gae_boundary(packet.terminated, packet.truncated,
                                      endpoint_identity=identity, next_value_identity=next_identity))
        rewards.append(packet.reward)
        values.append(value)

    trainer.record_step(torch.tensor(rewards), torch.tensor([b["done"] for b in boundaries]),
                        torch.tensor(values), truncated=torch.tensor([b["truncated"] for b in boundaries]),
                        bootstrap_mask=torch.tensor([b["bootstrap_mask"] for b in boundaries]),
                        executed_actions=actions, collection_token=token)
