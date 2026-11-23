"""Collect the original per-episode native sources under one shared Isaac clock."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from .tabletop_native_teacher_action_history import BatchPolicySnapshot
from .tabletop_dual_hand_reset import SIDES, prepare_reset_state
from .tabletop_native_teacher_context import ContextIdentity
from .tabletop_native_teacher_profile import control_source_scope
from .tabletop_teacher_batch_inputs import pack_teacher_group
from .tabletop_teacher_batch_rollout import CONTROL_SUBSTEPS, GAE_FIELDS, SOLVER_SUBSTEPS, record_group_window
from .tabletop_teacher_physical_group import TeacherPhysicalGroup
from .tabletop_training_reward import SourceFailure, TrainingReward


def translated_seed(seed, origin):
    """Derive lane placement while preserving the authenticated authored recipe."""
    from .tabletop_dual_hand_reset import _seed

    _seed(seed)
    offset = np.asarray(origin, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all() or offset[2] != 0.:
        raise ValueError("finite horizontal lane offset required")
    return prepare_reset_state(
        hand_roots=dict(zip(SIDES, seed["hand_roots"], strict=True)),
        origin=np.asarray(seed["origin"]) + offset, root_xy=seed["root_xy"],
        initial_offset_m=seed["initial_offset_m"], joint_positions=seed["joint_positions"],
        joint_velocities=seed["joint_velocities"], source_description=seed["source_description"],
    )


@dataclass
class _LaneCost:
    last_sdk_step: int
    solver_calls: int = 0
    rod_tick: int = 0
    resets: int = 0
    reset_id: int = 0


class BatchBudget:
    """Charge one shared SDK step and each independently completed rod call."""

    def __init__(self, env_ids, *, start_sdk_step, max_sdk_step):
        ids = tuple(env_ids)
        if (ids != tuple(sorted(set(ids))) or not 2 <= len(ids) <= 64
                or type(start_sdk_step) is not int or type(max_sdk_step) is not int
                or not 0 <= start_sdk_step <= max_sdk_step):
            raise ValueError("ordered native lanes and finite SDK budget required")
        self.start_sdk_step = self.sdk_step = start_sdk_step
        self.max_sdk_step = max_sdk_step
        self.lanes = {env_id: _LaneCost(start_sdk_step) for env_id in ids}

    def require_window(self):
        if self.sdk_step + CONTROL_SUBSTEPS > self.max_sdk_step:
            raise RuntimeError("complete control window exceeds the SDK budget")
        if any(lane.last_sdk_step != self.sdk_step or not lane.resets for lane in self.lanes.values()):
            raise ValueError("all lanes must be synchronized on the shared SDK clock")

    def advanced_sdk(self, sdk_step):
        if sdk_step != self.sdk_step + 1 or sdk_step > self.max_sdk_step:
            raise ValueError("exactly one measured shared SDK advance required")
        self.sdk_step = sdk_step

    def advanced_rod(self, clock):
        lane = self.lanes[clock["env_id"]]
        if (clock["physics_step"] != self.sdk_step or lane.last_sdk_step + 1 != self.sdk_step
                or clock["rod_tick"] != lane.rod_tick + SOLVER_SUBSTEPS or clock["reset_id"] != lane.reset_id):
            raise ValueError("unexpected same-lane solver progression")
        lane.last_sdk_step = self.sdk_step
        lane.solver_calls += 1
        lane.rod_tick = clock["rod_tick"]

    def reset_lane(self, env_id, clock):
        lane = self.lanes[env_id]
        if (clock["env_id"] != env_id or clock["physics_step"] != self.sdk_step
                or lane.last_sdk_step != self.sdk_step or clock["rod_tick"] != 0
                or clock["reset_id"] != lane.reset_id + 1):
            raise ValueError("lane Reset must preserve the shared SDK boundary and increment its own ID")
        lane.reset_id = clock["reset_id"]
        lane.rod_tick = 0
        lane.resets += 1

    def snapshot(self):
        return dict(sdk_step=self.sdk_step, measured_sdk_steps=self.sdk_step - self.start_sdk_step,
                    solver_calls=sum(lane.solver_calls for lane in self.lanes.values()),
                    lanes={i: dict(solver_calls=lane.solver_calls, rod_tick=lane.rod_tick,
                                   resets=lane.resets, reset_id=lane.reset_id)
                           for i, lane in self.lanes.items()})


class NativeTeacherBatch:
    """Bind actual lane owners to one trainer. The provider owns App and Reset."""

    def __init__(self, owners, trainer, *, reset_lane, journal, budget, compact=False, log_windows=True):
        owners = tuple(owners)
        self.trainer, self.reset_lane, self.journal, self.budget = trainer, reset_lane, journal, budget
        self.compact, self.log_windows = compact, log_windows
        self.owners = {owner.rod.env_id: owner for owner in owners}
        if (tuple(self.owners) != tuple(budget.lanes) or len(owners) != trainer.num_envs
                or len(self.owners) != trainer.num_envs or trainer._pending is not None
                or any(owner.sim is not owners[0].sim or owner.contract is not owners[0].contract
                       for owner in owners)):
            raise ValueError("one synchronized actual lane owner per PPO batch row required")
        self.group = TeacherPhysicalGroup(owner.episode._exchange for owner in owners)
        self.contract = owners[0].contract
        if (trainer.config != self.contract.ppo_config or trainer.rollout_steps != self.contract.config.rollout_steps
                or trainer.model.model_identity["grasp"] != self.contract.config.preflight["style"]):
            raise ValueError("grasp, PPO config, and rollout length must match the lane source contract")
        if compact and self.contract.config.preflight["style"] != "pinch":
            raise ValueError("compact measured history currently requires the Pinch source adapter")
        self.rewards = {}
        for env_id, owner in self.owners.items():
            if compact:
                owner.history.enable_compact_batch()
            self.rewards[env_id] = self._reward(owner)
            budget.reset_lane(env_id, owner.rod.clock())
        self.failed = False

    def _reward(self, owner):
        reward = TrainingReward(self.contract.reward_config)
        owner.sources.reset_reward(reward)
        return reward

    def collect(self, *, seal_evidence=None):
        if self.failed:
            raise RuntimeError("failed grouped Teacher rollout cannot continue")
        started = perf_counter()
        timings = {}
        try:
            with control_source_scope(self.contract.backend_profile) as verify_source:
                report = self._collect(verify_source, timings)
            return self._complete(report, started, timings, seal_evidence)
        except BaseException as exc:
            self.failed = True
            self.journal.emit("batch_failure", dict(type=type(exc).__name__, error=str(exc),
                                                    budget=self.budget.snapshot(), phase_seconds=timings))
            raise

    def _collect(self, verify_source, timings):
        self.budget.require_window()
        started = perf_counter()
        trainer = self.trainer
        states, packets = {}, []
        for env_id, owner in self.owners.items():
            pre = owner.pre_action_state()
            owner.history.capture_pre_action(pre)
            context = owner.context.freeze_pre_action(ContextIdentity.from_clock(pre["common"]["identity"]),
                                                       history_owner=owner.history)
            states[env_id] = pre
            packets.append((pre, owner.sources._raw, owner.sources.parameters, context))
        obs = pack_teacher_group(trainer.model, packets, backend_profile=self.contract.backend_profile)
        timings["pre_action_pack_s"] = perf_counter() - started
        started = perf_counter()
        sample = trainer.collect_step(obs, obs)
        token = sample["collection_token"]
        policy = BatchPolicySnapshot(trainer)
        policy_uid = policy.policy["model_weights_sha256"]
        for index, (env_id, owner) in enumerate(self.owners.items()):
            decision = owner.history.begin(states[env_id], trainer, lane_index=index, policy_snapshot=policy)
            lane_sample = dict(sample, actions=sample["actions"][index:index + 1])
            request = owner.sources.submitted_request(lane_sample, decision)
            self.rewards[env_id].begin(request, remaining_substeps=CONTROL_SUBSTEPS)
            owner.ledger.emit("decision", dict(collection_token=token, history=decision))

        timings["policy_and_history_s"] = perf_counter() - started
        rows = {env_id: [] for env_id in self.owners}
        lane_metrics = {}

        def observe(row):
            env_id = row["rod"]["env_id"]
            owner, reward = self.owners[env_id], self.rewards[env_id]
            rows[env_id].append(row)
            event = owner.episode.record_physical_row(row, owned=self.compact)
            if row["guard_failure"] is not None or "native_teacher_source_error" in row:
                return None
            reward.observe(owner.sources.observe(event))
            if reward._terminal:
                return "reward:" + "|".join(reward._reasons)
            if row["substep"] == CONTROL_SUBSTEPS - 1:
                if row["rod"]["timestamp_s"] - owner.sources._reset_timestamp_s >= self.contract.config.episode_cap_s:
                    return "time_limit"
            return None

        def commit(report):
            settlements, bootstrap, contexts = {}, {}, {}
            for index, (env_id, owner) in enumerate(self.owners.items()):
                reward = self.rewards[env_id]
                truncated = report["terminal_reasons"].get(env_id) == "time_limit"
                history = owner.history
                physical = history.end(dict(control_step=owner.episode._control_index,
                    physical_samples=len(rows[env_id]), termination_reason=None,
                    training_reward=None, teacher_admission=False))
                owner.episode.finish_group_window(terminal=False)
                packet = reward.settle_once(token, truncated=truncated,
                    buffer_full=not truncated and not reward._terminal and len(trainer.rollout) + 1 == trainer.rollout_steps)
                settlements[env_id] = packet
                lane_metrics[env_id] = dict(reward=packet.reward, terminated=packet.terminated,
                                            truncated=packet.truncated)
                context = owner.context.advance_physical_endpoint(owner.sources.context_state(packet),
                    history_owner=history, collection_token=token)
                contexts[env_id] = context
                issued = history.issued_for_pending(trainer, physical, lane_index=index, policy_snapshot=policy)
                if not packet.terminated:
                    pre = owner.pre_action_state()
                    identity = ContextIdentity.from_clock(pre["common"]["identity"])
                    if identity != context.metadata.identities[0]:
                        raise SourceFailure("critic bootstrap does not own the pre-Reset endpoint")
                    critic, source = owner.sources.pack_recorded(trainer.model, pre, context, purpose="bootstrap")
                    with torch.no_grad():
                        value = float(trainer.model.value(critic).item())
                    bootstrap[env_id] = (value, dict(
                        {key: getattr(identity, key) for key in GAE_FIELDS}, phase="post_action_pre_reset"))
                else:
                    source = None
                owner.ledger.emit("batch_settlement", dict(reward=packet, history=physical, issued_actions=issued,
                                                          bootstrap_source=source))
            verify_source()
            policy.verify()
            record_group_window(trainer, sample, report, rows, settlements, bootstrap,
                                policy_uid=policy_uid, reward_config=self.contract.reward_config)
            for env_id, owner in self.owners.items():
                owner.context.commit_post_context(contexts[env_id])
                if env_id in report["terminal_reasons"]:
                    owner.episode._state = "closed"

        report = self.group.control(sample["actions"], observe=observe, commit=commit, progress=self.budget,
                                    timings=timings, owned_rows=self.compact)
        if report["aborted"]:
            raise SourceFailure("shared window rejected: " + repr(report["terminal_reasons"]))
        return dict(report, lane_metrics=lane_metrics)

    def _complete(self, report, started, timings, seal_evidence):
        before = perf_counter()
        if report["terminal_reasons"] and self.budget.sdk_step + CONTROL_SUBSTEPS <= self.budget.max_sdk_step:
            self.group.reset_done(self._reset)
        timings["lane_reset_s"] = perf_counter() - before
        before = perf_counter()
        artifact = seal_evidence("committed") if seal_evidence is not None else None
        timings["evidence_seal_s"] = perf_counter() - before
        metrics = dict(window_seconds=perf_counter() - started, budget=self.budget.snapshot(),
                       phase_seconds=timings, evidence_file=artifact.name if artifact else None, **report)
        if self.log_windows:
            self.journal.emit("batch_control", metrics)
        return metrics

    def _reset(self, env_id, exchange):
        old = self.owners[env_id]
        if exchange is not old.episode._exchange:
            raise ValueError("Reset selected the wrong lane exchange")
        fresh = self.reset_lane(env_id, old)
        if self.compact:
            fresh.history.enable_compact_batch()
        self.budget.reset_lane(env_id, fresh.rod.clock())
        self.owners[env_id] = fresh
        self.rewards[env_id] = self._reward(fresh)
        return fresh.episode._exchange

    def update(self):
        if self.failed or not self.trainer.rollout.full:
            raise RuntimeError("complete unquarantined rollout required for PPO")
        stats = self.trainer.update()
        self.journal.emit("batch_ppo_update", dict(metrics=stats, budget=self.budget.snapshot()))
        return stats
