"""Prepared no-camera collection; real entry stays denied at current source gaps.

CPUFixtureCollector is explicitly a CPU-double verification harness. It composes
the original physical owner, R0 evaluator and PPO storage/GAE. Its synthetic
scalar never changes diagnostic R0 receipts or authorizes native training.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from .tabletop_actual_backend_teacher import load_adapter_config
from .tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode
from .tabletop_deformable_privileged_ppo import DeformablePPOTrainer, verify_base_ppo_source
from .tabletop_deformable_r0_replay import PhysicalR0Diagnostic, _copy, _digest, encode_owned
from .tabletop_deformable_rollout_admission import gae_boundary
from .tabletop_flexible_measured_common_teacher import common_selection

VERSION = "dh116-no-camera-on-policy-collector-cpu-v1"
SCOPE = "cpu_fixture_only"
EPOCH = ("run_uuid","env_id","episode_uid","reset_id","rod_uid")
ENDPOINT = ("env_id","episode_uid","reset_id","rod_uid","physics_step","rod_tick","timestamp_s")
TASK = "configs/dh116_tasks/flexible_dual_hand_grasp_v1.json"


def actual_admission(project, grasp):
    """Recheck available source/config bytes, never accept a native_ready switch."""
    project = Path(project)
    if grasp not in ("pinch","tripod","power"):
        raise ValueError("original per-style identity required")
    reasons = ["mesh_reset_source_unqualified", "formal_R0_training_source_missing"]
    config = project/f"configs/dh116_training/actual_measured_teacher_{grasp}_v1.json"
    task_bytes = (project/TASK).read_bytes()
    task = json.loads(task_bytes)
    coefficients = task["reward"]["coefficients"]
    # Current frozen source is deliberately not an approved training recipe.
    if any(value is not None for value in coefficients.values()):
        reasons.append("unexpected_frozen_reward_coefficients")
    binding = None
    source_error = None
    try:
        binding = load_adapter_config(config, project)
    except (ValueError, OSError, KeyError) as exc:
        source_error = type(exc).__name__ + ": " + str(exc)
        reasons.append("actual_model_source_unavailable")
    return dict(version=VERSION, grasp=grasp, admitted=False, blocked_reasons=reasons,
        projection_valid_fields=len(common_selection()),
        model_binding_sha256=None if binding is None else binding["binding_sha256"],
        model_source_error=source_error, model_input_available=binding is not None,
        task_sha256=hashlib.sha256(task_bytes).hexdigest(),
        coefficient_source=task["reward"]["coefficient_source"],
        reset_mesh_qualified=False, training_reward_qualified=False,
        native_qualification=False, teacher_admission=False,
        scope="source_audit_only_not_rollout_authorization")


def create_actual_collector(project, grasp, *, policy_factory, trainer_factory, episode_factory):
    """Fail before any policy, optimizer or physical owner construction.

    A future independently reviewed reset/source and reward contract must replace
    this boundary. Factory arguments document order, not a caller-supplied bypass.
    """
    gate = actual_admission(project, grasp)
    raise RuntimeError("actual collector admission denied: " + ", ".join(gate["blocked_reasons"]))


class ExclusiveJournal:
    """Each delivered physical sample is owned, flushed and fsynced before use."""
    def __init__(self, path):
        self.path = Path(path)
        self._stream = self.path.open("x", encoding="utf-8")
        self._sequence = 0

    def emit(self, operation, payload):
        row = dict(version=VERSION, scope=SCOPE, sequence=self._sequence,
                   operation=operation, payload=encode_owned(payload))
        self._stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._sequence += 1

    def close(self):
        self._stream.close()


class _ObserveTerminal:
    """Preserve the original physical control loop and return R0 stop reasons.

    DualHandPhysicalEpisode's recorder does not forward a callback return value.
    This narrow exchange wrapper runs that original observer first (which durably
    writes the row), then returns the already evaluated R0 terminal reason.
    """
    def __init__(self, inner, reason, owners):
        self._inner, self._reason, self._owners = inner, reason, owners
        self._closed = False

    def control(self, action, *, observe):
        if self._closed:
            raise RuntimeError("closed collector exchange")
        def recorded(row):
            observe(row)
            return self._reason()
        step = self._inner._step
        def checked(action, substep):
            self._owners.validate()
            return step(action, substep)
        self._inner._step = checked
        try:
            return self._inner.control(action, observe=recorded)
        finally:
            self._inner._step = step

    def close(self):
        self._closed = True


def _model_weights(trainer):
    return _digest(trainer.model.state_dict())


def _origin(trainer):
    identity = trainer.model.model_identity
    return dict(kind="fresh_CPU_fixture_initialization", checkpoint_path=None,
        checkpoint_sha256=None, checkpoint_namespace=identity["checkpoint_namespace"],
        model_identity_sha256=identity["model_identity_sha256"],
        initial_weights_sha256=_model_weights(trainer), session_id=trainer.session_id,
        policy_version=trainer.policy_version, counters=deepcopy(trainer.counters),
        optimizer_updates=0, real_checkpoint_loaded=False)


def _endpoint(identity):
    result = {key:identity[key] for key in ENDPOINT}
    result["phase"] = "post_action_pre_reset"
    return result


class CPUFixtureCollector:
    """One bounded policy epoch on SDK/rod doubles; no update or native entry.

    pack_observation and fixture_reward are explicit synthetic fixture providers,
    never actual-source attestations. Production would use pack_actual and a
    qualified training-R0 adapter, neither of which this scope claims to supply.
    """
    def __init__(self, episode, trainer, r0, *, pack_observation, fixture_reward,
                 journal_path, max_controls, fixture_label):
        if (type(episode) is not DualHandPhysicalEpisode
                or type(trainer) is not DeformablePPOTrainer
                or not isinstance(r0, PhysicalR0Diagnostic)):
            raise ValueError("original episode, structured PPO and R0 observer required")
        if (episode._state != "active" or r0._state != "active"
                or trainer._pending is not None or len(trainer.rollout)
                or trainer.num_envs != 1 or trainer.counters["updates"] != 0
                or trainer.policy_version != 0):
            raise ValueError("fresh CPU fixture policy and completed reset required")
        if (type(max_controls) is not int or not 1 <= max_controls <= trainer.rollout_steps
                or not isinstance(fixture_label,str) or not fixture_label.strip()
                or not callable(pack_observation) or not callable(fixture_reward)):
            raise ValueError("explicit bounded synthetic fixture providers required")
        if any(value.device.type != "cpu" for value in
               (*trainer.model.parameters(), *trainer.model.buffers())):
            raise ValueError("CPU fixture model only")
        # Reject the actual backend and actual-input model from this fixture API.
        from .tabletop_actual_backend_teacher import ActualBackendMeasuredTeacher
        from .tabletop_native_parameter_endpoints import BACKEND_SHA
        if isinstance(trainer.model, ActualBackendMeasuredTeacher) or episode._boundary._sha256 == BACKEND_SHA:
            raise ValueError("actual native input requires the denied actual admission path")
        from .tabletop_cpu_fixture_owners import CPUFixtureOwners
        self._owners = CPUFixtureOwners(episode)
        verify_base_ppo_source()
        self.origin = _origin(trainer)
        self._origin_digest = _digest(self.origin)
        self._policy = (trainer.session_id, trainer.policy_version, deepcopy(trainer.counters))
        self._weights = self.origin["initial_weights_sha256"]
        self._model_identity = _digest(trainer.model.model_identity)
        self._episode, self._trainer, self._r0 = episode, trainer, r0
        self._pack, self._reward = pack_observation, fixture_reward
        self._limit, self._count, self._closed, self._active = max_controls, 0, False, False
        self._rows, self._r0_reason, self._raw_terminal = [], None, None
        self._journal = ExclusiveJournal(journal_path)
        self.journal_path = self._journal.path
        self._journal.emit("origin", dict(origin=self.origin, fixture_label=fixture_label,
            frozen_R0_training_reward=None, cpu_owner_protocol=self._owners.validate(),
            native_qualification=False, teacher_admission=False))
        old_record = episode._record
        def record(event):
            # Complete delivered row precedes validation/settlement or any stop.
            self._journal.emit("physical", event)
            old_record(_copy(event))
            if not self._active or event.get("kind") != "physical":
                raise RuntimeError("physical sample outside a pending policy decision")
            if self._raw_terminal is not None or self._r0_reason is not None:
                raise RuntimeError("physical sample after terminal")
            self._r0.record(event)
            self._rows.append(_copy(event))
            self._raw_terminal = event["exchange"]["guard_failure"]
            reasons = self._r0._reducer._reasons
            self._r0_reason = None if not reasons else "r0:" + "|".join(reasons)
        episode._record = record
        episode._exchange = _ObserveTerminal(episode._exchange, lambda:self._r0_reason, self._owners)

    def _policy_check(self):
        tr = self._trainer
        if any(value.device.type != "cpu" for value in (*tr.model.parameters(), *tr.model.buffers())):
            raise ValueError("CPU policy parameters and buffers required before weight digest")
        if (_digest(self.origin) != self._origin_digest
                or (tr.session_id,tr.policy_version,tr.counters) != self._policy
                or _model_weights(tr) != self._weights
                or _digest(tr.model.model_identity) != self._model_identity):
            raise ValueError("policy epoch, weights or checkpoint origin changed")

    def collect(self):
        if self._closed or self._active:
            raise RuntimeError("closed or active collector")
        self._active = True
        self._rows, self._r0_reason, self._raw_terminal = [], None, None
        tr, ep = self._trainer, self._episode
        try:
            self._owners.validate()
            self._policy_check()
            pre = ep.pre_action_state()
            identity = deepcopy(pre["common"]["identity"])
            obs = self._pack(_copy(pre))
            # Packing must not advance/reset the physical owner.
            if _digest(ep.pre_action_state()) != _digest(pre):
                raise ValueError("same pre-action boundary changed while packing")
            sample = tr.collect_step(obs,obs)
            self._r0.begin(pre,sample["actions"])
            self._journal.emit("decision", dict(pre_action_identity=identity,
                pre_action_sha256=_digest(pre), action24=sample["actions"],
                collection_token=sample["collection_token"], policy_version=sample["policy_version"],
                origin_sha256=self._origin_digest, observation_sha256=_digest(obs)))
            result = ep.control(sample["actions"])
            count = len(self._rows)
            reason = self._raw_terminal or self._r0_reason
            if (result["control_step"] != identity["policy_step"]
                    or result["physical_samples"] != count
                    or result["termination_reason"] != reason
                    or not 1 <= count <= 8 or (count != 8 and reason is None)):
                raise ValueError("control result/window differs from observed physical samples")
            # The original replay validator compares its physical guard only.
            # Preserve both the actual control result and this explicit projection.
            r0_input = deepcopy(result)
            r0_input["termination_reason"] = self._raw_terminal
            settled = self._r0.end(r0_input)
            if settled["control_receipt"] is None:
                raise ValueError("R0 cannot settle this partial terminal; no PPO reward invented")
            reward = self._reward(_copy(settled))
            if type(reward) not in (int,float) or not math.isfinite(reward):
                raise ValueError("finite explicit synthetic reward required; unknown is not zero")
            last = self._rows[-1]["exchange"]["action_readback"]["post"]["identity"][0]
            if any(last[key] != identity[key] for key in EPOCH):
                raise ValueError("post-action endpoint reset/episode differs")
            terminated = reason is not None or settled["control_receipt"]["terminated"]
            truncated = not terminated and (self._count+1 == self._limit
                                             or settled["control_receipt"]["truncated"])
            endpoint = _endpoint(last)
            next_endpoint = None
            value = torch.zeros(1)
            if not terminated:
                next_pre = ep.pre_action_state()
                now = next_pre["common"]["identity"]
                if (any(now[key] != last[key] for key in EPOCH)
                        or _endpoint(now) != endpoint
                        or now["solver_time_s"] != last["solver_time_s"]
                        or now["policy_step"] != identity["policy_step"]+1
                        or now["substep"] != 0 or now["clock_valid"] is not True
                        or now["phase"] != identity["phase"]):
                    raise ValueError("bootstrap endpoint changed before reset")
                next_obs = self._pack(_copy(next_pre))
                if _digest(ep.pre_action_state()) != _digest(next_pre):
                    raise ValueError("bootstrap endpoint advanced during packing")
                with torch.no_grad():
                    value = tr.model.value(next_obs).detach().cpu()
                next_endpoint = _endpoint(now)
            boundary = gae_boundary(terminated=bool(terminated),truncated=bool(truncated),
                endpoint_identity=endpoint,next_value_identity=next_endpoint)
            # Entire episode/observer retains the sampled action as declared;
            # original R0 replay already checked each issued backend target.
            self._policy_check()
            receipt = dict(control_index=self._count, physical_samples=count,
                termination_reason=reason, terminated=bool(terminated),truncated=bool(truncated),
                done=bool(terminated or truncated), bootstrap_mask=not terminated,
                endpoint_identity=endpoint, next_value_identity=next_endpoint,
                fixture_reward=float(reward), training_reward=None,
                ppo_record_status="prepared_before_trainer_record",
                r0_settlement=settled, original_control_result=result, r0_result_input=r0_input,
                gae_boundary=boundary, collection_token=sample["collection_token"],
                origin_sha256=self._origin_digest, optimizer_updates=0,
                native_qualification=False, teacher_admission=False)
            self._journal.emit("settlement",receipt)
            tr.record_step(torch.tensor([reward]),torch.tensor([terminated or truncated]),
                value, truncated=torch.tensor([truncated]),
                bootstrap_mask=torch.tensor([not terminated]),
                executed_actions=sample["actions"], collection_token=sample["collection_token"])
            # Original counters increment environment steps while recording.
            self._policy = (tr.session_id,tr.policy_version,deepcopy(tr.counters))
            self._count += 1
            self._closed = bool(terminated or truncated)
            if self._closed:
                ep._exchange.close()
            self._journal.emit("transition_recorded",dict(
                collection_token=sample["collection_token"],rollout_rows=len(tr.rollout),
                session_id=tr.session_id,policy_version=tr.policy_version,optimizer_updates=0))
            return deepcopy(receipt)
        except Exception as exc:
            self._closed = True
            ep._exchange.close()
            self._journal.emit("failure",dict(error=type(exc).__name__+": "+str(exc),
                observed_physical_samples=len(self._rows), pending_transition=tr._pending is not None,
                no_transition_fabricated=True, native_qualification=False))
            raise
        finally:
            self._active = False

    def close(self):
        self._closed = True
        self._episode._exchange.close()
        self._journal.close()
