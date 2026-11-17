"""Opt-in controller memory and actual-request/PPO binding, with no simulator calls.

This owner starts at a NEW explicit reset. It cannot reconstruct historical
previous_action, qualify a reset/reward/model, or add valid 371D sensor fields.
Use wrap_recorder around the existing durable recorder: advanced samples must be
persisted before this checker can fail. The callback's return is not a stop signal.
"""
from copy import deepcopy
import uuid

import numpy as np
import torch

from .tabletop_deformable_dual_evidence import _Sequence, _clock_row
from .tabletop_deformable_privileged_ppo import DeformablePPOTrainer
from .tabletop_deformable_r0_replay import _copy, _digest
from .tabletop_readonly_control_cost import _physical_source

VERSION = "dh116-controller-action24-history-v1"
SOURCE = "new_controller_owned_normalized_action24_memory"
SCOPE = "memory_and_issued_request_consistency_not_native_training_admission"
EPOCH = ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        _require(value.device.type == "cpu", "CPU-owned policy evidence required")
    elif isinstance(value, dict):
        for child in value.values():
            _cpu_tree(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _cpu_tree(child)


def _action(value):
    _cpu_tree(value)
    if isinstance(value, torch.Tensor):
        _require(value.dtype == torch.float32, "explicit float32 normalized action24 required")
        value = value.detach().numpy()
    result = np.array(value, copy=True)
    _require(result.dtype == np.float32 and result.shape == (1,24)
             and np.isfinite(result).all() and np.all(abs(result) <= 1),
             "one finite normalized float32 action24 required")
    return result


def _policy(trainer):
    _require(type(trainer) is DeformablePPOTrainer and trainer.num_envs == 1,
             "original single-environment DeformablePPOTrainer required")
    pending = trainer._pending
    _require(isinstance(pending,dict), "current sampled policy transition required")
    for tensor in (*trainer.model.parameters(), *trainer.model.buffers()):
        _require(tensor.device.type == "cpu", "CPU policy evidence required before hashing")
    _cpu_tree(pending)
    action = _action(pending["actions"])
    _require(pending["session_id"] == trainer.session_id
             and pending["policy_version"] == trainer.policy_version,
             "pending policy epoch differs")
    identity = trainer.model.model_identity
    _require(isinstance(identity.get("model_identity_sha256"), str)
             and len(identity["model_identity_sha256"]) == 64
             and isinstance(identity.get("checkpoint_namespace"), str),
             "explicit model/checkpoint namespace required")
    return dict(collection_token=pending["collection_token"],session_id=trainer.session_id,
        policy_version=trainer.policy_version,model_identity=identity,
        model_weights_sha256=_digest(trainer.model.state_dict()),
        pending_sha256=_digest(pending)), action


class Action24HistoryOwner:
    """Own one reset/episode's action history, separately from target execution."""

    def __init__(self, *, initial_action24, seed_source, physics_budget_end, solver_substeps):
        seed = _action(initial_action24)
        _require(not np.any(seed), "original controller reset uses an explicit zero action seed")
        _require(isinstance(seed_source,str) and bool(seed_source.strip()), "explicit reset seed source required")
        self._seed = seed
        self._seed_sha = _digest(seed)
        self._source = seed_source
        self._owner_uid = uuid.uuid4().hex
        self._sequence = _Sequence(physics_budget_end,solver_substeps)
        self._previous = seed.copy()
        self._previous_sha = _digest(self._previous)
        self._reset = self._captured = self._decision = self._settled = None
        self._last_hands = self._backend = None
        self._used_tokens = set()
        self._failed = False

    def _check(self):
        if self._failed:
            raise RuntimeError("failed action history owner; no continuation or backfill")
        if _digest(self._seed) != self._seed_sha or _digest(self._previous) != self._previous_sha:
            self._failed = True
            raise ValueError("controller seed or history memory changed outside its owner")

    def wrap_controller_reset(self, original):
        """Preserve the original callback's source/status; add ONLY this buffer readback."""
        _require(callable(original), "original controller reset callback required")
        def reset(context):
            self._check()
            try:
                _require(self._reset is None, "fresh action history owner required for every reset")
                clock = _clock_row(dict(context,policy_step=0,substep=0))
                _require(clock["reset_id"] >= 1 and clock["rod_tick"] == 0
                         and clock["physics_step"] == 2, "fresh original reset boundary required")
                prior = deepcopy(original(deepcopy(context)))
                _require(prior.get("owner") == "controller"
                         and prior.get("episode_uid") == clock["episode_uid"]
                         and prior.get("status") in ("cleared","not_present","unavailable")
                         and isinstance(prior.get("source"),str) and bool(prior["source"])
                         and "action24_history" not in prior,
                         "original controller reset receipt differs")
                self._previous = self._seed.copy()  # actual new controller-memory write
                self._previous_sha = _digest(self._previous)
                self._reset = dict(version=VERSION,source=SOURCE,scope=SCOPE,owner_uid=self._owner_uid,
                    identity=clock,seed_source=self._source,seed_action24=self._seed.tolist(),
                    previous_action24=self._previous.tolist(),dtype="float32",
                    seed_sha256=self._seed_sha,readback_sha256=self._previous_sha,
                    source_seed_sha256=context["source_seed_sha256"],
                    original_controller_receipt_sha256=_digest(prior),
                    native_qualification=False,teacher_admission=False)
                return dict(prior,action24_history=deepcopy(self._reset))
            except BaseException:
                self._failed = True
                raise
        return reset

    def wrap_recorder(self, original):
        """The original recorder must persist first; no callback return can stop physics."""
        _require(callable(original), "existing physical recorder required")
        def record(row):
            owned = _copy(row)
            try:
                original(_copy(owned))
                self.record(owned)
            except BaseException:
                self._failed = True
                raise
        return record

    def record(self, row):
        self._check()
        try:
            row = _copy(row)
            kind = row["kind"]
            _require(kind in ("setup","reset","initial_geometry","physical"),
                     "original physical owner event required")
            if kind == "reset":
                _require(self._reset is not None, "new controller reset callback was not executed")
                receipt = deepcopy(row["reset"]["owner_resets"]["controller"])
                nested = receipt.pop("action24_history",None)
                _require(_digest(nested) == _digest(self._reset)
                         and _digest(receipt) == self._reset["original_controller_receipt_sha256"],
                         "reset controller memory/source differs")
                actual = _clock_row(dict(row["reset"]["after_clock"],policy_step=0,substep=0))
                _require(actual == self._reset["identity"], "reset readback identity differs")
                self._backend = row["rod_boundary"]["rod_snapshot"]["backend_sha256"]
            if kind == "physical":
                _require(self._decision is not None, "sampled decision required before actual physics")
                _require(row["rod_boundary"]["rod_snapshot"]["backend_sha256"] == self._backend,
                         "actual backend identity differs")
                hands = _physical_source(row["exchange"],self._common,self._last_hands,
                                         not self._row_hashes)
                request = np.asarray(row["exchange"]["action_readback"]["request"]["action24"])
                # The original physical reader owns float64 action24; require exact
                # sampled float32 -> float64 equality, never tolerate rounding here.
                _require(np.array_equal(request,self._sample_action.astype(np.float64)),
                         "canonical actual request differs from sampled policy action")
                self._sequence.accept(kind,row)
                self._last_hands = hands
                self._issued = request.astype(np.float32,copy=True)
                self._row_hashes.append(_digest(row))
                self._target_hashes.append(_digest(
                    row["exchange"]["action_readback"]["submission"]["expected_backend_targets"]))
                self._projection_hashes.append(_digest(row["exchange"]["wrist_projection"]))
            else:
                self._sequence.accept(kind,row)
        except BaseException:
            self._failed = True
            raise

    def capture_pre_action(self, pre):
        self._check()
        _require(self._reset is not None and self._decision is None and self._settled is None,
                 "settled new-reset controller boundary required")
        # Reuse the original exact clock/66D source checker without opening a window.
        probe = deepcopy(self._sequence)
        probe.accept("begin",dict(pre_action_state=pre,action=self._previous))
        _require(pre["rod_boundary"]["rod_snapshot"]["backend_sha256"] == self._backend,
                 "pre-action backend differs")
        packet = dict(version=VERSION,source=SOURCE,scope=SCOPE,owner_uid=self._owner_uid,
            identity=deepcopy(pre["common"]["identity"]),pre_action_sha256=_digest(pre),
            previous_action24=self._previous.copy(),previous_action_sha256=self._previous_sha,
            reset=deepcopy(self._reset),native_qualification=False,teacher_admission=False)
        self._captured = _copy(packet)
        return _copy(packet)

    def begin(self, pre, trainer):
        self._check()
        try:
            _require(self._captured is not None and self._decision is None and self._settled is None,
                     "capture controller history before policy sampling")
            _require(_digest(pre) == self._captured["pre_action_sha256"],
                     "pre-action changed between history read and policy sample")
            policy, action = _policy(trainer)
            _require(policy["collection_token"] not in self._used_tokens,
                     "policy collection token already issued")
            self._sequence.accept("begin",dict(pre_action_state=pre,action=action))
            self._decision = dict(version=VERSION,source=SOURCE,owner_uid=self._owner_uid,
                pre_action_sha256=_digest(pre),previous_action_sha256=self._previous_sha,
                history_sha256=_digest(self._captured),policy=policy,sampled_action24=action.copy())
            self._sample_action = action
            self._common = _copy(pre["common"])
            self._row_hashes,self._target_hashes,self._projection_hashes = [],[],[]
            return _copy(self._decision)
        except BaseException:
            self._failed = True
            raise

    def end(self, result):
        self._check()
        try:
            _require(self._decision is not None, "one active sampled window required")
            self._sequence.accept("end",result)
            self._previous = self._issued.copy()
            self._previous_sha = _digest(self._previous)
            self._settled = dict(self._decision,scope=SCOPE,issued_action24=self._issued.copy(),
                physical_samples=len(self._row_hashes),physical_rows_sha256=list(self._row_hashes),
                execution_targets_sha256=list(self._target_hashes),
                wrist_projection_sha256=list(self._projection_hashes),
                termination_reason=result["termination_reason"],endpoint=deepcopy(self._sequence.clock),
                training_reward=None,native_qualification=False,teacher_admission=False)
            self._decision = self._captured = None
            return _copy(self._settled)
        except BaseException:
            self._failed = True
            raise

    def issued_for_pending(self, trainer, settlement):
        """Return actual request-derived actions for ORIGINAL record_step; no PPO update.

        Caller supplies reward/done/bootstrap from their independent admitted
        sources. This helper grants none of those contracts.
        """
        self._check()
        try:
            _require(self._settled is not None and _digest(settlement) == _digest(self._settled),
                     "owned unconsumed physical settlement required")
            policy, action = _policy(trainer)
            _require(policy == self._settled["policy"]
                     and np.array_equal(action,self._settled["issued_action24"]),
                     "sample/token/model/checkpoint identity changed after actual execution")
            issued = torch.from_numpy(self._settled["issued_action24"].copy())
            self._used_tokens.add(policy["collection_token"])
            self._settled = None
            return issued
        except BaseException:
            self._failed = True
            raise
