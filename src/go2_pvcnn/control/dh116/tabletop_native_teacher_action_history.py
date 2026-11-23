"""Explicit v2 history at actual no-step Reset boundaries, no synthetic setup.

The old one-shot v1 owner is unchanged. Policy/token/weights/request/SDK target
checking and physical sequencing are reused verbatim through inheritance.
"""
from contextlib import contextmanager
from copy import deepcopy
import numpy as np
from . import tabletop_action24_history_owner as original
from .tabletop_action24_history_owner import Action24HistoryOwner,_require
from .tabletop_deformable_dual_evidence import _Sequence,_clock_row,_rod_boundary
from .tabletop_deformable_r0_replay import _digest

VERSION="dh116-native-reset-bound-action24-history-v2"
SOURCE="new_actual_reset_bound_controller_normalized_action24_memory"
SCOPE="memory_issued_request_consistency_not_native_admission"


class _ResetBoundSequence(_Sequence):
    def __init__(self,physics_budget_end,solver_substeps,previous_reset):
        super().__init__(physics_budget_end,solver_substeps)
        self.previous_reset=deepcopy(previous_reset)

    def accept(self,operation,row):
        if operation=="setup":
            raise ValueError("scene setup belongs to outer session, never replay setup=2")
        if operation!="reset":
            return super().accept(operation,row)
        from .tabletop_dual_hand_reset import _no_step
        receipt=row["reset"]
        _require(self.clock is None and receipt["status"]=="complete",
                 "one actual completed reset per fresh history owner required")
        clock=_clock_row(dict(receipt["after_clock"],policy_step=0,substep=0))
        _no_step(receipt["before_clock"],receipt["after_actual_clocks"])
        _require(clock["rod_tick"]==0 and clock["reset_id"]>=1
            and clock["physics_step"]==receipt["before_clock"]["sdk_physics_step"]
            and receipt["after_clock"]==receipt["after_actual_clocks"]["rod"],
            "actual absolute SDK/fresh reset boundary required")
        before=receipt["before_clock"]["rod"]
        _require(clock["reset_id"]==before["reset_id"]+1
            and clock["episode_uid"]!=before["episode_uid"]
            and clock["solver_time_s"]==before["solver_time_s"],
            "fresh UID/next reset ID and unchanged actual solver time required")
        prior=self.previous_reset
        if prior is not None:
            _require(all(clock[k]==prior[k] for k in ("run_uuid","env_id","rod_uid","physics_step","timestamp_s","solver_time_s"))
                and clock["reset_id"]==prior["reset_id"]+1 and clock["episode_uid"]!=prior["episode_uid"],
                "same scene absolute clocks and fresh reset identity required")
        _rod_boundary(row["rod_boundary"],clock,"reset")
        self.clock=clock;self.rows+=1


class BatchPolicySnapshot:
    """Seal one pending batch once and verify the full identity before admission."""

    def __init__(self, trainer):
        from .tabletop_deformable_privileged_ppo import DeformablePPOTrainer

        _require(type(trainer) is DeformablePPOTrainer, "original structured PPO trainer required")
        self._trainer = trainer
        self._pending = trainer._pending
        _require(isinstance(self._pending, dict), "current sampled policy transition required")
        original._cpu_tree(self._pending)
        identity = trainer.model.model_identity
        _require(isinstance(identity.get("model_identity_sha256"), str)
                 and len(identity["model_identity_sha256"]) == 64
                 and isinstance(identity.get("checkpoint_namespace"), str),
                 "explicit model/checkpoint namespace required")
        self.policy = dict(collection_token=self._pending["collection_token"], session_id=trainer.session_id,
                           policy_version=trainer.policy_version, model_identity=identity,
                           model_weights_sha256=_digest(trainer.model.state_dict()),
                           pending_sha256=_digest(self._pending))
        self._check_pending()

    def _check_pending(self):
        trainer, pending = self._trainer, self._pending
        _require(trainer._pending is pending and pending["actions"].shape == (trainer.num_envs, 24)
                 and pending["session_id"] == trainer.session_id == self.policy["session_id"]
                 and pending["policy_version"] == trainer.policy_version == self.policy["policy_version"]
                 and pending["collection_token"] == self.policy["collection_token"],
                 "pending PPO batch or epoch changed during native batch")

    def read(self, trainer, lane_index):
        _require(trainer is self._trainer and type(lane_index) is int and 0 <= lane_index < trainer.num_envs,
                 "explicit original PPO batch row required")
        self._check_pending()
        return (dict(self.policy, batch_row=lane_index),
                original._action(self._pending["actions"][lane_index:lane_index + 1]))

    def verify(self):
        self._check_pending()
        _require(_digest(self._trainer.model.state_dict()) == self.policy["model_weights_sha256"]
                 and _digest(self._pending) == self.policy["pending_sha256"],
                 "policy weights or pending tensors changed during native batch")


@contextmanager
def _batch_policy(trainer, lane_index, snapshot):
    """Supply the checked native lane row to the unchanged history protocol."""
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

    epoch = BatchPolicySnapshot(trainer) if snapshot is None else snapshot
    with bindings((original, "_policy", lambda actual: epoch.read(actual, lane_index))):
        yield


class NativeAction24HistoryOwner(Action24HistoryOwner):
    version=VERSION
    source=SOURCE
    def __init__(self,*,initial_action24,seed_source,physics_budget_end,solver_substeps,previous_reset=None):
        super().__init__(initial_action24=initial_action24,seed_source=seed_source,
            physics_budget_end=physics_budget_end,solver_substeps=solver_substeps)
        self._sequence=_ResetBoundSequence(physics_budget_end,solver_substeps,previous_reset)
        self._previous_reset=deepcopy(previous_reset)
        self.compact_batch = False
        self._live_event = None

    def enable_compact_batch(self):
        if self._decision is not None or self._sequence.controls:
            raise ValueError("compact history must be selected before policy collection")
        self.compact_batch = True

    def record(self, row):
        if not self.compact_batch or row["kind"] != "physical":
            return super().record(row)
        from .tabletop_readonly_control_cost import _physical_source

        self._check()
        try:
            _require(self._decision is not None, "sampled decision required before actual physics")
            _require(self._live_event is None, "prior physical event was not consumed")
            _require(row["rod_boundary"]["rod_snapshot"]["backend_sha256"] == self._backend,
                     "actual backend identity differs")
            exchange = row["exchange"]
            # Normalize only the small action trace, matching the legacy JSON-shaped validation.
            action_trace = original._copy(exchange["action_readback"])
            hands = _physical_source(dict(exchange, action_readback=action_trace), self._common,
                                     self._last_hands, self._sequence.substep == 0)
            request = np.asarray(exchange["action_readback"]["request"]["action24"])
            _require(np.array_equal(request, self._sample_action.astype(np.float64)),
                     "canonical actual request differs from sampled policy action")
            self._sequence.accept("physical", row)
            self._last_hands = hands
            self._issued = request.astype(np.float32, copy=True)
            self._live_event = row
        except BaseException:
            self._failed = True
            raise

    def claim_compact_physical(self, event):
        """Accept the exact owned event once, within the synchronous batch observer."""
        self._check()
        _require(self.compact_batch and self._decision is not None and self._live_event is event,
                 "owning recorded physical event required")
        self._live_event = None
        return dict(substep=self._sequence.substep - 1, endpoint=deepcopy(self._sequence.clock),
                    verification="same_window_ownership_and_measured_targets")

    def end(self, result):
        if not self.compact_batch:
            return super().end(result)
        from .tabletop_deformable_r0_replay import _copy

        self._check()
        try:
            _require(self._decision is not None and self._live_event is None,
                     "every compact physical event must be consumed before settlement")
            samples = self._sequence.substep
            self._sequence.accept("end", result)
            self._previous = self._issued.copy()
            self._previous_sha = _digest(self._previous)
            self._settled = dict(self._decision, scope=SCOPE, issued_action24=self._issued.copy(),
                physical_samples=samples, physical_verification="ordered_owned_rows_and_measured_targets",
                termination_reason=result["termination_reason"], endpoint=deepcopy(self._sequence.clock),
                training_reward=None, native_qualification=False, teacher_admission=False)
            self._decision = self._captured = None
            return _copy(self._settled)
        except BaseException:
            self._failed = True
            raise

    def wrap_controller_reset(self,original):
        _require(callable(original),"original controller reset callback required")
        def reset(context):
            self._check()
            try:
                _require(self._reset is None,"fresh reset-bound history owner required")
                clock=_clock_row(dict(context,policy_step=0,substep=0))
                _require(clock["reset_id"]>=1 and clock["rod_tick"]==0,
                    "actual fresh reset boundary required")
                prior=self._previous_reset
                if prior is not None:
                    _require(all(clock[k]==prior[k] for k in ("run_uuid","env_id","rod_uid","physics_step","timestamp_s","solver_time_s"))
                        and clock["reset_id"]==prior["reset_id"]+1 and clock["episode_uid"]!=prior["episode_uid"],
                        "same absolute clock/fresh UID required")
                original_receipt=deepcopy(original(deepcopy(context)))
                _require(original_receipt.get("owner")=="controller"
                    and original_receipt.get("episode_uid")==clock["episode_uid"]
                    and original_receipt.get("status") in ("cleared","not_present","unavailable")
                    and isinstance(original_receipt.get("source"),str) and bool(original_receipt["source"])
                    and "action24_history" not in original_receipt,"original reset owner receipt required")
                self._previous=self._seed.copy();self._previous_sha=_digest(self._previous)
                self._reset=dict(version=VERSION,source=SOURCE,scope=SCOPE,owner_uid=self._owner_uid,
                    identity=clock,seed_source=self._source,seed_action24=self._seed.tolist(),
                    previous_action24=self._previous.tolist(),dtype="float32",seed_sha256=self._seed_sha,
                    readback_sha256=self._previous_sha,source_seed_sha256=context["source_seed_sha256"],
                    original_controller_receipt_sha256=_digest(original_receipt),
                    native_qualification=False,teacher_admission=False)
                return dict(original_receipt,action24_history=deepcopy(self._reset))
            except BaseException:
                self._failed=True
                raise
        return reset

    def capture_pre_action(self,pre):
        from .tabletop_deformable_r0_replay import _copy
        packet=super().capture_pre_action(pre)
        packet.update(version=VERSION,source=SOURCE,scope=SCOPE)
        self._captured=_copy(packet)
        return _copy(packet)

    def begin(self,pre,trainer,*,lane_index=None, policy_snapshot=None):
        from .tabletop_deformable_r0_replay import _copy
        if lane_index is None:
            super().begin(pre,trainer)
        else:
            with _batch_policy(trainer, lane_index, policy_snapshot):
                super().begin(pre,trainer)
        self._decision.update(version=VERSION,source=SOURCE)
        return _copy(self._decision)

    def issued_for_pending(self, trainer, settlement, *, lane_index=None, policy_snapshot=None):
        if lane_index is None:
            return super().issued_for_pending(trainer, settlement)
        with _batch_policy(trainer, lane_index, policy_snapshot):
            return super().issued_for_pending(trainer, settlement)
