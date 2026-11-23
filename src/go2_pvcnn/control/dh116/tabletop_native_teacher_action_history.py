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


@contextmanager
def _batch_policy(trainer, lane_index):
    """Supply the checked native lane row to the unchanged history protocol."""
    from .tabletop_deformable_privileged_ppo import DeformablePPOTrainer
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings

    def read(actual):
        _require(actual is trainer and type(trainer) is DeformablePPOTrainer
                 and type(lane_index) is int and 0 <= lane_index < trainer.num_envs,
                 "explicit original PPO batch row required")
        pending = trainer._pending
        _require(isinstance(pending, dict), "current sampled policy transition required")
        original._cpu_tree(pending)
        _require(pending["actions"].shape == (trainer.num_envs, 24)
                 and pending["session_id"] == trainer.session_id
                 and pending["policy_version"] == trainer.policy_version,
                 "pending policy batch or epoch differs")
        identity = trainer.model.model_identity
        _require(isinstance(identity.get("model_identity_sha256"), str)
                 and len(identity["model_identity_sha256"]) == 64
                 and isinstance(identity.get("checkpoint_namespace"), str),
                 "explicit model/checkpoint namespace required")
        policy = dict(collection_token=pending["collection_token"], session_id=trainer.session_id,
                      policy_version=trainer.policy_version, model_identity=identity,
                      model_weights_sha256=_digest(trainer.model.state_dict()),
                      pending_sha256=_digest(pending), batch_row=lane_index)
        return policy, original._action(pending["actions"][lane_index:lane_index + 1])

    with bindings((original, "_policy", read)):
        yield


class NativeAction24HistoryOwner(Action24HistoryOwner):
    version=VERSION
    source=SOURCE
    def __init__(self,*,initial_action24,seed_source,physics_budget_end,solver_substeps,previous_reset=None):
        super().__init__(initial_action24=initial_action24,seed_source=seed_source,
            physics_budget_end=physics_budget_end,solver_substeps=solver_substeps)
        self._sequence=_ResetBoundSequence(physics_budget_end,solver_substeps,previous_reset)
        self._previous_reset=deepcopy(previous_reset)

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

    def begin(self,pre,trainer,*,lane_index=None):
        from .tabletop_deformable_r0_replay import _copy
        if lane_index is None:
            super().begin(pre,trainer)
        else:
            with _batch_policy(trainer, lane_index):
                super().begin(pre,trainer)
        self._decision.update(version=VERSION,source=SOURCE)
        return _copy(self._decision)

    def issued_for_pending(self, trainer, settlement, *, lane_index=None):
        if lane_index is None:
            return super().issued_for_pending(trainer, settlement)
        with _batch_policy(trainer, lane_index):
            return super().issued_for_pending(trainer, settlement)
