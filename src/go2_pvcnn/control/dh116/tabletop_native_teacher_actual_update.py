"""Seal one fresh Teacher update over an already constructed C04 collector."""
from pathlib import Path
import os
from tempfile import TemporaryDirectory

from .tabletop_deformable_r0_replay import _digest
from .tabletop_native_teacher_checkpoint import audit_checkpoint_bytes, checkpoint_identity, save_checkpoint
from .tabletop_native_teacher_collector import NativeTeacherCollector
from .tabletop_native_teacher_training import TrainingStopped, run_updates
from .tabletop_native_teacher_worker_contract import fresh_smoke_limits


ROLLOUT_STEPS = 32
FRESH_ORIGIN = {"kind": "fresh"}
CHECKPOINT_NAME = "teacher_clean.pt"


def _seal_checkpoint(path, trainer, contract):
    """Publish only the exact private byte snapshot accepted by C05."""
    with TemporaryDirectory(prefix=".teacher-audit-", dir=path.parent) as directory:
        staged = Path(directory) / CHECKPOINT_NAME
        saved = save_checkpoint(staged, trainer, contract)
        raw = staged.read_bytes()
        audited = audit_checkpoint_bytes(raw, trainer, contract, expected_origin=FRESH_ORIGIN)
        if (audited["checkpoint_sha256"] != saved["sha256"]
                or audited["counters"] != trainer.counters
                or audited["policy_version"] != trainer.policy_version
                or audited["policy_weights_sha256"] != _digest(trainer.model.state_dict())
                or audited["accepted_minibatch_updates"] != trainer._native_accepted_minibatches):
            raise TrainingStopped("audited checkpoint does not match the completed PPO update")
        verified = Path(directory) / "verified.pt"
        with verified.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(verified, path)
    return audited


def _clock(collector, *, controls, sdk_step, rod_tick):
    clock = collector.owner.episode._read_state_clock()
    expected = dict(env_id=0, policy_step=controls, substep=0, physics_step=sdk_step, rod_tick=rod_tick)
    if (clock.get("clock_valid") is not True
            or any(type(clock.get(key)) is not int or clock[key] != value for key, value in expected.items())):
        raise ValueError("fresh Teacher update has an unexpected SDK/rod clock boundary")
    return clock


def run_fresh_update(collector, *, checkpoint_contract, controls=ROLLOUT_STEPS):
    """Train over one fixed-horizon episode and bind the checkpoint to its ledger.

    The caller owns scene construction, authorization and teardown. Successful
    serialization is training evidence only. Native source and outer-process
    validation remain independent.
    """
    if type(collector) is not NativeTeacherCollector:
        raise ValueError("original C04 collector required")
    limits, _ = fresh_smoke_limits(controls)
    physical_steps = limits["measured_sdk_steps"]
    trainer = collector.trainer
    checkpoint = collector.journal.path.parent / CHECKPOINT_NAME
    if os.path.lexists(checkpoint):
        raise FileExistsError(checkpoint)
    if (collector._closed or collector._active or collector.quarantined or collector.episode_done
            or collector._student_stereo is not None
            or collector.transitions != 0 or collector.consumed_sdk_steps != 0
            or trainer.num_envs != 1 or trainer.rollout_steps != controls
            or trainer.policy_version != 0 or trainer.counters != {"updates": 0, "environment_steps": 0}
            or trainer._pending is not None or len(trainer.rollout)
            or getattr(trainer, "_native_training_origin", FRESH_ORIGIN) != FRESH_ORIGIN
            or getattr(trainer, "_native_accepted_minibatches", 0) != 0):
        raise ValueError("fresh camera-free Teacher, one environment and the exact unused rollout required")
    identity = checkpoint_identity(trainer, checkpoint_contract)
    scope = collector.sources.scope
    if (identity["training_contract"]["scope"] != scope or collector.journal.scope != scope):
        raise ValueError("checkpoint, collector and journal must share the same source scope")
    start = _clock(collector, controls=0, sdk_step=2, rod_tick=0)
    try:
        training = run_updates(collector, additional_updates=1)
        end = _clock(collector, controls=controls, sdk_step=2 + physical_steps,
                     rod_tick=limits["max_internal_solver_substeps"])
        if (collector.transitions != controls or collector.consumed_sdk_steps != physical_steps
                or trainer.counters != {"updates": 1, "environment_steps": controls}
                or any(end[key] != start[key] for key in ("run_uuid", "episode_uid", "rod_uid", "reset_id"))):
            raise TrainingStopped("exact one-episode control/SDK PPO update required")
        input_audit = {}
        if scope == "owner_source_unqualified":
            from .tabletop_native_teacher_actual_inputs import audit_teacher_input_sources

            input_audit["input_source_audit"] = audit_teacher_input_sources(
                collector.journal.path.read_bytes(), backend_profile=collector.owner.backend_profile,
                scope=scope, controls=controls)
        audited = _seal_checkpoint(checkpoint, trainer, checkpoint_contract)
        result = dict(training, clean_checkpoint=dict(path=checkpoint.name, sha256=audited["checkpoint_sha256"],
                      policy_weights_sha256=audited["policy_weights_sha256"]),
                      physical=dict(start_sdk_step=start["physics_step"], end_sdk_step=end["physics_step"],
                                    start_rod_tick=start["rod_tick"], end_rod_tick=end["rod_tick"]),
                      native_qualification=False, teacher_admission=False, **input_audit)
        collector.journal.emit("fresh_update_checkpoint_sealed", result)
        return result
    except BaseException:
        trainer._native_training_failed = True
        raise
