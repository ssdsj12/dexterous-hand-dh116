"""Seal one fresh or resumed Teacher update over an owned C04 collector."""
import hashlib
from pathlib import Path
import os
from tempfile import TemporaryDirectory

from .tabletop_deformable_r0_replay import _digest
from .tabletop_native_teacher_checkpoint import (
    _rng_state, audit_checkpoint_bytes, checkpoint_identity, save_checkpoint,
)
from .tabletop_native_teacher_collector import NativeTeacherCollector
from .tabletop_native_teacher_training import TrainingStopped, run_updates
from .tabletop_native_teacher_worker_contract import fresh_smoke_limits


ROLLOUT_STEPS = 32
FRESH_ORIGIN = {"kind": "fresh"}
CHECKPOINT_NAME = "teacher_clean.pt"


def _seal_checkpoint(path, trainer, contract, *, expected_origin=FRESH_ORIGIN):
    """Publish only the exact private byte snapshot accepted by C05."""
    with TemporaryDirectory(prefix=".teacher-audit-", dir=path.parent) as directory:
        staged = Path(directory) / CHECKPOINT_NAME
        saved = save_checkpoint(staged, trainer, contract)
        raw = staged.read_bytes()
        audited = audit_checkpoint_bytes(raw, trainer, contract, expected_origin=expected_origin)
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
    """Use one unused episode and an optimizer with no accepted updates."""
    return _run_update(collector, checkpoint_contract=checkpoint_contract, controls=controls, origin=None)


def run_resumed_update(collector, *, checkpoint_contract, origin, controls=ROLLOUT_STEPS):
    """Advance a restored optimizer by one update in a distinct, freshly reset episode."""
    if type(origin) is not dict:
        raise ValueError("resume requires a verified A checkpoint and its episode origin")
    return _run_update(collector, checkpoint_contract=checkpoint_contract, controls=controls, origin=origin)


def _run_update(collector, *, checkpoint_contract, controls, origin):
    if type(collector) is not NativeTeacherCollector:
        raise ValueError("original C04 collector required")
    limits, _ = fresh_smoke_limits(controls)
    physical_steps = limits["measured_sdk_steps"]
    trainer = collector.trainer
    checkpoint = collector.journal.path.parent / CHECKPOINT_NAME
    if os.path.lexists(checkpoint):
        raise FileExistsError(checkpoint)
    previous_checkpoint = None
    counters = {"updates": 0, "environment_steps": 0}
    expected_origin = FRESH_ORIGIN
    previous_accepted = 0
    if origin is not None:
        previous_checkpoint = Path(origin["checkpoint_path"])
        if (not previous_checkpoint.is_absolute() or previous_checkpoint.resolve(strict=True) != previous_checkpoint
                or not previous_checkpoint.is_file() or previous_checkpoint.parent == checkpoint.parent
                or previous_checkpoint.name != CHECKPOINT_NAME
                or hashlib.sha256(previous_checkpoint.read_bytes()).hexdigest() != origin["checkpoint_sha256"]):
            raise ValueError("verified unchanged A checkpoint in a separate directory required")
        expected_origin = {"kind": "resume", "checkpoint_sha256": origin["checkpoint_sha256"]}
        previous_accepted = origin["loaded_accepted_minibatch_updates"]
        counters = {"updates": 1, "environment_steps": controls}
        if (type(previous_accepted) is not int or previous_accepted < 1
                or _digest(trainer.model.state_dict()) != origin["policy_weights_sha256"]):
            raise ValueError("resumed policy weights or accepted minibatch count differs from A")
    if (collector._closed or collector._active or collector.quarantined or collector.episode_done
            or collector._student_stereo is not None
            or collector.transitions != 0 or collector.consumed_sdk_steps != 0
            or trainer.num_envs != 1 or trainer.rollout_steps != controls
            or trainer.policy_version != counters["updates"] or trainer.counters != counters
            or trainer._pending is not None or len(trainer.rollout)
            or getattr(trainer, "_native_training_origin", FRESH_ORIGIN) != expected_origin
            or getattr(trainer, "_native_accepted_minibatches", 0) != previous_accepted):
        raise ValueError("fresh camera-free episode with the exact fresh or resumed trainer required")
    identity = checkpoint_identity(trainer, checkpoint_contract)
    scope = collector.sources.scope
    if (identity["training_contract"]["scope"] != scope or collector.journal.scope != scope):
        raise ValueError("checkpoint, collector and journal must share the same source scope")
    start = _clock(collector, controls=0, sdk_step=2, rod_tick=0)
    restored_rng = None
    if origin is not None:
        if start["episode_uid"] == origin["producer_episode_uid"]:
            raise ValueError("resumed update requires a distinct physical episode from A")
        restored_rng = _digest(_rng_state())
        if restored_rng != origin["rng_state_sha256"]:
            raise ValueError("restored optimizer RNG differs from A")
    try:
        training = run_updates(collector, additional_updates=1)
        end = _clock(collector, controls=controls, sdk_step=2 + physical_steps,
                     rod_tick=limits["max_internal_solver_substeps"])
        expected_counts = dict(updates=counters["updates"] + 1,
                               environment_steps=counters["environment_steps"] + controls)
        if (collector.transitions != controls or collector.consumed_sdk_steps != physical_steps
                or trainer.counters != expected_counts or trainer.policy_version != expected_counts["updates"]
                or training["loaded_updates"] != counters["updates"]
                or training["target_updates"] != expected_counts["updates"]
                or training["accepted_minibatch_updates"] != previous_accepted + training["metrics"]["minibatch_updates"]
                or any(end[key] != start[key] for key in ("run_uuid", "episode_uid", "rod_uid", "reset_id"))):
            raise TrainingStopped("exact one-episode control/SDK PPO update required")
        input_audit = {}
        if scope == "owner_source_unqualified":
            from .tabletop_native_teacher_actual_inputs import audit_teacher_input_sources

            input_audit["input_source_audit"] = audit_teacher_input_sources(
                collector.journal.path.read_bytes(), backend_profile=collector.owner.backend_profile,
                scope=scope, controls=controls,grasp=trainer.model.model_identity["grasp"])
        if (previous_checkpoint is not None
                and hashlib.sha256(previous_checkpoint.read_bytes()).hexdigest() != origin["checkpoint_sha256"]):
            raise TrainingStopped("original A checkpoint changed during the resumed update")
        audited = _seal_checkpoint(checkpoint, trainer, checkpoint_contract, expected_origin=expected_origin)
        result = dict(training, clean_checkpoint=dict(path=checkpoint.name, sha256=audited["checkpoint_sha256"],
                      policy_weights_sha256=audited["policy_weights_sha256"]),
                      physical=dict(start_sdk_step=start["physics_step"], end_sdk_step=end["physics_step"],
                                    start_rod_tick=start["rod_tick"], end_rod_tick=end["rod_tick"]),
                      native_qualification=False, teacher_admission=False, **input_audit)
        if restored_rng is not None:
            result.update(restored_rng_sha256=restored_rng, resume_origin=expected_origin)
        operation = "fresh_update_checkpoint_sealed" if origin is None else "resume_update_checkpoint_sealed"
        collector.journal.emit(operation, result)
        return result
    except BaseException:
        trainer._native_training_failed = True
        raise
