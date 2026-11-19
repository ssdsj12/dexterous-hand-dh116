"""Derive the fresh Worker's evidence views from its closed original journals."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from .tabletop_native_teacher_worker_artifacts import PRIMARY_OPERATIONS, _journal, _json, _jsonl
from .tabletop_native_teacher_worker_contract import (
    RESULT_SCHEMA, validate_teacher_recipe_v2, validate_teacher_training_result,
)


_LIFECYCLE_FILES = ("result.json", "cleanup_stages.json", "app_close_returned.json",
                    "stop_commit.json", "ownership.jsonl")
_VIEWS = {"physics.jsonl": "physical", "transitions.jsonl": "transition_recorded", "ownership.jsonl": None}
_CLOSE_STAGES = ("stop", "manager_before_close", "ownership_before_close", "close", "close_returned_marker")


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_line(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _normal_close(raw):
    result, ledger, returned, stop = (_json(raw[name]) for name in _LIFECYCLE_FILES[:-1])
    if not all(type(record) is dict for record in (result, ledger, returned, stop)):
        raise ValueError("complete R06 close records required")
    stages = ledger.get("stages")
    if (not isinstance(stages, dict) or any(
            not isinstance(stages.get(name), dict) or stages[name].get("attempted") is not True
            or stages[name].get("returned") is not True or stages[name].get("error") is not None
            for name in _CLOSE_STAGES)
            or any(record.get("close_mode") != "normal" for record in (result, ledger, returned))
            or any(record.get("failure", True) is not None for record in (result, ledger))
            or result.get("lifetime_released_before_close") is not True
            or result.get("native_qualification") is not False
            or result.get("process_exit_verified") is not False
            or ledger.get("qualification_granted") is not False
            or returned.get("app_close_returned") is not True
            or returned.get("qualification_granted") is not False
            or returned.get("process_exit_verified") is not False
            or stop.get("stop_commit_returned") is not True
            or type(stop.get("explicit_physics_steps_requested")) is not int
            or stop["explicit_physics_steps_requested"] != 0):
        raise ValueError("R06 normal close, released ownership and zero advancing STOP required")
    acquired = result.get("acquisition")
    if type(acquired) is not dict:
        raise ValueError("R06 completed acquisition result missing")
    return acquired


def _publish(output, files):
    # All validation and staging complete before linking any derived file.
    # Link the result last. Exclusive linking cannot erase existing evidence.
    with TemporaryDirectory(dir=output, prefix=".derived-") as directory:
        staging = Path(directory)
        for name, raw in files.items():
            with (staging / name).open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        for name in files:
            os.link(staging / name, output / name)
        handle = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)


def derive_actual_artifacts(*, spec, output):
    """Write fresh pre-exit Worker evidence. The raw journal and R06 stay intact."""
    from scripts.run_dh116_work_packages import native_artifact

    validate_teacher_recipe_v2(spec)
    output = Path(output)
    if not output.is_dir() or output.resolve(strict=True) != output:
        raise ValueError("canonical Worker output directory required")
    for name in (*_VIEWS, "shutdown_complete.json", "training_result.json"):
        if os.path.lexists(output / name):
            raise FileExistsError("derived output already exists: " + name)
    lifecycle = {}
    for name in _LIFECYCLE_FILES:
        relative = "lifecycle/" + name
        try:
            lifecycle[name] = native_artifact(output, relative).read_bytes()
        except (OSError, ValueError) as error:
            raise ValueError("regular R06 evidence required: " + relative) from error
    if not lifecycle["ownership.jsonl"] or not lifecycle["ownership.jsonl"].endswith(b"\n"):
        raise ValueError("complete R06 ownership stream required")
    acquisition = _normal_close(lifecycle)
    physical = acquisition.get("physical")
    expected_clock = dict(start_sdk_step=2, end_sdk_step=258, start_rod_tick=0, end_rod_tick=3072)
    if (type(physical) is not dict or any(type(physical.get(key)) is not int or physical[key] != value
                                         for key, value in expected_clock.items())):
        raise ValueError("R06 acquisition requires the complete fresh SDK and rod window")
    contract = spec["teacher_contract"]
    limits = contract["limits"]["jsonl_limits"]
    raw_name = contract["journal_contract"]["raw_file"]
    raw = native_artifact(output, raw_name).read_bytes()
    rows = _jsonl(raw, name=raw_name, maximum=limits[raw_name])
    checkpoint = acquisition.get("clean_checkpoint")
    if type(checkpoint) is not dict:
        raise ValueError("sealed fresh checkpoint reference required in R06 acquisition")
    checkpoint_sha = _digest(native_artifact(output, "teacher_clean.pt").read_bytes())
    training = ("loaded_updates", "target_updates", "updates", "accepted_minibatch_updates", "parameter_delta_l2")
    result = dict(schema=RESULT_SCHEMA, semantic_sha256=contract["semantic_sha256"],
        training={name: acquisition.get(name) for name in training},
        clean_checkpoint={name: checkpoint.get(name) for name in ("path", "sha256")},
        process_exit_verified=False, native_qualification=False, teacher_admission=False,
        raw_journal_sha256=_digest(raw), lifecycle_sha256={name: _digest(data) for name, data in lifecycle.items()})
    validate_teacher_training_result(result, expected_semantic_sha256=contract["semantic_sha256"],
                                     artifact_sha256={"teacher_clean.pt": checkpoint_sha})
    _journal(rows, result)

    files = {}
    for name, operation in _VIEWS.items():
        filtered = [row for row in rows if (row["operation"] == operation if operation is not None
                                           else row["operation"] not in PRIMARY_OPERATIONS)]
        expected = spec["validator"]["line_counts"].get(name)
        if len(filtered) > limits[name] or (expected is not None and len(filtered) != expected):
            raise ValueError("derived journal view exceeds required scope or row budget: " + name)
        files[name] = b"".join(_json_line(dict(raw_sequence=row["sequence"], payload=row["payload"]))
                               for row in filtered)
    files["shutdown_complete.json"] = _json_line(dict(close_returned=True, process_exit_verified=False,
        pending_simulation_callback_error=None, callback_scope="observed_R06_STOP_and_close",
        lifecycle_sha256=result["lifecycle_sha256"], qualification_granted=False))
    files["training_result.json"] = _json_line(result)
    _publish(output, files)
    return result
