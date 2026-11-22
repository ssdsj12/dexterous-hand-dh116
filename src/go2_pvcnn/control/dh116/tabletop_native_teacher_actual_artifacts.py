"""Derive actual Worker evidence views from its closed original journals."""
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from .tabletop_native_teacher_worker_artifacts import PRIMARY_OPERATIONS, _journal, _json, _jsonl
from .tabletop_native_teacher_worker_contract import (
    RESULT_SCHEMA, SMOKE_PROFILE, CONTACT_PROFILE, _sha, validate_teacher_recipe_v2, validate_teacher_training_result,
)


_LIFECYCLE_FILES = ("result.json", "cleanup_stages.json", "app_close_returned.json",
                    "stop_commit.json", "ownership.jsonl")
_VIEWS = {"physics.jsonl": "physical", "transitions.jsonl": "transition_recorded", "ownership.jsonl": None}
_RESUME_RESULT_SCHEMA = "dh116-native-teacher-actual-resume-result-v1"
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


def _originals(spec, output):
    from scripts.run_dh116_work_packages import native_artifact

    validate_teacher_recipe_v2(spec)
    output = Path(output)
    if not output.is_dir() or output.resolve(strict=True) != output:
        raise ValueError("canonical Worker output directory required")
    lifecycle = {}
    for name in _LIFECYCLE_FILES:
        relative = "lifecycle/" + name
        try:
            lifecycle[name] = native_artifact(output, relative).read_bytes()
        except (OSError, ValueError) as error:
            raise ValueError("regular R06 evidence required: " + relative) from error
    if not lifecycle["ownership.jsonl"] or not lifecycle["ownership.jsonl"].endswith(b"\n"):
        raise ValueError("complete R06 ownership stream required")
    raw_name = spec["teacher_contract"]["journal_contract"]["raw_file"]
    raw = native_artifact(output, raw_name).read_bytes()
    checkpoint = native_artifact(output, "teacher_clean.pt").read_bytes()
    return lifecycle, raw, checkpoint


def _resume_result(result, acquisition, rows, origin, *, contract, checkpoint_sha):
    """Bind B's report to the exact loaded policy, counters and raw update seal."""
    if (type(origin) is not dict or origin.get("origin_scope") != "actual_resume_diagnostic_unqualified"
            or origin.get("native_qualification") is not False or origin.get("teacher_admission") is not False
            or origin.get("source_closure_sha256") != contract["source_sha256"]
            or origin.get("semantic_sha256") != contract["semantic_sha256"]
            or any(type(origin.get(key)) is not int or origin[key] != value for key, value in (
                ("loaded_updates", 1), ("loaded_policy_version", 1),
                ("loaded_environment_steps", contract["limits"]["controls"])))
            or any(not _sha(origin.get(key)) or origin[key]=="0"*64 for key in
                   ("checkpoint_sha256", "policy_weights_sha256", "rng_state_sha256"))):
        raise ValueError("verified same-contract actual A diagnostic origin required")
    expected_origin = dict(kind="resume", checkpoint_sha256=origin["checkpoint_sha256"])
    training = result["training"]
    accepted_before = origin.get("loaded_accepted_minibatch_updates")
    seal_rows = [row for row in rows if row["operation"] == "resume_update_checkpoint_sealed"]
    start_checks = dict(resume_origin=expected_origin, restored_rng_sha256=origin["rng_state_sha256"],
                        native_qualification=False, teacher_admission=False)
    if (type(accepted_before) is not int or accepted_before<1
            or any(type(training.get(key)) is not int or training[key] != value for key, value in (
                ("loaded_updates", 1), ("target_updates", 2), ("updates", 2)))
            or type(training.get("accepted_minibatch_updates")) is not int
            or training["accepted_minibatch_updates"] <= accepted_before
            or type(training.get("parameter_delta_l2")) not in (int, float)
            or not math.isfinite(training["parameter_delta_l2"]) or training["parameter_delta_l2"]<=0
            or result["clean_checkpoint"] != dict(path="teacher_clean.pt", sha256=checkpoint_sha)
            or not _sha(checkpoint_sha) or checkpoint_sha == origin["checkpoint_sha256"]
            or any(type(acquisition.get(key)) is not type(value) or acquisition[key] != value
                   for key, value in start_checks.items())
            or len(seal_rows) != 1 or any(row["operation"] == "fresh_update_checkpoint_sealed" for row in rows)
            or _json_line(seal_rows[0]["payload"]) != _json_line(
                {key:value for key,value in acquisition.items() if key!="physics_rows"})
            or "physics_rows" in acquisition and (type(acquisition["physics_rows"]) is not int
                or acquisition["physics_rows"] != contract["limits"]["controls"]*8)
            or any(row["sequence"] >= seal_rows[0]["sequence"] for row in rows
                   if row["operation"] == "optimizer_update_accepted")):
        raise ValueError("resumed result differs from the same-contract loaded state or original journal seal")
    result.update(schema=_RESUME_RESULT_SCHEMA, resume_origin=expected_origin,
                  restored_rng_sha256=origin["rng_state_sha256"])
    return _journal(rows, result, controls=contract["limits"]["controls"], initial_version=1,
                    prior_accepted=accepted_before, prior_weights=origin["policy_weights_sha256"],grasp=contract["style"],profile=contract["profile"])


def _derived_payloads(spec, lifecycle, raw, checkpoint_bytes, *, origin=None):
    acquisition = _normal_close(lifecycle)
    physical = acquisition.get("physical")
    contract = spec["teacher_contract"]
    bounds = contract["limits"]
    prime_steps = bounds["warmup_sdk_steps"] if contract["profile"]==CONTACT_PROFILE else 0
    expected_clock = dict(start_sdk_step=bounds["setup_sdk_steps"]+prime_steps, end_sdk_step=spec["budget"]["physics_steps"],
                          start_rod_tick=12*prime_steps, end_rod_tick=bounds["max_internal_solver_substeps"])
    if (type(physical) is not dict or any(type(physical.get(key)) is not int or physical[key] != value
                                         for key, value in expected_clock.items())):
        raise ValueError("R06 acquisition requires the complete fresh SDK and rod window")
    limits = bounds["jsonl_limits"]
    raw_name = contract["journal_contract"]["raw_file"]
    rows = _jsonl(raw, name=raw_name, maximum=limits[raw_name])
    checkpoint = acquisition.get("clean_checkpoint")
    if type(checkpoint) is not dict:
        raise ValueError("sealed fresh checkpoint reference required in R06 acquisition")
    checkpoint_sha = _digest(checkpoint_bytes)
    training = ("loaded_updates", "target_updates", "updates", "accepted_minibatch_updates", "parameter_delta_l2")
    result = dict(schema=RESULT_SCHEMA, semantic_sha256=contract["semantic_sha256"],
        training={name: acquisition.get(name) for name in training},
        clean_checkpoint={name: checkpoint.get(name) for name in ("path", "sha256")},
        process_exit_verified=False, native_qualification=False, teacher_admission=False,
        raw_journal_sha256=_digest(raw), lifecycle_sha256={name: _digest(data) for name, data in lifecycle.items()})
    if origin is None:
        validate_teacher_training_result(result, expected_semantic_sha256=contract["semantic_sha256"],
                                         artifact_sha256={"teacher_clean.pt": checkpoint_sha})
        epoch = _journal(rows, result, controls=bounds["controls"],grasp=contract["style"],profile=contract["profile"])
    else:
        epoch = _resume_result(result, acquisition, rows, origin, contract=contract,
                               checkpoint_sha=checkpoint_sha)

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
    return files, result, rows, epoch


def derive_actual_artifacts(*, spec, output):
    """Write fresh pre-exit Worker evidence. The raw journal and R06 stay intact."""
    return _derive(spec, output, origin=None)


def derive_actual_resume_artifacts(*, spec, output, origin):
    """Write resumed pre-exit evidence with distinct counters and origin."""
    if type(origin) is not dict:
        raise ValueError("actual A origin required for B evidence")
    return _derive(spec, output, origin=origin)


def _derive(spec, output, *, origin):
    output = Path(output)
    for name in (*_VIEWS, "shutdown_complete.json", "training_result.json"):
        if os.path.lexists(output / name):
            raise FileExistsError("derived output already exists: " + name)
    files, result, _, _ = _derived_payloads(spec, *_originals(spec, output), origin=origin)
    _publish(output, files)
    return result


def freeze_actual_fresh_artifacts(*, spec, output):
    """Read each file once and recompute the A views over those immutable bytes."""
    return _freeze(spec, output, origin=None)


def freeze_actual_resume_artifacts(*, spec, output, origin):
    """Read B's immutable view bytes and recompute its separate result contract."""
    if type(origin) is not dict:
        raise ValueError("actual A origin required for B evidence")
    return _freeze(spec, output, origin=origin)


def _freeze(spec, output, *, origin):
    from scripts.run_dh116_work_packages import native_artifact

    output = Path(output)
    lifecycle, raw, checkpoint = _originals(spec, output)
    expected, result, rows, epoch = _derived_payloads(spec, lifecycle, raw, checkpoint, origin=origin)
    digests = {}
    for name, canonical in expected.items():
        saved = native_artifact(output, name).read_bytes()
        if saved != canonical:
            raise ValueError("derived artifact differs from frozen original evidence: " + name)
        digests[name] = _digest(saved)
    digests[spec["teacher_contract"]["journal_contract"]["raw_file"]] = result["raw_journal_sha256"]
    digests["teacher_clean.pt"] = result["clean_checkpoint"]["sha256"]
    return dict(result=result, rows=rows, raw_journal=raw, checkpoint_bytes=checkpoint,
                artifact_sha256=digests, policy_epoch=epoch)


def _audit_sealed_packs(frozen, profile, *, controls=32, seal_operation="fresh_update_checkpoint_sealed",grasp="pinch",
                        context_mode=SMOKE_PROFILE):
    from .tabletop_native_teacher_actual_inputs import SCOPE, audit_teacher_input_sources

    seams = [(index, row["payload"]) for index, row in enumerate(frozen["rows"])
             if row["operation"] == seal_operation]
    if len(seams) != 1:
        raise ValueError("source ledger requires exactly one checkpoint seal")
    index, seal = seams[0]
    checkpoint = seal.get("clean_checkpoint")
    if (type(checkpoint) is not dict or any(checkpoint.get(key) != value
            for key, value in frozen["result"]["clean_checkpoint"].items())):
        raise ValueError("checkpoint differs from the original journal seal")
    prefix = b"".join(frozen["raw_journal"].splitlines(keepends=True)[:index])
    rebuilt = audit_teacher_input_sources(prefix, backend_profile=profile, scope=SCOPE, controls=controls,grasp=grasp,profile=context_mode)
    if rebuilt != seal.get("input_source_audit"):
        raise ValueError("source audit differs from the recorded checkpoint seal")
    return rebuilt



def _audit_fresh_setup(rows, *, profile=SMOKE_PROFILE, end_sdk_step=258):
    from .tabletop_deformable_dual_evidence import _Sequence, _clock_row
    from .tabletop_dual_hand_reset import _no_step
    from .tabletop_deformable_r0_replay import decode_owned
    from .tabletop_deformable_bimanual_step import DT

    setup = [r for r in rows if r["operation"] == "setup"]
    reset = [r for r in rows if r["operation"] == "owner_event" and r["payload"].get("kind") == "reset"]
    first_action = next((r for r in rows if r["operation"] == "decision"), None)
    first_physical = next((r for r in rows if r["operation"] == "physical"), None)
    if (len(setup) != 2 or len(reset) != 1 or first_action is None or first_physical is None
            or not setup[0]["sequence"] < setup[1]["sequence"] < reset[0]["sequence"]
                < first_action["sequence"] < first_physical["sequence"]):
        raise ValueError("ordered SDK setup and exactly one Reset before the first action required")
    try:
        sequence = _Sequence(end_sdk_step, 12)
        for row in setup:
            sequence.accept("setup", row["payload"])
        data = decode_owned(reset[0]["payload"])
        sequence.accept("reset", data)
        receipt = data["reset"]
        before, after = receipt["before_clock"], receipt["after_actual_clocks"]
        _no_step(before, after)
        last_setup = setup[-1]["payload"]
        if (any(type(row["sdk_physics_step"]) is not int or row["sdk_physics_step"] != 2
                or row["sdk_timestamp_s"] != last_setup["sdk_time_s"] or row["sdk_dt_s"] != DT
                for row in (before, after))
                or any(type(receipt[key]) is not int or receipt[key] != 0
                       for key in ("new_sdk_steps", "new_solver_steps"))
                or after["rod"] != receipt["after_clock"]
                or data["clearance_passed"] is not True):
            raise ValueError("actual SDK setup and zero-step Reset readbacks differ")
        expected_policy_clock=sequence.clock
        if profile==CONTACT_PROFILE:
            from .tabletop_native_teacher_worker_artifacts import _contact_prime
            prime=_contact_prime(rows,end_sdk_step=end_sdk_step)
            expected_policy_clock=prime["clock"]
        elif profile!=SMOKE_PROFILE:
            raise ValueError("known setup profile required")
        pre = first_physical["payload"]["exchange"]["action_readback"]["pre"]["identity"]
        if len(pre) != 1 or _clock_row(pre[0]) != expected_policy_clock:
            raise ValueError("Reset identity differs from the first executed pre-action boundary")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("incomplete actual setup and Reset readback") from error
    result = dict(setup_sdk_steps=sequence.setup, episode_reset_sdk_steps=receipt["new_sdk_steps"],
                episode_uid=sequence.clock["episode_uid"], reset_id=sequence.clock["reset_id"])
    if profile==CONTACT_PROFILE:
        result.update(source_priming_sdk_steps=1,first_policy_sdk_step=prime["clock"]["physics_step"])
    return result



def _audit_fresh_budget(rows, envelope, *, episode_uid, episode_cap_s, local_cap_s, end_sdk_step=258,
                        role_prefix="fresh", profile=SMOKE_PROFILE):

    samples = [row for row in rows if row["operation"] == role_prefix + "_update_wall_budget"]
    decisions = [row["sequence"] for row in rows if row["operation"] == "decision"]
    seals = [row["sequence"] for row in rows if row["operation"] == role_prefix + "_update_checkpoint_sealed"]
    if (len(samples) != 2 or not decisions or len(seals) != 1
            or not samples[0]["sequence"] < decisions[0] <= decisions[-1] < seals[0] < samples[1]["sequence"]
            or episode_uid != envelope.nonce + ":0"):
        raise ValueError("parent-bound wall budget samples must bracket the fresh update")
    deadline = envelope.deadline_monotonic_s
    before, after = [row["payload"] for row in samples]
    start_sdk_step=3 if profile==CONTACT_PROFILE else 2
    if profile not in (SMOKE_PROFILE,CONTACT_PROFILE):
        raise ValueError("known wall budget profile required")
    for row, phase, step in zip((before, after), ("before_update", "after_update"), (start_sdk_step, end_sdk_step)):
        numbers = ("episode_cap_s", "local_cap_s", "observed_s", "started_s", "parent_deadline_s", "remaining_s")
        if (any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) for k in numbers)
                or row.get("phase") != phase or row.get("episode_uid") != episode_uid
                or type(row.get("sdk_step")) is not int or row["sdk_step"] != step
                or row["episode_cap_s"] != episode_cap_s or row["local_cap_s"] != local_cap_s
                or row["parent_deadline_s"] != deadline
                or not envelope.issued_monotonic_s <= row["started_s"] <= row["observed_s"] < deadline):
            raise ValueError("fresh update budget identity, cap or parent deadline differs")
        remaining = min(local_cap_s - (row["observed_s"] - row["started_s"]), deadline - row["observed_s"])
        if remaining <= 0. or not math.isclose(row["remaining_s"], remaining, abs_tol=1e-9, rel_tol=0.):
            raise ValueError("fresh update budget differs from the live local and parent clocks")
    if before["started_s"] != after["started_s"] or before["observed_s"] > after["observed_s"]:
        raise ValueError("fresh update budget clocks do not belong to one session")
    return dict(parent_deadline_s=deadline, episode_cap_s=episode_cap_s, session_local_cap_s=local_cap_s,
                remaining_before_s=before["remaining_s"], remaining_after_s=after["remaining_s"])


def audit_actual_fresh_outputs(spec, output, *, repository, package, envelope):
    """Audit fresh Pinch output after the parent has observed the real exit.

    Full admission also requires independent receipt validation.
    The returned data can describe a complete measured window while it remains
    ineligible for publishing a qualified native receipt.
    """
    return _audit_actual_outputs(spec, output, repository=repository, package=package,
                                 envelope=envelope, role="a_fresh")


def audit_actual_resume_outputs(spec, output, *, repository, package, envelope):
    """Audit B only after its outer process exit and parent origin readback."""
    return _audit_actual_outputs(spec, output, repository=repository, package=package,
                                 envelope=envelope, role="b_resume")


def _audit_actual_outputs(spec, output, *, repository, package, envelope, role):
    from .tabletop_native_teacher_actual_process import ParentLaunchEnvelope
    from .tabletop_native_teacher_profile import BackendProfile, checked_file
    from .tabletop_native_teacher_worker_artifacts import (
        _audit_fresh_checkpoint, _audit_role_checkpoint, _current_recipe_context, _physical_journal,
    )

    context = _current_recipe_context(spec, repository, package)
    launch = repository.actual_launch(context["package"])
    if launch is None or launch["role"] != role:
        raise ValueError("matching registered actual companion role required for this audit")
    pin = spec["teacher_contract"]
    if (type(envelope) is not ParentLaunchEnvelope or envelope.role != role
            or envelope.package_name != context["package_name"] or
            envelope.semantic_sha256 != pin["semantic_sha256"] or envelope.core_source_sha256 != pin["source_sha256"]
            or envelope.frozen_config != pin["training_config"]):
        raise ValueError("parent launch envelope differs from the approved actual budget")
    config = json.loads(checked_file(envelope.frozen_config["path"], envelope.frozen_config["sha256"]))
    origin = envelope.checkpoint_origin
    if role == "a_fresh":
        if origin is not None or launch["checkpoint_origin"] is not None:
            raise ValueError("fresh output audit rejects a resume origin")
        frozen = freeze_actual_fresh_artifacts(spec=spec, output=output)
        prefix, initial_version = "fresh", 0
    else:
        reference = launch["checkpoint_origin"]
        if (type(origin) is not dict or type(reference) is not dict
                or set(reference) != {"producer_package", "receipt_sha256"}
                or any(origin.get(key) != reference[key] for key in reference)):
            raise ValueError("resumed output origin differs from the registered A reference")
        frozen = freeze_actual_resume_artifacts(spec=spec, output=output, origin=origin)
        prefix, initial_version = "resume", 1
    setup = _audit_fresh_setup(frozen["rows"],profile=pin["profile"],end_sdk_step=spec["budget"]["physics_steps"])
    if origin is not None and setup["episode_uid"] == origin["producer_episode_uid"]:
        raise ValueError("resumed output must contain a distinct physical Reset from A")
    budget = _audit_fresh_budget(frozen["rows"], envelope, episode_uid=setup["episode_uid"],
        episode_cap_s=config["episode_cap_s"], local_cap_s=pin["limits"]["inner_wall_timeout_s"],
        end_sdk_step=spec["budget"]["physics_steps"], role_prefix=prefix,profile=pin["profile"])
    physical = _physical_journal(frozen["rows"], reward_scope="unqualified_source",
        expected_backend_sha256=pin["backend_sha256"], controls=pin["limits"]["controls"],
        initial_policy_version=initial_version,profile=pin["profile"])
    if physical["physical_nodes"] != 41:
        raise ValueError("actual Pinch Worker requires 41 measured material nodes")
    backend_profile = BackendProfile.load(repository.root)
    model_input = _audit_sealed_packs(frozen, backend_profile,
        controls=pin["limits"]["controls"], seal_operation=prefix + "_update_checkpoint_sealed",grasp=pin["style"],context_mode=pin["profile"])
    result = frozen["result"]
    checkpoint_args = dict(checkpoint_bytes=frozen["checkpoint_bytes"],
        checkpoint_binding=result["clean_checkpoint"], training=result["training"], epoch=frozen["policy_epoch"],
        include_rng_digest=True)
    if origin is None:
        checkpoint = _audit_fresh_checkpoint(spec, repository, context, **checkpoint_args)
    else:
        checkpoint = _audit_role_checkpoint(spec, repository, context, **checkpoint_args,
            expected_origin=dict(kind="resume", checkpoint_sha256=origin["checkpoint_sha256"]),
            environment_windows=2)
    from .tabletop_native_teacher_lift_trial import audit_recorded_lift, read_lift_trial
    from .tabletop_native_teacher_preflight import _binding
    from .tabletop_training_reward import load_config

    reward_path, _ = _binding(config["preflight"]["reward_config"], repository.root)
    reward_config = load_config(reward_path)
    lift_trial = read_lift_trial(reward_config)
    lift_audit = {}
    if lift_trial is not None:
        lift_audit["lift_reward_audit"] = audit_recorded_lift(
            frozen["rows"], reward_config, lift_trial, backend_profile=backend_profile)
    return dict(artifact_sha256=frozen["artifact_sha256"], lifecycle_sha256=result["lifecycle_sha256"],
        teacher_training=result["training"], physical_evidence=physical, checkpoint_audit=checkpoint,
        input_source_audit=model_input, physical_to_model_snapshot_binding_verified=True,
        setup_evidence=setup, budget_evidence=budget, actual_setup_source_verified=True, cap_source_verified=True,
        native_ready=False, native_qualification=False, teacher_admission=False, **lift_audit)
