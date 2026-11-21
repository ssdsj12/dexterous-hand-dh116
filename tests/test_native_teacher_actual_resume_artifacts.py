"""Exercise the resumed artifact role with CPU fixtures only."""
import json

import pytest

from test_native_teacher_actual_artifacts import attempt, cpu_native_source_gate
from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_artifacts as artifacts


@pytest.mark.parametrize("damage", [None, "wrong_rng", "wrong_weights", "old_epoch",
                                    "wrong_cumulative", "missing_seal", "wrong_origin"])
def test_b_derives_role_specific_views_from_original_journal(attempt, damage):
    spec, output, rows, journal, lifecycle = attempt
    pin = spec["teacher_contract"]
    origin = dict(origin_scope="actual_resume_diagnostic_unqualified", checkpoint_sha256="a"*64,
                  policy_weights_sha256="b"*64, rng_state_sha256="c"*64,
                  loaded_updates=1, loaded_policy_version=1, loaded_environment_steps=32,
                  loaded_accepted_minibatch_updates=2, source_closure_sha256=pin["source_sha256"],
                  semantic_sha256=pin["semantic_sha256"], native_qualification=False,
                  teacher_admission=False)
    close_path = lifecycle / "result.json"
    close = json.loads(close_path.read_text())
    acquisition = close["acquisition"]
    acquisition.update(loaded_updates=1, target_updates=2, updates=2, accepted_minibatch_updates=3,
                       resume_origin=dict(kind="resume", checkpoint_sha256=origin["checkpoint_sha256"]),
                       restored_rng_sha256=origin["rng_state_sha256"],
                       native_qualification=False, teacher_admission=False)
    for row in rows:
        payload = row["payload"]
        if row["operation"] in ("collector_origin", "decision"):
            payload["policy"].update(policy_version=1, weights=origin["policy_weights_sha256"])
        elif row["operation"] == "policy_epoch":
            payload["policy_version"] = 2
        elif row["operation"] == "optimizer_update_accepted":
            payload.update(updates=2, accepted_minibatch_updates=3)
    if damage != "missing_seal":
        rows.append(dict(sequence=len(rows), scope="cpu_fixture",
                         operation="resume_update_checkpoint_sealed", payload=dict(acquisition)))
    if damage == "wrong_rng":
        acquisition["restored_rng_sha256"] = "e"*64
    elif damage == "wrong_weights":
        origin["policy_weights_sha256"] = "e"*64
    elif damage == "old_epoch":
        next(row for row in rows if row["operation"] == "policy_epoch")["payload"]["policy_version"] = 1
    elif damage == "wrong_cumulative":
        next(row for row in rows if row["operation"] == "optimizer_update_accepted")["payload"]["metrics"]["minibatch_updates"] = 3
    elif damage == "wrong_origin":
        acquisition["resume_origin"] = dict(kind="resume", checkpoint_sha256="e"*64)
    acquisition["physics_rows"] = 256
    close_path.write_text(json.dumps(close))
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original = journal.read_bytes()
    if damage:
        with pytest.raises(ValueError):
            artifacts.derive_actual_resume_artifacts(spec=spec, output=output, origin=origin)
        assert not (output / "training_result.json").exists()
    else:
        result = artifacts.derive_actual_resume_artifacts(spec=spec, output=output, origin=origin)
        readback = artifacts.freeze_actual_resume_artifacts(spec=spec, output=output, origin=origin)
        assert journal.read_bytes() == original and readback["raw_journal"] == original
        assert readback["rows"] == rows and readback["result"] == result
        assert result["native_qualification"] is result["teacher_admission"] is False
        assert readback["policy_epoch"]["policy_version"] == 2
        assert result["training"]["accepted_minibatch_updates"] == 3
        with pytest.raises(ValueError):
            artifacts.freeze_actual_fresh_artifacts(spec=spec, output=output)


@pytest.mark.parametrize("damage", [None, "a_role", "same_episode", "policy_zero", "mismatched_origin"])
def test_b_post_exit_audits_same_clock_and_new_episode(tmp_path, monkeypatch, damage):
    from types import SimpleNamespace
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_process as process
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_worker_artifacts as common
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile

    pin = dict(style="pinch",source_sha256="a"*64, semantic_sha256="b"*64, backend_sha256="c"*64,
               training_config={"path":"/fixed", "sha256":"d"*64},
               limits=dict(controls=8, inner_wall_timeout_s=180))
    spec = dict(teacher_contract=pin, budget=dict(physics_steps=66))
    origin = dict(producer_episode_uid="old_A:0", checkpoint_sha256="e"*64, producer_package="teacher.a",
                  receipt_sha256="f"*64, loaded_policy_version=1)
    role = "a_fresh" if damage == "a_role" else "b_resume"
    reference = {key:origin[key] for key in ("producer_package", "receipt_sha256")}
    envelope = process.ParentLaunchEnvelope(process.ENVELOPE_SCHEMA, "teacher.b", {}, {}, {},
        pin["source_sha256"], pin["semantic_sha256"], pin["training_config"], None, "/unused.sock", "new_B",
        100., 280., role, {}, origin)
    repo = SimpleNamespace(root=tmp_path,
        actual_launch=lambda package: dict(role="b_resume", checkpoint_origin=reference))
    package = {}
    monkeypatch.setattr(common, "_current_recipe_context", lambda *_: dict(
        package=package, package_name="teacher.b"))
    monkeypatch.setattr(artifacts, "freeze_actual_resume_artifacts", lambda **_: dict(
        rows=[], result=dict(clean_checkpoint=dict(sha256="1"*64), training=dict(updates=2),
        lifecycle_sha256={}), policy_epoch=dict(policy_version=2), checkpoint_bytes=b"CPU only", artifact_sha256={}))
    monkeypatch.setattr(artifacts, "_audit_fresh_setup", lambda _: dict(
        episode_uid=origin["producer_episode_uid"] if damage=="same_episode" else "new_B:0"))
    monkeypatch.setattr(artifacts, "_audit_fresh_budget", lambda *a, **kw: dict())
    monkeypatch.setattr(artifacts, "_audit_sealed_packs", lambda *a, **kw: dict())
    monkeypatch.setattr(BackendProfile, "load", lambda *_: None)
    import go2_pvcnn.control.dh116.tabletop_native_teacher_profile as profile
    monkeypatch.setattr(profile, "checked_file", lambda *_: b'{"episode_cap_s":2}')
    monkeypatch.setattr(common, "_audit_role_checkpoint", lambda *a, **kw: dict(
        expected_origin=kw["expected_origin"]), raising=False)

    def physical(rows, *, initial_policy_version, **kwargs):
        if initial_policy_version == 0 or damage == "policy_zero":
            raise ValueError("physical policy version differs")
        return dict(physical_nodes=41)

    monkeypatch.setattr(common, "_physical_journal", physical)
    if damage == "mismatched_origin":
        reference["receipt_sha256"] = "9"*64
    if damage:
        with pytest.raises(ValueError):
            artifacts.audit_actual_resume_outputs(spec, tmp_path, repository=repo,
                                                 package=package, envelope=envelope)
    else:
        report = artifacts.audit_actual_resume_outputs(spec, tmp_path, repository=repo,
                                                       package=package, envelope=envelope)
        assert report["native_qualification"] is report["teacher_admission"] is False
        assert report["checkpoint_audit"]["expected_origin"] == {"kind":"resume", "checkpoint_sha256":"e"*64}
