"""Production selection and sealed candidate loading, without an Isaac process."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

MODULE = "go2_pvcnn.control.dh116.tabletop_cuda_geometry"


def production():
    assert importlib.util.find_spec(MODULE) is not None, "production CUDA geometry factory is missing"
    return importlib.import_module(MODULE)


def owner():
    names = tuple("link"+str(i) for i in range(23))
    grid = dict(values=np.indices((3, 3, 3))[0]*.01, origin=np.zeros(3), spacing=np.array(.01))
    return SimpleNamespace(_geometry={s: {"source": {"link0": np.zeros((2, 3))}}
        for s in ("left", "right")}, REQUIRED_PADS=("link0",),
        _grids={s: {"link0": grid} for s in ("left", "right")},
        sources={s: {"sdf_binding": {"link0": {"sha256": "a"*64}}} for s in ("left", "right")},
        _sdf=lambda *a, **k: None, _pose_matrix=lambda *a: None, _transform=lambda *a: None), (names, names)


def test_production_factory_exists_and_default_preserves_eager(monkeypatch):
    mod = production()
    monkeypatch.delenv("DH116_NATIVE_READBACK_BACKEND", raising=False)
    monkeypatch.delenv("DH116_TENSOR_GEOMETRY_BACKEND", raising=False)
    source, names = owner()
    geometry = mod.make_tensor_geometry(source, names, device="cpu")
    assert type(geometry).__name__ == "TensorGeometry"
    assert geometry.backend == "eager"
    assert geometry.source_bindings == {(s, "link0"): {"sha256": "a"*64} for s in (0, 1)}
    assert mod.geometry_measurement_identity(geometry)["actual_measurement_backend"] == "eager"
    assert mod.geometry_measurement_identity(None)["actual_measurement_backend"] is None


def test_factory_selects_custom_measurement_with_explicit_eager_guard(monkeypatch):
    mod = production()
    source, names = owner()
    calls = []
    class Candidate:
        @classmethod
        def from_policy_geometry(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return "candidate"
    monkeypatch.setattr(mod, "CudaMeasurementGeometry", Candidate)
    monkeypatch.setenv("DH116_TENSOR_GEOMETRY_BACKEND", "cuda_cabi")
    assert mod.make_tensor_geometry(source, names, device="cuda:0") == "candidate"
    assert calls == [((source, names), dict(device="cuda:0", backend="eager"))]
    with pytest.raises(ValueError, match="backend"):
        mod.make_tensor_geometry(source, names, device="cpu", backend="invalid")
    assert mod.make_tensor_geometry(source, names, device="cpu", backend="eager").backend == "eager"


def test_compiled_factory_is_disabled_without_calling_compiler(monkeypatch):
    mod = production()
    source, names = owner()
    monkeypatch.setattr(torch, "compile", lambda *a, **k: pytest.fail("runtime compiler called"))
    with pytest.raises(ValueError, match="compiled.*disabled"):
        mod.make_tensor_geometry(source, names, device="cpu", backend="compiled")


def test_checked_profile_and_reject_tampered_manifest(tmp_path):
    mod = production()
    profile = mod.CudaGeometryProfile.load()
    assert len(profile.binary_sha256) == 64
    assert profile.native_ready is False and profile.teacher_admission is False
    assert profile.verify() is profile
    manifest = json.loads(mod.DEFAULT_MANIFEST.read_text())
    for key, value in (("schema", "wrong"), ("sm", 80), ("version", "wrong"),
                       ("fmad", True), ("native_ready", True), ("teacher_admission", True),
                       ("binary_sha256", "0"*64)):
        bad = dict(manifest, **{key: value})
        path = tmp_path/(key+".json")
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            mod.CudaGeometryProfile.load(path)
    bad = dict(manifest)
    bad["repo_source_sha256"] = dict(manifest["repo_source_sha256"])
    key = next(iter(bad["repo_source_sha256"]))
    bad["repo_source_sha256"][key] = "0"*64
    path = tmp_path/"source.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="source.*SHA"):
        mod.CudaGeometryProfile.load(path)
    bad = dict(manifest)
    bad["source_preservation_evidence"] = dict(manifest["source_preservation_evidence"], sha256="0"*64)
    path = tmp_path/"guard-proof.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="guard preservation.*SHA"):
        mod.CudaGeometryProfile.load(path)


def test_custom_constructor_rejects_cpu_and_non_eager_guards():
    mod = production()
    source, names = owner()
    with pytest.raises(ValueError, match="CUDA"):
        mod.CudaMeasurementGeometry.from_policy_geometry(source, names, device="cpu", backend="eager")
    with pytest.raises(ValueError, match="eager"):
        mod.CudaMeasurementGeometry([], device="cpu", backend="compiled")


def test_collector_reuses_cached_actual_radius_tensor():
    from go2_pvcnn.control.dh116.tabletop_tensor_collector import NativeTensorCollector
    c = NativeTensorCollector.__new__(NativeTensorCollector)
    c.radius = .005
    c.radii = torch.full((2,), c.radius, dtype=torch.float64)
    c.state = {"link_pose_xyzw": torch.zeros(2, 2, 23, 7)}
    c.rod_state = lambda: (torch.zeros(2, 41, 3), torch.zeros(2, 41, 3))
    seen = []
    def capture(poses, nodes, *, frame, radius):
        seen.append(radius)
        return SimpleNamespace(source_complete=torch.ones(2, dtype=torch.bool),
            geometry_passed=torch.ones(2, dtype=torch.bool), positive_pad_mean_m=torch.ones(2, 2))
    c.geometry = SimpleNamespace(capture=capture)
    c.distance("one")
    c.distance("two", reset=True)
    assert seen[0] is c.radii and seen[1] is c.radii


def test_reset_still_rejects_changed_actual_radius():
    from go2_pvcnn.control.dh116.tabletop_tensor_collector import NativeTensorCollector
    c = NativeTensorCollector.__new__(NativeTensorCollector)
    c.radius = .005
    c.source = SimpleNamespace(_boundary=lambda: None)
    c.runtime = SimpleNamespace(solver=SimpleNamespace(parameter_snapshot=lambda: {"contact_radius_m": .005001}))
    with pytest.raises(ValueError, match="Reset changed.*radius"):
        c._reset_refs((0,))


def test_new_build_directory_cannot_replace_history(tmp_path):
    from scripts.build_dh116_cuda_geometry_candidate import build
    # Existing directory must be rejected before invoking any compiler.
    with pytest.raises(ValueError, match="new independent"):
        build(tmp_path, Path(__file__))


def test_unqualified_cuda_geometry_cannot_admit_ppo(tmp_path):
    import os
    import subprocess
    import sys
    args = [sys.executable, "-B", "-m", "scripts.train_dh116_tensor_native_teacher",
        "--grasp", "pinch", "--num-envs", "128", "--gpu", "3", "--receipt", "unopened.json",
        "--receipt-sha256", "a"*64, "--output", str(tmp_path/"forbidden"), "--updates", "2"]
    result = subprocess.run(args, env=dict(os.environ, CUDA_VISIBLE_DEVICES="",
        DH116_TENSOR_GEOMETRY_BACKEND="cuda_cabi"), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode == 2
    assert "CUDA geometry requires verified native lifecycle evidence before PPO" in result.stdout
    assert not (tmp_path/"forbidden").exists()


def diagnostic_receipts(tmp_path, monkeypatch):
    """Synthetic unit fixtures, never accepted as real project evidence."""
    import hashlib
    mod = production()
    from go2_pvcnn.control.dh116.tabletop_runtime_source_audit import SOURCE_NAMES
    identity = dict(actual_measurement_backend="cuda_cabi", binary_sha256="a"*64,
        manifest_sha256="b"*64, source_sha256={"fixture.py": "c"*64},
        binary_path="/unit-fixture-only.so", native_ready=False, teacher_admission=False)
    profile = SimpleNamespace(identity=lambda: identity)
    profile.verify = lambda: profile
    monkeypatch.setattr(mod.CudaGeometryProfile, "load", lambda: profile)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipts = []
    for count in (2, 128):
        parent = tmp_path/str(count)
        native = parent/"native"
        native.mkdir(parents=True)
        steps = 218 if count == 2 else 18
        actual = dict(style="pinch", measurement_scope="bounded_diagnostic", updates=0,
            native_readback_mode="raw", source_cache_metadata=None,
            geometry_measurement=identity, actual_measurement_backend="cuda_cabi",
            environments=count, controls=27 if count == 2 else 2, sdk_step=steps,
            teacher_admission=False, grasp_qualified=False,
            reset_ids=[2, 1] if count == 2 else [1]*128, window_seconds=[.1]*(27 if count == 2 else 2))
        if count == 2:
            actual.update(ppo_rollout_rows=0, ppo_rollout_admission=False,
                contact_coverage=dict(hand_pair_samples=[524, 0], nonzero_captured_hand_steps=[14, 0],
                    nonzero_applied_hand_steps=[14, 0]),
                reset_checks=[dict(sdk_step=210, selected=[0], peers=[1], native_mesh_verified=True, reset_ids=[2, 1])],
                post_reset_next_sdk=dict(sdk_step=211, source_sdk_step=210, reset_ids=[2, 1], rod_ticks_pre=[0, 2496],
                    applied=dict(nonzero_hand_force=[False, False])))
        else:
            actual["throughput_probe_only"] = True
        result = dict(acquisition=actual, physics_steps=steps, failure=None, close_mode="normal",
            lifetime_released_before_close=True, teacher_admission=False)
        (native/"result.json").write_text(json.dumps(result))
        (native/"app_close_returned.json").write_text(json.dumps(dict(app_close_returned=True, close_mode="normal")))
        sources = {name: sha(mod.ROOT/name) for name in SOURCE_NAMES}
        attempt = parent/"attempt.json"
        attempt.write_text(json.dumps(dict(unit_fixture_only=True, source_sha256=sources)))
        (native/"source_freeze_audit.json").write_text(json.dumps(dict(
            version="dh116-attempt-source-freeze-audit-v1", attempt_sha256=sha(attempt),
            admissible=True, drift=[], source_sha256=sources, startup_checked=True,
            process_exit_verified=False, qualification_granted=False)))
        (native/"timeline_release.json").write_text(json.dumps(dict(
            version="dh116-local-timeline-release-v1", stop_commit_returned=True,
            local_timeline_weakref_supported=True, local_timeline_weakref_alive=False,
            local_timeline_release_verified=True, native_release_verified=False,
            process_exit_verified=False, qualification_granted=False)))
        (parent/"native.log").write_text("unit fixture, no process executed\n")
        outer = dict(gdb_exit_code=0, inferior_status=dict(exit_code=0, signal=None), forced_exit=False,
            log_sha256=sha(parent/"native.log"), artifacts={name: sha(native/name)
                for name in ("result.json", "app_close_returned.json", "source_freeze_audit.json", "timeline_release.json")})
        outer["artifacts"]["attempt.json"] = sha(attempt)
        path = parent/"outer.json"
        path.write_text(json.dumps(outer))
        receipts.append((path, sha(path)))
    return mod, receipts, sha


def test_receipt_gate_requires_both_actual_diagnostics(tmp_path, monkeypatch):
    mod, receipts, _ = diagnostic_receipts(tmp_path, monkeypatch)
    result = mod.verify_measurement_native_receipts(receipts, style="pinch")
    assert result["scope"] == "bounded_diagnostic"
    assert result["native_ready"] is False and result["teacher_admission"] is False
    with pytest.raises(ValueError, match="N2.*N128"):
        mod.verify_measurement_native_receipts([receipts[0], receipts[0]], style="pinch")
    with pytest.raises(ValueError, match="Pinch"):
        mod.verify_measurement_native_receipts(receipts, style="power")


@pytest.mark.parametrize("case", ("signal11", "fake_none", "forced", "no_exit", "wrong_style", "wrong_profile",
    "no_contact", "no_reset", "no_continuation", "artifact_sha", "no_close", "admission", "wrong_readback"))
def test_receipt_gate_rejects_failed_or_mismatched_evidence(tmp_path, monkeypatch, case):
    mod, receipts, sha = diagnostic_receipts(tmp_path, monkeypatch)
    path, _ = receipts[0]
    outer = json.loads(path.read_text())
    result_path = path.parent/"native/result.json"
    result = json.loads(result_path.read_text())
    actual = result["acquisition"]
    if case == "signal11": outer["inferior_status"]["signal"] = "SIGSEGV"
    if case == "fake_none": outer["inferior_status"]["signal"] = "None"
    if case == "forced": outer["forced_exit"] = True
    if case == "no_exit": outer["inferior_status"].pop("exit_code")
    if case == "wrong_style": actual["style"] = "power"
    if case == "wrong_profile": actual["geometry_measurement"]["manifest_sha256"] = "d"*64
    if case == "no_contact": actual["contact_coverage"]["hand_pair_samples"] = [0, 0]
    if case == "no_reset": actual["reset_checks"] = []
    if case == "no_continuation": actual["post_reset_next_sdk"] = None
    if case == "no_close": result["close_mode"] = "force"
    if case == "admission": actual["teacher_admission"] = True
    if case == "wrong_readback": actual["native_readback_mode"] = "official_cache"
    result_path.write_text(json.dumps(result))
    outer["artifacts"]["result.json"] = "0"*64 if case == "artifact_sha" else sha(result_path)
    path.write_text(json.dumps(outer))
    receipts[0] = path, sha(path)
    with pytest.raises(ValueError):
        mod.verify_measurement_native_receipts(receipts, style="pinch")


def test_raw_receipts_cannot_qualify_official_cache(tmp_path, monkeypatch):
    mod, receipts, _ = diagnostic_receipts(tmp_path, monkeypatch)
    monkeypatch.setenv("DH116_NATIVE_READBACK_BACKEND", "official_cache")
    with pytest.raises(ValueError, match="readback"):
        mod.verify_measurement_native_receipts(receipts, style="pinch")


def cache_diagnostic_receipts(tmp_path, monkeypatch):
    mod, receipts, sha = diagnostic_receipts(tmp_path, monkeypatch)
    monkeypatch.setenv("DH116_NATIVE_READBACK_BACKEND", "official_cache")
    for index, (path, _) in enumerate(receipts):
        result_path = path.parent/"native/result.json"
        result = json.loads(result_path.read_text())
        actual = result["acquisition"]
        actual.update(native_readback_mode="official_cache", source_cache_metadata={"mode": "official_cache", "sdk": "unit-fixture"})
        actual["readback_audit"] = dict(enabled=index == 0, diagnostic_only=True,
            side_snapshots=42 if index == 0 else 0, mismatches=[], extra_raw_reads_per_side_snapshot=2 if index == 0 else 0)
        result_path.write_text(json.dumps(result))
        outer = json.loads(path.read_text())
        outer["artifacts"]["result.json"] = sha(result_path)
        path.write_text(json.dumps(outer))
        receipts[index] = path, sha(path)
    return mod, receipts, sha


def test_cache_receipts_require_n2_dual_raw_oracle_but_n128_can_time(tmp_path, monkeypatch):
    mod, receipts, _ = cache_diagnostic_receipts(tmp_path, monkeypatch)
    result = mod.verify_measurement_native_receipts(receipts, style="pinch")
    assert result["native_readback_mode"] == "official_cache"
    assert result["native_ready"] is False and result["teacher_admission"] is False


@pytest.mark.parametrize("case", ("missing", "disabled", "mismatch", "no_snapshots", "nondiagnostic", "no_raw_oracle"))
def test_cache_receipts_reject_missing_or_failed_dual_oracle(tmp_path, monkeypatch, case):
    mod, receipts, sha = cache_diagnostic_receipts(tmp_path, monkeypatch)
    path, _ = receipts[0]
    result_path = path.parent/"native/result.json"
    result = json.loads(result_path.read_text())
    actual = result["acquisition"]
    if case == "missing": actual.pop("readback_audit")
    if case == "disabled": actual["readback_audit"]["enabled"] = False
    if case == "mismatch": actual["readback_audit"]["mismatches"] = [{"side": "left", "mismatches": ["q"]}]
    if case == "no_snapshots": actual["readback_audit"]["side_snapshots"] = 0
    if case == "nondiagnostic": actual["readback_audit"]["diagnostic_only"] = False
    if case == "no_raw_oracle": actual["readback_audit"]["extra_raw_reads_per_side_snapshot"] = 0
    result_path.write_text(json.dumps(result))
    outer = json.loads(path.read_text())
    outer["artifacts"]["result.json"] = sha(result_path)
    path.write_text(json.dumps(outer))
    receipts[0] = path, sha(path)
    with pytest.raises(ValueError, match="dual raw.*oracle"):
        mod.verify_measurement_native_receipts(receipts, style="pinch")


@pytest.mark.parametrize("case", ("missing_freeze_hash", "missing_timeline_hash", "missing_attempt_hash",
    "audit_unadmissible", "audit_drift", "audit_wrong_version", "audit_wrong_attempt", "audit_no_startup",
    "audit_exit_qualification", "audit_grants_qualification",
    "attempt_bytes_changed", "attempt_source_missing", "attempt_source_wrong", "audit_source_wrong",
    "audit_source_missing", "source_no_longer_current", "timeline_unsupported", "timeline_alive",
    "timeline_unverified", "timeline_no_stop", "timeline_native_qualification", "timeline_exit_qualification",
    "timeline_grants_qualification"))
def test_receipt_gate_rejects_source_drift_or_unreleased_timeline(tmp_path, monkeypatch, case):
    mod, receipts, sha = diagnostic_receipts(tmp_path, monkeypatch)
    path, _ = receipts[0]
    native = path.parent/"native"
    outer = json.loads(path.read_text())
    attempt = json.loads((path.parent/"attempt.json").read_text())
    audit = json.loads((native/"source_freeze_audit.json").read_text())
    timeline = json.loads((native/"timeline_release.json").read_text())
    source = next(iter(audit["source_sha256"]))
    if case == "audit_unadmissible": audit["admissible"] = False
    if case == "audit_drift": audit["drift"] = [{"path": source, "before": "a"*64, "after": "b"*64}]
    if case == "audit_wrong_version": audit["version"] = "unknown"
    if case == "audit_no_startup": audit["startup_checked"] = False
    if case == "audit_exit_qualification": audit["process_exit_verified"] = True
    if case == "audit_grants_qualification": audit["qualification_granted"] = True
    if case == "attempt_source_missing": attempt["source_sha256"].pop(source)
    if case == "attempt_source_wrong": attempt["source_sha256"][source] = "0"*64
    if case == "audit_source_wrong": audit["source_sha256"][source] = "0"*64
    if case == "audit_source_missing": audit["source_sha256"].pop(source)
    if case == "source_no_longer_current":
        attempt["source_sha256"][source] = audit["source_sha256"][source] = "0"*64
    if case == "timeline_unsupported": timeline["local_timeline_weakref_supported"] = False
    if case == "timeline_alive": timeline["local_timeline_weakref_alive"] = True
    if case == "timeline_unverified": timeline["local_timeline_release_verified"] = False
    if case == "timeline_no_stop": timeline["stop_commit_returned"] = False
    if case == "timeline_native_qualification": timeline["native_release_verified"] = True
    if case == "timeline_exit_qualification": timeline["process_exit_verified"] = True
    if case == "timeline_grants_qualification": timeline["qualification_granted"] = True
    (path.parent/"attempt.json").write_text(json.dumps(attempt))
    outer["artifacts"]["attempt.json"] = sha(path.parent/"attempt.json")
    audit["attempt_sha256"] = sha(path.parent/"attempt.json")
    if case == "audit_wrong_attempt": audit["attempt_sha256"] = "0"*64
    if case == "attempt_bytes_changed": (path.parent/"attempt.json").write_text("{}")
    (native/"source_freeze_audit.json").write_text(json.dumps(audit))
    (native/"timeline_release.json").write_text(json.dumps(timeline))
    for name in ("source_freeze_audit.json", "timeline_release.json"):
        outer["artifacts"][name] = sha(native/name)
    if case == "missing_freeze_hash": outer["artifacts"].pop("source_freeze_audit.json")
    if case == "missing_timeline_hash": outer["artifacts"].pop("timeline_release.json")
    if case == "missing_attempt_hash": outer["artifacts"].pop("attempt.json")
    path.write_text(json.dumps(outer))
    receipts[0] = path, sha(path)
    with pytest.raises(ValueError):
        mod.verify_measurement_native_receipts(receipts, style="pinch")
