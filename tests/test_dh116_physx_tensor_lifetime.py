"""CPU ownership and immutable-evidence regression; imports no native SDK."""
import ast
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest


def helpers():
    from go2_pvcnn.control.dh116 import tabletop_native_lifetime_audit as audit
    return audit


class Resource:
    def __repr__(self):
        raise AssertionError("diagnostic must not repr native owners")


def test_snapshot_reports_python_alias_without_retaining_it():
    audit = helpers()
    view = Resource()
    owner = {"_physics_sim_view": view}
    ref = weakref.ref(view)
    report = audit.snapshot_weakrefs({"view": ref})
    assert report["resources"]["view"]["alive"] is True
    assert any("_physics_sim_view" in row.get("matching_keys", [])
               for row in report["resources"]["view"]["referrers"])
    json.dumps(report, allow_nan=False)
    del owner, view
    gc.collect()
    assert ref() is None
    assert report["resources"]["view"]["alive"] is True


def test_snapshot_dead_and_unknown_are_distinct():
    audit = helpers()
    obj = Resource()
    ref = weakref.ref(obj)
    del obj
    report = audit.snapshot_weakrefs({"dead": ref, "unsupported": None})
    assert report["resources"]["dead"]["alive"] is False
    assert report["resources"]["unsupported"]["alive"] is None
    assert report["native_release_verified"] is False


def test_snapshot_rejects_strong_inputs_and_bounds_output():
    audit = helpers()
    obj = Resource()
    with pytest.raises(TypeError, match="weak"):
        audit.snapshot_weakrefs({"native": obj})
    with pytest.raises(ValueError, match="bound"):
        audit.snapshot_weakrefs({"native": weakref.ref(obj)}, max_referrers=0)
    aliases = [{"owned": obj} for _ in range(12)]
    result = audit.snapshot_weakrefs({"native": weakref.ref(obj)}, max_referrers=2)
    assert len(result["resources"]["native"]["referrers"]) == 2
    assert result["resources"]["native"]["referrers_truncated"] is True
    assert len(aliases) == 12


def test_stored_snapshot_closure_is_visible_then_collectable():
    audit = helpers()
    def make():
        hand = Resource()
        def snapshot():
            return hand
        return weakref.ref(hand), snapshot
    ref, callback = make()
    result = audit.snapshot_weakrefs({"hand": ref})
    assert any(row["kind"] == "closure_cell" for row in result["resources"]["hand"]["referrers"])
    del callback
    gc.collect()
    assert ref() is None


def test_finished_failure_frame_is_visible_without_materializing_locals():
    audit = helpers()
    refs = []
    def fail():
        resource = Resource()
        refs.append(weakref.ref(resource))
        raise RuntimeError("fixture")
    try:
        fail()
    except RuntimeError as error:
        saved = error
    result = audit.snapshot_weakrefs({"resource": refs[0]})
    assert any(row["kind"] == "frame" and row["function"] == "fail"
               for row in result["resources"]["resource"]["referrers"])
    saved.__traceback__ = None
    del saved
    gc.collect()
    assert refs[0]() is None


def test_original_snapshot_closures_alone_do_not_retain_owners(tmp_path):
    # Execute only the two original nested Python definitions, never _run_impl.
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import run_owned_call
    from go2_pvcnn.control.dh116.tabletop_deformable_rod_bridge import NativeOwnershipTrace
    source = Path(__file__).parents[1] / "scripts/verify_dh116_deformx_pinch_contact.py"
    impl = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_run_impl")
    nested = [n for n in impl.body if isinstance(n, ast.FunctionDef) and n.name in ("snapshot", "mesh_poses")]
    assert [n.name for n in nested] == ["snapshot", "mesh_poses"]
    scaffold = ast.parse("""def operation(owners):
    hand = owners.keep("hand", Resource())
    view = owners.keep("view", Resource())
    composer = owners.keep("composer", SimpleNamespace(reset=lambda: None))
    rope = owners.keep("runtime", Resource())
    hand.view = view
    view.getter = lambda: hand
    hand.composer = composer
    composer.asset = hand
    for name, obj in (("hand", hand), ("view", view), ("rope", rope)):
        refs[name] = weakref.ref(obj)
    bodies = []
    host = lambda value: value
    return {"failure": "mimic_error"}
""").body[0]
    scaffold.body[-1:-1] = nested
    namespace = dict(Resource=Resource, SimpleNamespace=SimpleNamespace, weakref=weakref, refs={})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[scaffold], type_ignores=[])), str(source), "exec"), namespace)
    trace = NativeOwnershipTrace(tmp_path / "ownership.jsonl")
    result = run_owned_call(namespace["operation"], synchronize=lambda: None, ownership=trace)
    assert result == {"failure": "mimic_error"}
    assert all(ref() is None for ref in namespace["refs"].values())


LOG = """DH116_GDB_FATAL_BACKTRACE SIGSEGV
DH116_GDB_COMMAND bt 32
#0  0x0000000000109718 in ?? () from /sdk/tensor.so
#1  0x000000000020c1de in ?? () from /sdk/binding.so
#2  0x0000000000300000 in Py_FinalizeEx ()
DH116_GDB_COMMAND thread apply all bt 20
Thread 1
#0  0x0000000000109718 in ?? () from /sdk/tensor.so
DH116_GDB_COMMAND info sharedlibrary
0x100000 0x110000 Yes (*) /sdk/tensor.so
DH116_GDB_COMMAND info proc mappings
          Start Addr           End Addr       Size     Offset  Perms  objfile
            0x100000           0x110000    0x10000    0x50000  r-xp   /sdk/tensor.so
            0x200000           0x210000    0x10000    0x20000  r-xp   /sdk/binding.so
DH116_GDB_INFERIOR_STATUS {"exit_code": null, "signal": 11}
"""


def test_first_fatal_offsets_use_same_log_map_file_offsets():
    audit = helpers()
    evidence = audit.first_fatal_capture(LOG)
    assert evidence["capture_complete"] is True
    assert evidence["frames"][0]["file_offset"] == "0x59718"
    assert evidence["frames"][1]["file_offset"] == "0x2c1de"
    assert evidence["python_finalize_in_stack"] is True


@pytest.mark.parametrize("log", [
    "DH116_GDB_FATAL_BACKTRACE SIGSEGV\nValueError: missing output\n",
    LOG.replace("DH116_GDB_COMMAND info proc mappings", "MISSING"),
    LOG.replace("0x100000           0x110000", "0x900000           0x910000"),
])
def test_marker_or_unmapped_pc_does_not_qualify_capture(log):
    assert helpers().first_fatal_capture(log)["capture_complete"] is False


def test_global_manager_release_does_not_prove_asset_release():
    rows = [
        dict(phase="owned hand and solver references dropped", python_liveness={"hand": True, "hand_view": True, "composer": True, "rope": True}),
        dict(phase="SDK STOP", manager_view_released=True),
        dict(phase="app.close returned, outer exit pending", python_liveness={"hand": True, "hand_view": True, "composer": True, "rope": True}),
    ]
    result = helpers().summarize_ownership(rows)
    assert result["manager_view_released"] is True
    assert result["live_before_stop"] == ["composer", "hand", "hand_view", "rope"]
    assert result["live_after_close"] == ["composer", "hand", "hand_view", "rope"]
    assert result["root_cause_proven"] is False


def test_pinned_json_rejects_changed_evidence(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text('{"value": 1}')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert helpers().read_pinned_json(path, sha) == {"value": 1}
    path.write_text('{"value": 2}')
    with pytest.raises(ValueError, match="SHA"):
        helpers().read_pinned_json(path, sha)


def test_diagnostic_does_not_create_a_reported_closure_owner():
    obj = Resource()
    report = helpers().snapshot_weakrefs({"plain": weakref.ref(obj)})
    for row in report["resources"]["plain"]["referrers"]:
        assert row["kind"] != "closure_cell"


def test_archive_audit_reopens_both_hash_bound_inputs(tmp_path):
    from scripts.audit_dh116_physx_tensor_lifetime import audit_run
    output = tmp_path / "native"
    output.mkdir()
    ownership = output / "ownership.jsonl"
    ownership.write_text(json.dumps(dict(phase="SDK STOP", manager_view_released=True)) + "\n")
    log = tmp_path / "native.log"
    log.write_text(LOG)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(dict(
        output=str(output), log_sha256=digest(log), tested_head="head",
        code_config_sha256="fingerprint", outer_exit_code=255, duration_seconds=33.,
        debugger_status_verified=True, debugger_inferior_signal=11)))
    receipt_sha, ownership_sha = digest(receipt), digest(ownership)
    result = audit_run(receipt, receipt_sha, ownership_sha)
    assert result["fatal"]["capture_complete"] is True
    assert result["ownership"]["manager_view_released"] is True
    log.write_text(LOG + "changed")
    with pytest.raises(ValueError, match="log SHA"):
        audit_run(receipt, receipt_sha, ownership_sha)
    log.write_text(LOG)
    ownership.write_text(ownership.read_text() + "{}\n")
    with pytest.raises(ValueError, match="ownership SHA"):
        audit_run(receipt, receipt_sha, ownership_sha)


@pytest.mark.parametrize("empty_section", ["threads", "libraries"])
def test_empty_stack_or_library_command_output_is_incomplete(empty_section):
    if empty_section == "threads":
        log = LOG.replace("Thread 1\n#0  0x0000000000109718 in ?? () from /sdk/tensor.so\n", "")
    else:
        log = LOG.replace("0x100000 0x110000 Yes (*) /sdk/tensor.so\n", "")
    assert helpers().first_fatal_capture(log)["capture_complete"] is False
