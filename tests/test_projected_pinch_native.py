"""Independent projected path, unmodified guards and bounded execution."""
from pathlib import Path
import importlib
import pytest
from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
ROOT=Path(__file__).resolve().parents[1]
def api():return importlib.import_module("scripts.prepare_dh116_projected_pinch_native")
def test_scopes_restore_and_do_not_install_by_default():
    op=api()
    from scripts import prepare_dh116_dual_pinch_director_interval as d
    from scripts import prepare_dh116_left_pinch_approach_native as old
    from scripts import dh116_pinch_margin_contact_audit as audit
    before=(d.DirectorExchange,old.VERSION,old.make_geometry,audit.inspect_closure)
    for fail in (False,True):
        try:
            with op.scenario_scope():
                assert d.DirectorExchange is not before[0]
                assert old.VERSION==op.VERSION
                assert audit.operation is op
                assert audit.inspect_closure is op.inspect_projected_closure
                assert old.action_schedule().equal(op.plan.action_schedule())
                if fail:raise RuntimeError("fixture")
        except RuntimeError:assert fail
        assert before==(d.DirectorExchange,old.VERSION,old.make_geometry,audit.inspect_closure)

def test_clean_preflight_new_identity_and_complete_sources():
    e=importlib.import_module("scripts.verify_dh116_projected_pinch")
    c=e.preflight(ROOT)
    assert c["version"]==api().VERSION and c["physics_steps"]==234
    assert c["native_execution_performed"] is False
    for p in (e.ENTRY,e.WRAPPER,"scripts/prepare_dh116_projected_pinch_native.py",
            "src/go2_pvcnn/control/dh116/tabletop_wrist_projection_evidence.py",
            "src/go2_pvcnn/control/dh116/tabletop_projected_pinch_geometry.py"):
        assert p in c["sources"]["local_sources"]
    assert e.PARENT_SHA=="f0767fc970b02d0e4024f57558982eb01bd3c7c6a9698e2289adef71f4974901"

@pytest.mark.parametrize("key,value",[("physics_steps",235),("updates",1),("environments",2),("gpu_index",2),("timeout_seconds",181)])
def test_no_budget_expansion(key,value):
    from test_left_pinch_approach_native import recipe
    w=importlib.import_module("scripts.run_dh116_projected_pinch_work_packages")
    r=recipe(w.entry);r["source_files"]=list(w.AUDIT_SOURCES)
    with bindings((w.base,"ENTRY",w.entry.ENTRY),(w.base,"AUDIT_SOURCES",w.AUDIT_SOURCES)):
        assert w.cleanup_case(r)=="explicit"
        r["budget"][key]=value
        with pytest.raises(ValueError):w.cleanup_case(r)

def test_postexit_requires_new_evidence_marker(monkeypatch,tmp_path):
    w=importlib.import_module("scripts.run_dh116_projected_pinch_work_packages")
    from scripts import validate_dh116_native_parameter_task as validator
    monkeypatch.setattr(validator,"audit_outcome",lambda *a,**k:dict(lifecycle_qualified=True))
    with pytest.raises(FileNotFoundError):w.audit_outcome(tmp_path,tmp_path/"outer","explicit")
