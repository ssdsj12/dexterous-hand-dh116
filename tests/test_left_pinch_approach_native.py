"""Independent native candidate routing/budget tests; no Isaac launch."""
import importlib
from copy import deepcopy
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]

def test_candidate_scope_restores_original_modules_and_plan():
    op=importlib.import_module("scripts.prepare_dh116_left_pinch_approach_native")
    from scripts import prepare_dh116_effective_right_native as old
    from scripts import prepare_dh116_effective_right_pinch as old_plan
    from scripts import prepare_dh116_dual_pinch_closure_interval as closure
    from scripts import validate_dh116_effective_right_contact as validator
    from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
    before=(old.VERSION,old.CONFIG,old.load_pair,old.SCOPE,validator.SCOPE,closure.make_geometry,builder.PAIR)
    old_bytes=old_plan.load_pair()
    with op.scenario_scope(ROOT):
        assert old.VERSION==op.VERSION and old.SCOPE==op.SCOPE
        assert validator.SCOPE==op.SCOPE
        assert builder.PAIR==ROOT/op.CONFIG
        assert closure.action_schedule().shape==(29,24)
        assert old.load_pair()["version"]==op.plan.VERSION
        assert old_plan.load_pair()==old_bytes
        assert closure.make_geometry is op.make_geometry
    assert before==(old.VERSION,old.CONFIG,old.load_pair,old.SCOPE,validator.SCOPE,closure.make_geometry,builder.PAIR)
    with pytest.raises(RuntimeError):
        with op.scenario_scope(ROOT):raise RuntimeError("test unwind")
    assert before==(old.VERSION,old.CONFIG,old.load_pair,old.SCOPE,validator.SCOPE,closure.make_geometry,builder.PAIR)

def test_preflight_freezes_new_sources_and_original_parent():
    entry=importlib.import_module("scripts.verify_dh116_left_pinch_approach")
    c=entry.preflight(ROOT)
    assert c["version"]==entry.VERSION
    assert c["physics_steps"]==234 and c["parameter_endpoint_count"]==233
    assert c["native_execution_performed"] is c["teacher_admission"] is False
    assert c["sources"]["external_sources"][str(entry.operation.plan.PROOF)]==entry.operation.plan.PROOF_SHA
    for name in (entry.ENTRY,entry.WRAPPER,entry.operation.CONFIG,
            "scripts/prepare_dh116_left_pinch_approach.py",
            "src/go2_pvcnn/control/dh116/tabletop_left_pinch_approach_geometry.py"):
        assert name in c["sources"]["local_sources"]
    assert entry.PARENT_SHA=="f0767fc970b02d0e4024f57558982eb01bd3c7c6a9698e2289adef71f4974901"

def recipe(entry):
    from scripts import run_dh116_left_pinch_approach_work_packages as w
    return dict(entrypoint=entry.ENTRY,source_files=list(w.AUDIT_SOURCES),debugger="gdb_on_segv",argv=["--cleanup","explicit","--approved-code-config-sha256","a"*64],
        budget=dict(physics_steps=234,updates=0,environments=1,gpu_index=3,timeout_seconds=180),
        validator=dict(equals=dict(case="explicit",version=entry.VERSION,code_config_sha256="a"*64)),
        dependency_receipt=dict(package=entry.PARENT_PACKAGE,qualification_scope=entry.PARENT_SCOPE))

@pytest.mark.parametrize("key,value",[("physics_steps",235),("updates",1),("environments",2),("gpu_index",2),("timeout_seconds",181)])
def test_no_budget_or_identity_expansion(key,value):
    wrapper=importlib.import_module("scripts.run_dh116_left_pinch_approach_work_packages")
    r=recipe(wrapper.entry)
    assert cleanup(wrapper,r)=="explicit"
    r["budget"][key]=value
    with pytest.raises(ValueError):cleanup(wrapper,r)

def test_foreign_scope_rejected():
    wrapper=importlib.import_module("scripts.run_dh116_left_pinch_approach_work_packages")
    r=recipe(wrapper.entry);r["validator"]["equals"]["version"]="old"
    with pytest.raises(ValueError):cleanup(wrapper,r)

def cleanup(wrapper,spec):
    from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
    with bindings((wrapper.base,"ENTRY",wrapper.entry.ENTRY),(wrapper.base,"AUDIT_SOURCES",wrapper.AUDIT_SOURCES)):
        return wrapper.cleanup_case(spec)
