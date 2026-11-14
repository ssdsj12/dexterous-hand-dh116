"""Post-exit CPU fixtures are explicit; no native execution or promotion."""
import ast
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import pytest
from test_dh116_dual_pinch_postexit import fixture as original_fixture,change,write
from test_dh116_dual_pinch_builder import prepared,geometry

ROOT=Path(__file__).resolve().parents[1]
MODULE="scripts.validate_dh116_dual_pinch_closure_interval"
PROFILE="67fe4ac5a231355aac627d2ce4e7bd8186b138273d80df3c30a5c5406c4e0498"
def api():return importlib.import_module(MODULE)

def test_lifetime_helper_keeps_all_frozen_checks_except_explicit_step_parameter():
    helper=importlib.import_module("scripts.dh116_dual_pinch_lifetime_evidence")
    old=ast.parse((ROOT/"scripts/validate_dh116_dual_pinch_interval.py").read_text())
    old=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=="_lifetime")
    new=ast.parse(Path(helper.__file__).read_text())
    new=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=="lifetime_evidence")
    class Expected(ast.NodeTransformer):
        def visit_Constant(self,node):
            return ast.Name(id="expected_steps",ctx=ast.Load()) if type(node.value) is int and node.value==10 else node
    assert ast.dump(ast.Module(body=[Expected().visit(n) for n in old.body],type_ignores=[]))==ast.dump(ast.Module(body=new.body[1:],type_ignores=[]))

@pytest.fixture(scope="module")
def templates(tmp_path_factory,prepared,geometry):
    from scripts import prepare_dh116_dual_pinch_closure_interval as operation
    from test_dh116_dual_pinch_closure_interval import test_actual_episode_with_one_held_nonzero_interval as build
    result={}
    for stop in (False,True):
        path=tmp_path_factory.mktemp("closure-terminal" if stop else "closure-complete")
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(operation,"PROFILE_SHA",PROFILE,raising=False)
            build(monkeypatch,path,prepared,geometry,stop)
        result[stop]=path/"closure"
    return result

def fixture(tmp_path,templates,*,stop=False):
    out=original_fixture(tmp_path)
    for name in ("physical.jsonl","result.json","before_app_close.json","app_close_returned.json"):
        shutil.copyfile(templates[stop]/name,out/name)
    clock=templates[stop]/"sdk_clock.jsonl"
    if clock.exists():shutil.copyfile(clock,out/"sdk_clock.jsonl")
    else:(out/"sdk_clock.jsonl").unlink()
    # Synthetic liveness/GDB from the original CPU audit fixture; not native data.
    change(out/"result.json","lifetime_released_before_close",False)
    change(out/"before_app_close.json","lifetime_released",False)
    return out

def audit(out,tmp_path,**kwargs):
    return api().audit_outcome(out,tmp_path/"outer_exit.json","explicit",profile_sha256=PROFILE,**kwargs)

def test_complete_18_step_cpu_operation_and_independent_source_geometry(tmp_path,templates):
    out=fixture(tmp_path,templates)
    r=audit(out,tmp_path)
    assert r["lifecycle_qualified"] is True,r
    assert (r["physics_steps"],r["measurement_steps"],r["rod_steps"])==(18,16,16)
    assert r["profile_sha256"]==PROFILE and set(r["initial_geometry"])=={"left","right"}
    assert r["simulation_liveness"]["before_app_close"] is True
    assert r["same_rod_contact_qualified"] is r["teacher_admission"] is False
    assert r["hand_table_contacts"] is None

def test_guarded_partial_keeps_valid_journal_diagnostics_without_qualification(tmp_path,templates):
    out=fixture(tmp_path,templates,stop=True)
    r=audit(out,tmp_path)
    assert r["lifecycle_qualified"] is False
    assert r["partial_journal"]["measured_physics_rows"]==9
    assert r["partial_journal"]["terminal_reason"]=="post_step_source_clearance:right"
    assert r["observed_result"]["physics_steps"]==11
    assert r["same_rod_contact_qualified"] is r["teacher_admission"] is False
    assert r["simulation_liveness"]["before_app_close"] is True

@pytest.mark.parametrize("damage",["outer","duration","gdb","stop","live","unknown","ready",
    "clock","clock_boolean","clock_timestamp_s","clock_policy_step","clock_substep","clock_sdk_time_s","budget","code","profile","schedule","missing_profile_expectation","version","floor"])
def test_no_gate_is_waived_by_complete_physical_rows(tmp_path,templates,damage):
    out=fixture(tmp_path,templates)
    if damage=="outer":change(tmp_path/"outer_exit.json","outer_exit_code",1)
    elif damage=="duration":change(tmp_path/"outer_exit.json","duration_seconds",181)
    elif damage=="gdb":(out/"gdb_finalize.registers.txt").unlink()
    elif damage=="stop":(out/"ownership.jsonl").write_text(json.dumps(dict(phase="SDK STOP",manager_view_released=False))+chr(10))
    elif damage in ("live","unknown"):
        r=json.loads((out/"before_app_close.refs.json").read_text())
        r["resources"]["right_view"]["alive"]=True if damage=="live" else None
        write(out/"before_app_close.refs.json",r)
    elif damage=="ready":change(out/"simulation_clear_receipt.json","unsubscribe_returned",False)
    elif damage=="clock":(out/"sdk_clock.jsonl").write_text(json.dumps(dict(phase_step=1,sdk_step=99))+chr(10))
    elif damage.startswith("clock_"):
        clock_rows=[json.loads(line) for line in (out/"sdk_clock.jsonl").read_text().splitlines()]
        if damage=="clock_boolean":clock_rows[0]["phase_step"]=True
        else:clock_rows[0][damage.removeprefix("clock_")]+=1
        (out/"sdk_clock.jsonl").write_text(chr(10).join(json.dumps(row) for row in clock_rows)+chr(10))
    elif damage=="budget":change(out/"result.json","physics_budget",{})
    elif damage=="code":change(out/"result.json","code_config_sha256","b"*64)
    elif damage in ("profile","schedule"):
        r=json.loads((out/"result.json").read_text())
        if damage=="profile":r["acquisition"]["profile_sha256"]="d"*64
        else:r["acquisition"]["action_schedule"][1][14]=.01
        write(out/"result.json",r)
    elif damage=="version":change(out/"result.json","version","dh116-dual-pinch-one-interval-preparation-v1")
    elif damage=="floor":
        lines=[json.loads(s) for s in (out/"physical.jsonl").read_text().splitlines()]
        first=next(x for x in lines if x.get("operation")=="physical")
        first["payload"]["exchange"]["post_step_clearance"]["floor_m"]=.0019
        (out/"physical.jsonl").write_text(chr(10).join(json.dumps(x) for x in lines)+chr(10))
    if damage=="missing_profile_expectation":r=api().audit_outcome(out,tmp_path/"outer_exit.json","explicit")
    else:r=audit(out,tmp_path)
    assert r["lifecycle_qualified"] is False,r
    assert r["reason"] and r["same_rod_contact_qualified"] is r["teacher_admission"] is False

def test_old_ten_step_fixture_cannot_acquire_closure_scope(tmp_path):
    out=original_fixture(tmp_path)
    r=audit(out,tmp_path)
    assert r["lifecycle_qualified"] is False
    assert r["observed_result"]["physics_steps"]==10

def test_missing_or_truncated_journal_remains_diagnostic(tmp_path,templates):
    out=fixture(tmp_path,templates)
    (out/"physical.jsonl").write_text("{")
    r=audit(out,tmp_path)
    assert r["lifecycle_qualified"] is False and r["partial_journal"] is None
    assert r["partial_journal_error"]


def test_actual_v4_ten_step_receipt_is_negative_only():
    attempt=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f7-dual-pinch-zero-interval-v4-native/9e95d33e8b0565133c05d8fae5eb5c5eb68de269806ae561e4c5bd022bd9e202/attempt-fz50kogx")
    receipt=json.loads((attempt/"receipt.json").read_text())
    assert receipt["exit_code"]==0
    result=api().audit_outcome(receipt["output"],attempt/"outer_exit.json","explicit",profile_sha256=PROFILE)
    assert result["lifecycle_qualified"] is False
    assert result["observed_result"]["physics_steps"]==10
    assert result["same_rod_contact_qualified"] is result["teacher_admission"] is False
