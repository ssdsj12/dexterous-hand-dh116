"""Scope isolation and raw lifecycle-validator preservation, no native run."""
import ast, inspect
from pathlib import Path
import pytest
from scripts import prepare_dh116_tripod_native_interval as operation
from scripts import prepare_dh116_tripod_owned_approach as plan
from scripts import prepare_dh116_native_parameter_task as parameter
from scripts import prepare_dh116_dual_pinch_director_interval as director
from scripts import prepare_dh116_dual_pinch_closure_interval as closure
from scripts import dh116_pinch_margin_contact_audit as audit
from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("fail",[False,True])
def test_original_owner_restored_after_success_or_exception(fail):
    saved=(parameter.VERSION,parameter.inspect_interval,director.DirectorExchange,
           builder.load_pair_proposal,closure.make_geometry,audit.operation)
    try:
        with operation.scenario_scope(ROOT):
            assert parameter.VERSION==operation.VERSION
            assert director.DirectorExchange is operation.TripodExchange
            assert parameter.inspect_interval is operation.inspect_interval_bound
            assert builder.load_pair_proposal()==plan.load_pair()
            assert audit.operation is operation
            assert closure.make_geometry is operation.make_geometry
            if fail:raise RuntimeError("fixture")
    except RuntimeError:assert fail
    assert saved==(parameter.VERSION,parameter.inspect_interval,director.DirectorExchange,
                   builder.load_pair_proposal,closure.make_geometry,audit.operation)

def test_postexit_original_checks_only_explicit_operation_import_changes():
    from scripts import validate_dh116_bilateral_owned_contact as old
    from scripts import validate_dh116_tripod_owned_contact as new
    a=ast.parse(inspect.getsource(old.validate_artifacts))
    b=ast.parse(inspect.getsource(new.validate_artifacts))
    a.body[0].body[0].names[0].name="prepare_dh116_tripod_native_interval"
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)

def test_director_all_original_checks_retained_with_one_explicit_terminal_extension():
    from scripts import dh116_tripod_contact_audit as new
    a=inspect.getsource(audit.inspect_directors)
    marker='        require(exchange["guard_failure"]==expected_guard,'
    extension='        require(exchange["quiet_prior_guard_failure"]==expected_guard,\n                "quiet layer must retain original material/geometry guard")\n        expected_guard=operation.observation_failure(exchange)\n'
    assert a.count(marker)==1
    expected=ast.parse(a.replace(marker,extension+marker))
    assert ast.dump(expected,include_attributes=False)==ast.dump(ast.parse(inspect.getsource(new.inspect_directors)),include_attributes=False)

def test_scenario_calls_existing_owner_once_and_original_class_not_replaced_permanently(monkeypatch):
    calls=[]
    monkeypatch.setattr(parameter,"run_director",lambda *a,**k:calls.append((k["project"],director.DirectorExchange)) or "owner")
    old=director.DirectorExchange
    assert operation.run_existing_app(project=ROOT)=="owner"
    assert calls==[(ROOT,operation.TripodExchange)] and director.DirectorExchange is old
