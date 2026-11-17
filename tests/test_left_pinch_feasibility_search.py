"""CPU optimizer changes must preserve every original physical constraint."""
import numpy as np
import pytest
from scripts import solve_dh116_left_pinch_tracking as m

def test_independent_non_linear_search_reaches_actual_constraints():
    def query(x):
        return dict(x=list(x),constraints={"lower":float(x[0]-.2)*.001,"upper":float(.8-x[0])*.001})
    result=m.nonlinear_search(query,np.array([-.5]),maximum_evaluations=30,maximum_seconds=2.)
    assert result["best"]["admission"]["static_constraints_passed"]
    assert .2 <= result["best"]["x"][0] <= .8
    assert result["optimizer_status_is_not_qualification"]

def test_search_budget_counts_finite_difference_queries():
    calls=[]
    def query(x):
        calls.append(x.copy());return dict(x=list(x),constraints={"impossible":-1.})
    result=m.nonlinear_search(query,np.zeros(9),maximum_evaluations=5,maximum_seconds=2.)
    assert len(calls)<=5
    assert not result["best"]["admission"]["static_constraints_passed"]
    assert result["termination"]=="budget_exhausted"

def test_search_rejects_changing_or_nonfinite_guards():
    with pytest.raises(ValueError):
        m.nonlinear_search(lambda x:dict(x=list(x),constraints={"bad":float("nan")}),np.zeros(1),maximum_evaluations=5,maximum_seconds=2.)

def test_v2_bounds_and_physical_contract_equal_v1():
    assert m.NONLINEAR_CONFIG["source_config"]==m.CONFIG
    assert m.NONLINEAR_CONFIG["maximum_evaluations"]==600
    assert m.NONLINEAR_CONFIG["maximum_search_seconds"]==60.
    assert m.NONLINEAR_CONFIG["native_execution"] is False
