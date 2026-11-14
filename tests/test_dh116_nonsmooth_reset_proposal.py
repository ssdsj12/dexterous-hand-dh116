import numpy as np
import pytest
from scripts.solve_dh116_right_pinch_reset_direction import local_pass

def discontinuous(x):
    return np.r_[x-.02,.5-x,.5 if x[0]<=0 else .6]

def test_old_default_still_rejects_nonsmooth_jacobian():
    r=local_pass(discontinuous,np.zeros(4),-np.ones(4),np.ones(4),trust_radius=.1)
    assert r["reason"]=="constraint_jacobian_nonsmooth"
    assert r["linear_solver_calls"]==0 and r["best_violation"]>0

def test_inactive_proposal_only_row_keeps_actual_full_nonlinear_result():
    r=local_pass(discontinuous,np.zeros(4),-np.ones(4),np.ones(4),trust_radius=.1,
                 proposal_only_rows=[8])
    assert r["proposal_only_rows"]==[8] and r["full_nonlinear_constraints_always_verified"]
    assert r["best_violation"]==0 and r["jacobian_relative_error"]>.2
    assert r["proposal_jacobian_relative_error"]<1e-6
    assert len(r["trials"])==4 and all(len(t["constraints"])==9 for t in r["trials"])
    assert r["measurements"]==21

def test_nonlinear_failure_cannot_be_hidden_by_zero_proposal_derivative():
    def f(x):return np.r_[x-.02,.5-x,.3 if x[0]<=0 else -1.]
    r=local_pass(f,np.zeros(4),-np.ones(4),np.ones(4),trust_radius=.1,proposal_only_rows=[8])
    assert r["best_violation"]==pytest.approx(.02)
    assert r["best_x"]==[0,0,0,0]
    assert all(t["violation"]==1 for t in r["trials"])

@pytest.mark.parametrize("rows",[[9],[-1],[8,8],[True],["8"]])
def test_proposal_rows_require_distinct_existing_integer_indices(rows):
    with pytest.raises(ValueError):
        local_pass(discontinuous,np.zeros(4),-np.ones(4),np.ones(4),proposal_only_rows=rows)

@pytest.mark.parametrize("margin",[-.1,0,.249999])
def test_active_or_small_margin_cannot_drop_its_proposal_derivative(margin):
    def f(x):return np.r_[x-.02,.5-x,margin]
    with pytest.raises(ValueError,match="inactive"):
        local_pass(f,np.zeros(4),-np.ones(4),np.ones(4),proposal_only_rows=[8])
