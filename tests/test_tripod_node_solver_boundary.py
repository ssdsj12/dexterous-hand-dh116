import numpy as np
from scripts import solve_dh116_tripod_node_closure as m

def test_bounded_solver_moves_from_exact_zero_lower_bound():
    assert hasattr(m,"bounded_fit"), "explicit boundary-safe solver route missing"
    calls=[]
    def evaluate(x):
        calls.append(x.copy())
        assert np.all(x>=0) and np.all(x<=1)
        return dict(residual=(x-np.array([.7,.8])).tolist())
    fit=m.bounded_fit(evaluate,np.zeros(2),np.ones(2))
    assert fit.success
    assert np.linalg.norm(fit.x-[.7,.8])<1e-5
    assert np.array_equal(calls[0],np.zeros(2))
    assert len(calls)<100
