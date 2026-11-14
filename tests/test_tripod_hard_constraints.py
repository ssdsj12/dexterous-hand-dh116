import importlib
import numpy as np
import pytest

def module():
    name="scripts.solve_dh116_tripod_hard_constraints"
    assert importlib.util.find_spec(name) is not None, "hard constrained Tripod study missing"
    return importlib.import_module(name)

def row():
    return dict(table_gap_by_shape_m={"source:finger32_link":.003},
        nonrequired_gap_lower_m={"source:finger13_link":.001},
        required_hull_intervals_m=[[-.00001,.00001]]*3,
        node_sdf_gap_m=[-.00001]*3,node_gradient_norm=[1]*3,
        geometric_opposition=[-.6,-.6],sdf_opposition=[-.6,-.6])

def test_all_thresholds_are_constraints_not_weighted_penalties():
    r=row();q=module().contract(r)
    assert np.all(np.asarray(q["constraints"])>0)
    for key,val in [("table_gap_by_shape_m",{"source:finger32_link":.00209}),
                    ("nonrequired_gap_lower_m",{"source:finger13_link":-.00001}),
                    ("node_sdf_gap_m",[.0001,0,0]),
                    ("required_hull_intervals_m",[[-.00006,.00001]]*3),
                    ("geometric_opposition",[-.19,-.6])]:
        assert min(module().contract({**r,key:val})["constraints"])<0
    assert q["native_ready"] is False

def test_unknown_node_measurement_cannot_become_zero():
    r=row();r["node_sdf_gap_m"][0]=None
    q=module().contract(r)
    assert q["measurements_supported"] is False
    assert min(q["constraints"])<0

def test_solver_does_not_trade_collision_for_a_better_objective():
    def packet(x):
        return dict(objective=(x[0]-1)**2,constraints=[.4-x[0]],labels=["collision"])
    s=module().Study(packet,maximum=70)
    fit=module().solve(s,np.array([.1]),np.array([0.]),np.array([1.]),iterations=6)
    assert fit.success
    assert fit.x[0]<=.400000001
    assert abs(fit.x[0]-.4)<1e-6

def test_cache_does_not_consume_duplicate_geometry_budget():
    count=[]
    s=module().Study(lambda x:count.append(1) or dict(objective=1.,constraints=[1.],labels=["a"]),maximum=2)
    s(np.zeros(2));s(np.zeros(2));s(np.ones(2))
    assert s.count==len(count)==len(s.rows)==2
    with pytest.raises(module().EvaluationLimit):s(np.ones(2)*.5)

def test_joint_bound_derivatives_never_leave_original_box():
    def packet(x):
        assert np.all(x>=0) and np.all(x<=1)
        return dict(objective=x[0]**2,constraints=[x[1]-.2],labels=["c"])
    s=module().Study(packet,maximum=8)
    j=s.derivatives(np.array([0.,1.]),np.zeros(2),np.ones(2))
    assert j["objective"]==pytest.approx([.0005,0])
    assert j["constraints"][0]==pytest.approx([0,1])

def test_changing_constraint_identity_invalidates_derivative():
    def packet(x):
        return dict(objective=0.,constraints=[1.],labels=["a" if x[0]==0 else "b"])
    s=module().Study(packet,maximum=10)
    with pytest.raises(ValueError):s.derivatives(np.zeros(1),-np.ones(1),np.ones(1))

def test_existing_output_is_not_overwritten(tmp_path):
    with pytest.raises(FileExistsError):module().main(["--output",str(tmp_path)])
