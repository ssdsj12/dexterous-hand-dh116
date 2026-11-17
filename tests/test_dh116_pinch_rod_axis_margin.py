import math
import numpy as np
import pytest
from scripts import audit_dh116_pinch_rod_axis_margin as m
from scripts import audit_dh116_pinch_rod_axis_clearance as base

def test_retained_constraints_are_original_rows():
    rng=np.random.default_rng(19)
    rows=np.column_stack((rng.normal(size=(100,2)),np.repeat([.1,.2],50)))
    ids=m.extreme_indices(rows)
    assert len(ids)<100 and len(set(ids))==len(ids)
    assert np.array_equal(rows[ids],np.array([rows[i] for i in ids]))
    for theta in np.linspace(-math.pi,math.pi,1001):
        values=rows[:,0]*np.cos(theta)+rows[:,1]*np.sin(theta)-rows[:,2]
        assert values.min()==pytest.approx(values[ids].min(),abs=1e-12)

@pytest.mark.parametrize("rows",[
    [[1,0,0]],[[1,0,0],[2,0,0]],[[1,0,0],[2,0,0],[3,0,0]],
    [[1,0,0],[1,0,0],[1,0,.1]]])
def test_degenerate_sets_keep_valid_actual_indices(rows):
    rows=np.array(rows,float);ids=m.extreme_indices(rows)
    assert len(ids)>0 and np.array_equal(rows[ids],rows[ids].copy())
    assert all(0<=i<len(rows) for i in ids)

def test_maximum_margin_reaches_symmetric_known_optimum():
    rows=np.array([[.01,.02,.001],[.01,-.02,.001]])
    intervals=base.feasible_intervals(rows,[-.2,.2])
    result=m.select_margin(rows,intervals)
    assert abs(result["angle_rad"])<1e-7
    assert result["lower_margin_m"]==pytest.approx(.009,abs=1e-8)
    assert result["upper_margin_m"]-result["lower_margin_m"]<1e-8
    assert result["full_constraint_rechecks"]>0
    assert result["native_ready"] is False

def test_margin_selection_does_not_regress_to_boundary():
    rows=np.array([[.01,.03,.009],[.01,-.03,.009]])
    intervals=base.feasible_intervals(rows,[-.2,.2])
    old=base.choose_angle(intervals)
    result=m.select_margin(rows,intervals)
    assert result["lower_margin_m"]>=min(rows[:,0]*np.cos(old)+rows[:,1]*np.sin(old)-rows[:,2])-1e-12

def test_false_subset_feasibility_gets_full_fallback(monkeypatch):
    rows=np.array([[.01,.03,.009],[.01,-.03,.009]])
    monkeypatch.setattr(m,"extreme_indices",lambda _:np.array([0]))
    result=m.select_margin(rows,base.feasible_intervals(rows,[-.2,.2]))
    assert result["full_fallback_calls"]>0
    assert result["lower_margin_m"]>=0
    x=result["angle_rad"]
    assert min(rows[:,0]*np.cos(x)+rows[:,1]*np.sin(x)-rows[:,2])>=result["lower_margin_m"]-1e-12

def test_empty_feasible_domain_has_no_candidate():
    result=m.select_margin(np.array([[0,0,.1]]),[])
    assert result["angle_rad"] is None and result["native_ready"] is False

def test_nan_and_unbounded_input_rejected():
    with pytest.raises(ValueError):m.select_margin([[float("nan"),0,0]],[[-.1,.1]])

def test_existing_output_is_not_overwritten(tmp_path):
    p=tmp_path/"already";p.mkdir();(p/"raw").write_text("keep")
    with pytest.raises(FileExistsError):m.main(["--output",str(p)])
    assert (p/"raw").read_text()=="keep"

def test_path_gate_rejects_unknown_depth_and_excursion():
    def rows(values):
        return [dict(pads={name:dict(minimum_gap_m=v) for name,v in zip(base.PADS,pair)})
                for pair in values]
    assert m.sampled_depth_gate(rows([[.001,.002],[-.0002,-.00025]]))
    assert not m.sampled_depth_gate(rows([[.001,.002],[-.00031,-.00025]]))
    assert not m.sampled_depth_gate(rows([[None,.002],[-.0002,-.00025]]))
    assert not m.sampled_depth_gate([])
