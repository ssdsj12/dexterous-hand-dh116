import importlib
import numpy as np
import pytest

def module():
    name="scripts.solve_dh116_tripod_node_closure"
    assert importlib.util.find_spec(name) is not None, "node-aware bounded Tripod closure missing"
    return importlib.import_module(name)

def parent():
    return dict(independent_style="tripod",side="left",
        independent_seed_sha256=module().TRIPOD_SHA,
        base_pose_wxyz=[0,0,.4,1,0,0,0],
        initial_q_rad=[.4,.4,.5,.5,.7,.95])

def test_decode_retains_identity_and_unused_fingers():
    p=parent();x=np.zeros(10);x[1]=.5;x[7:]=[.1,.2,.3]
    r=module().decode(p,x)
    assert r["side"]=="left" and r["independent_style"]=="tripod"
    assert r["base_pose_wxyz"][1]==pytest.approx(.0025)
    assert r["active_q_rad"]==pytest.approx([.4,.41,.52,.53,.7,.95])
    assert r["native_ready"] is False

def test_zero_vector_is_verified_quiet_start_not_old_closure_endpoint():
    r=module().decode(parent(),np.zeros(10))
    assert r["active_q_rad"]==parent()["initial_q_rad"]

@pytest.mark.parametrize("change",["pinch","wrong_side","seed","bounds","negative_close"])
def test_wrong_identity_or_cpu_search_limits_rejected(change):
    p=parent();x=np.zeros(10)
    if change=="pinch":p["independent_style"]="pinch"
    elif change=="wrong_side":p["side"]="right"
    elif change=="seed":p["independent_seed_sha256"]="0"*64
    elif change=="bounds":x[0]=1.01
    else:x[7]=-.01
    with pytest.raises(ValueError):module().decode(p,x)

def test_original_joint_limits_restrict_parameter_bounds():
    p=parent();low,high=module().active_limits()
    p["initial_q_rad"][1]=float(high[1]-.02)
    lo,hi=module().bounds(p)
    assert hi[7]==pytest.approx(.2)
    assert lo[7]==0

def test_budget_stops_and_preserves_only_completed_rows():
    calls=[]
    b=module().Budget(lambda x: calls.append(x.copy()) or dict(residual=[float(x[0])]),maximum=2)
    assert b(np.zeros(10))["residual"]==[0.]
    b(np.ones(10))
    with pytest.raises(module().EvaluationLimit):b(np.zeros(10))
    assert b.count==len(b.rows)==len(calls)==2

def test_jacobian_respects_box_and_absolute_steps():
    def evaluate(x):
        assert (x>=0).all() and (x<=1).all()
        return dict(residual=[x[0]**2+3*x[1],4*x[0]-x[1]**2])
    x=np.array([0.,.4]);j=module().jacobian(evaluate,x,np.zeros(2),np.ones(2),h=.001)
    assert j[:,0]==pytest.approx([.001,4])
    assert j[:,1]==pytest.approx([3,-.8])

def test_endpoint_gate_needs_all_independent_geometric_checks():
    m=module()
    good=dict(minimum_table_gap_m=m.CLEARANCE_M+.0001,
              minimum_nonrequired_gap_lower_m=.0001,
              required_hull_intervals_m=[[-.00001,.00001]]*3,
              node_sdf_gap_m=[-.00001]*3,node_gradient_norm=[1]*3,
              geometric_opposition=[-.5,-.5],sdf_opposition=[-.5,-.5])
    assert m.endpoint_gate(good) is True
    for name,value in [("minimum_table_gap_m",.002),("minimum_nonrequired_gap_lower_m",-.001),
                       ("node_sdf_gap_m",[.002,0,0]),("geometric_opposition",[.1,-.5]),
                       ("sdf_opposition",[.1,-.5]),("node_gradient_norm",[0,1,1])]:
        assert m.endpoint_gate({**good,name:value}) is False

def test_existing_output_is_not_overwritten(tmp_path):
    with pytest.raises(FileExistsError):module().main(["--output",str(tmp_path)])
