from copy import deepcopy
import numpy as np
import pytest
from scripts import solve_dh116_tripod_joint_root_direction as m


def parent(side="left"):
    return dict(candidate_id=m.PARENTS[side],side=side,independent_style="tripod",
        independent_seed_sha256=m.TRIPOD_SHA,base_pose_wxyz=[.53,.13 if side=="left" else -.19,.39,1.,0.,0.,0.],
        active_q_rad=[.5,.3,.6,.6,.7,.7],active_delta_rad=[0,.025,.05,.05,0,0])


def test_joint_root_codec_moves_only_declared_variables_without_mirroring():
    p=parent();saved=deepcopy(p);x=np.ones(10)*.1
    candidate=m.decode(p,x)
    assert p==saved
    np.testing.assert_allclose(candidate["base_pose_wxyz"][:3],np.array(p["base_pose_wxyz"][:3])+m.SCALES[:3]*x[:3])
    np.testing.assert_allclose(candidate["active_q_rad"][:4],np.array(p["active_q_rad"][:4])+p["active_delta_rad"][:4]+m.SCALES[6:]*x[6:])
    assert candidate["active_q_rad"][4:]==p["active_q_rad"][4:]
    assert candidate["side"]=="left" and candidate["reset_verified"] is False
    assert np.linalg.norm(candidate["base_pose_wxyz"][3:])==pytest.approx(1)


def test_codec_refuses_outside_declared_box_or_original_joint_limit():
    with pytest.raises(ValueError):m.decode(parent(),np.ones(10)*1.01)
    p=parent();p["active_q_rad"][0]=0
    with pytest.raises(ValueError):m.decode(p,np.r_[np.zeros(6),-1.,0,0,0])


def test_two_scale_central_difference_has_exact_bound_and_no_fake_stability():
    calls=[]
    def evaluate(x):
        calls.append(x.copy())
        return {"residual":np.r_[x[:3]*2,x[3:]*3].tolist()}
    fit=m.local_direction(evaluate,lower=-np.ones(10),upper=np.ones(10))
    assert len(calls)==41
    assert fit["jacobian_relative_difference"]<1e-12
    assert fit["model_supported"] is False  # Already at zero residual; no invented descent.


def test_bounded_direction_improves_true_residual_without_additional_jacobians():
    calls=[]
    def evaluate(x):
        calls.append(x.copy())
        return {"residual":(x-np.ones(10)*.25).tolist()}
    result=m.local_direction(evaluate,lower=-np.ones(10),upper=np.ones(10))
    assert len(calls)<=45
    assert result["model_supported"] is True
    assert result["observed_descent"] is True
    assert result["best_objective"]<result["baseline_objective"]
    assert all(np.max(abs(x))<=1 for x in calls)


def test_evaluation_budget_rejects_before_next_query():
    calls=[]
    b=m.EvaluationBudget(lambda x:calls.append(x) or {"residual":[1.]},maximum=2)
    b(np.zeros(10));b(np.zeros(10))
    with pytest.raises(m.EvaluationLimit):b(np.zeros(10))
    assert len(calls)==2 and b.count==2


def test_failed_static_bound_never_becomes_endpoint_success():
    row=dict(required_gap_intervals_m=[[-.0002,.00001]]*3,
        geometric_thumb_partner_dot=[-.5,-.5],minimum_nonrequired_support_gap_m=.001,
        minimum_table_gap_m=.0022)
    r=m.endpoint_gates(row)
    assert r["coarse_endpoint_passed"] is True
    assert r["strict_endpoint_passed"] is False
    assert r["native_ready"] is False and r["reset_verified"] is False


def test_2p1mm_runtime_floor_is_kept_separate_from_static_1p9():
    row=dict(required_gap_intervals_m=[[.00001,.00002]]*3,
        geometric_thumb_partner_dot=[-.5,-.5],minimum_nonrequired_support_gap_m=.001,
        minimum_table_gap_m=.002)
    r=m.endpoint_gates(row)
    assert r["static_1p9mm_passed"] is True and r["runtime_2p1mm_passed"] is False
    assert r["strict_endpoint_passed"] is True
    assert r["candidate_all_endpoint_constraints_passed"] is False


def test_path_unknown_and_missing_side_cannot_be_bilateral_reset_proposal():
    rows={"left":{"candidate_all_endpoint_constraints_passed":True,"hand_y_bounds_m":[.1,.2]}}
    assert m.pair_gates(rows)["bilateral_endpoint_passed"] is False
    rows["right"]={"candidate_all_endpoint_constraints_passed":True,"hand_y_bounds_m":[-.2,-.1]}
    result=m.pair_gates(rows)
    assert result["bilateral_endpoint_passed"] is True
    assert result["continuous_path_verified"] is False and result["reset_proposal_created"] is False


def test_tampered_parent_receipt_refused(tmp_path):
    p=tmp_path/"wrong.json";p.write_text("{}")
    with pytest.raises(ValueError):m.load_parents(p,tmp_path,{})


def test_missing_geometric_direction_cannot_pass_endpoint_gate():
    row=dict(required_gap_intervals_m=[[.00001,.00002]]*3,
        geometric_thumb_partner_dot=[None,-.5],minimum_nonrequired_support_gap_m=.001,
        minimum_table_gap_m=.0022)
    assert m.endpoint_gates(row)["candidate_all_endpoint_constraints_passed"] is False


def test_envelope_support_does_not_miss_interior_between_axis_heights():
    vertices=np.array([[x,y,z] for x in (.429,.431) for y in (-.01,.01) for z in (.306,.307)])
    layout=dict(wire_points=[[.43,-.25,.308],[.43,.25,.308]],
                table_center=[.6,0,.25],table_size=[.8,1.2,.1])
    r=m.envelope_support(vertices,layout,.005)
    assert r["surface_gap_lower_bound_m"]<0
    assert r["direction_world"][1]==0


def test_bad_identity_is_not_relabelled_as_original_tripod():
    p=parent();p["independent_seed_sha256"]="0"*64
    with pytest.raises(ValueError):m.decode(p,np.zeros(10))


def test_unstable_finite_difference_does_not_call_solver_or_claim_direction():
    calls=[]
    def evaluate(x):
        calls.append(x.copy())
        return {"residual":[1.+x[0]**3]}
    result=m.local_direction(evaluate,lower=-np.ones(10),upper=np.ones(10))
    assert len(calls)==41 and result["jacobian_relative_difference"]>.2
    assert result["linear_solver_calls"]==0 and result["model_supported"] is False
    assert result["observed_descent"] is False


def test_partial_keeps_completed_queries_and_attempted_count(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(m,"load_geometry",lambda project,side:dict(source_sha256={}))
    parents={s:parent(s) for s in m.SIDES}
    monkeypatch.setattr(m,"load_parents",lambda *a:(parents,{}))
    calls=[]
    def measure(g,p,x):
        calls.append(x.copy())
        if len(calls)==3:raise m.BudgetExpired()
        return {"residual":[1.]}
    monkeypatch.setattr(m,"measure",measure)
    output=tmp_path/"out"
    assert m.main(["--output",str(output),"--wall-seconds","10"])==0
    result=json.loads((output/"result.json").read_text())
    assert result["status"]=="partial_timeout"
    assert result["side_evaluation_attempts"]=={"left":3}
    assert len(result["evaluation_records"])==2 and result["sides"]=={}
    assert result["pair"]["bilateral_endpoint_passed"] is False
    assert len((output/"progress.jsonl").read_text().splitlines())==2


def test_output_is_exclusive_and_preserves_existing_evidence(tmp_path):
    marker=tmp_path/"keep";marker.write_text("preserve")
    with pytest.raises(FileExistsError):m.main(["--output",str(tmp_path)])
    assert marker.read_text()=="preserve"
