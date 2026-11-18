from copy import deepcopy
import numpy as np
import pytest
from scripts import solve_dh116_power_joint_palm_direction as m


def parent(side="left"):
    return dict(candidate_id=m.PARENT_IDS[side],side=side,archive_start_index=m.SELECTED[side],
        independent_style="power",independent_seed_sha256=m.POWER_SHA,
        mode="minimum_table_clearance_root",base_pose_wxyz=[.52,.18 if side=="left" else -.19,.42,1.,0.,0.,0.],
        active_q_rad=[.2,.2,.6,.6,.7,.7],
        initial_rod_settling_gaps_m={"source":{"finger22_link":.003},"cooked":{"finger14_link":.005}},
        initial_all_shape_table_clearance_m=.002)


def test_joint_palm_codec_covers_all_six_active_and_fixed_world_y_rotation():
    p=parent();saved=deepcopy(p);x=np.arange(10)/20
    row=m.decode(p,x)
    assert p==saved
    np.testing.assert_allclose(row["base_pose_wxyz"][:3],np.array(p["base_pose_wxyz"][:3])+x[:3]*.015)
    np.testing.assert_allclose(row["active_q_rad"],np.array(p["active_q_rad"])+x[4:]*.1)
    assert row["base_pose_wxyz"][4]==0 and row["base_pose_wxyz"][6]==0
    assert row["side"]=="left" and row["reset_verified"] is False
    assert np.linalg.norm(row["base_pose_wxyz"][3:])==pytest.approx(1)


@pytest.mark.parametrize("edit",[
    lambda p:p.update(independent_seed_sha256="0"*64),
    lambda p:p.update(candidate_id="pinch"),
    lambda p:p.update(archive_start_index=12),
    lambda p:p["initial_rod_settling_gaps_m"]["source"].update(finger22_link=-.001),
    lambda p:p.update(initial_all_shape_table_clearance_m=.0018),
])
def test_only_original_nonoverlapping_independent_power_seed_allowed(edit):
    p=parent();edit(p)
    with pytest.raises(ValueError):m.validate_parent(p)


def test_parameter_bounds_use_original_six_joint_limits_and_no_previous_closure_delta():
    p=parent("right");p["active_q_rad"][1]=.02
    lo,hi=m.parameter_bounds(p)
    assert lo.shape==hi.shape==(10,)
    assert -.2<lo[5]<0 and np.all(hi<=1)
    zero=m.decode(p,np.zeros(10))
    assert zero["active_q_rad"]==p["active_q_rad"]


def test_power_gate_uses_all_five_pads_and_four_directions():
    r=dict(required_gap_intervals_m=[[.00001,.00002]]*5,
        geometric_thumb_partner_dot=[-.5]*4,minimum_nonrequired_support_gap_m=.001,
        minimum_table_gap_m=.0022)
    good=m.power_endpoint_gates(r)
    assert good["candidate_all_endpoint_constraints_passed"] is True
    assert "five" in good["scoped_constraints"]
    r["required_gap_intervals_m"][-1]=[.001,.0011]
    assert m.power_endpoint_gates(r)["candidate_all_endpoint_constraints_passed"] is False
    r["required_gap_intervals_m"][-1]=[.00001,.00002]
    r["geometric_thumb_partner_dot"][-1]=None
    assert m.power_endpoint_gates(r)["candidate_all_endpoint_constraints_passed"] is False


@pytest.mark.parametrize("count", [0,3,4,6])
def test_power_cannot_silently_reduce_to_tripod(count):
    r=dict(required_gap_intervals_m=[[0.,.00001]]*count,
        geometric_thumb_partner_dot=[-.5]*4,minimum_nonrequired_support_gap_m=.001,
        minimum_table_gap_m=.0022)
    with pytest.raises(ValueError):m.power_endpoint_gates(r)


def test_shared_two_scale_solver_is_reused_and_remains_45_per_side():
    from scripts.solve_dh116_tripod_joint_root_direction import local_direction,EvaluationBudget
    assert m.local_direction is local_direction and m.EvaluationBudget is EvaluationBudget
    calls=[]
    b=m.EvaluationBudget(lambda x:calls.append(x.copy()) or {"residual":(x-.2).tolist()})
    result=m.local_direction(b,lower=-np.ones(10),upper=np.ones(10))
    assert b.count==len(calls)==45 and result["observed_descent"] is True


def test_tampered_report_rejected_before_new_queries(tmp_path):
    p=tmp_path/"bad.json";p.write_text("{}")
    with pytest.raises(ValueError):m.load_power_parents(p,tmp_path,{})


def test_source_failure_and_one_side_never_emit_reset_proposal():
    assert m.pair_gates({})["reset_proposal_created"] is False
    assert m.pair_gates({})["bilateral_endpoint_passed"] is False


def test_output_exclusive(tmp_path):
    marker=tmp_path/"keep";marker.write_text("preserve")
    with pytest.raises(FileExistsError):m.main(["--output",str(tmp_path)])
    assert marker.read_text()=="preserve"


def test_partial_retains_exact_queries_without_promoting_one_side(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(m,"load_geometry",lambda project,side:dict(source_sha256={}))
    monkeypatch.setattr(m,"load_power_parents",lambda *a:({s:parent(s) for s in m.SIDES},{}))
    calls=[]
    def measure(g,p,x):
        calls.append(x.copy())
        if len(calls)==3:raise m.BudgetExpired()
        return {"residual":[1.]}
    monkeypatch.setattr(m,"measure",measure)
    out=tmp_path/"out"
    assert m.main(["--output",str(out),"--wall-seconds","10"])==0
    r=json.loads((out/"result.json").read_text())
    assert r["status"]=="partial_timeout" and r["side_evaluation_attempts"]=={"left":3}
    assert len(r["evaluation_records"])==2 and r["sides"]=={}
    assert r["pair"]["bilateral_endpoint_passed"] is False
