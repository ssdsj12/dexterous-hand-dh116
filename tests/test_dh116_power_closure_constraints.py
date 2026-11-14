from copy import deepcopy
import numpy as np
import pytest
from scripts import audit_dh116_power_closure_constraints as m


def samples(gap=.002):
    return [dict(active_delta_rad=float(a),capsule_surface_gap_m=gap) for a in np.linspace(0,.1,9)]


def test_continuous_certificate_deducts_half_interval_travel_and_numeric_margin():
    c=m.certify_curve(samples(),maximum_delta=.1,travel_bound=.008,support_lower_bounds=[r["capsule_surface_gap_m"] for r in samples()])
    assert c["continuous_surface_gap_lower_bound_m"]==pytest.approx(.001498)
    assert c["fixed_root_contact_excluded"] is True
    assert c["necessary_root_translation_m"]==pytest.approx(.000998)
    assert c["global_infeasible"] is False and c["native_ready"] is False


def test_sampled_positive_gap_alone_cannot_exclude_unsampled_contact():
    c=m.certify_curve(samples(.0008),maximum_delta=.1,travel_bound=.008,support_lower_bounds=[.0008]*9)
    assert c["continuous_surface_gap_lower_bound_m"]<.0005
    assert c["fixed_root_contact_excluded"] is False


@pytest.mark.parametrize("change",[
    lambda r:r.pop(),
    lambda r:r[0].update(active_delta_rad=.001),
    lambda r:r[3].update(active_delta_rad=.039),
    lambda r:r[5].update(capsule_surface_gap_m=float("nan")),
])
def test_curve_requires_original_nine_full_interval_finite_samples(change):
    r=samples();change(r)
    with pytest.raises(ValueError):m.certify_curve(r,maximum_delta=.1,travel_bound=.008,support_lower_bounds=[.002]*9)


@pytest.mark.parametrize("delta,bound",[(0,.01),(.11,.01),(.1,-.001),(.1,float("nan"))])
def test_invalid_budget_or_displacement_bound_rejected(delta,bound):
    with pytest.raises(ValueError):m.certify_curve(samples(),maximum_delta=delta,travel_bound=bound,support_lower_bounds=[.002]*9)


def test_joint_budget_is_not_mislabeled_as_joint_limit():
    q=np.array([.2,.1,.6,.6,.6,.6])
    record=m.joint_budget(q)
    assert record["allowed_positive_delta_rad"]==pytest.approx([.1]*6)
    assert all(record["upper_headroom_rad"][i]>.1 for i in range(6))
    assert all(record["limited_by_control_duration"])
    assert record["maximum_velocity_rad_s"]==.3
    assert record["duration_s"]==pytest.approx(1/3)


def test_tampered_prior_refused_before_geometry(tmp_path):
    p=tmp_path/"bad.json";p.write_text("{}")
    with pytest.raises(ValueError):m.load_selection(p,tmp_path,{})


def test_new_left_curve_but_reuse_right_avoid_duplicate_queries(monkeypatch):
    calls=[]
    def measure(g,c,ch):
        calls.append((c["side"],c["archive_start_index"],ch))
        return dict(closure_grid={"new":[]})
    monkeypatch.setattr(m,"measure_closure_grid",measure)
    saved=dict(closure_grid={"old":[1]},candidate_id="right")
    prior={"sides":{"right":{"closure_grid_probes":[
        {"archive_start_index":1,"grid_performed":True,"candidate":saved}]}}}
    row=dict(side="right",archive_start_index=1)
    original=deepcopy(saved)
    result,origin=m.obtain_curves({},row,prior)
    assert result==saved and saved==original and origin=="immutable_prior_nine_angle_measurements"
    result,origin=m.obtain_curves({},dict(side="left",archive_start_index=15),prior)
    assert calls==[("left",15,(1,2,3,4,5))]
    assert origin=="new_left_five_digit_nine_angle_measurements"


def test_initial_source_overlap_prevents_closure_curve_execution(monkeypatch):
    monkeypatch.setattr(m,"initial_geometry",lambda g,c:dict(
        source_rod_overlaps={"finger32_link":-.002},cooked_rod_overlaps={}))
    monkeypatch.setattr(m,"obtain_curves",lambda *a:pytest.fail("unsafe initial seed must stop before curve"))
    r=m.audit_one({},dict(side="right",archive_start_index=12),{})
    assert r["closure_evaluated"] is False
    assert r["blockers"]==["initial_source_or_cooked_rod_overlap"]
    assert r["native_ready"] is False


def test_cli_exclusive_output(tmp_path):
    marker=tmp_path/"keep";marker.write_text("a")
    with pytest.raises(FileExistsError):m.main(["--output",str(tmp_path)])
    assert marker.read_text()=="a"


def test_timeout_preserves_original_root_identity_without_fake_completion(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(m,"load_geometry",lambda project,side:dict(source_sha256={}))
    selected=[dict(side=s,archive_start_index=i) for s,i in m.SELECTION]
    monkeypatch.setattr(m,"load_selection",lambda *a:(selected,{}))
    monkeypatch.setattr(m,"source_inventory",lambda:[])
    calls=[]
    def audit(g,r,p):
        calls.append((r["side"],r["archive_start_index"]))
        if len(calls)==3:raise m.BudgetExpired()
        return dict(r,closure_evaluated=False,native_ready=False)
    monkeypatch.setattr(m,"audit_one",audit)
    output=tmp_path/"out"
    assert m.main(["--output",str(output),"--wall-seconds","10"])==0
    result=json.loads((output/"result.json").read_text())
    assert result["status"]=="partial_timeout" and len(result["rows"])==2
    assert result["remaining_original_roots"]==[list(x) for x in m.SELECTION[2:]]
    assert result["new_left_closure_curves"]==0 and result["reused_right_closure_curves"]==0
    assert result["reset_proposal_created"] is False
    assert len((output/"progress.jsonl").read_text().splitlines())==2


def test_qp_estimates_alone_cannot_be_used_as_lower_bound():
    with pytest.raises(TypeError):
        m.certify_curve(samples(),maximum_delta=.1,travel_bound=.008)


def test_analytic_lower_bound_not_feasible_qp_estimate_controls_certificate():
    c=m.certify_curve(samples(.02),maximum_delta=.1,travel_bound=.008,
                      support_lower_bounds=[.0001]*9)
    assert c["fixed_root_contact_excluded"] is False


def test_support_plane_is_bounded_by_known_convex_box_distance():
    vertices=np.array([[x,y,z] for x in (2.,3.) for y in (-.2,.2) for z in (-.2,.2)])
    axis=np.array([[0.,-1.,0.],[0.,1.,0.]])
    r=m.support_plane_bound(vertices,axis,.5)
    assert r["surface_gap_lower_bound_m"]<=1.5
    assert r["surface_gap_lower_bound_m"]==pytest.approx(1.5,abs=1e-10)
    assert np.linalg.norm(r["direction_world"])<=1
    overlap=m.support_plane_bound(vertices-2.5*np.array([1.,0.,0.]),axis,.5)
    assert overlap["distance_lower_bound_m"]==0


def test_tampered_frozen_recheck_cannot_reuse_prior_number(tmp_path):
    p=tmp_path/"wrong.json";p.write_text("{}")
    with pytest.raises(ValueError):m.read_frozen_report(p,tmp_path,{})


def test_recheck_restores_signal_handler_after_complete_frozen_rows(tmp_path,monkeypatch):
    import json,signal
    before=signal.getsignal(signal.SIGALRM)
    monkeypatch.setattr(m,"load_geometry",lambda project,side:dict(source_sha256={}))
    selected=[dict(side=s,archive_start_index=i) for s,i in m.SELECTION]
    monkeypatch.setattr(m,"load_selection",lambda *a:(selected,{}))
    monkeypatch.setattr(m,"source_inventory",lambda:[])
    monkeypatch.setattr(m,"read_frozen_report",lambda *a:dict(rows=selected))
    monkeypatch.setattr(m,"recheck_row",lambda g,r,old:dict(r,curve_source="frozen_report_samples_no_new_QP"))
    output=tmp_path/"out"
    assert m.main(["--output",str(output),"--recheck-report",str(tmp_path/"input")])==0
    assert signal.getsignal(signal.SIGALRM)==before
    result=json.loads((output/"result.json").read_text())
    assert result["status"]=="complete" and len(result["rows"])==5
    assert result["new_closure_curve_queries"]==0
