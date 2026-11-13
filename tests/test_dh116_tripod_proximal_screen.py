from copy import deepcopy
import numpy as np
import pytest
from scripts import screen_dh116_tripod_proximal as m


def parent(side):
    return dict(candidate_id=m.PARENTS[side],side=side,independent_style="tripod",
        independent_seed_sha256=m.TRIPOD_SHA,active_joint_names=list(m.ACTIVE),
        active_q_rad=[.5,.44,.65,.70,.7,.95] if side=="left" else [.14,.032,.63,.66,.95,.2],
        base_pose_wxyz=[.53,.13 if side=="left" else -.19,.39,1.,0.,0.,0.])


def diagnostic(side):
    return dict(link=m.BLOCKER[side],geometric_axis_to_hull_direction=
        [-.632,0.,.775] if side=="left" else [1.,0.,0.])


def test_exact_120_independent_candidates_preserve_parent_and_three_digit_channels():
    parents={s:parent(s) for s in m.SIDES};old=deepcopy(parents)
    rows=[r for s in m.SIDES for r in m.candidates(parents[s],diagnostic(s))]
    assert len(rows)==120 and len({r["candidate_id"] for r in rows})==120
    for side in m.SIDES:
        selected=[r for r in rows if r["side"]==side]
        assert len(selected)==60
        np.testing.assert_allclose(selected[0]["base_pose_wxyz"],parents[side]["base_pose_wxyz"],atol=1e-15)
        assert selected[0]["active_q_rad"]==parents[side]["active_q_rad"]
        assert all(r["parent_candidate_id"]==m.PARENTS[side] for r in selected)
        assert all(r["active_q_rad"][1:]==parents[side]["active_q_rad"][1:] for r in selected)
        assert all(r["independent_style"]=="tripod" for r in selected)
    assert parents==old


@pytest.mark.parametrize("edit",[
    lambda p,d:p.update(candidate_id="pinch-left"),
    lambda p,d:p.update(independent_seed_sha256="0"*64),
    lambda p,d:d.update(link="finger13_link"),
    lambda p,d:d.update(geometric_axis_to_hull_direction=[0,1,0]),
])
def test_wrong_parent_seed_blocker_or_direction_refused(edit):
    p,d=parent("left"),diagnostic("left");edit(p,d)
    with pytest.raises(ValueError):list(m.candidates(p,d))


def test_parent_receipt_digest_rejected_before_geometry(tmp_path):
    p=tmp_path/"p.json";p.write_text("{}")
    with pytest.raises(ValueError):m.load_parents(p,tmp_path,{})


def test_original_tripod_endpoint_tolerance_and_three_digits(monkeypatch):
    calls=[]
    def measure(g,c,channels):
        calls.append(channels)
        return dict(c,coarse_geometry_passed=False,coarse_blockers=["fixture"],
                    max_required_endpoint_abs_gap_m=.00049,endpoint_all_shape_table_clearance_m=.002)
    monkeypatch.setattr(m,"measure_closure_grid",measure)
    monkeypatch.setattr(m,"endpoint_blocker",lambda g,c:dict(capsule_surface_gap_m=-.001))
    r=m.measure({},parent("left"))
    assert calls==[(1,2,3)]
    assert r["required_endpoint_tolerance_m"]==.0005
    assert r["coarse_geometry_passed"] is False


def path(side,passed=True,clearance=.002):
    return dict(candidate_id=side,geometry_preparation_passed=passed,
        continuous_table_clearance_lower_bound_m=dict(source={"a":clearance},cooked={"a":clearance+.001}),
        continuous_hand_y_bounds_m=[.1,.2] if side=="left" else [-.2,-.1])


def test_pairing_requires_both_complete_passed_paths_and_separation():
    assert m.pair_result({"left":path("left")})["bilateral_geometry_preparation_passed"] is False
    assert m.pair_result({"left":path("left"),"right":path("right",False)})["bilateral_geometry_preparation_passed"] is False
    right=path("right");right["continuous_hand_y_bounds_m"]=[.15,.3]
    assert m.pair_result({"left":path("left"),"right":right})["bilateral_geometry_preparation_passed"] is False


def test_static_1p9mm_success_does_not_grant_2p1mm_runtime_floor():
    result=m.pair_result({s:path(s) for s in m.SIDES})
    assert result["bilateral_geometry_preparation_passed"] is True
    assert result["runtime_floor_m"]==.0021
    assert result["runtime_floor_passed"] is False
    assert result["minimum_runtime_floor_margin_m"]==pytest.approx(-.0001)
    assert result["native_ready"] is False and result["teacher_admitted"] is False


def test_no_partial_or_one_side_proposal(tmp_path):
    result=dict(pair=m.pair_result({"left":path("left")}),status="complete")
    assert m.write_proposal(tmp_path,result)==False
    assert not (tmp_path/"static-reset-proposal.json").exists()
    result=dict(pair=m.pair_result({s:path(s) for s in m.SIDES}),status="partial_timeout")
    assert m.write_proposal(tmp_path,result)==False


def test_paths_bounded_and_unmeasured_endpoint_never_qualifies(monkeypatch):
    rows=[dict(candidate_id=str(i),side="left",coarse_geometry_passed=True,
               targeted_endpoint=dict(capsule_surface_gap_m=-.0001),max_required_endpoint_abs_gap_m=0) for i in range(3)]
    calls=[]
    monkeypatch.setattr(m,"verify_candidate_path",lambda g,c:calls.append(c) or path("left",False))
    checks=[]
    assert m.check_side_paths({},rows,checks,lambda r:None)==None
    assert calls==[] and checks==[]


def test_timeout_retains_completed_rows_and_exact_remaining_identity(tmp_path,monkeypatch):
    import json
    fake=dict(source_sha256={},reference=dict(reference_layout=dict(
        wire_points=[[.43,-.25,.308],[.43,.25,.308]],table_center=[.6,0,.25],table_size=[.8,1.2,.1])))
    monkeypatch.setattr(m,"load_geometry",lambda project,side:deepcopy(fake))
    parents={s:parent(s) for s in m.SIDES}
    prior=dict(sides={s:dict(selected=dict(path={})) for s in m.SIDES})
    monkeypatch.setattr(m,"load_parents",lambda *a:(parents,prior))
    monkeypatch.setattr(m,"endpoint_blocker",lambda g,p:diagnostic(p["side"]))
    calls=[]
    def measure(g,c):
        calls.append(c["candidate_id"])
        if len(calls)==3:
            raise m.BudgetExpired()
        return dict(c,coarse_geometry_passed=False)
    monkeypatch.setattr(m,"measure",measure)
    output=tmp_path/"out"
    assert m.main(["--output",str(output),"--wall-seconds","10"])==0
    result=json.loads((output/"result.json").read_text())
    ledger=[json.loads(line) for line in (output/"progress.jsonl").read_text().splitlines()]
    assert result["status"]=="partial_timeout" and result["unfinished_stage"]=="endpoint_screen"
    assert result["completed_candidates"]==2 and len(result["remaining_candidate_ids"])==118
    assert [r["candidate_id"] for r in result["rows"]]==calls[:2]
    assert [r["candidate_id"] for r in ledger]==calls[:2]
    assert result["remaining_candidate_ids"][0]==calls[2]
    assert result["static_reset_proposal_created"] is False


def test_exclusive_output_is_never_overwritten(tmp_path):
    marker=tmp_path/"keep";marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        m.main(["--output",str(tmp_path)])
    assert marker.read_text()=="preserve"


@pytest.mark.parametrize("wall",["0","51","nan"])
def test_unbounded_budget_is_refused(tmp_path,wall):
    with pytest.raises(SystemExit):
        m.main(["--output",str(tmp_path/"out"),"--wall-seconds",wall])
    assert not (tmp_path/"out").exists()
