"""CPU arithmetic fixtures; no native owner, model or optimizer."""
from copy import deepcopy
import importlib
import math
import pytest

def module():
    return importlib.import_module("go2_pvcnn.control.dh116.tabletop_approach_control_composer")

def window(index=0, count=8, *, terminal=None, previous=None):
    m=module()
    epoch=dict(env_id=0,episode_uid="fixture:one",reset_id=1,rod_uid="rod0")
    start=2+index*8
    rows=[];physical=[];r0rows=[];costrows=[]
    for i in range(count):
        step=start+i+1
        identity=dict(epoch,run_uuid="fixture",physics_step=step,rod_tick=(step-2)*12,
            policy_step=index,substep=i,timestamp_s=step/480,solver_time_s=(step-2)/480,clock_valid=True)
        reason=terminal if i==count-1 else None
        event=dict(exchange=dict(physics_step=step,policy_step=index,guard_failure=reason,
            action_readback=dict(post=dict(identity=[identity],hands={side:dict(qd=[[0.]*17]) for side in ("left","right")}))))
        joint=float(reason is not None and reason.endswith(":joint_limit"))
        r=dict(epoch,grasp="pinch",physics_step=step,rod_tick=identity["rod_tick"],
            timestamp_s=identity["timestamp_s"],terms=dict(
              approach_progress=dict(raw=.0001,unit="m",valid=True,source="fixture:distance"),
              joint_limit=dict(raw=joint,unit="event",valid=True,source="fixture:joint")),
            training_reward=None)
        wrapped=dict(r0=r,physical_guard_failure=reason)
        live=dict(identity=identity,actual_event_sha256=m._digest(event),r0_row=wrapped)
        physical.append(event);r0rows.append(wrapped);rows.append(live)
        costrows.append(dict(physics_step=step,rod_tick=identity["rod_tick"],policy_step=index,
            substep=i,actual_max_abs_joint_velocity_numeric=0.,guard_failure=reason,
            components=dict(time=-.1/480,action_rate=None if previous is None else 0.,velocity_excess=0.),
            training_reward=None))
    result=dict(control_step=index,physical_samples=count,termination_reason=terminal,
        training_reward=None,teacher_admission=False)
    receipt=dict(epoch,grasp="pinch",control_index=index,physical_samples=count,
        action24=[0.]*24,physics_step_start=start,physics_step_end=start+count,
        rod_tick_start=(start-2)*12,rod_tick_end=(start+count-2)*12,
        timestamp_start_s=start/480,timestamp_end_s=(start+count)/480,
        expected_full_interval_samples=8,partial_interval_due_to_terminal=terminal is not None and count<8,
        terminated=terminal is not None,training_reward=None,
        terms={key:dict(unit=unit,valid_samples=count,expected_samples=count,
            raw_sum=math.fsum(r["r0"]["terms"][key]["raw"] for r in r0rows))
            for key,unit in (("approach_progress","m"),("joint_limit","event"))})
    costs=dict(control_index=index,physical_samples=count,terminal_reason=terminal,
        complete_eight_sample_window=count==8,action=[[0.]*24],
        previous_action=previous,rows=costrows,training_reward=None)
    return dict(r0_end=dict(control_receipt=receipt,physical_rows=r0rows,control_result=result,
                training_reward=None),live_rows=rows,physical_rows=physical,
                physical_result=result,cost_control=costs)

def composer():
    m=module()
    return m.ApproachControlComposer(m.candidate_spec(100.))

def test_first_window_remains_unknown():
    c=composer();r=c.compose(window())
    assert r["candidate_reward"] is None
    assert r["components"]["approach_progress"] == pytest.approx(.08)
    assert r["components"]["action_rate"] is None
    assert r["missing_terms"] == ["action_rate"]
    assert r["training_reward"] is None

def test_known_window_formula_and_no_extra_dt_or_divide_eight():
    c=composer();c.compose(window())
    r=c.compose(window(1,previous=[[0.]*24]))
    assert r["candidate_reward"] == pytest.approx(.08-.1*8/480)
    assert r["components"]["joint_limit_terminal"]==0.
    assert r["physical_samples"]==8

def test_negative_progress_is_retained():
    w=window();m=module()
    for i in range(8):
        w["r0_end"]["physical_rows"][i]["r0"]["terms"]["approach_progress"]["raw"]=-.0001
        w["live_rows"][i]["r0_row"]=deepcopy(w["r0_end"]["physical_rows"][i])
    w["r0_end"]["control_receipt"]["terms"]["approach_progress"]["raw_sum"]=-.0008
    assert composer().compose(w)["components"]["approach_progress"]==pytest.approx(-.08)

def test_terminal_partial_only_pays_one_joint_event():
    c=composer();c.compose(window())
    r=c.compose(window(1,3,terminal="left:joint_limit",previous=[[0.]*24]))
    assert r["candidate_reward"]==pytest.approx(.03-.1*3/480-25.)
    assert r["physical_samples"]==3
    with pytest.raises(ValueError,match="closed"):
        c.compose(window(2,previous=[[0.]*24]))

def test_no_default_coefficient_and_bad_units():
    m=module()
    with pytest.raises(TypeError):m.candidate_spec()
    for value in (None,True,0.,-1.,float("inf"),float("nan")):
        with pytest.raises(ValueError):m.candidate_spec(value)
    s=m.candidate_spec(100.);s["approach_coefficient_unit"]="points/s"
    with pytest.raises(ValueError):m.ApproachControlComposer(s)

@pytest.mark.parametrize("damage",[
    "first_previous_filled","missing_row","cost_order","epoch","event_hash","unit",
    "sum","terminal","early_guard","partial_nonterminal","action","cost_component",
    "duplicate_component","joint_repeated","joint_without_terminal","source_missing",
    "row_clock","receipt_clock","physical_result","row_shadow"])
def test_misaligned_or_changed_input_rejected(damage):
    w=window();m=module()
    if damage=="first_previous_filled":
        w["cost_control"]["previous_action"]=[[0.]*24]
        for r in w["cost_control"]["rows"]:r["components"]["action_rate"]=0.
    elif damage=="missing_row":w["live_rows"].pop()
    elif damage=="cost_order":w["cost_control"]["rows"].reverse()
    elif damage=="epoch":w["live_rows"][0]["identity"]["episode_uid"]="other"
    elif damage=="event_hash":w["live_rows"][0]["actual_event_sha256"]="0"*64
    elif damage=="unit":w["r0_end"]["control_receipt"]["terms"]["approach_progress"]["unit"]="mm"
    elif damage=="sum":w["r0_end"]["control_receipt"]["terms"]["approach_progress"]["raw_sum"]+=.1
    elif damage=="terminal":w["cost_control"]["terminal_reason"]="left:joint_limit"
    elif damage=="early_guard":
        w["physical_rows"][0]["exchange"]["guard_failure"]="left:joint_limit"
        w["live_rows"][0]["actual_event_sha256"]=m._digest(w["physical_rows"][0])
    elif damage=="partial_nonterminal":w=window(count=3)
    elif damage=="action":w["cost_control"]["action"][0][0]=.1
    elif damage=="cost_component":w["cost_control"]["rows"][0]["components"]["time"]=0.
    elif damage=="duplicate_component":w["cost_control"]["rows"][0]["components"]["failure"]=-25.
    elif damage=="joint_repeated":
        for i in (0,1):
            w["r0_end"]["physical_rows"][i]["r0"]["terms"]["joint_limit"]["raw"]=1.
            w["live_rows"][i]["r0_row"]=deepcopy(w["r0_end"]["physical_rows"][i])
        w["r0_end"]["control_receipt"]["terms"]["joint_limit"]["raw_sum"]=2.
    elif damage=="joint_without_terminal":
        w["r0_end"]["physical_rows"][-1]["r0"]["terms"]["joint_limit"]["raw"]=1.
        w["live_rows"][-1]["r0_row"]=deepcopy(w["r0_end"]["physical_rows"][-1])
        w["r0_end"]["control_receipt"]["terms"]["joint_limit"]["raw_sum"]=1.
    elif damage=="source_missing":
        w["r0_end"]["physical_rows"][0]["r0"]["terms"]["approach_progress"]["source"]=None
        w["live_rows"][0]["r0_row"]=deepcopy(w["r0_end"]["physical_rows"][0])
    elif damage=="row_clock":w["cost_control"]["rows"][0]["rod_tick"]+=12
    elif damage=="receipt_clock":w["r0_end"]["control_receipt"]["physics_step_end"]+=1
    elif damage=="physical_result":w["physical_result"]=dict(w["physical_result"],physical_samples=7)
    else:w["live_rows"][0]["r0_row"]=dict(w["live_rows"][0]["r0_row"],extra=True)
    with pytest.raises(ValueError):composer().compose(w)

def test_duplicate_window_and_after_failure_cannot_settle():
    c=composer();w=window();c.compose(w)
    with pytest.raises(ValueError):c.compose(w)
    with pytest.raises(ValueError,match="invalid|closed"):c.compose(window(1,previous=[[0.]*24]))

def test_unknown_approach_not_zero_filled():
    w=window()
    w["r0_end"]["physical_rows"][0]["r0"]["terms"]["approach_progress"].update(raw=None,valid=False,source=None)
    w["live_rows"][0]["r0_row"]=deepcopy(w["r0_end"]["physical_rows"][0])
    w["r0_end"]["control_receipt"]["terms"]["approach_progress"].update(raw_sum=None,valid_samples=7)
    r=composer().compose(w)
    assert r["candidate_reward"] is None and r["components"]["approach_progress"] is None

def test_spec_drift_between_windows_rejected():
    c=composer();c.compose(window())
    c.spec["approach_points_per_m"]=200.
    with pytest.raises(ValueError,match="spec|candidate"):c.compose(window(1,previous=[[0.]*24]))

def test_pinned_reader_parses_only_checked_bytes_and_rejects_change(tmp_path):
    m=module();p=tmp_path/"one.json";p.write_bytes(b'{"value":1}')
    import hashlib
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    assert m._pinned(p,sha)==b'{"value":1}'
    p.write_bytes(b'{"value":2}')
    with pytest.raises(ValueError,match="SHA"):m._pinned(p,sha)

def test_archive_rejects_changed_audit_proof_before_replay(tmp_path,monkeypatch):
    m=module();p=tmp_path/"proof.json";p.write_text("{}")
    monkeypatch.setattr(m,"PROOF",p)
    with pytest.raises(ValueError,match="SHA"):m.evaluate_archive(approach_points_per_m=100.)

def test_cli_existing_output_rejected_before_evaluation(tmp_path,monkeypatch):
    cli=importlib.import_module("scripts.evaluate_dh116_approach_control_candidate")
    p=tmp_path/"existing.json";p.write_text("keep")
    monkeypatch.setattr(cli,"evaluate_archive",lambda **kwargs:pytest.fail("must reject before replay"))
    with pytest.raises(SystemExit):cli.main(["--output",str(p),"--approach-points-per-m","100"])
    assert p.read_text()=="keep"

def test_cli_requires_explicit_weight():
    cli=importlib.import_module("scripts.evaluate_dh116_approach_control_candidate")
    with pytest.raises(SystemExit):cli.main(["--output","unused.json"])

def test_pinned_historical_232_rows_preserve_first_na():
    result=module().evaluate_archive(approach_points_per_m=100.)
    assert result["physical_samples"]==232
    assert result["known_reward_windows"]==28
    assert result["complete_candidate_reward"] is None
    assert result["controls"][0]["missing_terms"]==["action_rate"]
    assert result["known_reward_sum"]==pytest.approx(-.01859693043286804,abs=1e-14)
    assert result["totals"]["approach_progress"]["complete_sum"]==pytest.approx(.028162961825728416)
    assert result["training_reward"] is None and result["geometry_recomputed"] is False
