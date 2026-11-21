import copy,json
import numpy as np
import pytest
from scripts import verify_dh116_902_fixed_wrist_open_reset as m
from scripts import verify_dh116_902_flexible_reset as old

CANDIDATE=old.ROOT/"configs/dh116_tasks/flexible_902_tripod_fixed_wrist_open_cpu_v1.json"

def test_only_right_initial_pose_changes_and_float32_matches_screen():
    _,original,_=old.load_candidate("tripod")
    config,files=m.reset_config(CANDIDATE)
    expected=copy.deepcopy(original)
    candidate=json.loads(CANDIDATE.read_bytes())
    request=json.loads((old.ROOT/candidate["source_request"]).read_bytes())
    expected["hands"]["right"]["pose_wxyz"]=request["target"]["pose_wxyz"]
    expected["hands"]["right"]["q_rad"]=np.asarray(candidate["opening_active_q_rad"],dtype=np.float32).astype(float).tolist()
    assert config==expected
    assert old.CAMERA in files and old.HAND_ORDER_CAMERA in files
    assert config["hands"]["left"]==original["hands"]["left"]

def proof():
    return dict(initial_passed=True,initial={side:dict(passed=True,shape_count=22,frame_count=239)
       for side in ("left","right")},closure_geometry_passed=True,closure_passed=False,
       endpoint_contact_candidate_passed=False,native_admissible=False,teacher_admission=False,
       native_runs=0,ppo_updates=0,elapsed_seconds=12.3)

def test_reset_admission_preserves_failed_contact_and_is_deterministic():
    row=m.checked_reset_proof(proof())
    assert "elapsed_seconds" not in row
    assert row["closure_passed"] is False and row["teacher_admission"] is False
    assert m.checked_reset_proof({**proof(),"elapsed_seconds":3})==row

@pytest.mark.parametrize("field,value",[("initial_passed",False),("native_admissible",True),
    ("teacher_admission",True),("native_runs",1)])
def test_incomplete_or_escalated_evidence_cannot_freeze(field,value):
    p=proof();p[field]=value
    with pytest.raises(ValueError):m.checked_reset_proof(p)

def test_partial_archive_rejected():
    p=proof();p["initial"]["right"]["frame_count"]=238
    with pytest.raises(ValueError):m.checked_reset_proof(p)

def test_native_runner_keeps_all_gates_and_exact_budget():
    from pathlib import Path
    import ast
    ast.parse(Path(m.__file__).read_text())
    assert m.SDK_END==240 and m.SOLVER_STEPS==2856
    assert m.BUDGET["control_steps"]==0 and m.BUDGET["ppo_updates"]==0


def test_child_screen_preserves_failed_contact_exit_one(monkeypatch,tmp_path):
    def child(argv,**kwargs):
        from types import SimpleNamespace
        from pathlib import Path
        Path(argv[argv.index("--output")+1]).write_text(json.dumps(proof()))
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(m.subprocess,"run",child)
    result=m.run_screen(CANDIDATE,tmp_path/"receipt")
    assert not result["closure_passed"] and result["initial_passed"]

def test_missing_child_proof_is_not_accepted(monkeypatch,tmp_path):
    from types import SimpleNamespace
    monkeypatch.setattr(m.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=1))
    with pytest.raises(ValueError):m.run_screen(CANDIDATE,tmp_path/"receipt")


def test_frozen_receipt_must_match_fresh_screen(tmp_path):
    receipt=tmp_path/"receipt";receipt.write_text("x")
    frozen={str(receipt.resolve()):"new"}
    with pytest.raises(ValueError):
        m.check_frozen_proof({"settling_outer_sha256":"old","source_sha256":{}},frozen,receipt)

def test_frozen_source_must_match_fresh_screen(tmp_path):
    receipt=tmp_path/"receipt";receipt.write_text("x")
    frozen={str(receipt.resolve()):"same",str(CANDIDATE.resolve()):"new"}
    with pytest.raises(ValueError):
        m.check_frozen_proof({"settling_outer_sha256":"same","source_sha256":{str(CANDIDATE.relative_to(m.ROOT)):"old"}},frozen,receipt)

@pytest.mark.parametrize("bad",["failure","missing","short","pair"])
def test_natural_exit_is_not_internal_success(tmp_path,bad):
    result=dict(sdk_steps=240,measured_exchanges=238,solver_substeps=2856,
        pair_counts={"left.pad":0,"right.pad":0,"support_0":12,"support_1":15},teacher_admission=False,ppo_updates=0,rods_per_env=1)
    if bad=="short":result["measured_exchanges"]=237
    if bad=="pair":result["pair_counts"]["right.pad"]=1
    if bad!="missing":(tmp_path/"tripod_shared_result.json").write_text(json.dumps(result))
    if bad=="failure":(tmp_path/"tripod_failure.json").write_text("{}")
    assert not m.completed_interval(tmp_path)


def test_complete_no_contact_interval_is_separate_from_grasp(tmp_path):
    row=dict(sdk_steps=240,measured_exchanges=238,solver_substeps=2856,
        pair_counts={"left.pad":0,"right.pad":0,"support_0":12,"support_1":15},teacher_admission=False,ppo_updates=0,rods_per_env=1)
    (tmp_path/"tripod_shared_result.json").write_text(json.dumps(row))
    for sdk in range(3,241):(tmp_path/f"tripod_exchange_{sdk}.json").write_text("{}")
    assert m.completed_interval(tmp_path)
    (tmp_path/"tripod_exchange_100.json").unlink()
    assert not m.completed_interval(tmp_path)


@pytest.mark.parametrize("bad",["missing_support","unsupported","no_hands"])
def test_support_evidence_is_required(tmp_path,bad):
    counts={"left.pad":0,"right.pad":0,"support_0":12,"support_1":15}
    if bad=="missing_support":counts.pop("support_1")
    if bad=="unsupported":counts["support_1"]=0
    if bad=="no_hands":counts={k:v for k,v in counts.items() if k.startswith("support")}
    row=dict(sdk_steps=240,measured_exchanges=238,solver_substeps=2856,
        pair_counts=counts,teacher_admission=False,ppo_updates=0,rods_per_env=1)
    (tmp_path/"tripod_shared_result.json").write_text(json.dumps(row))
    for sdk in range(3,241):(tmp_path/f"tripod_exchange_{sdk}.json").write_text("{}")
    assert not m.completed_interval(tmp_path)
