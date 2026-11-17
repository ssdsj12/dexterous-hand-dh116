"""CPU fixtures and rejection boundaries; no policy or optimizer is created."""
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned, encode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_dual_evidence import DualHandEvidenceJournal
from test_deformable_r0_replay import setup
from test_dh116_dual_hand_evidence_journal import connect

MODULE = "go2_pvcnn.control.dh116.tabletop_readonly_control_cost"

def api():
    assert importlib.util.find_spec(MODULE) is not None, "read-only evaluator not implemented"
    return importlib.import_module(MODULE)

def encoded(rows):
    return b"".join(json.dumps(encode_owned(r), allow_nan=False).encode()+b"\n" for r in rows)

@pytest.fixture(scope="module")
def ledger(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("control-cost")
    with pytest.MonkeyPatch.context() as patch:
        episode, _, _, _, rows, _ = setup(patch, tmp, mode="empty")
        path = tmp/"cpu.jsonl"
        with DualHandEvidenceJournal(path, physics_budget_end=18, solver_substeps=12) as journal:
            connect(journal, episode, rows[0])
            for i in range(2):
                action = torch.zeros((1,24), dtype=torch.float64)
                action[0,0] = i*.2
                journal.begin(episode.pre_action_state(), action)
                journal.end(episode.control(action))
    return [decode_owned(json.loads(x)) for x in path.read_bytes().splitlines()]

def evaluate(rows):
    raw = encoded(rows)
    return api().evaluate_bytes(raw, expected_sha256=hashlib.sha256(raw).hexdigest(), grasp="pinch")

def physical(rows):
    return [r["payload"]["exchange"] for r in rows if r.get("operation")=="physical"]

def test_exact_original_numeric_formula_and_mixed_dof_semantics():
    r = api().formula_components(np.ones((1,24)), np.zeros((1,24)), np.full((2,17),6.))
    assert r == pytest.approx(dict(time=-.1/480, action_rate=-.125/480, velocity_excess=-.1/480))
    r = api().formula_components(np.zeros((1,24)), None, np.zeros((2,17)))
    assert r["action_rate"] is None
    assert r["time"] < 0 and r["velocity_excess"] == 0

def test_two_windows_unknown_first_previous_action_is_not_zero(ledger):
    r = evaluate(ledger)
    assert r["physical_samples"] == 16 and len(r["controls"]) == 2
    assert r["controls"][0]["action_rate"]["contribution_sum"] is None
    assert r["controls"][1]["previous_action"] == [[0.]*24]
    assert r["controls"][1]["action_rate"]["contribution_sum"] == pytest.approx(-.5*(.2/2)**2/24/60)
    assert r["totals"]["action_rate"]["valid_samples"] == 8
    assert r["totals"]["action_rate"]["contribution_sum"] is None
    assert r["training_reward"] is None and r["optimizer_updates"] == r["new_physics_steps"] == 0
    assert not r["teacher_admission"] and not r["native_qualification"]
    assert r["velocity_semantics"] == "max_abs_full_17DOF_per_hand_numeric_mixed_m_per_s_and_rad_per_s"

@pytest.mark.parametrize("damage", [
    "missing_qd","nan_qd","short_qd","joint_names","active_indices","target_indices",
    "target_value","owner_changed","same_left_right_owner","path_changed","row_index",
    "pre_post_jump","first_common_q","action","episode","clock","drop","duplicate",
    "incomplete_nonterminal","after_terminal","missing_end","partial_line",
])
def test_rejects_missing_source_identity_or_window_corruption(ledger, damage):
    rows = deepcopy(ledger); xs = physical(rows); x = xs[0]; a=x["action_readback"]
    hand=a["post"]["hands"]["left"]
    if damage=="missing_qd": del hand["qd"]
    elif damage=="nan_qd": hand["qd"][0,0]=np.nan
    elif damage=="short_qd": hand["qd"]=hand["qd"][:,:16]
    elif damage=="joint_names": hand["joint_names"][0]="invented"
    elif damage=="active_indices": hand["active_indices"]=[]
    elif damage=="target_indices": a["submission"]["expected_backend_targets"]["left"]["joint_indices"]=[]
    elif damage=="target_value": a["submission"]["expected_backend_targets"]["left"]["values"][0,0]+=.1
    elif damage=="owner_changed": hand["view_object_id"]+=1
    elif damage=="same_left_right_owner":
        for phase in ("pre","post"):
            a[phase]["hands"]["right"]["instance_object_id"]=a[phase]["hands"]["left"]["instance_object_id"]
    elif damage=="path_changed": hand["prim_paths"]=["/wrong"]
    elif damage=="row_index": hand["row_indices"]=[1]
    elif damage=="pre_post_jump": xs[1]["action_readback"]["pre"]["hands"]["left"]["qd"][0,0]+=.01
    elif damage=="first_common_q":
        for phase in ("pre","post"): a[phase]["hands"]["left"]["q"][0,0]+=.01
    elif damage=="action": a["request"]["action24"][0,0]=.8
    elif damage=="episode": a["post"]["identity"][0]["episode_uid"]="other"
    elif damage=="clock": a["post"]["identity"][0]["rod_tick"]+=1
    elif damage=="drop": rows.remove(next(r for r in rows if r.get("operation")=="physical"))
    elif damage=="duplicate": rows.insert(7,deepcopy(rows[7]))
    elif damage=="incomplete_nonterminal":
        rows=rows[:8]+[dict(operation="end",payload=dict(control_step=0,physical_samples=1,termination_reason=None,training_reward=None,teacher_admission=False))]
    elif damage=="after_terminal": x["guard_failure"]="fixture_guard"
    elif damage=="missing_end": rows=rows[:-1]
    raw=encoded(rows)
    if damage=="partial_line":raw=raw[:-1]
    with pytest.raises(ValueError):
        api().evaluate_bytes(raw,expected_sha256=hashlib.sha256(raw).hexdigest(),grasp="pinch")

def test_terminal_partial_keeps_actual_rows_without_padding(ledger):
    rows=deepcopy(ledger[:8]); physical(rows)[0]["guard_failure"]="fixture_guard"
    rows.append(dict(operation="end",payload=dict(control_step=0,physical_samples=1,
        termination_reason="fixture_guard",training_reward=None,teacher_admission=False)))
    r=evaluate(rows)
    assert r["physical_samples"]==1
    assert r["controls"][0]["physical_samples"]==1
    assert r["controls"][0]["terminal_reason"]=="fixture_guard"
    assert r["totals"]["time"]["contribution_sum"]==pytest.approx(-.1/480)
    assert r["controls"][0]["training_reward"] is None

def test_sha_rejected_before_json_parse():
    with pytest.raises(ValueError,match="SHA"):
        api().evaluate_bytes(b"not json",expected_sha256="0"*64,grasp="pinch")

def test_source_change_rejected(tmp_path):
    with pytest.raises(ValueError,match="source|contract"):
        api().load_contract(tmp_path)

def test_cli_exclusive_output_before_loading_input(tmp_path):
    out=tmp_path/"existing.json";out.write_text("retain")
    script=Path(__file__).resolve().parents[1]/"scripts/evaluate_dh116_readonly_control_cost.py"
    assert script.exists(), "read-only CLI not implemented"
    p=subprocess.run([sys.executable,str(script),"--grasp","power","--output",str(out)],capture_output=True,text=True)
    assert p.returncode!=0 and out.read_text()=="retain"

@pytest.mark.parametrize("which", ["action_nan","action_range","previous_nan","previous_range","velocity_nan","velocity_short"])
def test_formula_refuses_missing_or_invalid_input(which):
    action=np.zeros((1,24));previous=action.copy();velocity=np.zeros((2,17))
    if which=="action_nan":action[0,0]=np.nan
    if which=="action_range":action[0,0]=1.1
    if which=="previous_nan":previous[0,0]=np.nan
    if which=="previous_range":previous[0,0]=-1.1
    if which=="velocity_nan":velocity[0,0]=np.inf
    if which=="velocity_short":velocity=velocity[:,:16]
    with pytest.raises(ValueError):
        api().formula_components(action,previous,velocity)

def test_mixed_numeric_velocity_includes_linear_wrist_without_relabeling():
    velocity=np.zeros((2,17));velocity[0,0]=18.
    r=api().formula_components(np.zeros((1,24)),None,velocity)
    assert r["velocity_excess"]==pytest.approx(-.1*25/480)

def test_changed_frozen_source_is_rejected(tmp_path):
    module=api();contract=module.load_contract()
    for name in [module.CONTRACT,*contract["source_sha256"]]:
        dst=tmp_path/name;dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_bytes((module.ROOT/name).read_bytes())
    name="src/go2_pvcnn/control/dh116/tabletop_tensor_env.py"
    with (tmp_path/name).open("ab") as stream:stream.write(b"\n# changed\n")
    with pytest.raises(ValueError,match="source SHA"):
        module.load_contract(tmp_path)

@pytest.mark.parametrize("text", ['{"version":1,"version":2}\n','{"version":NaN}\n'])
def test_duplicate_or_nonstandard_json_rejected(text):
    raw=text.encode()
    with pytest.raises(ValueError):
        api().evaluate_bytes(raw,expected_sha256=hashlib.sha256(raw).hexdigest(),grasp="pinch")

def test_power_without_actual_data_stays_unknown():
    r=api().evaluate_archive("power")
    assert r["status"]=="unavailable" and r["physical_samples"] is r["totals"] is None
    assert r["training_reward"] is None and not r["teacher_admission"]

@pytest.mark.parametrize("style,n,valid,expected_time,expected_action,last",[
    ("pinch",232,224,-.04833333333333334,-9.621746570074074e-05,8),
    ("tripod",197,189,-.04104166666666667,-.00012787752263806782,5),
])
def test_exact_historical_archive_sums_and_unknown_first_window(style,n,valid,expected_time,expected_action,last):
    r=api().evaluate_archive(style)
    assert r["physical_samples"]==n
    assert r["totals"]["time"]["contribution_sum"]==pytest.approx(expected_time,abs=1e-15)
    assert r["totals"]["action_rate"]["valid_samples"]==valid
    assert r["totals"]["action_rate"]["observed_sum"]==pytest.approx(expected_action,abs=1e-15)
    assert r["totals"]["action_rate"]["contribution_sum"] is None
    assert r["totals"]["velocity_excess"]["contribution_sum"]==0.
    assert r["controls"][-1]["physical_samples"]==last
    assert r["controls"][-1]["terminal_reason"]==("closure_geometry_guard" if style=="tripod" else None)
    assert r["training_reward"] is None and r["new_physics_steps"]==r["optimizer_updates"]==0

def test_hand_source_helper_rejects_duplicate_side_before_dictionary_collapse(ledger):
    common=deepcopy(next(r["payload"]["pre_action_state"]["common"] for r in ledger if r.get("operation")=="begin"))
    common["source_packets"].append(deepcopy(common["source_packets"][0]))
    snapshot=physical(ledger)[0]["action_readback"]["pre"]
    with pytest.raises(ValueError,match="two|unique|duplicate"):
        api()._hands(snapshot,common)

def test_extreme_finite_velocity_is_explicit_numeric_rejection():
    with pytest.raises(ValueError,match="overflow|numeric"):
        api().formula_components(np.zeros((1,24)),None,np.full((2,17),1e308))
