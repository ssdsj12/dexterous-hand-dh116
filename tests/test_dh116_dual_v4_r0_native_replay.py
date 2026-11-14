"""The SHA-pinned native dual v4 ledger, never a generated SDK fixture."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import pytest

def api():
    return importlib.import_module("scripts.replay_dh116_dual_v4_r0")

@pytest.fixture(scope="module")
def replay():
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import FlexibleGraspEpisode
    calls=[]
    prior=FlexibleGraspEpisode.step
    def step(self,snapshot,measurements):
        calls.append(snapshot["rod_snapshot"]["physics_step"])
        return prior(self,snapshot,measurements)
    FlexibleGraspEpisode.step=step
    try:
        report,bundle,summary=api().replay_native_interval()
    finally:
        FlexibleGraspEpisode.step=prior
    assert calls==list(range(3,11))*2
    return report,bundle,summary

def test_exact_real_window_and_original_reducer_once_per_pass(replay):
    report,bundle,summary=replay
    assert summary["ledger_sha256"]=="ead0dbf2c15b83864a1c440848ed9cf159b86c88a6ade0e4275208efebcf778c"
    assert summary["physical_steps"]==list(range(3,11))
    assert summary["rod_ticks"]==list(range(12,97,12))
    assert summary["setup_steps"]==2 and summary["physical_samples"]==8
    assert summary["recorded_control_windows"]==summary["settled_control_windows"]==1
    assert summary["strict_json_bundle_replay_equal"] is True
    assert summary["new_physics_steps"]==summary["optimizer_updates"]==0
    assert summary["training_reward"] is summary["diagnostic_reward"] is None
    assert summary["teacher_admission"] is summary["native_qualification"] is False
    assert [r["op"] for r in bundle["events"]]==["record","begin"]+["record"]*8+["end"]

def test_actual_known_zero_and_unknown_are_separate(replay):
    report,bundle,summary=replay
    known={"required_contact":0.,"contact_persistence":0.,"grasp_geometry":0.,"joint_limit":0.}
    for row in report["controls"][0]["physical_rows"]:
        for key,value in known.items():
            assert row["r0"]["terms"][key]["raw"]==value
        assert row["measurements"]["table_contact"]["value"] is True
        assert row["r0"]["terms"]["external_support_violation"]["raw"]==1.
        assert row["r0"]["prohibited_support_latched"] is False
        assert row["r0"]["success_candidate"] is False
        for key in ("approach_progress","lift_progress","hold_progress","material_anchor_slip","instability"):
            assert row["r0"]["terms"][key]["raw"] is None
        for key in ("external_support","authored_force","attachment","stable","approach_distance_m"):
            assert row["measurements"][key]["value"] is None
        for side in ("left","right"):
            assert row["measurements"][side]["value"]["pads"]==[]
            assert row["measurements"][side]["value"]["material_s_m"] is None
    assert summary["hand_table_source_available"] is False
    assert summary["no_hand_table_contact_proven"] is False
    assert bundle["inputs"]["radius_sample"]=={"value":None,"source":None}

def test_coefficients_are_exact_labelled_existing_fixture_and_task_remains_unknown(replay):
    report,bundle,summary=replay
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import R0_TERMS
    assert bundle["inputs"]["diagnostic_coefficients"]=={n:float(i+1) for i,n in enumerate(R0_TERMS)}
    assert set(bundle["inputs"]["task"]["reward"]["coefficients"].values())=={None}
    assert summary["coefficient_scope"]=="existing_CPU_fixture_only_not_physical_reward"
    assert summary["budget_scope"]=="existing_CPU_reducer_capacity_not_native_run_authorization"

def test_changed_native_ledger_rejected_before_reducer(tmp_path):
    path=tmp_path/"physical.jsonl"
    path.write_bytes(api().LEDGER.read_bytes()+b"\n")
    with pytest.raises(ValueError,match="SHA"):
        api().load_native_rows(path)

@pytest.mark.parametrize("damage",["missing","extra","swapped"])
def test_exact_native_operation_boundary_cannot_be_padded_or_reordered(damage):
    rows,inspection=api().load_native_rows()
    rows=deepcopy(rows)
    if damage=="missing":rows.pop(7)
    elif damage=="extra":rows.insert(7,deepcopy(rows[7]))
    else:rows[6],rows[7]=rows[7],rows[6]
    with pytest.raises(ValueError,match="operation"):
        api().validate_operations(rows)

def test_existing_replay_cli_reads_real_bundle_on_disk_without_native_imports(replay,tmp_path):
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned
    report,bundle,summary=replay
    src=tmp_path/"bundle.json";out=tmp_path/"receipt.json"
    src.write_text(json.dumps(encode_owned(bundle),allow_nan=False))
    proc=subprocess.run([sys.executable,str(api().ROOT/"scripts/replay_dh116_flexible_r0.py"),
        "--input",str(src),"--output",str(out)],capture_output=True,text=True,timeout=120)
    assert proc.returncode==0,proc.stdout+proc.stderr
    receipt=json.loads(out.read_text())
    assert receipt["replay_complete"] and receipt["native_imports"]==[]
    assert receipt["report"]==encode_owned(report)

def test_cli_preserves_existing_output_before_reading_archive(tmp_path,monkeypatch):
    output=tmp_path/"prior.json";output.write_text("preserve")
    monkeypatch.setattr(api(),"replay_native_interval",lambda *a:pytest.fail("replay must not start"))
    with pytest.raises(SystemExit):
        api().main(["--output",str(output),"--bundle-output",str(tmp_path/"bundle.json")])
    assert output.read_text()=="preserve"
