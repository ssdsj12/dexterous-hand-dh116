"""Original physical owner and R0 reducer with explicit CPU SDK/solver doubles."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_dh116_dual_hand_episode import setup as episode_setup
from test_dh116_deformable_control_step import inputs
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import (
    PhysicalR0Diagnostic, encode_owned, decode_owned, replay_diagnostic_bundle,
)


def setup(monkeypatch,tmp_path,mode="absent"):
    episode,rod,hands,sim,seed,owners,rows=episode_setup(monkeypatch,tmp_path)
    rod.env_id=0
    for b in episode._bindings:
        path=next(iter(b.env_paths.values()))
        b.env_paths.clear();b.env_paths[0]=path
    for hand in hands:
        hand.data.body_link_pose_w=hand.link_poses[..., [0,1,2,6,3,4,5]]
        hand.data.body_lin_vel_w=torch.zeros((1,23,3))
        hand.data.body_ang_vel_w=torch.zeros((1,23,3))
        hand.root_physx_view.get_link_velocities=lambda:torch.zeros((1,23,6))
        previous=hand.update
        def update(dt,hand=hand,previous=previous):
            previous(dt)
            hand.data.body_link_pose_w=hand.link_poses[..., [0,1,2,6,3,4,5]]
        hand.update=update
    if mode!="absent":
        old=rod.rope.step_free
        slots=[] if mode=="empty" else [5] if mode=="table" else [0,1,6,7]
        def step(dt,substeps):
            old(dt,substeps)
            rod.rope.force[:]=0;rod.rope.wrenches[:]=0
            rod.rope.force[0,0,2]=len(slots)
            for slot in slots:rod.rope.wrenches[0,slot,2]=-1.
        rod.rope.step_free=step
        def packet():
            ids=np.asarray([(sub,0,slot) for sub in range(12) for slot in slots],dtype=np.int64).reshape(-1,3)
            vectors=np.zeros((len(ids),12));vectors[:,5]=1.;vectors[:,8]=1.
            return dict(substeps=12,mesh_count=11,capacity=128,attempted=len(ids),overflow=False,
                indices=ids,vectors=vectors,node_forces_on_rod_world_n=rod.rope.force[0].copy(),
                mesh_wrenches_on_mesh_world=rod.rope.wrenches[0].copy())
        rod.rope.contact_pair_samples=packet
    task,budget,weights=inputs()
    original=deepcopy(task)
    diagnostic=PhysicalR0Diagnostic(task,"pinch",env_id=0,budget_config=budget,
        diagnostic_coefficients=weights,backend_artifact=episode._boundary._artifact,
        backend_sha256=episode._boundary._sha256,
        radius_sample={"value":.005,"source":"explicit CPU radius fixture, not native measurement"})
    episode.reset(seed,episode_uid="run-cpu:r0-replay",owner_resets=owners)
    diagnostic.record(rows[0])
    assert task==original
    return episode,rod,hands,sim,rows,diagnostic


def advance(episode,rows,diagnostic,*,aux=False):
    pre=episode.pre_action_state()
    action=torch.zeros((1,24),dtype=torch.float64)
    diagnostic.begin(pre,action)
    start=len(rows)
    result=episode.control(action)
    for event in rows[start:]:
        extra=None
        if aux:
            row=event["exchange"]["rod"]
            extra=dict(identity={k:row[k] for k in (
                "run_uuid","env_id","episode_uid","reset_id","rod_uid","physics_step","rod_tick","timestamp_s")},
                samples={name:{"value":([.1,.1] if name=="approach_distance_m" else name=="stable"),
                    "source":"explicit CPU auxiliary fixture, not native attestation"} for name in (
                        "approach_distance_m","external_support","authored_force","attachment","stable")})
        diagnostic.record(event,auxiliary=extra)
    return diagnostic.end(result)


def test_actual_owner_rows_replay_with_missing_pairs_and_frozen_unknowns(monkeypatch,tmp_path):
    episode,rod,hands,sim,rows,diagnostic=setup(monkeypatch,tmp_path)
    result=advance(episode,rows,diagnostic)
    assert result["control_receipt"]["physical_samples"]==8
    assert result["control_receipt"]["training_reward"] is None
    assert result["control_receipt"]["diagnostic_reward"] is None
    assert result["physical_rows"][0]["measurements"]["left"]["value"] is None
    assert result["physical_rows"][0]["measurements"]["table_contact"]["value"] is None
    assert result["control_receipt"]["terms"]["joint_limit"]["valid_samples"]==8
    assert result["diagnostic_backend_binding"]["differs_from_frozen_task"]
    assert result["r0_native_reset_complete"] is result["teacher_admission"] is False
    assert sim.current_time_step_index==10 and rod.rope.steps==8
    with pytest.raises(RuntimeError):diagnostic.end({"physical_samples":8})


def test_complete_empty_pairs_never_make_material_slip_zero_or_first_progress_valid(monkeypatch,tmp_path):
    episode,rod,hands,sim,rows,diagnostic=setup(monkeypatch,tmp_path,mode="empty")
    first=advance(episode,rows,diagnostic,aux=True)
    row=first["physical_rows"][0]
    assert row["measurements"]["left"]["value"]["pads"]==[]
    assert row["measurements"]["left"]["value"]["opposed"] is False
    assert row["measurements"]["left"]["value"]["material_s_m"] is None
    assert row["r0"]["terms"]["approach_progress"]["raw"] is None
    assert row["r0"]["terms"]["material_anchor_slip"]["raw"] is None
    assert first["control_receipt"]["terms"]["approach_progress"]["valid_samples"]==7
    assert first["control_receipt"]["terms"]["material_anchor_slip"]["valid_samples"]==0
    second=advance(episode,rows,diagnostic,aux=True)
    assert second["control_receipt"]["terms"]["approach_progress"]["valid_samples"]==8
    assert second["control_receipt"]["blocked_terms"]==["material_anchor_slip"]
    assert second["control_receipt"]["diagnostic_reward"] is None
    assert second["teacher_admission"] is False


def test_multpad_counts_rest_stations_and_normals_do_not_become_opposed_or_slip(monkeypatch,tmp_path):
    episode,rod,hands,sim,rows,diagnostic=setup(monkeypatch,tmp_path,mode="multi")
    result=advance(episode,rows,diagnostic)
    row=result["physical_rows"][0]
    assert row["pair_diagnostics"]["simultaneous_required_pad_substeps"]==12
    assert row["pair_diagnostics"]["hands"]["left"]["per_pad"]["finger14_link"]["loaded_node_rest_stations_m"]==[0.]
    assert row["measurements"]["left"]["value"] is None
    assert row["pair_diagnostics"]["effective_opposed_contact"] is None
    assert row["r0"]["terms"]["material_anchor_slip"]["raw"] is None
    assert row["r0"]["terms"]["required_contact"]["raw"] is None


def test_table_pair_is_separate_and_cannot_imply_no_external_support(monkeypatch,tmp_path):
    episode,rod,hands,sim,rows,diagnostic=setup(monkeypatch,tmp_path,mode="table")
    result=advance(episode,rows,diagnostic)
    measurement=result["physical_rows"][0]["measurements"]
    assert measurement["table_contact"]["value"] is True
    assert measurement["external_support"]["value"] is None
    assert measurement["authored_force"]["value"] is None
    assert measurement["stable"]["value"] is None


@pytest.mark.parametrize("damage",["action","rod_clock","pre_clock","backend","duplicate","aux_clock","slip_injection"])
def test_foreign_or_fabricated_sources_fail_closed(monkeypatch,tmp_path,damage):
    episode,rod,hands,sim,rows,diagnostic=setup(monkeypatch,tmp_path,mode="empty")
    pre=episode.pre_action_state()
    if damage=="pre_clock":
        pre["common"]["identity"]["policy_step"]=1
        with pytest.raises(ValueError):diagnostic.begin(pre,torch.zeros((1,24)))
        return
    diagnostic.begin(pre,torch.zeros((1,24)))
    episode.control(torch.zeros((1,24)))
    event=deepcopy(rows[1]);auxiliary=None
    if damage=="action":event["exchange"]["action_readback"]["request"]["action24"][0,0]=.5
    elif damage=="rod_clock":event["rod_boundary"]["rod_snapshot"]["physics_step"]+=1
    elif damage=="backend":event["rod_boundary"]["rod_snapshot"]["backend_sha256"]="0"*64
    elif damage=="duplicate":
        diagnostic.record(event)
    else:
        row=event["exchange"]["rod"]
        auxiliary=dict(identity={k:row[k] for k in (
            "run_uuid","env_id","episode_uid","reset_id","rod_uid","physics_step","rod_tick","timestamp_s")},samples={})
        if damage=="aux_clock":auxiliary["identity"]["physics_step"]+=1
        else:auxiliary["samples"]["left"]={"value":{},"source":"invented slip injection"}
    with pytest.raises((ValueError,RuntimeError)):diagnostic.record(event,auxiliary=auxiliary)
    with pytest.raises(RuntimeError):diagnostic.record(rows[2])


def test_joint_guard_terminal_settles_once_without_more_sdk_steps(monkeypatch,tmp_path):
    episode,rod,hands,sim,rows,diagnostic=setup(monkeypatch,tmp_path,mode="empty")
    step=sim.step
    def guarded(*,render):
        step(render=render)
        hands[1].measured[0,hands[1].joint_names.index("finger12")]=-.05
    sim.step=guarded
    result=advance(episode,rows,diagnostic)
    assert result["control_receipt"]["terminated"]
    assert result["control_receipt"]["terminal_reasons"]==["joint_limit"]
    assert result["control_receipt"]["physical_samples"]==1
    assert sim.current_time_step_index==3 and rod.rope.steps==1


def test_strict_disk_codec_preserves_unknown_contact_and_replays_same_receipt(monkeypatch,tmp_path):
    episode,rod,hands,sim,rows,diagnostic=setup(monkeypatch,tmp_path,mode="empty")
    advance(episode,rows,diagnostic,aux=True)
    bundle=diagnostic.bundle()
    path=tmp_path/"recorded-r0.json"
    path.write_text(json.dumps(encode_owned(bundle),allow_nan=False))
    loaded=decode_owned(json.loads(path.read_text()))
    assert np.isnan(loaded["events"][0]["event"]["rod_boundary"]["rod_snapshot"]["distributed_contact"]).all()
    before=sim.current_time_step_index,rod.rope.steps
    replay=replay_diagnostic_bundle(loaded)
    assert encode_owned(replay)==encode_owned(diagnostic.report())
    assert (sim.current_time_step_index,rod.rope.steps)==before
    assert replay["new_physics_steps"]==0 and replay["optimizer_updates"]==0


@pytest.mark.parametrize("damage",["result_guard","recorded_guard","provider_velocity","provider_curvature"])
def test_replay_rejects_forged_guard_or_derived_provider_state(monkeypatch,tmp_path,damage):
    episode,rod,hands,sim,rows,diagnostic=setup(monkeypatch,tmp_path,mode="empty")
    diagnostic.begin(episode.pre_action_state(),torch.zeros((1,24)))
    result=episode.control(torch.zeros((1,24)))
    if damage=="result_guard":
        for event in rows[1:]:diagnostic.record(event)
        result["termination_reason"]="right:mimic_error"
        with pytest.raises(ValueError):diagnostic.end(result)
    else:
        event=deepcopy(rows[1])
        if damage=="recorded_guard":event["exchange"]["guard_failure"]="right:mimic_error"
        elif damage=="provider_velocity":event["rod_boundary"]["rod_snapshot"]["velocities"][0,0,0]=3
        else:event["rod_boundary"]["rod_snapshot"]["curvature"][0,1]=3
        with pytest.raises(ValueError):diagnostic.record(event)


def test_mimic_guard_partial_window_is_recorded_but_never_settled(monkeypatch,tmp_path):
    episode,rod,hands,sim,rows,diagnostic=setup(monkeypatch,tmp_path,mode="empty")
    step=sim.step
    def guarded(*,render):
        step(render=render)
        hands[1].measured[0,hands[1].joint_names.index("finger13")]=.1
    sim.step=guarded
    result=advance(episode,rows,diagnostic)
    assert result["control_result"]["termination_reason"]=="right:mimic_error"
    assert result["control_receipt"] is None
    assert result["settlement_blocker"]=="physical_guard_closed_partial_window_without_original_R0_terminal_source"
    assert result["physical_rows"][0]["measurements"]["stable"]["value"] is None
    assert diagnostic.report()["settled_control_windows"]==0
    assert sim.current_time_step_index==3 and rod.rope.steps==1


def test_actual_recorder_callback_consumes_one_complete_window(monkeypatch,tmp_path):
    episode,rod,hands,sim,rows,diagnostic=setup(monkeypatch,tmp_path,mode="empty")
    episode._record=diagnostic.record
    diagnostic.begin(episode.pre_action_state(),torch.zeros((1,24)))
    output=diagnostic.end(episode.control(torch.zeros((1,24))))
    assert output["control_receipt"]["physical_samples"]==8
    assert len(rows)==1
    assert sim.current_time_step_index==10 and rod.rope.steps==8


def test_cli_replays_strict_json_in_real_cpu_subprocess(monkeypatch,tmp_path):
    import os
    import subprocess
    import sys
    episode,rod,hands,sim,rows,diagnostic=setup(monkeypatch,tmp_path,mode="empty")
    advance(episode,rows,diagnostic,aux=True)
    source=tmp_path/"input.json";out=tmp_path/"out.json"
    source.write_text(json.dumps(encode_owned(diagnostic.bundle()),allow_nan=False))
    root=Path(__file__).resolve().parents[1]
    cmd=[sys.executable,str(root/"scripts/replay_dh116_flexible_r0.py"),
         "--input",str(source),"--output",str(out)]
    proc=subprocess.run(cmd,cwd=root,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"),
                        capture_output=True,text=True,timeout=30)
    assert proc.returncode==0,proc.stderr
    receipt=json.loads(out.read_text())
    assert receipt["replay_complete"] is True
    assert receipt["report"]==encode_owned(diagnostic.report())
    assert receipt["input_sha256"]==hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt["native_imports"]==[]
    original=out.read_bytes()
    repeated=subprocess.run(cmd,cwd=root,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"),
                            capture_output=True,text=True,timeout=30)
    assert repeated.returncode==2
    assert out.read_bytes()==original
    source.write_text(source.read_text().replace('"physics_step": 3','"physics_step": 99',1))
    failed_out=tmp_path/"out-failed.json"
    cmd[-1]=str(failed_out)
    proc=subprocess.run(cmd,cwd=root,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"),
                        capture_output=True,text=True,timeout=30)
    assert proc.returncode==1
    failed=json.loads(failed_out.read_text())
    assert out.read_bytes()==original
    assert failed["replay_complete"] is False and failed["report"] is None
    assert failed["teacher_admission"] is False


@pytest.mark.parametrize("damage",["post_path","post_joint_order","next_pre_q","extra_instance"])
def test_every_pre_post_instance_and_joint_state_keeps_original_identity(monkeypatch,tmp_path,damage):
    episode,rod,hands,sim,rows,diagnostic=setup(monkeypatch,tmp_path,mode="empty")
    diagnostic.begin(episode.pre_action_state(),torch.zeros((1,24)))
    episode.control(torch.zeros((1,24)))
    event=deepcopy(rows[1])
    if damage=="post_path":
        event["exchange"]["action_readback"]["post"]["hands"]["left"]["prim_paths"]=["/foreign/left"]
    elif damage=="post_joint_order":
        names=event["exchange"]["action_readback"]["post"]["hands"]["left"]["joint_names"]
        names[0],names[1]=names[1],names[0]
    elif damage=="extra_instance":
        identity=event["exchange"]["action_readback"]["post"]["identity"]
        identity.append(deepcopy(identity[0]))
    else:
        diagnostic.record(event)
        event=deepcopy(rows[2])
        event["exchange"]["action_readback"]["pre"]["hands"]["left"]["q"][0,0]=.1
    with pytest.raises(ValueError):diagnostic.record(event)


@pytest.mark.parametrize("damage",["active_order","empty_target","target_order","target_nan","backend_shape","instance_id"])
def test_guard_and_target_indices_are_rebuilt_from_original_joint_names(monkeypatch,tmp_path,damage):
    episode,rod,hands,sim,rows,diagnostic=setup(monkeypatch,tmp_path,mode="empty")
    diagnostic.begin(episode.pre_action_state(),torch.zeros((1,24)))
    episode.control(torch.zeros((1,24)))
    event=deepcopy(rows[1]);action=event["exchange"]["action_readback"]
    hand=action["post"]["hands"]["left"]
    target=action["submission"]["expected_backend_targets"]["left"]
    if damage=="active_order":hand["active_indices"]=list(reversed(hand["active_indices"]))
    elif damage=="empty_target":
        target["joint_indices"]=[];target["values"]=np.empty((1,0))
    elif damage=="target_order":
        target["joint_indices"]=list(reversed(target["joint_indices"]))
        target["values"]=hand["backend_targets"][:,target["joint_indices"]].copy()
    elif damage=="target_nan":target["values"][0,0]=float("nan")
    elif damage=="backend_shape":hand["backend_targets"]=np.zeros((2,17))
    else:hand["instance_object_id"]+=1
    with pytest.raises(ValueError):diagnostic.record(event)
