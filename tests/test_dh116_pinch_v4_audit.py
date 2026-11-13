"""CPU post-exit tests; native/geometry producers are explicit historical test doubles."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    path.write_text(json.dumps(value,allow_nan=False))


def lines(path):
    return [json.loads(v) for v in path.read_text().splitlines()]


def write_lines(path,rows):
    path.write_text("".join(json.dumps(v,allow_nan=False)+"\n" for v in rows))


def clear_receipt():
    return dict(version="dh116-simulation-owner-cleanup-cpu-v1",
        subscription_status="explicitly_unsubscribed",unsubscribe_returned=True,
        subscription_field_cleared=True,original_clear_instance_returned=True,
        extra_gc_calls=0,native_release_verified=False,simulation_liveness_after_return=None,
        constructor_frame_owner_resolved=False,physics_steps_executed=0)


@pytest.fixture
def case(tmp_path,monkeypatch):
    # Reuse the actual frozen 238-step measurement loop with CPU SDK/Rope doubles.
    from tests.test_dh116_pinch_v2 import (
        test_independent_disk_audit_accepts_complete_cpu_fixture_and_rejects_damage)
    from scripts.dh116_pinch_startup_v3 import BOUNDARY_VERSION
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,expand
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import (
        candidate_action,integrate_pinch_action)
    test_independent_disk_audit_accepts_complete_cpu_fixture_and_rejects_damage(tmp_path,monkeypatch,None)
    out=tmp_path/"native"
    cfg=read(ROOT/"configs/dh116_tasks/flexible_pinch_candidate_v1.json")
    cfg["initial_pose"]=read(ROOT/"configs/dh116_tasks/flexible_pinch_pose_candidate_v2.json")
    cfg["action"].update(thumb_velocity_normalized=.3,index_velocity_normalized=.4)
    cfg["scope"]="CPU explicit successor config fixture"
    active=[JOINTS.index(n) for n in ACTIVE]
    target=np.asarray(cfg["initial_pose"]["active_q_rad"],dtype=float)
    rows=lines(out/"physics.jsonl")
    for step,row in enumerate(rows,1):
        row["pre_state"]["q"]=expand(target).tolist()
        action=candidate_action(cfg,step)
        target=integrate_pinch_action(target,target,action,side="left",dt=1/480)
        row["canonical_action24"]=action.tolist()
        row["sdk_joint_target_rad"]=expand(target).tolist()
        row["post_state"]["q"]=expand(target).tolist()
    write_lines(out/"physics.jsonl",rows)
    result=read(out/"result.json");result["version"]="dh116-single-left-pinch-v4"
    write(out/"result.json",result)
    write(out/"before_app_launcher.json",dict(version=BOUNDARY_VERSION,
        phase="before_isaaclab_app_import",loaded_usd_modules=[],native_execution_performed=False))
    write(out/"simulation_clear_receipt.json",clear_receipt())
    from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS
    names=["base_link"]+[f"finger{d}{j}_link" for d,count in ((1,4),(2,3),(3,3),(4,3),(5,3))
                           for j in range(1,count+1)]
    geometry=dict(source_colliders={n:dict(table=dict(top_clearance_m=.002,intersects=False),
        rod=dict(capsule_surface_gap_m=.003)) for n in names},
        locked_pads={n:dict(table=dict(top_clearance_m=.002,intersects=False),
        rod=dict(capsule_surface_gap_m=.003)) for n in DISTAL_PAD_LINKS})
    write(out/"initial_geometry.json",geometry)
    return out,tmp_path/"outer_exit.json",cfg


def audit(case):
    from scripts.validate_dh116_pinch_v4 import audit_outcome
    out,outer,cfg=case
    return audit_outcome(out,outer,"explicit",cfg=cfg)


def inject_pairs(out,phase,entries):
    """entries=(substep,node,mesh,normal_load); maintain real raw reduction identities."""
    from scripts.dh116_pinch_evidence_v2 import phase_pair_record
    from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS
    from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import contact_metrics
    rows=lines(out/"physics.jsonl");pairs=lines(out/"pairs.jsonl")
    row=rows[phase-1]
    raw=deepcopy(pairs[phase-1]["raw"])
    raw["attempted"]=len(entries)
    ids=[];vectors=[];nodes=np.zeros((41,3));wrench=np.zeros((6,6))
    for sub,node,mesh,load in entries:
        normal=np.array([1.,0,0]) if mesh!=1 else np.array([-1.,0,0])
        force=load*normal
        ids.append([sub,node,mesh]);vectors.append([0.,0.,0.,*normal,*force,0.,0.,0.])
        nodes[node]+=force/12
        wrench[mesh,:3]-=force/12
    raw.update(indices=np.array(ids,dtype=np.int64).reshape(-1,3),
        vectors=np.array(vectors,dtype=float).reshape(-1,12),
        node_forces_on_rod_world_n=nodes,mesh_wrenches_on_mesh_world=wrench)
    row["rod_node_contact_forces_w"]=nodes.tolist()
    row["rod_mesh_wrenches_w"]={n:w.tolist() for n,w in zip((*DISTAL_PAD_LINKS,"table"),wrench)}
    row["contact_node_ids"]=np.flatnonzero(np.linalg.norm(nodes,axis=-1)>1e-5).tolist()
    row["pair_force_residual_n"]=float(np.linalg.norm(nodes.astype(np.float32).sum(0)+wrench[:,:3].sum(0)))
    row["rod_on_hand_com_w"]=wrench[:5].tolist()
    row["r0_raw_measurements"]=contact_metrics(row["rod_mesh_wrenches_w"])
    pairs[phase-1]=phase_pair_record(raw,row)
    previous=np.zeros((5,6)).tolist()
    for row in rows:
        row["applied_wrench_com_w"]=previous
        previous=row["rod_on_hand_com_w"]
    write_lines(out/"physics.jsonl",rows);write_lines(out/"pairs.jsonl",pairs)


def test_successor_config_and_zero_contact_are_separate_from_lifecycle(case):
    from scripts import validate_dh116_pinch_v2 as old
    from go2_pvcnn.control.dh116 import tabletop_deformable_grasp_runtime as runtime
    before=old.VERSION,old.validate_action_targets,runtime.validate_pinch_ledger
    original=deepcopy(case[2]);result=audit(case)
    assert result["lifecycle_qualified"] is True,result.get("reason")
    assert result["contact_observation_qualified"] is False
    assert result["contact_evidence"]["hand_pair_samples"]==0
    assert result["contact_evidence"]["first_hand_contact"] is None
    assert result["simulation_clear_receipt_audit"]["api_sequence_verified"] is True
    assert result["simulation_released"] is None
    assert result["native_release_verified"] is result["grasp_qualified"] is result["teacher_admission"] is False
    assert (old.VERSION,old.validate_action_targets,runtime.validate_pinch_ledger)==before
    assert case[2]==original


def test_postquiet_measured_thumb_index_sync_is_only_contact_observation(case):
    inject_pairs(case[0],49,[(2,10,0,2.),(2,10,1,3.),(3,11,0,1.),(4,11,1,1.)])
    result=audit(case)
    assert result["lifecycle_qualified"] is True,result.get("reason")
    stats=result["contact_evidence"]
    assert stats["hand_pair_samples"]==4
    assert stats["pads"]["finger14_link"]["pair_samples"]==2
    assert stats["pads"]["finger23_link"]["pair_samples"]==2
    assert stats["pads"]["finger33_link"]["first_contact"] is None
    assert stats["first_hand_contact"]["phase_step"]==49
    assert stats["first_hand_contact"]["solver_substep"]==2
    assert stats["thumb_index_synchronous_positive_load_substeps"]==1
    assert stats["synchronous_load_samples"][0]["normal_load_n"]["finger14_link"]==2.
    assert stats["synchronous_load_samples"][0]["normal_load_n"]["finger23_link"]==3.
    assert result["contact_observation_qualified"] is True
    assert result["effective_opposed_contact"] is result["material_anchor_slip_m_s"] is None
    assert result["grasp_qualified"] is result["teacher_admission"] is False


@pytest.mark.parametrize("kind",["table_only","different_substeps","zero_load"])
def test_table_or_asynchronous_or_zero_load_cannot_qualify_pinch(case,kind):
    entries={"table_only":[(0,3,5,1.)],
        "different_substeps":[(0,3,0,1.),(1,3,1,1.)],
        "zero_load":[(0,3,0,0.),(0,3,1,0.)]}[kind]
    inject_pairs(case[0],49,entries)
    result=audit(case)
    assert result["lifecycle_qualified"] is True,result.get("reason")
    assert result["contact_observation_qualified"] is False
    assert result["contact_evidence"]["thumb_index_synchronous_positive_load_substeps"]==0


@pytest.mark.parametrize("damage",["quiet","version","boundary","cleanup","missing_cleanup",
    "release_claim","source_clearance","cooked_clearance","rod_gap","missing_shape","target",
    "action","force_source","old_cfg","clock","binding","exit"])
def test_full_disk_gate_rejects_damage(case,damage):
    out,outer,cfg=case
    if damage=="quiet":inject_pairs(out,48,[(0,1,0,1.)])
    elif damage=="version":
        data=read(out/"result.json");data["version"]="dh116-single-left-pinch-v3";write(out/"result.json",data)
    elif damage=="boundary":
        data=read(out/"before_app_launcher.json");data["loaded_usd_modules"]=["pxr"];write(out/"before_app_launcher.json",data)
    elif damage in ("cleanup","release_claim"):
        data=read(out/"simulation_clear_receipt.json")
        data["unsubscribe_returned" if damage=="cleanup" else "native_release_verified"]=damage=="release_claim"
        write(out/"simulation_clear_receipt.json",data)
    elif damage=="missing_cleanup":(out/"simulation_clear_receipt.json").unlink()
    elif damage in ("source_clearance","cooked_clearance","rod_gap","missing_shape"):
        data=read(out/"initial_geometry.json")
        group="locked_pads" if damage=="cooked_clearance" else "source_colliders"
        name=next(iter(data[group]))
        if damage=="rod_gap":data[group][name]["rod"]["capsule_surface_gap_m"]=-1e-8
        elif damage=="missing_shape":data[group].pop(name)
        else:data[group][name]["table"]["top_clearance_m"]=.00189999
        write(out/"initial_geometry.json",data)
    elif damage=="old_cfg":
        cfg["action"]["index_velocity_normalized"]=.25
    elif damage=="exit":
        data=read(outer);data["outer_exit_code"]=1;write(outer,data)
    else:
        rows=lines(out/"physics.jsonl")
        if damage=="action":rows[49]["canonical_action24"][0]+=.001
        elif damage=="target":rows[49]["sdk_joint_target_rad"][0]+=.001
        elif damage=="force_source":rows[49]["pair_force_residual_n"]=.1
        elif damage=="clock":rows[49]["rod_tick"]+=12
        elif damage=="binding":
            a=rows[49]["actual_pad_bindings"]
            a["finger14_link"]["body_index"],a["finger23_link"]["body_index"]=a["finger23_link"]["body_index"],a["finger14_link"]["body_index"]
        write_lines(out/"physics.jsonl",rows)
    result=audit(case)
    assert result["lifecycle_qualified"] is False,result
    assert result["contact_observation_qualified"] is False
    assert result["grasp_qualified"] is result["teacher_admission"] is False


def test_wrong_cfg_cannot_be_overridden_by_output_claim(case):
    data=read(case[0]/"plan.json");data["cfg"]=deepcopy(case[2]);write(case[0]/"plan.json",data)
    case[2]["action"]["thumb_velocity_normalized"]=-.3
    result=audit(case)
    assert result["lifecycle_qualified"] is False
    assert "action" in result["reason"]


def test_import_is_cpu_and_does_not_load_usd_or_sdk():
    code="""
import sys,importlib.abc
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('pxr','PhysxSchema','isaaclab','isaacsim','ropesim'):
   raise RuntimeError('CPU import guard:'+fullname)
sys.meta_path.insert(0,Guard())
from scripts.validate_dh116_pinch_v4 import audit_outcome,VERSION
assert VERSION=='dh116-single-left-pinch-v4'
"""
    result=subprocess.run([sys.executable,"-c",code],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr

def test_binding_restored_when_successor_action_validation_raises(case,monkeypatch):
    from scripts import validate_dh116_pinch_v2 as old
    from go2_pvcnn.control.dh116 import tabletop_deformable_grasp_runtime as runtime
    original=runtime.validate_pinch_ledger
    def fail(*args):
        raise ValueError("CPU injected action validator")
    monkeypatch.setattr(old,"validate_action_targets",fail)
    result=audit(case)
    assert not result["lifecycle_qualified"] and "CPU injected" in result["reason"]
    assert old.validate_action_targets is fail and runtime.validate_pinch_ledger is original
    assert old.VERSION=="dh116-single-left-pinch-v2"


def test_cli_loads_only_project_successor_config(case,tmp_path,monkeypatch,capsys):
    from scripts import validate_dh116_pinch_v4 as module
    project=tmp_path/"cpu-project"
    path=project/module.CONFIG;path.parent.mkdir(parents=True);write(path,case[2])
    (project/"scripts").mkdir()
    monkeypatch.setattr(module,"__file__",str(project/"scripts"/"validate_dh116_pinch_v4.py"))
    assert module.main(["--output",str(case[0]),"--outer-exit",str(case[1])])==0
    value=json.loads(capsys.readouterr().out)
    assert value["lifecycle_qualified"] is True and value["contact_observation_qualified"] is False
    wrong=deepcopy(case[2]);wrong["action"]["index_velocity_normalized"]=.25
    write(path,wrong)
    assert module.main(["--output",str(case[0]),"--outer-exit",str(case[1])])==1
    value=json.loads(capsys.readouterr().out)
    assert value["lifecycle_qualified"] is False
