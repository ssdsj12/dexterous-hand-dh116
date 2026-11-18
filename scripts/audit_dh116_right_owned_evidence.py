#!/usr/bin/env python3
"""Read-only, SHA-bound audit of actual right Pinch execution and contact."""
import argparse
from collections import Counter
import hashlib,json,os,time
from pathlib import Path
import numpy as np
from scripts import replay_dh116_segment_contact_r0 as ledger
from scripts.replay_dh116_pinch_contact_timeline import SOURCE
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,expand
from go2_pvcnn.control.dh116.tabletop_deformable_pair_source_basis import ContactTimeline,verify_contact_source
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,encode_owned

RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f8-right-pinch-owned-contact-native/2fcb2478d3c04d8e52d47638a10ec0bbecc4f34247d308a058b662d45d38d873/attempt-z3xlac81/receipt.json")
RECEIPT_SHA="3e12bb3204f46df748e1de7b35b77306e57be016e6a7f626249f0790d466c83d"
VERSION="dh116-right-owned-native-evidence-audit-v1"

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
 return h.hexdigest()

def stat(values):
 a=np.asarray(values,dtype=float)
 if not a.size:return None
 if a.ndim!=1 or not np.isfinite(a).all():raise ValueError("finite scalar observations required")
 return dict(count=len(a),min=float(a.min()),max=float(a.max()),
  p50=float(np.quantile(a,.5)),p90=float(np.quantile(a,.9)),p95=float(np.quantile(a,.95)))

def measured_tracking(hand):
 if (hand["measured_source"]!="physx_backend_get_dof_positions/get_dof_velocities"
     or hand["backend_source"]!="physx_backend_get_dof_position_targets"):
  raise ValueError("actual PhysX measured and target sources required")
 names=hand["joint_names"];indices=[names.index(n) for n in ACTIVE]
 if indices!=hand["active_indices"]:raise ValueError("canonical active mapping differs")
 q=np.asarray(hand["q"]);qd=np.asarray(hand["qd"]);target=np.asarray(hand["backend_targets"])
 if any(a.shape!=(1,len(names)) or not np.isfinite(a).all() for a in (q,qd,target)):
  raise ValueError("finite measured shape required")
 active=q[0,indices];expected=expand(active)
 passive=[n for n in JOINTS if n not in ACTIVE]
 residual=max(abs(q[0,names.index(n)]-expected[list(JOINTS).index(n)]) for n in passive)
 return dict(active_positions_rad=active.copy(),active_targets_rad=target[0,indices].copy(),
  active_error_rad=active-target[0,indices],active_velocity_rad_s=qd[0,indices].copy(),
  passive_coupling_error_rad=float(residual),hardware_measured=False)

def run(output):
 if RECEIPT.resolve(strict=True)!=RECEIPT or sha(RECEIPT)!=RECEIPT_SHA:raise ValueError("actual receipt changed")
 r=json.loads(RECEIPT.read_bytes());native=Path(r["output"])
 if (r["native_qualification"] is not True or r["outer_exit_code"]!=0
   or r["debugger_inferior_signal"] is not None or r["duration_seconds"]>180):
  raise ValueError("bounded natural execution required")
 for name,expected in r["artifact_sha256"].items():
  if sha(native/name)!=expected:raise ValueError("native artifact changed: "+name)
 audit_path=RECEIPT.parent/"right-pinch-owned-post-exit-audit.json"
 if sha(audit_path)!=r["lifetime_post_exit_audit_sha256"]:raise ValueError("post-exit audit changed")
 postexit=json.loads(audit_path.read_bytes())["evidence"]
 if postexit["lifecycle_qualified"] is not True:raise ValueError("independent lifecycle not qualified")
 p0=decode_owned(json.loads((native/"director_reset.json").read_bytes())["payload"])
 source=verify_contact_source(SOURCE);counts=Counter();contacts=Counter();observer=None;last=None
 sides={s:dict(active_errors=[],passive_errors=[],base_z=[],table=[],other=[],depth=[],finger12=[]) for s in ("left","right")}
 base0={};rod0=None;lift=[];pending=None;ordinal=0;peak=None
 shape_clearances={side:{} for side in sides}
 with (output/"physics.jsonl").open("x") as phys,(output/"contact_substeps.jsonl").open("x") as timeline:
  for event in ledger.stream_events(native/"physical.jsonl",r["artifact_sha256"]["physical.jsonl"]):
   op=event.get("operation");counts[str(op)]+=1
   if op is None:continue
   p=event["payload"]
   if op=="reset":
    radius=ledger.old.check_radius_packet(p0,p["reset"]["rod_reset"])
    observer=ContactTimeline(grasp="pinch",radius_sample=dict(value=radius["value"],
      source="actual_parameter_metadata:"+ledger.old.BACKEND_SHA),max_physics_rows=232)
    rod0=np.asarray(p["reset"]["rod_reset"]["node_positions"])[0].copy()
    # Actual measured Reset body poses, not requested seed, are supplied by the
    # first post-reset pre-action sample below.
   elif op=="begin":
    if pending is not None:raise ValueError("unsettled policy window")
    pending=np.asarray(p["action"])
   elif op=="physical":
    if observer is None or pending is None:raise ValueError("reset and owner action required")
    x=p["exchange"];a=x["action_readback"];identity=a["post"]["identity"][0];ordinal+=1
    if identity["physics_step"]!=ordinal+2 or not np.array_equal(a["request"]["action24"],pending):
     raise ValueError("action/physics ordering differs")
    if x["guard_failure"] is not None:raise ValueError("qualified source contains terminal")
    row=dict(identity=identity,physics_step=x["physics_step"],action24=pending,side={})
    for side in sides:
     t=measured_tracking(a["post"]["hands"][side]);g=x["closure_geometry"]["sides"][side]
     pos=np.asarray(g["actual_base_pose_xyzw"])[:3]
     if side not in base0:base0[side]=pos.copy()
     # Displacement is explicitly relative to first completed measurement
     # (SDK3), not mislabeled as physical Reset-relative palm displacement.
     dz=float(pos[2]-base0[side][2]);s=sides[side]
     s["base_z"].append(dz);s["active_errors"].extend(np.abs(t["active_error_rad"]).tolist())
     s["passive_errors"].append(t["passive_coupling_error_rad"]);s["finger12"].append(float(t["active_positions_rad"][1]))
     s["table"].append(float(g["minimum_table_clearance_m"]));s["other"].append(float(g["minimum_non_target_gap_lower_m"]))
     depths=[v["minimum_gap_m"] for v in g["required_pad_sdf"].values() if v["minimum_gap_m"] is not None]
     s["depth"].extend(depths)
     for shape,gap in g["table_clearance_m"].items():shape_clearances[side].setdefault(shape,[]).append(float(gap))
     row["side"][side]=dict(tracking=t,actual_base_world_m=pos,base_z_from_sdk3_m=dz,
        table_clearance_m=g["minimum_table_clearance_m"],required_pad_sdf=g["required_pad_sdf"])
    center=float(np.asarray(x["rod"]["node_positions"])[0,:,2].mean()-rod0[:,2].mean())
    lift.append(center);row["rod_mean_node_z_minus_reset_m"]=center
    if peak is None or center>peak["value_m"]:peak=dict(value_m=center,identity=identity)
    for pair in x["rod"]["rod_mesh_contact_pairs"]["pairs"]:contacts[pair["mesh_name"]]+=1
    for entry in observer.record(x["rod"],action_identity=identity):
     timeline.write(json.dumps(encode_owned(entry),allow_nan=False)+"\n")
    phys.write(json.dumps(encode_owned(row),allow_nan=False)+"\n");last=identity
   elif op=="end":
    if pending is None or last["substep"]!=7:raise ValueError("incomplete control")
    pending=None
  for f in (phys,timeline):f.flush();os.fsync(f.fileno())
 expected=Counter({"None":1,"setup":2,"reset":1,"initial_geometry":2,"begin":29,"physical":232,"end":29})
 if counts!=expected or pending is not None or last["physics_step"]!=234:raise ValueError("exact complete native sequence required")
 return dict(version=VERSION,status="complete",receipt_sha256=RECEIPT_SHA,
  recipe_sha256=r["command_spec_sha256"],code_config_sha256=r["code_config_sha256"],
  ledger_sha256=r["artifact_sha256"]["physical.jsonl"],postexit_sha256=r["lifetime_post_exit_audit_sha256"],
  actual_native_steps=234,new_native_steps=0,ppo_updates=0,outer_exit_code=0,
  native_duration_s=r["duration_seconds"],contact_counts=dict(contacts),operation_counts=dict(counts),
  contact_source=source,timeline=observer.summary(),sides={side:{key:stat(values) for key,values in value.items()} for side,value in sides.items()},
  rod_mean_z_minus_reset_m=stat(lift),rod_mean_z_peak=peak,
  table_clearance_by_shape_m={side:{shape:stat(v) for shape,v in shapes.items()} for side,shapes in shape_clearances.items()},
  actual_execution_qualified=True,native_qualified=False,teacher_admission=False,grasp_success=False,
  hand_table_contact_source=None,material_anchor_slip=None,stable_hold_qualified=False,
  base_displacement_reference="first completed measurement SDK3; actual base_link, not commanded palm or physical Reset",
  limitations=["table supports rod throughout; no free grasp or Lift qualification","contact opposition sign is not force closure or stability","no 5s hold budget"])

def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 try:result=run(a.output)
 except Exception as exc:result=dict(version=VERSION,status="blocked",reason=type(exc).__name__+": "+str(exc),new_native_steps=0,ppo_updates=0,teacher_admission=False)
 result.update(elapsed_s=time.monotonic()-start,script_sha256=sha(__file__))
 with (a.output/"result.json").open("x") as f:
  json.dump(encode_owned(result),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
 print(json.dumps({k:result.get(k) for k in ("status","reason","contact_counts","elapsed_s")}))
 return int(result["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
