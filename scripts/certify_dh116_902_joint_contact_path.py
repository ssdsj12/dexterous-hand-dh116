"""Certify an opt-in small 902 alignment and full float32 fixed-wrist closure."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scripts.search_dh116_902_joint_contact import (
    ROOT,ORIGINAL_REQUEST,ORIGINAL_REQUEST_SHA256,decode_candidate,candidate_passed)
from scripts.screen_dh116_902_fixed_wrist_open import compile_closure,reset_geometry

VERSION="dh116-902-joint-contact-candidate-v1"
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def select_target(config,search_path,receipt_sha):
    if (config.get("version")!=VERSION or config.get("source_request")!=ORIGINAL_REQUEST
            or config.get("source_request_sha256")!=ORIGINAL_REQUEST_SHA256
            or config.get("receipt_sha256")!=receipt_sha
            or config.get("native_admissible") is not False or config.get("teacher_admission") is not False
            or sha(search_path)!=config.get("search_sha256")
            or sha(ROOT/ORIGINAL_REQUEST)!=ORIGINAL_REQUEST_SHA256):
        raise ValueError("source-bound independent CPU candidate required")
    search=json.loads(Path(search_path).read_bytes())
    if search.get("source_request_sha256")!=ORIGINAL_REQUEST_SHA256 or search.get("receipt_sha256")!=receipt_sha:
        raise ValueError("search and candidate source identities differ")
    rows=[row for row in search["rows"] if row["ordinal"]==config["selected_ordinal"]]
    if len(rows)!=1 or rows[0].get("endpoint_candidate_passed") is not True:
        raise ValueError("one accepted search candidate required")
    row=rows[0];target=json.loads((ROOT/ORIGINAL_REQUEST).read_bytes())["target"]
    pose,q=decode_candidate(row["x"],target)
    if (not np.array_equal(pose,config["pose_wxyz"]) or not np.array_equal(q,config["active_q_rad"])
            or not np.array_equal(pose,row["pose_wxyz"]) or not np.array_equal(q,row["active_q_rad"])):
        raise ValueError("target changed after bounded source-anchored search")
    return row


def certify(candidate_path,search_path,receipt,receipt_sha):
    from scripts.audit_dh116_wire_contact_witness import shapes_world
    from scripts.audit_dh116_pinch_initial_pose import load_geometry
    from scripts.screen_dh116_902_command_contact import _measure_hand
    from scripts.screen_dh116_tripod_supported_reset import archived_tube
    from scripts.verify_dh116_902_flexible_reset import REFERENCE
    from go2_pvcnn.control.dh116.tabletop_supported_rod_reference import load_parked_reference,_read_artifact
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_scene_preparation as scene
    candidate_path=Path(candidate_path).resolve(strict=True);candidate_path.relative_to(ROOT)
    config=json.loads(candidate_path.read_bytes());selected=select_target(config,search_path,receipt_sha)
    receipt=Path(receipt).resolve(strict=True)
    if sha(receipt)!=receipt_sha:raise ValueError("same measured rod source required")
    outer=json.loads(receipt.read_bytes());output=(receipt.parent/"native").resolve(strict=True)
    frozen,_=_read_artifact(output,outer["artifact_sha256"],"frozen_plan.json")
    if frozen["version"]!="dh116-902-fixed-wrist-open-reset-v1":
        raise ValueError("actual independent open Reset source required")
    scene_config=frozen["reset_config"]
    initial=decode_owned(_read_artifact(output,outer["artifact_sha256"],"tripod_reset_source_1.json")[0])
    refs=[load_parked_reference(receipt,receipt_sha,sdk,task="tripod") for sdk in range(3,241)]
    nodes=np.stack([initial["rod_boundary"]["rod_snapshot"]["node_positions"][0],*[ref.node_positions for ref in refs]])
    boxes=[(np.asarray(b["center_m"])-np.asarray(b["size_m"])/2,
            np.asarray(b["center_m"])+np.asarray(b["size_m"])/2) for b in refs[0].support_boxes]
    reference=json.loads(REFERENCE.read_bytes());table=reference["table"]
    top=table["center_m"][2]+table["size_m"][2]/2
    models={s:load_geometry(ROOT,side=s) for s in ("left","right")}
    if any(m["usd"]["sha256"]!=reference["hand_usd_sha256"][s] for s,m in models.items()):
        raise ValueError("original side-specific USD sources required")
    pose=np.asarray(config["pose_wxyz"]);goal=np.asarray(config["active_q_rad"])
    opening=np.asarray(scene_config["hands"]["right"]["q_rad"])
    initial_audit={}
    for side in models:
        p=pose if side=="right" else scene_config["hands"][side]["pose_wxyz"]
        q=opening if side=="right" else scene_config["hands"][side]["q_rad"]
        initial_audit[side]=reset_geometry(shapes_world(models[side],p,q),nodes,boxes,table_top=top)
    actions,states=compile_closure(opening,goal,windows=64)
    grids=scene.load_sdf_arrays(scene.prepare_scene(ROOT))
    request=json.loads((ROOT/ORIGINAL_REQUEST).read_bytes());tail=nodes[-9:];tube=archived_tube(tail)
    samples=[]
    for i,q in enumerate(states):
        row=_measure_hand("right",(models["right"],{}),pose,q,tail,tube,boxes,grids,top,request)
        row["ordinal"]=i+1
        row["geometry_passed"]=bool(row["noncontact_rod_clearance_m"]>.00025
            and row["support_clearance_m"]>=.001 and row["table_clearance_m"]>=.0021
            and row["contact_rod_clearance_m"]>=-.00015 and row["all_envelopes_certified"]
            and min(row["pad_gaps_m"])>=-.00015)
        samples.append(row)
    dependencies=[Path(__file__),candidate_path,ROOT/ORIGINAL_REQUEST,REFERENCE,
        ROOT/"scripts/search_dh116_902_joint_contact.py",ROOT/"scripts/screen_dh116_902_fixed_wrist_open.py",
        ROOT/"scripts/screen_dh116_902_command_contact.py",ROOT/"scripts/audit_dh116_pinch_initial_pose.py"]
    return dict(version="dh116-902-joint-contact-path-cpu-v1",scope="CPU_only_against_saved_actual_rod",
        candidate_sha256=sha(candidate_path),search_sha256=sha(search_path),receipt_sha256=receipt_sha,
        selected_ordinal=selected["ordinal"],initial=initial_audit,
        initial_passed=all(row["passed"] for row in initial_audit.values()),
        closure_geometry_passed=all(row["geometry_passed"] for row in samples),
        endpoint_contact_candidate_passed=candidate_passed(samples[-1]),samples=samples,
        actions=actions.tolist(),action_sha256=hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(),
        pose_wxyz=pose.tolist(),opening_active_q_rad=opening.tolist(),target_active_q_rad=goal.tolist(),
        float32_endpoint_active_q_rad=states[-1].tolist(),control_windows=64,physical_substeps=512,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in dependencies},
        geometry_sources={side:m["source_sha256"] for side,m in models.items()},
        archive_sha256=[ref.sample_sha256 for ref in refs],
        native_runs=0,native_contact_verified=False,force_qualified=False,grasp_qualified=False,teacher_admission=False,
        limitations=["Static archived geometry is not live rod deformation or contact force",
            "New wrist Reset requires its own native receipt before control qualification"])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",type=Path,required=True);p.add_argument("--search",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True);p.add_argument("--receipt-sha256",required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args(argv)
    if a.output.exists():raise ValueError("fresh full-path proof required")
    result=certify(a.candidate,a.search,a.receipt,a.receipt_sha256)
    with a.output.open("x") as f:json.dump(result,f,indent=2,allow_nan=False)
    return int(not(result["initial_passed"] and result["closure_geometry_passed"] and result["endpoint_contact_candidate_passed"]))


if __name__=="__main__":raise SystemExit(main())
