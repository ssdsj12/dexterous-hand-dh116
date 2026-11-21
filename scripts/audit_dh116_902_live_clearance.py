"""Audit saved supported Tripod Reset using measured base poses/q and original USD FK.

All bounds are derived geometry, not independent per-link native measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scripts.screen_dh116_tripod_supported_reset import archived_tube,separation
from scripts.certify_dh116_tripod_distal_route import support_clearance_lower

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def temporal_shape_bound(vertices,nodes,supports,*,table_top):
    v=np.asarray(vertices,dtype=float);n=np.asarray(nodes,dtype=float)
    if (v.ndim!=3 or v.shape[2]!=3 or v.shape[1]<4 or v.shape[0]!=n.shape[0]
            or n.ndim!=3 or n.shape[1:]!=(41,3) or not len(n)
            or not np.isfinite(v).all() or not np.isfinite(n).all() or not np.isfinite(table_top)
            or not len(supports)):
        raise ValueError("complete finite same-clock moving convex and rod frames required")
    for lo,hi in supports:
        if np.shape(lo)!=(3,) or np.shape(hi)!=(3,) or not np.isfinite([lo,hi]).all() or not (lo<hi).all():
            raise ValueError("finite ordered support bounds required")
    motion=float(np.linalg.norm(v-v[0],axis=-1).max())
    return dict(rod_gap_lower_m=separation(v[0],*archived_tube(n))-motion,
        support_gap_lower_m=min(support_clearance_lower(v[0],lo,hi) for lo,hi in supports)-motion,
        table_gap_m=float(v[:,:,2].min()-table_top),max_vertex_displacement_m=motion)


def sdk_matrix(pose):
    xyz,xyzw=map(np.asarray,pose)
    if xyz.shape!=(3,) or xyzw.shape!=(4,) or not np.isfinite([*xyz,*xyzw]).all() or abs(np.linalg.norm(xyzw)-1)>1e-5:
        raise ValueError("finite SDK xyz/xyzw pose required")
    out=np.eye(4);out[:3,:3]=Rotation.from_quat(xyzw).as_matrix();out[:3,3]=xyz
    return out


def reconstruct(model,state,poses,side):
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,MIMIC
    names=state["names"];q=np.asarray(state["position"]).reshape(-1)
    if len(names)!=17 or len(set(names))!=17 or q.shape!=(17,) or not np.isfinite(q).all():
        raise ValueError("complete named native joint state required")
    active=np.asarray([q[names.index(name)] for name in ACTIVE])
    mimic=max(abs(q[names.index(f)]-q[names.index(m)]*ratio-offset) for f,m,ratio,offset in MIMIC)
    if mimic>1e-4:raise ValueError("measured mimic diverges from original FK")
    fk={k.rsplit("/",1)[-1]:v for k,v in model["fk"].poses(active).items()}
    delta=sdk_matrix(poses[side+".base_link"])@np.linalg.inv(fk["base_link"])
    frames={k:delta@v for k,v in fk.items()}
    errors=[]
    for key,value in poses.items():
        if not key.startswith(side+"."):continue
        name=key.split(".",1)[1];actual=sdk_matrix(value);predicted=frames[name]
        errors.append(dict(link=name,position_error_m=float(np.linalg.norm(actual[:3,3]-predicted[:3,3])),
            rotation_error_rad=float(Rotation.from_matrix(actual[:3,:3].T@predicted[:3,:3]).magnitude())))
    shapes={}
    for group,records in (("source",model["source_colliders"]),("cooked",model["cooked_pads"])):
        for name,record in records.items():
            t=frames[name];shapes[group+":"+name]=np.asarray(record["vertices"])@t[:3,:3].T+t[:3,3]
    return shapes,errors,float(mimic)


def validate_recipe(frozen):
    if frozen.get("version")=="dh116-902-fixed-wrist-open-reset-v1":
        return
    if frozen.get("version")!="dh116-902-joint-contact-open-reset-v1":
        raise ValueError("independent measured open Reset recipe required")
    if frozen["screen"].get("version")!="dh116-902-joint-contact-path-cpu-v1":
        raise ValueError("new Reset must carry the exact full path CPU proof")
    for name,key in (("joint_search","search_sha256"),("joint_path_proof","supplied_path_proof_sha256")):
        record=frozen[name];artifact=Path(record["path"]).resolve(strict=True)
        if (record["sha256"]!=sha(artifact) or frozen["frozen_files"].get(str(artifact))!=record["sha256"]
                or frozen["screen"].get(key)!=record["sha256"]):
            raise ValueError("new Reset artifact differs from full path evidence")
    record=frozen["joint_candidate"];path=Path(record["path"]).resolve(strict=True)
    path.relative_to(ROOT)
    digest=sha(path)
    if (record["sha256"]!=digest or frozen["frozen_files"].get(str(path))!=digest
            or frozen["screen"]["candidate_sha256"]!=digest):
        raise ValueError("frozen joint candidate bytes differ")
    joint=json.loads(path.read_bytes())
    config=frozen["reset_config"]
    if (joint.get("native_admissible") is not False or joint.get("teacher_admission") is not False
            or config["hands"]["right"]["pose_wxyz"]!=joint["pose_wxyz"]
            or config["hands"]["right"]["q_rad"]!=frozen["screen"]["opening_active_q_rad"]):
        raise ValueError("new measured Reset differs from source-bound open candidate")


def audit(receipt,expected_sha):
    from go2_pvcnn.control.dh116.tabletop_supported_rod_reference import load_parked_reference,_read_artifact
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    from scripts.audit_dh116_pinch_initial_pose import load_geometry
    from scripts.screen_dh116_902_fixed_wrist_open import REQUIRED_SHAPES
    receipt=Path(receipt).resolve(strict=True)
    if sha(receipt)!=expected_sha:raise ValueError("explicit outer receipt SHA differs")
    outer=json.loads(receipt.read_bytes());output=receipt.parent/"native";manifest=outer["artifact_sha256"]
    frozen,_=_read_artifact(output,manifest,"frozen_plan.json")
    validate_recipe(frozen)
    if (output/"tripod_failure.json").exists():raise ValueError("internally failed Reset cannot qualify")
    scene,_=_read_artifact(output,manifest,"tripod_support_scene.json")
    models={side:load_geometry(ROOT,side=side) for side in ("left","right")}
    for model in models.values():
        if frozen["frozen_files"].get(str(Path(model["usd"]["path"]).resolve()))!=model["usd"]["sha256"]:
            raise ValueError("audit USD differs from actual frozen native source")
    initial_raw,initial_sha=_read_artifact(output,manifest,"tripod_reset_source_1.json")
    initial=decode_owned(initial_raw)
    initial_rod=initial["rod_boundary"]["rod_snapshot"]
    if initial_rod["physics_step"]!=2 or initial_rod["rod_tick"]!=0:
        raise ValueError("same-clock measured initial Reset required")
    samples=[np.asarray(initial_rod["node_positions"])[0]];series={s:{name:[] for name in REQUIRED_SHAPES} for s in models}
    errors={s:[] for s in models};mimic={s:0. for s in models}
    hashes={"2":initial_sha}
    for side,model in models.items():
        hand=initial["reset"]["hands"][side]
        body=np.asarray(hand["body_poses_xyzw"])[0]
        initial_poses={side+"."+name:(body[i,:3],body[i,3:])
            for i,name in enumerate(hand["body_names"]) if name in model["source_colliders"]}
        shapes,check,residual=reconstruct(model,dict(names=hand["joint_names"],position=hand["q"]),initial_poses,side)
        for name,value in shapes.items():series[side][name].append(value)
        errors[side].extend(check);mimic[side]=residual
    for sdk in range(3,241):
        reference=load_parked_reference(receipt,expected_sha,sdk,task="tripod")
        raw,digest=_read_artifact(output,manifest,f"tripod_exchange_{sdk}.json")
        hashes[str(sdk)]=digest;row=decode_owned(raw);samples.append(reference.node_positions)
        for side in models:
            shapes,check,residual=reconstruct(models[side],row["native_joint_state"][side],row["hand_link_poses"],side)
            if set(shapes)!=REQUIRED_SHAPES:raise ValueError("all original 22 geometries required")
            for name,value in shapes.items():series[side][name].append(value)
            errors[side].extend(check);mimic[side]=max(mimic[side],residual)
    supports=scene["supports"]
    boxes=[(np.asarray(b["center_m"])-np.asarray(b["size_m"])/2,np.asarray(b["center_m"])+np.asarray(b["size_m"])/2) for b in supports]
    config=frozen["reset_config"]
    # Original table is independently frozen through load_candidate's REFERENCE.
    from scripts.verify_dh116_902_flexible_reset import REFERENCE
    if frozen["frozen_files"].get(str(REFERENCE.resolve()))!=sha(REFERENCE):raise ValueError("original table source differs")
    table=json.loads(REFERENCE.read_bytes())["table"];top=table["center_m"][2]+table["size_m"][2]/2
    results={}
    for side in models:
        bounds={name:temporal_shape_bound(np.stack(v),np.stack(samples),boxes,table_top=top) for name,v in series[side].items()}
        position=max(e["position_error_m"] for e in errors[side]);rotation=max(e["rotation_error_rad"] for e in errors[side])
        passed=(position<=1e-5 and rotation<=1e-4 and all(b["rod_gap_lower_m"]>.001
            and b["support_gap_lower_m"]>.001 and b["table_gap_m"]>=.0021 for b in bounds.values()))
        results[side]=dict(bounds=bounds,max_recorded_link_position_error_m=position,
            max_recorded_link_rotation_error_rad=rotation,max_mimic_error_rad=mimic[side],
            recorded_link_checks=len(errors[side]),passed=bool(passed))
    return dict(scope="saved_native_base_and_q_original_USD_FK_derived_clearance",
        sdk_range=[2,240],frame_count=239,measured_exchange_frames=238,hands=results,geometry_passed=all(v["passed"] for v in results.values()),
        receipt_sha256=expected_sha,sample_sha256=hashes,source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__)},
        geometry_sources={side:model["source_sha256"] for side,model in models.items()},
        all_links_independently_measured=False,rod_steady_qualified=False,grasp_qualified=False,native_runs=0,ppo_updates=0,teacher_admission=False,
        limitations=["Unrecorded link poses derive from original USD FK and measured active q",
            "Recorded link FK residual is evidence, not a bound on every unrecorded link",
            "No self/inter-hand collision or contact/Lift/Hold qualification"])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt",type=Path,required=True);p.add_argument("--receipt-sha256",required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args(argv)
    result=audit(a.receipt,a.receipt_sha256)
    with a.output.open("x") as f:json.dump(result,f,indent=2,allow_nan=False)
    return int(not result["geometry_passed"])


if __name__=="__main__":raise SystemExit(main())
