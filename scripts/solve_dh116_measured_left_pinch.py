"""Bounded CPU proposals from the moving-left interval, never native admission."""
from copy import deepcopy
from pathlib import Path
import argparse,json,os,time
import numpy as np
from scripts import audit_dh116_projected_pinch_tracking as source
from scripts import solve_dh116_left_pinch_tracking as old
from scripts import prepare_dh116_left_pinch_approach as plan
from scripts import solve_dh116_pinch_tracking_targets as observed
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,expand
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned

ROOT=Path(__file__).resolve().parents[1]
VERSION="dh116-moving-left-native-source-cpu-proposals-v1"
CONFIG=dict(version=VERSION,maximum_evaluations=600,maximum_search_seconds=60.,
    search_physics_steps=[211,222,234],validation_physics_steps=list(range(211,235)),
    original_candidate_limits=deepcopy(old.CONFIG),native_execution=False,
    physics_steps=0,ppo_updates=0,historical_errors_are_predictions_for_new_actions=True,
    scope="independent reset and endpoint proposal, unchanged original finger increment")

def validate_packet(p):
    if (p["side"]!="left" or p["executed_left_actions_zero"] is not False
        or p["left_motion_observed"] is not True or p["native_ready"] or p["teacher_admission"]
        or p["source_sha256"]["receipt"]!=source.RECEIPT_SHA):
        raise ValueError("frozen moving-left native source, never native candidate admission")
    first=None
    if [s["identity"]["physics_step"] for s in p["scenarios"]]!=CONFIG["validation_physics_steps"]:
        raise ValueError("complete adjacent source window")
    for s in p["scenarios"]:
        key=tuple(s["identity"][k] for k in ("run_uuid","env_id","episode_uid","reset_id","rod_uid"))
        if (s["identity"]!=s["joint_identity"] or s["identity"]!=s["rod_identity"]
            or first is not None and key!=first or s["source_hand"]!="left" or not s["empirical_only"]):
            raise ValueError("same left physical/joint/rod identities")
        first=key
        for name,shape in (("active_error_rad",(6,)),("measured_active_q_rad",(6,)),("rod_nodes_m",(41,3)),
                            ("root_translation_error_m",(3,)),("world_rotation_error_xyzw",(4,))):
            a=np.asarray(s[name],float)
            if a.shape!=shape or not np.isfinite(a).all():raise ValueError("complete finite source "+name)
        if not np.allclose(np.asarray(p["anchor"]["q_rad"])+s["active_error_rad"],
                           s["measured_active_q_rad"],atol=1e-14,rtol=0):
            raise ValueError("same executed command residual")
    return True

def load_packet():
    source_packet=source.load_source();pair=source_packet["plan"]
    end=plan.expected_boundary(pair,232);candidate=pair["left"]["candidate"]
    rows=source_packet["rows"]
    moved=any(np.any(x["action_readback"]["request"]["action24"][0,:6]) for x in rows)
    if not moved:raise ValueError("held-left source cannot replace actual moving-left episode")
    anchor=dict(root_pose_wxyz=end["root_poses_w"][0].tolist(),q_rad=end["left_q_rad"].tolist())
    scenarios=[]
    for x in rows[-24:]:
        raw=x["closure_geometry"]["raw"];identity=source.validate_boundary(x)
        j=raw["joint_readback"]["hands"]["left"];h=raw["hands"]["left"]
        root=h["link_poses_xyzw"][0,h["body_names"].index("base_link")]
        measured=np.r_[root[:3],root[6],root[3:6]];q=source.reuse.active_values(j)
        full=np.array([j["q"][0,j["joint_names"].index(n)] for n in JOINTS])
        scenarios.append(dict(identity=deepcopy(identity),joint_identity=deepcopy(raw["joint_readback"]["identity"][0]),
            rod_identity=deepcopy(identity),source_hand="left",empirical_only=True,
            root_translation_error_m=(measured[:3]-end["root_poses_w"][0,:3]).tolist(),
            world_rotation_error_xyzw=(observed.rotation(measured)*observed.rotation(end["root_poses_w"][0]).inv()).as_quat().tolist(),
            active_error_rad=(q-end["left_q_rad"]).tolist(),rod_nodes_m=raw["rod_nodes_m"].tolist(),
            measured_root_wxyz=measured.tolist(),measured_active_q_rad=q.tolist(),
            observed_coupling_residual_rad=(full-expand(q)).tolist()))
    p=dict(side="left",anchor=anchor,scenarios=scenarios,original_candidate=deepcopy(candidate),
        original_joint_delta_rad=deepcopy(candidate["active_delta_rad"]),executed_left_actions_zero=False,
        left_motion_observed=True,source_sha256=dict(receipt=source.RECEIPT_SHA,
            ledger=source_packet["source"]["physical_sha256"],
            audit=source_packet["source"]["post_exit_audit_sha256"]),
        interpretation="actual projected moving-left residuals; transfer to new controls is unqualified",
        native_ready=False,teacher_admission=False)
    validate_packet(p);return p

def run(output):
    packet=load_packet();geometry=old.Geometry(packet)
    baseline=geometry.evaluate(np.zeros(9),packet["scenarios"])
    selected=[s for s in packet["scenarios"] if s["identity"]["physics_step"] in CONFIG["search_physics_steps"]]
    with (output/"search.jsonl").open("x") as stream:
        def query(x):
            row=geometry.evaluate(x,selected)
            stream.write(json.dumps(encode_owned(row),allow_nan=False)+"\n");stream.flush()
            return row
        search=old.nonlinear_search(query,np.zeros(9),maximum_evaluations=CONFIG["maximum_evaluations"],
                                    maximum_seconds=CONFIG["maximum_search_seconds"])
        stream.flush();os.fsync(stream.fileno())
    validation=geometry.evaluate(np.asarray(search["best"]["x"]),packet["scenarios"])
    return dict(version=VERSION,status="complete",config=CONFIG,source=packet,baseline=baseline,
        search=search,validation=validation,
        qualification=observed.admission(validation["nominal"],list(validation["scenarios"].values())),
        geometry_source_sha256=geometry.g["source_sha256"],sdf_bindings=geometry.bindings,
        original_joint_increment_preserved=True,continuous_path_proved=False,
        native_ready=False,teacher_admission=False,new_physics_steps=0,ppo_updates=0)

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:r=run(a.output)
    except Exception as error:
        r=dict(version=VERSION,status="blocked",reason=type(error).__name__+": "+str(error),
            new_physics_steps=0,ppo_updates=0,native_ready=False,teacher_admission=False)
    r.update(elapsed_s=time.monotonic()-start,script_sha256=source.qualified.sha(__file__))
    with (a.output/"result.json").open("x") as f:
        json.dump(encode_owned(r),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:r.get(k) for k in ("status","reason","elapsed_s","qualification","native_ready")}))
    return int(r["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
