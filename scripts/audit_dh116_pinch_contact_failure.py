"""Read-only contact chronology and failure certificates for a frozen native ledger."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

DT=1/480
SUBDT=DT/12
IDENTITY=("run_uuid","env_id","episode_uid","reset_id","rod_uid")
PADS=("finger14_link","finger23_link","finger33_link","finger43_link","finger53_link")
ALLOWED={"table"}|{s+"."+p for s in ("left","right") for p in PADS}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()

def summarize_exchanges(exchanges):
    first=None;last_step=2;terminal=False;count=0;contacts={};table_count=0
    both_count=0;run=longest=0;min_table={s:float("inf") for s in ("left","right")}
    traces=[];max_unit_error=0.
    for x in exchanges:
        r=x["rod"];step=x["physics_step"]
        ident={k:r[k] for k in IDENTITY}
        if first is None:first=ident
        if (ident!=first or type(step) is not int or step!=last_step+1 or terminal
                or r["physics_step"]!=step or r["rod_tick"]!=(step-2)*12
                or not math.isclose(r["timestamp_s"],step*DT,rel_tol=0,abs_tol=2e-7)
                or x["policy_step"]!=(step-3)//8 or x["substep"]!=(step-3)%8):
            raise ValueError("non-contiguous or cross-episode terminal/clock evidence")
        post=x["action_readback"]["post"]["identity"]
        if len(post)!=1 or any(post[0].get(k)!=r[k] for k in IDENTITY+("physics_step","rod_tick","timestamp_s")):
            raise ValueError("action and physical identity differ")
        last_step=step;count+=1;terminal=x["guard_failure"] is not None
        packet=r["rod_mesh_contact_pairs"]
        if not math.isclose(packet["substep_dt_s"],SUBDT,rel_tol=0,abs_tol=1e-12):
            raise ValueError("solver substep dt differs")
        present=[set() for _ in range(12)]
        per_tick={}
        for p in packet["pairs"]:
            mesh=p["mesh_name"];sub=p["substep"]
            if mesh not in ALLOWED or type(sub) is not int or not 0<=sub<12 or not 0<=p["node_index"]<41:
                raise ValueError("unregistered contact partner or solver substep")
            force=np.asarray(p["force_on_rod_world_n"],float)
            impulse=np.asarray(p["impulse_on_rod_ns"],float)
            if (force.shape!=(3,) or impulse.shape!=(3,) or not np.isfinite(force).all()
                    or not np.isfinite(impulse).all() or not math.isfinite(p["normal_force_on_rod_n"])
                    or p["normal_force_on_rod_n"]<0):
                raise ValueError("finite actual force/impulse required")
            err=float(np.linalg.norm(impulse-force*SUBDT));max_unit_error=max(max_unit_error,err)
            if err>1e-8:raise ValueError("force and impulse/dt differ")
            if mesh=="table":table_count+=1
            else:
                c=contacts.setdefault(mesh,dict(count=0,first_physics_step=step,
                    first_solver_substep=sub,last_physics_step=step,last_solver_substep=sub,
                    max_normal_force_n=0.,max_force_n=0.))
                c["count"]+=1;c["last_physics_step"]=step;c["last_solver_substep"]=sub
                c["max_normal_force_n"]=max(c["max_normal_force_n"],p["normal_force_on_rod_n"])
                c["max_force_n"]=max(c["max_force_n"],float(np.linalg.norm(force)))
                per_tick[mesh]=per_tick.get(mesh,0)+1
                if p["normal_force_on_rod_n"]>0:present[sub].add(mesh)
        for meshes in present:
            simultaneous={"left.finger14_link","left.finger23_link"}<=meshes
            both_count+=int(simultaneous);run=run+1 if simultaneous else 0;longest=max(longest,run)
        geometry=x["closure_geometry"]
        for s in min_table:min_table[s]=min(min_table[s],geometry["sides"][s]["minimum_table_clearance_m"])
        traces.append(dict(physics_step=step,policy_step=x["policy_step"],substep=x["substep"],
            timestamp_s=r["timestamp_s"],hand_pair_count=per_tick,
            minimum_non_target_gap_lower_m={s:geometry["sides"][s]["minimum_non_target_gap_lower_m"] for s in min_table},
            failure_reasons=geometry["failure_reasons"],guard_failure=x["guard_failure"],
            original_guard_failure=x["original_guard_failure"]))
    if first is None:raise ValueError("no complete physical rows")
    return dict(identity=first,measured_physics_rows=count,first_physics_step=3,last_physics_step=last_step,
        terminal_reason=traces[-1]["guard_failure"],hand_contact_pairs=contacts,table_pair_count=table_count,
        both_left_contact_solver_steps=both_count,both_left_contiguous_solver_steps=longest,
        both_left_contiguous_s=longest*SUBDT,minimum_table_clearance_m=min_table,
        max_force_impulse_residual_ns=max_unit_error,trajectory=traces,
        source="native solver pair ledger; not SDF contact prediction",
        native_contact_observed=bool(contacts),teacher_admission=False,grasp_qualified=False)

def analyze(ledger,project):
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    from go2_pvcnn.control.dh116.tabletop_polyline_support_certificate import certify,validate
    from scripts.audit_dh116_pinch_initial_pose import load_geometry,pose_matrix,transform
    first_last={}
    def rows():
        with Path(ledger).open() as f:
            for line in f:
                if not line.endswith("\n"):raise ValueError("incomplete final JSONL row")
                d=json.loads(line)
                if d.get("operation")=="physical":
                    x=decode_owned(d["payload"])["exchange"]
                    first_last.setdefault("first",x);first_last["last"]=x
                    yield x
    result=summarize_exchanges(rows())
    first=first_last["first"];last=first_last["last"];certs=[]
    for reason in last["closure_geometry"]["failure_reasons"]:
        parts=reason.split(":")
        if len(parts)!=4 or parts[1]!="non_target_not_proven_clear":continue
        side,_,kind,name=parts
        geo=load_geometry(project,side=side)
        if geo["source_sha256"]!=last["closure_geometry"]["sources"][side]["source_sha256"]:
            raise ValueError("actual source mesh provenance differs")
        raw=last["closure_geometry"]["raw"];hand=raw["hands"][side]
        pose=hand["link_poses_xyzw"][0][hand["body_names"].index(name)]
        mesh=geo["source_colliders" if kind=="source" else "cooked_pads"][name]
        vertices=transform(mesh["vertices"],pose_matrix(np.r_[pose[:3],pose[6],pose[3:6]]))
        certificate=certify(vertices,raw["rod_nodes_m"])
        validate(certificate,vertices,raw["rod_nodes_m"])
        if not math.isclose(certificate["old_global_xz_lower_m"],
                last["closure_geometry"]["sides"][side]["non_target_surface_gap_lower_m"][kind+":"+name],
                rel_tol=0,abs_tol=1e-12):raise ValueError("historical bound not reproduced")
        certs.append(dict(reason=reason,physics_step=last["physics_step"],certificate=certificate,
            explanation="old XZ whole-rod bound was inconclusive; new full-3D segment proof is CPU only"))
    pads=("finger14_link","finger23_link","finger43_link","finger53_link")
    result.update(version="dh116-pinch-contact-failure-audit-v1",ledger_path=str(ledger),
        ledger_sha256=digest(ledger),refined_failure_certificates=certs,
        initial_left_clearance_m={p:{k:first["closure_geometry"]["sides"]["left"]["table_clearance_m"][k+":"+p]
                                    for k in ("source","cooked")} for p in pads},
        initial_left_penetration_m={p:max(0.,-first["closure_geometry"]["sides"]["left"]["table_clearance_m"]["cooked:"+p]) for p in pads},
        last_original_guard=last["original_guard_failure"],
        table_contact_force_source="hand-to-static-table NA; rod-to-table pair observed",
        immutable_history_result="failed; original receipt not rewritten")
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ledger",required=True,type=Path);p.add_argument("--sha256",required=True)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    if digest(a.ledger)!=a.sha256:raise ValueError("immutable native ledger SHA differs")
    result=analyze(a.ledger,Path(__file__).resolve().parents[1])
    a.output.mkdir(parents=True,exist_ok=False)
    trajectory=result.pop("trajectory")
    with (a.output/"trajectory.jsonl").open("x") as f:
        for row in trajectory:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
    result["trajectory_sha256"]=digest(a.output/"trajectory.jsonl")
    with (a.output/"result.json").open("x") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    print(json.dumps({k:result[k] for k in ("last_physics_step","hand_contact_pairs","both_left_contiguous_s","teacher_admission")}))
if __name__=="__main__":main()
