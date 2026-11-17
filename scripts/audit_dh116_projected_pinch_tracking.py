"""Read-only same-boundary geometry decomposition; no dynamic causal claim."""
from pathlib import Path
import argparse,json,math
import numpy as np
from scripts import audit_dh116_projected_pinch_contact as qualified
from scripts import audit_dh116_effective_right_tracking as reuse
from scripts import prepare_dh116_left_pinch_approach as plan
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,expand
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned,encode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_pad_region import bind_geometry

ROOT=Path(__file__).resolve().parents[1]
VERSION="dh116-projected-pinch-ordered-geometry-decomposition-v1"
RECEIPT=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/flex.f8-original-projected-pinch-native/d2b175b7654b4b30c67028a20a6b0e1a50eda98b3804414554c6d8c9fd3f7d58/attempt-zy2fao0h/receipt.json")
RECEIPT_SHA="04ebfe2460b6366960d87cda6f36200f503072e4fdfa2abffd25bc8dfcf775c3"
STAGES=("planned_nominal","sdk_target_nominal","measured_fk_nominal","measured_links_nominal","measured_links_actual_rod")

def named_sdk_active(record,target):
    names=record["joint_names"];indices=target["joint_indices"]
    values=np.asarray(target["values"],float)
    if (len(set(names))!=len(names) or len(set(indices))!=len(indices)
        or any(type(i) is not int or not 0<=i<len(names) for i in indices)
        or values.shape!=(1,len(indices)) or not np.isfinite(values).all()):
        raise ValueError("unique named finite actual SDK targets")
    indexed={names[i]:v for i,v in zip(indices,values[0])}
    if not set(ACTIVE)<=set(indexed):raise ValueError("all actual active SDK targets")
    return np.array([indexed[n] for n in ACTIVE])

def ellipse_differences(stages):
    vals=np.array([stages[k]["ellipse_value"] for k in STAGES],float)
    if not np.isfinite(vals).all():raise ValueError("finite ordered geometry")
    delta={a+" -> "+b:float(y-x) for a,b,x,y in zip(STAGES,STAGES[1:],vals,vals[1:])}
    return dict(ordered_deltas=delta,total=float(vals[-1]-vals[0]),causal_effect_established=False,
        ordering_matters=True,units="dimensionless squared original ellipse radius")

def validate_boundary(x):
    raw=x["closure_geometry"]["raw"];c=raw["identity"]
    if (x["guard_failure"] is not None or raw["joint_readback"]["identity"]!=[c]
        or x["action_readback"]["post"]["identity"]!=[c]
        or x["closure_geometry"]["pad_region_diagnostic"]["identity"]!=c
        or type(c["physics_step"]) is not int or not 3<=c["physics_step"]<=234
        or x["physics_step"]!=c["physics_step"] or x["policy_step"]!=(c["physics_step"]-3)//8
        or not math.isfinite(c["timestamp_s"]) or abs(c["timestamp_s"]-c["physics_step"]/480)>1e-9):
        raise ValueError("same measured nonterminal physical/joint/rod/action identity")
    nodes=np.asarray(raw["rod_nodes_m"],float)
    if nodes.shape!=(41,3) or not np.isfinite(nodes).all():raise ValueError("complete finite measured rod")
    for side in ("left","right"):
        h=raw["hands"][side];names=h["body_names"];poses=np.asarray(h["link_poses_xyzw"],float)
        if (len(names)!=len(set(names)) or "base_link" not in names or poses.shape!=(1,len(names),7)
            or not np.isfinite(poses).all()
            or not np.allclose(np.linalg.norm(poses[0,:,3:],axis=1),1.,atol=1e-5,rtol=0)):
            raise ValueError("complete finite own-side measured link poses")
    return c

def load_source(*,expected_sha=RECEIPT_SHA):
    if expected_sha!=RECEIPT_SHA:raise ValueError("exact frozen native receipt")
    summary=qualified.run(RECEIPT,expected_sha)
    r=json.loads(qualified.read_locked(RECEIPT,expected_sha))
    raw=qualified.read_locked(Path(r["output"])/"physical.jsonl",r["artifact_sha256"]["physical.jsonl"])
    rows=[decode_owned(e["payload"])["exchange"] for e in map(json.loads,raw.splitlines())
          if e.get("operation")=="physical"]
    for x in rows:validate_boundary(x)
    return dict(rows=rows,receipt=r,source=summary["source"],plan=plan.load_pair(ROOT))

class Geometry:
    def __init__(self):
        t=reuse.target;self.target=t;self.sides={}
        for side in ("left","right"):
            g=t.surface.base.load_geometry(ROOT,side=side)
            grids,bindings=t.surface.base.grids_for(g,t.surface.base.checked_json(
                t.surface.base.SDF_DIR/"result.json",t.surface.base.SDF_SHA))
            atlas={n:t.surface._patch(g["source_colliders"][n]["vertices"],side,n) for n in t.surface.PADS}
            self.sides[side]=(g,grids,atlas,bindings)
        self.nominal=np.stack([np.full(41,.43),np.linspace(-.25,.25,41),np.full(41,.305)],axis=1)

    def compare(self,x,packet):
        c=validate_boundary(x);raw=x["closure_geometry"]["raw"];t=self.target
        expected=plan.expected_boundary(packet,c["physics_step"]-2);nodes=np.asarray(raw["rod_nodes_m"],float)
        submissions=x["action_readback"]["submission"]
        sdk_root=np.asarray(submissions["projected_root_poses"],float)
        if sdk_root.shape!=(1,2,7) or not np.isfinite(sdk_root).all():raise ValueError("actual SDK root poses")
        pads={};hands={}
        for i,side in enumerate(("left","right")):
            g,grids,atlas,bindings=self.sides[side]
            h=raw["hands"][side];j=raw["joint_readback"]["hands"][side]
            q=reuse.active_values(j);sdkq=named_sdk_active(j,submissions["expected_backend_targets"][side])
            requested=np.asarray(x["action_readback"]["request"]["requested_finger_targets_rad"])[0,i*6:(i+1)*6]
            if not np.allclose(sdkq,requested,atol=1e-6,rtol=0):raise ValueError("SDK/issued joint mapping mismatch")
            poses={n:t.surface.base.pose_matrix(np.r_[p[:3],p[6],p[3:6]])
                for n,p in zip(h["body_names"],h["link_poses_xyzw"][0])}
            root=h["link_poses_xyzw"][0,h["body_names"].index("base_link")]
            rootw=np.r_[root[:3],root[6],root[3:6]]
            nominal=t.surface.base.fk_poses(g,expected["root_poses_w"][i],expected[side+"_q_rad"])
            sdk=t.surface.base.fk_poses(g,sdk_root[0,i],sdkq)
            fk=t.surface.base.fk_poses(g,rootw,q)
            full=np.array([j["q"][0,j["joint_names"].index(n)] for n in JOINTS])
            hands[side]=dict(active_names=list(ACTIVE),sdk_active_q_rad=sdkq.tolist(),actual_active_q_rad=q.tolist(),
                active_tracking_error_rad=(q-sdkq).tolist(),
                planned_active_error_rad=(q-expected[side+"_q_rad"]).tolist(),
                root_actual_minus_sdk_mm=((rootw[:3]-sdk_root[0,i,:3])*1000).tolist(),
                root_sdk_minus_planned_mm=((sdk_root[0,i,:3]-expected["root_poses_w"][i,:3])*1000).tolist(),
                all_joint_names=list(JOINTS),coupling_residual_rad=(full-expand(q)).tolist())
            for n in t.surface.PADS:
                name=side+"."+n
                bound=bind_geometry(raw,c,name,x["rod"]["mesh_bindings"][name])
                if not np.allclose(bound,poses[n],atol=1e-12,rtol=0):raise ValueError("own-side collision transform mismatch")
                states=((nominal,self.nominal),(sdk,self.nominal),(fk,self.nominal),(poses,self.nominal),(poses,nodes))
                values={key:t.predict_region(grids[n],state[n],rod,atlas[n])
                        for key,(state,rod) in zip(STAGES,states)}
                actual=[p for p in x["closure_geometry"]["pad_region_diagnostic"]["pairs"] if p["mesh_name"]==name]
                pads[name]=dict(stages=values,ordered_geometry_difference=ellipse_differences(values),
                    actual_region_pairs=len(actual),
                    actual_original_region_accepted=sum(p["region"]["accepted_by_original_patch"] for p in actual),
                    actual_region_pairs_by_substep=actual,
                    fk_minus_link_position_mm=((fk[n][:3,3]-poses[n][:3,3])*1000).tolist(),
                    region_prediction_is_not_contact=True)
        return dict(identity=c,actor_action24=x["action_readback"]["request"]["action24"],
            pads=pads,hands=hands,rod_displacement_max_abs_mm=(abs(nodes-self.nominal).max(axis=0)*1000).tolist(),
            native_counterfactual=False,teacher_admission=False)

def run():
    source=load_source();g=Geometry()
    # Last 24 adjacent physical boundaries, never different episodes' maxima.
    rows=[g.compare(x,source["plan"]) for x in source["rows"][-24:]]
    if [r["identity"]["physics_step"] for r in rows]!=list(range(211,235)):
        raise ValueError("complete terminal window")
    return dict(version=VERSION,source=source["source"],window=rows,
        source_geometry={s:{"source_sha256":v[0]["source_sha256"],"sdf_bindings":v[3]} for s,v in g.sides.items()},
        qualification="ordered geometry attribution only; not a causal or native counterfactual experiment",
        native_counterfactual=False,new_physics_steps=0,ppo_updates=0,teacher_admission=False,
        limitations=["Intermediate nominal states are CPU comparisons, not executed trajectories.",
            "Active FK uses original coupling; measured links retain actual passive dynamics.",
            "Original patch predictions do not establish contact, opposing normals, load or success.",
            "Actual solver pairs retain their own substep; post-step geometry is not substituted for raw forces."])

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    r=run()
    with a.output.open("x") as f:json.dump(encode_owned(r),f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(dict(output=str(a.output),sha256=qualified.sha(a.output),rows=len(r["window"]),
        native_counterfactual=False,teacher_admission=False)))
if __name__=="__main__":main()
