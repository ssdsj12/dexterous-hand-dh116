#!/usr/bin/env python3
"""Continuous static-path bounds for independently passing original-pad targets."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
from scripts import solve_dh116_pinch_effective_targets as targets
from scripts.solve_dh116_right_pinch_owned_approach import grid_lipschitz
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import encode_owned,decode_owned
from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
PARENT=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/pinch-effective-target-20261114-run03/result.json")
PARENT_SHA="cc6fcfcbf001d597fcc46d0a27768974f8e0f4b1714a872074ac8a6a65d8ddbe"
VERSION="dh116-effective-pinch-target-path-cpu-v3"
CONFIG=dict(version=VERSION,mode="cpu_only",path_samples=1025,maximum_seconds=60.,
    motion_steps=160,physics_hz=480,bow_world_x_m=.0001,native_execution=False,teacher_admission=False)

def bow(fraction,amplitude_m):
    if not np.isfinite([fraction,amplitude_m]).all() or not 0<=fraction<=1 or not 0<=amplitude_m<=.0001:
        raise ValueError("frozen100um geometric detour within original wrist limits")
    nodes=np.linspace(0,1,21)
    return float(np.interp(fraction,nodes,4*amplitude_m*nodes*(1-nodes)))

def wrist_path_lipschitz(points):
    points=np.asarray(points,dtype=float)
    if points.shape!=(21,3) or not np.isfinite(points).all():
        raise ValueError("twenty finite piecewise linear60Hz segments")
    return float(20*np.linalg.norm(np.diff(points,axis=0),axis=1).max())

def run(output,config):
    if config!=CONFIG:raise ValueError("frozen CPU-only path budget required")
    surface=targets.surface;base=surface.base
    p=decode_owned(base.checked_json(PARENT,PARENT_SHA));pair=targets.parent.load_pair(targets.ROOT)
    nodes=np.stack([np.full(41,.43),np.linspace(-.25,.25,41),np.full(41,.305)],axis=1)
    started=time.monotonic();out=dict(version=VERSION,status="complete",parent_sha256=PARENT_SHA,sides={},
        new_physics_steps=0,ppo_updates=0,native_ready=False,teacher_admission=False,
        actual_contact_observed=False,native_recipe_created=False)
    with (output/"path.jsonl").open("x") as f:
        for side in ("left","right"):
            record=p["sides"][side];end=record["search"]["best"]
            if not end["admission"]["static_constraints_passed"]:
                out["sides"][side]=dict(status="blocked",reason="parent endpoint failed",path_samples=0)
                continue
            try:
                old=pair[side]["candidate"];start,delta=targets.original_path(end,old,0.)
                rate=delta/(160/480);motion=np.array(old.get("root_translation_delta_m",[0,0,0]))
                if (np.any(abs(rate)>ACTION_CONTRACT["joint_rate_rad_s"]+1e-12)
                    or np.any(abs(motion/(160/480))>ACTION_CONTRACT["translation_rate_m_s"]+1e-12)):
                    raise ValueError("original action speed cap")
                g=base.load_geometry(targets.ROOT,side=side)
                grids,bindings=base.grids_for(g,base.checked_json(base.SDF_DIR/"result.json",base.SDF_SHA))
                if g["source_sha256"]!=record["source_geometry_sha256"] or bindings!=record["sdf_bindings"]:
                    raise ValueError("same original geometry and SDF required")
                travel={**{"source:"+n:v for n,v in base.path_displacement_bound(g,start["q_rad"],delta).items()},
                    **{"cooked:"+n:v for n,v in base.path_displacement_bound(g,start["q_rad"],delta,cooked=True).items()}}
                ts=np.linspace(0,1,21)
                root_path=ts[:,None]*motion[None,:]
                root_path[:,0]+=[bow(float(t),config["bow_world_x_m"]) for t in ts]
                root_deltas=np.diff(root_path,axis=0)
                root_rates=root_deltas/(8/480)
                if np.any(abs(root_rates)>ACTION_CONTRACT["translation_rate_m_s"]+1e-12):
                    raise ValueError("original per-axis wrist velocity cap")
                root_travel=wrist_path_lipschitz(root_path)
                travel={n:float(v+root_travel) for n,v in travel.items()}
                lipschitz={n:grid_lipschitz(grids[n]) for n in surface.PADS}
                denominator=2*(config["path_samples"]-1)
                worst=dict(table_m=float("inf"),other_m=float("inf"),depth_m=float("inf"))
                first=None;peak_id={};count=0
                for k,fraction in enumerate(np.linspace(0,1,config["path_samples"])):
                    if time.monotonic()-started>config["maximum_seconds"]:raise TimeoutError("CPU path budget exhausted")
                    pose,_=targets.original_path(end,old,float(fraction))
                    pose["root_pose_wxyz"][0]+=bow(float(fraction),config["bow_world_x_m"])
                    T=base.pose_matrix(pose["root_pose_wxyz"]);q=np.array(pose["q_rad"])
                    audit=surface.geometry_record(g,grids,T,q,nodes)
                    if k==0:
                        transforms=base.fk_poses(g,pose["root_pose_wxyz"],q)
                        initial={kind+":"+n:surface._separation(base.transform(v["vertices"],transforms[n]),nodes)
                            for kind,key in (("source","source_colliders"),("cooked","cooked_pads"))
                            for n,v in g[key].items()}
                        first=dict(all_shape_rod_lower_m=initial,minimum_m=min(initial.values()))
                    table={n:v-travel[n]/denominator for n,v in audit["table_m"].items()}
                    other={n:v-travel[n]/denominator for n,v in audit["non_target_rod_gap_m"].items()}
                    depth={n:audit["pads"][n]["minimum_gap_m"]-lipschitz[n]*travel["cooked:"+n]/denominator for n in surface.PADS}
                    row=dict(side=side,index=k,fraction=float(fraction),table_lower_m=table,other_lower_m=other,
                        required_sdf_lower_m=depth,reference_geometry_only=True)
                    for name,values in (("table_m",table),("other_m",other),("depth_m",depth)):
                        key,value=min(values.items(),key=lambda a:a[1])
                        if value<worst[name]:worst[name]=value;peak_id[name]=dict(index=k,shape=key,value_m=value)
                    f.write(json.dumps(row,allow_nan=False)+"\n");count+=1
                passed=first["minimum_m"]>=.00005 and worst["table_m"]>=.0021 and worst["other_m"]>=0 and worst["depth_m"]>=-.0003
                out["sides"][side]=dict(status="complete",path_samples=count,initial=first,worst=worst,worst_identity=peak_id,
                    static_path_passed=bool(passed),physical_joint_velocity_rad_s=rate.tolist(),
                    root_velocity_world_m_s=root_rates.tolist(),bow_world_x_m=config["bow_world_x_m"],
                    same_original_finger_action_amplitude=True,original_wrist_limits_preserved=True,
                    wrist_trajectory_changed=True,start=start,end=end,travel_bounds_m=travel,root_lipschitz_per_fraction_m=root_travel,sdf_lipschitz=lipschitz,
                    dynamic_tracking_qualified=False,rod_response_qualified=False,native_ready=False)
            except (ValueError,RuntimeError,TimeoutError) as e:
                out["sides"][side]=dict(status="blocked",reason=type(e).__name__+": "+str(e),native_ready=False)
            f.flush();os.fsync(f.fileno())
    out["elapsed_s"]=time.monotonic()-started
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--config",type=Path,default=targets.ROOT/"configs/dh116_tasks/flexible_pinch_effective_path_cpu_v3.json")
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:r=run(a.output,json.loads(a.config.read_bytes()))
    except Exception as e:r=dict(status="blocked",reason=type(e).__name__+": "+str(e),new_physics_steps=0,ppo_updates=0,native_ready=False)
    r.update(source_sha256=targets.surface.base.digest(__file__),config_sha256=targets.surface.base.digest(a.config),
        target_module_sha256=targets.surface.base.digest(targets.__file__))
    with (a.output/"result.json").open("x") as f:
        json.dump(encode_owned(r),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status=r["status"],reason=r.get("reason"),
        sides={s:{k:v.get(k) for k in ("status","reason","path_samples","static_path_passed","worst")} for s,v in r.get("sides",{}).items()})))
    return int(r["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
