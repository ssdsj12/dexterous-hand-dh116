#!/usr/bin/env python3
"""One CPU-only audit of the frozen failed right Reset's closest support."""
import argparse,json,os,signal,time
from pathlib import Path
from scripts.solve_dh116_right_pinch_reset_direction import Geometry,previous
from go2_pvcnn.control.dh116.tabletop_polyline_closest_support import refine,validate

PARENT=Path("/mnt/mydisk/xk/projects/dh116-autonomous-3c80166/right-reset-direction-v4-20261113-379rl4yl/run01/result.json")
PARENT_SHA="bd311cc064544f0b6c4ee295ac49d7b5e72f883dacd22c9f36ea6daa8787b865"

def run():
    if previous.digest(PARENT)!=PARENT_SHA:raise ValueError("fixed failed Reset SHA required")
    parent=previous.decode_owned(json.loads(PARENT.read_bytes()))
    if parent["static_path_passed"] is not False:raise ValueError("expected failed actual candidate")
    g=Geometry();v=g.shapes(g.poses(parent["best_x"],1))["source:finger13_link"]
    certificate=refine(v,g.anchor["nodes"],max_segments=4);validate(certificate,v,g.anchor["nodes"])
    old=certificate["original_certificate"]["lower_m"];new=certificate["lower_m"]
    return dict(status="complete",scope="CPU_frozen_right_source_finger13_endpoint_only",
        source_result_sha256=PARENT_SHA,source_ledger_sha256=g.anchor["ledger_sha256"],
        source_geometry=g.g["source_sha256"],certificate=certificate,
        old_lower_m=old,refined_lower_m=new,proof_improved=new>old,
        original_other_shape_safety_passed=new>=0,
        optimization_reserve_passed=new>=.0001,full_path_checked=False,
        required_pad_SDF_unchanged=parent["selected"]["boundaries"]["1"]["pads"],
        actual_contact_measurement=False,native_qualification=False,teacher_admission=False,
        native_recipe_created=False,new_physics_steps=0,ppo_updates=0)

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    out=dict(status="partial",new_physics_steps=0,ppo_updates=0,wall_budget_s=60)
    handler=signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError("60s CPU support audit exhausted")))
    signal.setitimer(signal.ITIMER_REAL,60)
    try:out.update(run())
    except Exception as exc:out.update(status="blocked",reason=type(exc).__name__+": "+str(exc))
    finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
    out.update(elapsed_s=time.monotonic()-start,source_sha256=previous.digest(__file__))
    with (a.output/"result.json").open("x") as f:
        json.dump(previous.encode_owned(out),f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    print(json.dumps({k:out.get(k) for k in ("status","reason","elapsed_s","old_lower_m","refined_lower_m","original_other_shape_safety_passed")}))
    return int(out["status"]!="complete")
if __name__=="__main__":raise SystemExit(main())
