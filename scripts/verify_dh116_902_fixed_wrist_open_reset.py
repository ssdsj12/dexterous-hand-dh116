"""One-shot measured settling of the independently screened 902 open Reset.

No closing actions, contact admission, camera acquisition or Teacher update.
"""
import argparse
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import numpy as np

from scripts import verify_dh116_902_flexible_reset as old
from scripts import screen_dh116_902_fixed_wrist_open as screen

ROOT=old.ROOT
SDK_END=240
SOLVER_STEPS=2856
BUDGET=dict(mode="902_tripod_fixed_wrist_open_settle",sdk_steps=SDK_END,
    sdk_setup_steps=2,control_steps=0,solver_steps=SOLVER_STEPS,
    teacher_queries=0,ppo_updates=0,model_loads=0,soft_wall_seconds=180,
    auto_retries=0,auto_kill=False,new_power_mesh_registrations=6)


def reset_config(candidate_path):
    candidate_path=Path(candidate_path).resolve(strict=True)
    candidate_path.relative_to(ROOT)
    candidate=json.loads(candidate_path.read_bytes())
    if (candidate.get("version")!=screen.VERSION or candidate.get("grasp")!="tripod"
            or candidate.get("side")!="right" or candidate.get("native_admissible") is not False
            or candidate.get("teacher_admission") is not False):
        raise ValueError("original CPU-only Tripod/right opening required")
    source,request=screen.checked_source_request(candidate,ROOT)
    q=np.asarray(candidate["opening_active_q_rad"],dtype=float)
    goal=np.asarray(request["target"]["active_q_rad"],dtype=float)
    screen.expand(q)
    if not np.array_equal(q[4:],goal[4:]):
        raise ValueError("non-grasp targets differ")
    _,config,files=old.load_candidate("tripod")
    config=deepcopy(config)
    config["hands"]["right"]["pose_wxyz"]=request["target"]["pose_wxyz"]
    config["hands"]["right"]["q_rad"]=q.astype(np.float32).astype(float).tolist()
    return config,(*files,candidate_path,source,Path(screen.__file__),Path(__file__))


def checked_reset_proof(proof):
    row=deepcopy(proof)
    if (row.get("initial_passed") is not True or set(row.get("initial",{}))!={"left","right"}
            or any(hand.get("passed") is not True or hand.get("shape_count")!=22
                or hand.get("frame_count")!=239 for hand in row["initial"].values())
            or row.get("native_admissible") is not False or row.get("teacher_admission") is not False
            or row.get("native_runs")!=0 or row.get("ppo_updates")!=0):
        raise ValueError("complete fresh initial geometry without native/Teacher admission required")
    row.pop("elapsed_seconds",None)
    return row


def run_screen(candidate,receipt):
    # Complete standalone USD use in a child before Kit imports its USD.
    with TemporaryDirectory(prefix="dh116-open-reset-") as folder:
        out=Path(folder)/"proof.json"
        child=subprocess.run([sys.executable,"-B","-m","scripts.screen_dh116_902_fixed_wrist_open",
            "--candidate",str(candidate),"--settling-receipt",str(receipt),"--output",str(out)],
            cwd=ROOT,timeout=210)
        # Exit 1 is expected for the separately reported uncontacted endpoint.
        if child.returncode not in (0,1) or not out.is_file():
            raise ValueError("fresh CPU screen did not produce its complete proof")
        return checked_reset_proof(json.loads(out.read_bytes()))


def check_frozen_proof(proof,frozen,receipt):
    expected={str(Path(receipt).resolve(strict=True)):proof["settling_outer_sha256"]}
    expected.update({str((ROOT/path).resolve(strict=True)):value
                     for path,value in proof["source_sha256"].items()})
    if any(frozen.get(path)!=value for path,value in expected.items()):
        raise ValueError("fresh CPU proof and frozen input hashes differ")


def completed_interval(output):
    output=Path(output)
    result=output/"tripod_shared_result.json"
    if (output/"tripod_failure.json").exists() or not result.is_file():
        return False
    row=json.loads(result.read_bytes())
    return bool(row.get("sdk_steps")==SDK_END and row.get("measured_exchanges")==SDK_END-2
        and row.get("solver_substeps")==SOLVER_STEPS and row.get("rods_per_env")==1
        and row.get("ppo_updates")==0 and row.get("teacher_admission") is False
        and row.get("pair_counts") and all(type(v) is int and v==0 for v in row["pair_counts"].values())
        and all((output/f"tripod_exchange_{sdk}.json").is_file() for sdk in range(3,SDK_END+1)))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--settling-receipt",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--plan",type=Path,required=True)
    for name in ("rays","left-bake","right-bake","proximal-capture","proximal-bake"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--approved-frozen-sha256")
    args=p.parse_args(argv)
    config,files=reset_config(args.candidate)
    proof=run_screen(args.candidate,args.settling_receipt)
    if (proof["right_wrist_pose_wxyz"]!=config["hands"]["right"]["pose_wxyz"]
            or proof["opening_active_q_rad"]!=config["hands"]["right"]["q_rad"]):
        raise ValueError("fresh proof and actual Reset initial targets differ")
    from scripts import verify_dh116_power_supported_task as supported
    from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary
    shared=supported.shared
    native,sources,reset=shared.native,shared.sources,shared.reset
    native.validate_execution_context(ROOT)
    plan=reset.freeze_plan(args.output)
    plan.update(version="dh116-902-fixed-wrist-open-reset-v1",screen=proof,
        reset_config=config,preserved_scope="independent Tripod open Reset and support settling only")
    plan["budget"].update(BUDGET)
    _,proximal=shared.proximal_registration_inputs(args.proximal_capture,args.proximal_bake)
    files=[*files,Path(supported.__file__),Path(shared.__file__),args.rays,
        args.settling_receipt,*proximal]
    for path,expected in proof["source_sha256"].items():
        source=ROOT/path
        if native.sha(source)!=expected:
            raise ValueError("CPU proof source changed before freezing: "+path)
        files.append(source)
    palms=sources.checked_json(args.rays,sources.RAYS_SHA)["palms"]
    for side,bake in (("left",args.left_bake),("right",args.right_bake)):
        _,_,record,_=sources.fixture(side,palms[side],bake)
        raw=args.rays.parent/(side+"_base_link.json")
        sources.checked_json(raw,record["gpu_row_sha256"])
        files.extend((raw,bake/"result.json",bake/(side+".base_link.npz")))
    plan["frozen_files"].update({str(path.resolve(strict=True)):native.sha(path) for path in files})
    check_frozen_proof(proof,plan["frozen_files"],args.settling_receipt)
    assert_clean_import_boundary()
    digest=native.digest(plan)
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan,plan)
        print(json.dumps(dict(frozen_sha256=digest,budget=plan["budget"])))
        return 0
    if (digest!=args.approved_frozen_sha256 or json.loads(args.plan.read_bytes())!=plan
            or args.output.exists()):
        raise ValueError("exact frozen one-shot recipe and fresh output required")
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
    args.verify_near_open_settling=True
    args.measure_contact_evidence=False
    acquire=partial(supported.acquire_supported_reset,config=config,sdk_end=SDK_END,task="tripod")
    reset.execute(plan,acquire_operation=partial(shared.run_shared,
        profile=BackendProfile.load(ROOT),args=args,supported_acquire=acquire,sdk_end=SDK_END,
        require_no_hand_contact=True,task="tripod",live_support_scene=config["support_scene"],
        action_after_settle=None))
    return int(not completed_interval(args.output))


if __name__=="__main__":
    raise SystemExit(main())
