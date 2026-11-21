"""One bounded source-separated supported Tripod closing diagnostic; zero PPO."""
import argparse,json,subprocess,sys
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from scripts import verify_dh116_power_supported_task as supported
from scripts import dh116_902_joint_closure_contract as contract
ROOT=contract.ROOT
shared=supported.shared
native,sources,reset=shared.native,shared.sources,shared.reset


def diagnostic_complete(output):
    output=Path(output)
    if (output/"tripod_failure.json").exists():return False
    names=("tripod_shared_result.json","tripod_control_result.json","tripod_distal_probe_summary.json")
    if any(not (output/name).is_file() for name in names):return False
    result=json.loads((output/names[0]).read_bytes())
    control=json.loads((output/names[1]).read_bytes())["result"]
    return bool(result.get("sdk_steps")==880 and result.get("solver_substeps")==10536
        and result.get("controlled_sdk_steps")==640 and result.get("ppo_updates")==0
        and result.get("teacher_admission") is False
        and control.get("physical_samples")==640 and control.get("termination_reason") is None
        and all((output/f"tripod_control_{sdk}.json").is_file()
            and (output/f"tripod_distal_probe_{sdk}.json").is_file() for sdk in range(241,881)))


def fresh_proof(args):
    with TemporaryDirectory(prefix="dh116-joint-execution-") as folder:
        out=Path(folder)/"proof.json"
        subprocess.run([sys.executable,"-B","-m","scripts.certify_dh116_902_joint_contact_path",
            "--candidate",str(args.candidate),"--search",str(args.search),
            "--receipt",str(args.execution_receipt),"--receipt-sha256",native.sha(args.execution_receipt),
            "--output",str(out)],cwd=ROOT,timeout=240,check=True)
        raw=args.execution_proof.read_bytes()
        proof=json.loads(out.read_bytes())
        if proof!=json.loads(raw):raise ValueError("fresh exact execution-source CPU proof required")
        return proof,contract.hashlib.sha256(raw).hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("candidate","search","search-receipt","execution-receipt","execution-proof",
                 "rays","left-bake","right-bake","proximal-capture","proximal-bake","output","plan"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--approved-frozen-sha256")
    args=p.parse_args(argv)
    native.validate_execution_context(ROOT)
    proof,proof_hash=fresh_proof(args)
    config,actions,contact,archive,files=contract.load_stage(args.candidate,args.search,args.search_receipt,
        args.execution_receipt,args.execution_proof)
    from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary
    plan=reset.freeze_plan(args.output)
    plan.update(version="dh116-902-joint-closure-native-v1",reset_config=config,
        search_receipt_sha256=proof["search_receipt_sha256"],
        execution_receipt_sha256=proof["execution_receipt_sha256"],
        execution_clearance_sha256=proof["execution_clearance_sha256"],
        action_schedule=actions.tolist(),closing_stage=contact,
        preserved_scope="original assets/physics; 64 closing + 16 diagnostic feedback windows; no Lift/PPO")
    plan["budget"].update(contract.BUDGET)
    _,proximal=shared.proximal_registration_inputs(args.proximal_capture,args.proximal_bake)
    files=[*files,*proximal,args.rays,Path(__file__),Path(shared.__file__),Path(supported.__file__),
        ROOT/"scripts/verify_dh116_supported_distal_transfer.py",
        ROOT/"scripts/screen_dh116_902_flexible_contact.py",
        ROOT/"scripts/certify_dh116_902_joint_contact_path.py"]
    palms=sources.checked_json(args.rays,sources.RAYS_SHA)["palms"]
    for side,bake in (("left",args.left_bake),("right",args.right_bake)):
        _,_,record,_=sources.fixture(side,palms[side],bake)
        raw=args.rays.parent/(side+"_base_link.json");sources.checked_json(raw,record["gpu_row_sha256"])
        files.extend((raw,bake/"result.json",bake/(side+".base_link.npz")))
    plan["frozen_files"].update({str(path.resolve(strict=True)):native.sha(path) for path in files})
    expected={str((ROOT/path).resolve(strict=True)):value for path,value in proof["source_sha256"].items()}
    expected.update(proof["external_source_sha256"])
    expected.update({str(args.candidate.resolve()):proof["candidate_sha256"],
        str(args.search.resolve()):proof["search_sha256"],
        str(args.search_receipt.resolve()):proof["search_receipt_sha256"],
        str(args.execution_receipt.resolve()):proof["execution_receipt_sha256"],
        str(args.execution_proof.resolve()):proof_hash})
    execution_folder=args.execution_receipt.resolve(strict=True).parent
    expected.update({str(execution_folder/"live-clearance.json"):proof["execution_clearance_sha256"],
        str(execution_folder/"native/tripod_exchange_240.json"):proof["execution_boundary_sha256"]})
    if any(plan["frozen_files"].get(path)!=value for path,value in expected.items()):
        raise ValueError("source changed after execution proof")
    assert_clean_import_boundary()
    digest=native.digest(plan)
    if args.approved_frozen_sha256 is None:
        native.write_new(args.plan,plan)
        print(json.dumps(dict(frozen_sha256=digest,budget=plan["budget"])))
        return 0
    if args.approved_frozen_sha256!=digest or json.loads(args.plan.read_bytes())!=plan or args.output.exists():
        raise ValueError("fresh exact one-shot closing recipe required")
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
    args.verify_near_open_settling=True
    args.measure_contact_evidence=False
    acquire=partial(supported.acquire_supported_reset,config=config,sdk_end=contract.SDK_END,task="tripod")
    reset.execute(plan,acquire_operation=partial(shared.run_shared,
        profile=BackendProfile.load(ROOT),args=args,supported_acquire=acquire,sdk_end=contract.SDK_END,
        require_no_hand_contact=True,task="tripod",live_support_scene=config["support_scene"],
        action_after_settle=actions,tripod_archive=archive,distal_contact=contact))
    return int(not diagnostic_complete(args.output))


if __name__=="__main__":raise SystemExit(main())
