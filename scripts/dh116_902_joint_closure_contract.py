"""Strict SDK240 supported-to-contact handoff; old contracts remain separate."""
import hashlib,json
from pathlib import Path
import numpy as np
from scripts.search_dh116_902_joint_contact import candidate_passed,ROOT
from scripts.certify_dh116_902_joint_contact_path import path_row_passed,select_target
from scripts.diagnose_dh116_teacher_initialization import digest
VERSION="dh116-902-joint-contact-transfer-v1"
SDK_END=880
SOLVER_STEPS=10536
BUDGET=dict(mode="902_tripod_fixed_wrist_closing_feedback",sdk_steps=880,sdk_setup_steps=2,
 control_steps=80,solver_steps=10536,teacher_queries=0,ppo_updates=0,model_loads=0,
 soft_wall_seconds=480,auto_retries=0,auto_kill=False,new_power_mesh_registrations=6,
 short_feedback_hold_seconds=16/60)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bounded_schedule(actions):
    a=np.asarray(actions,dtype=float)
    held=[*range(6),7,10,11,*range(12,24)]
    if (a.shape!=(80,24) or not np.isfinite(a).all() or np.max(abs(a))>1
            or np.count_nonzero(a[-16:]) or np.count_nonzero(a[:,held])):
        raise ValueError("exact 64-window closing and 16-window zero-action feedback schedule required")
    return a


def validate_contact(task,contact):
    from scripts.screen_dh116_902_command_contact import validate_locked_geometry,PADS
    from scripts.audit_dh116_certified_segment_distance import ENVELOPE_REFINEMENT_VERSION
    proof=contact.get("command_proof",{})
    if (task!="tripod" or contact.get("grasp")!=task or contact.get("contact_side")!="right"
            or contact.get("entry_contract")!=VERSION or contact.get("boundary_sdk")!=240
            or contact.get("archive") is not None or contact.get("bilateral") is not None
            or contact.get("seating") is not None or contact.get("max_pad_penetration_m")!=.00015
            or contact.get("envelope_refinement")!=ENVELOPE_REFINEMENT_VERSION
            or proof.get("version")!="dh116-902-joint-contact-path-cpu-v2"
            or proof.get("control_windows")!=80 or proof.get("physical_substeps")!=640
            or proof.get("closing_windows")!=64 or proof.get("hold_windows")!=16
            or proof.get("initial_passed") is not True
            or proof.get("closure_geometry_passed") is not True
            or proof.get("endpoint_contact_candidate_passed") is not True
            or any(proof.get(key) is not False for key in ("native_contact_verified","force_qualified",
                "grasp_qualified","teacher_admission"))
            or proof.get("search_receipt_sha256")==proof.get("execution_receipt_sha256")):
        raise ValueError("explicit source-separated joint-contact transfer proof required")
    actions=bounded_schedule(proof["actions"])
    if hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()!=proof["action_sha256"]:
        raise ValueError("submitted actions differ from the full CPU path")
    rows=proof.get("samples",[])
    if len(rows)!=640 or any(row.get("ordinal")!=i or not path_row_passed(row) for i,row in enumerate(rows,1)) or not candidate_passed(rows[-1]):
        raise ValueError("every compiled substep and final three-pad candidate must pass")
    expected={"right."+pad:sorted({row["node_ids"][i] for row in rows}) for i,pad in enumerate(PADS)}
    if contact.get("stations")!=expected or proof.get("stations")!=expected:
        raise ValueError("only CPU-proved original-side material stations are admitted")
    raw=contact.get("settled_archive",{})
    rod=raw.get("rod",{})
    if (digest(raw)!=contact.get("settled_archive_digest") or rod.get("physics_step")!=240
            or rod.get("rod_tick")!=2856 or rod.get("clock_valid") is not True
            or contact.get("execution_boundary_sha256")!=proof.get("execution_boundary_sha256")):
        raise ValueError("sealed actual SDK240 settling boundary required")
    validate_locked_geometry(proof,contact.get("continuous_cooked_hulls"))
    if contact.get("continuous_cooked_hulls")!=proof.get("continuous_cooked_hulls"):
        raise ValueError("probe must use the exact source-locked cooked hulls")
    return True


def load_stage(candidate,search,source_receipt,execution_receipt,proof_path):
    from scripts.audit_dh116_902_live_clearance import validate_recipe
    from scripts.screen_dh116_902_command_contact import locked_geometry
    from scripts.screen_dh116_902_fixed_wrist_open import compile_closure
    from go2_pvcnn.control.dh116.tabletop_supported_rod_reference import load_parked_reference,_read_artifact
    candidate,search,source_receipt,execution_receipt,proof_path=map(Path,
        (candidate,search,source_receipt,execution_receipt,proof_path))
    request=json.loads(candidate.read_bytes());select_target(request,search,sha(source_receipt))
    proof=json.loads(proof_path.read_bytes());receipt=execution_receipt.resolve(strict=True)
    if (proof["candidate_sha256"]!=sha(candidate) or proof["search_sha256"]!=sha(search)
            or proof["search_receipt_sha256"]!=sha(source_receipt)
            or proof["execution_receipt_sha256"]!=sha(receipt) or proof["receipt_sha256"]!=sha(receipt)):
        raise ValueError("immutable search source and actual execution source must both match")
    reference=load_parked_reference(receipt,sha(receipt),240,task="tripod")
    outer=json.loads(receipt.read_bytes());output=(receipt.parent/"native").resolve(strict=True)
    frozen,_=_read_artifact(output,outer["artifact_sha256"],"frozen_plan.json");validate_recipe(frozen)
    if frozen["version"]!="dh116-902-joint-contact-open-reset-v1" or frozen["joint_candidate"]["sha256"]!=sha(candidate):
        raise ValueError("new-wrist measured Reset is required before closing")
    archive,archive_sha=_read_artifact(output,outer["artifact_sha256"],"tripod_exchange_240.json")
    if archive_sha!=proof["execution_boundary_sha256"]:
        raise ValueError("execution boundary differs from CPU proof")
    audit_path=receipt.parent/"live-clearance.json"
    audit=json.loads(audit_path.read_bytes())
    if (sha(audit_path)!=proof["execution_clearance_sha256"] or not audit["geometry_passed"]
            or audit["receipt_sha256"]!=sha(receipt)):
        raise ValueError("same-receipt actual clearance audit required")
    config=frozen["reset_config"]
    base,_=compile_closure(config["hands"]["right"]["q_rad"],request["active_q_rad"],windows=64)
    actions=bounded_schedule(np.concatenate((base,np.zeros((16,24)))))
    if not np.array_equal(actions,np.asarray(proof["actions"])):
        raise ValueError("recompiled float32 actions differ from proof")
    contact=dict(entry_contract=VERSION,grasp="tripod",contact_side="right",boundary_sdk=240,
        archive=None,settled_archive=archive,settled_archive_digest=digest(archive),
        execution_boundary_sha256=archive_sha,stations=proof["stations"],
        max_pad_penetration_m=.00015,envelope_refinement="dh116-continuous-envelope-support-v1",
        continuous_cooked_hulls=proof["continuous_cooked_hulls"],command_proof=proof)
    validate_contact("tripod",contact)
    _,_,geometry_files=locked_geometry()
    files=[Path(__file__),candidate,search,source_receipt,receipt,proof_path,audit_path,*geometry_files,
        output/"frozen_plan.json",output/"tripod_support_scene.json",output/"tripod_shared_result.json",
        output/"result.json",*(output/f"tripod_exchange_{sdk}.json" for sdk in range(3,241))]
    files += [ROOT/name for name in proof["source_sha256"]]
    files += [Path(name) for name in proof.get("external_source_sha256",{})]
    return config,actions,contact,archive,tuple(files)
