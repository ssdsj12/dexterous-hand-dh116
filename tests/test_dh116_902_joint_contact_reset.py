import copy,json
from pathlib import Path
import pytest
from scripts import verify_dh116_902_fixed_wrist_open_reset as m
from scripts import certify_dh116_902_joint_contact_path as path

OPEN=m.ROOT/"configs/dh116_tasks/flexible_902_tripod_fixed_wrist_open_cpu_v1.json"
JOINT=m.ROOT/"configs/dh116_tasks/flexible_902_tripod_joint_contact_candidate_v1.json"

def proof(config):
    joint=json.loads(JOINT.read_bytes())
    return dict(version="dh116-902-joint-contact-path-cpu-v1",initial_passed=True,
      initial={s:dict(passed=True,shape_count=22,frame_count=239) for s in ("left","right")},
      closure_geometry_passed=True,endpoint_contact_candidate_passed=True,
      physical_substeps=512,control_windows=64,pose_wxyz=joint["pose_wxyz"],
      opening_active_q_rad=list(config["hands"]["right"]["q_rad"]),teacher_admission=False,native_runs=0,
      receipt_sha256="a"*64,source_sha256={},native_contact_verified=False,force_qualified=False,grasp_qualified=False)

def test_new_reset_changes_only_right_wrist_and_preserves_opening():
    config,_=m.reset_config(OPEN);original=copy.deepcopy(config)
    joint=json.loads(JOINT.read_bytes());p=proof(config)
    result,adapted=m.adapt_joint_reset(config,joint,p)
    expected=copy.deepcopy(original);expected["hands"]["right"]["pose_wxyz"]=joint["pose_wxyz"]
    assert result==expected and config==original
    assert adapted["right_wrist_pose_wxyz"]==joint["pose_wxyz"]
    assert adapted["settling_outer_sha256"]=="a"*64
    assert adapted["teacher_admission"] is False

def test_failed_closure_or_mismatched_opening_cannot_start_new_reset():
    config,_=m.reset_config(OPEN);joint=json.loads(JOINT.read_bytes())
    p=proof(config);p["closure_geometry_passed"]=False
    with pytest.raises(ValueError):m.adapt_joint_reset(config,joint,p)
    p=proof(config);p["opening_active_q_rad"][0]+=.01
    with pytest.raises(ValueError):m.adapt_joint_reset(config,joint,p)

def test_intermediate_node_identity_must_remain_on_right():
    row=dict(shape_count=22,noncontact_rod_clearance_m=.001,support_clearance_m=.002,
        table_clearance_m=.01,contact_rod_clearance_m=-.0001,all_envelopes_certified=True,
        pad_gaps_m=[.01,.01,.01],node_ids=[3,2,4])
    assert path.path_row_passed(row)
    row["node_ids"][1]=30
    assert not path.path_row_passed(row)


def test_new_recipe_audit_requires_candidate_bytes_and_actual_reset_config():
    from scripts import audit_dh116_902_live_clearance as audit
    from hashlib import sha256
    joint=json.loads(JOINT.read_bytes());digest=sha256(JOINT.read_bytes()).hexdigest()
    config,_=m.reset_config(OPEN)
    config["hands"]["right"]["pose_wxyz"]=joint["pose_wxyz"]
    frozen=dict(version="dh116-902-joint-contact-open-reset-v1",reset_config=config,
        joint_candidate=dict(path=str(JOINT),sha256=digest),
        frozen_files={str(JOINT.resolve()):digest},
        joint_search=dict(path=str(JOINT),sha256=digest),joint_path_proof=dict(path=str(JOINT),sha256=digest),
        screen=dict(version="dh116-902-joint-contact-path-cpu-v1",candidate_sha256=digest,
            search_sha256=digest,supplied_path_proof_sha256=digest,
            opening_active_q_rad=list(config["hands"]["right"]["q_rad"])))
    audit.validate_recipe(frozen)
    bad=copy.deepcopy(frozen);bad["joint_candidate"]["sha256"]="0"*64
    with pytest.raises(ValueError):audit.validate_recipe(bad)
    bad=copy.deepcopy(frozen);bad["reset_config"]["hands"]["right"]["pose_wxyz"][0]+=.01
    with pytest.raises(ValueError):audit.validate_recipe(bad)


def test_frozen_joint_artifact_hashes_cannot_change_between_screen_and_launch(tmp_path):
    files=[tmp_path/name for name in ("candidate","search","proof")]
    for p in files:p.write_text(p.name)
    frozen={str(p.resolve()):"a"*64 for p in files}
    evidence=dict(candidate_sha256="a"*64,search_sha256="b"*64,supplied_path_proof_sha256="a"*64)
    with pytest.raises(ValueError):m.check_joint_frozen_proof(evidence,frozen,*files)


def test_partial_opt_in_arguments_are_rejected_before_any_screen():
    argv=["--candidate",str(OPEN),"--settling-receipt","unused","--output","unused",
          "--plan","unused","--joint-candidate",str(JOINT)]
    for flag in ["--rays","--left-bake","--right-bake","--proximal-capture","--proximal-bake"]:
        argv += [flag,"unused"]
    with pytest.raises(SystemExit) as caught:m.main(argv)
    assert caught.value.code==2

def test_cpu_contact_proof_cannot_claim_force_qualification():
    config,_=m.reset_config(OPEN);joint=json.loads(JOINT.read_bytes());p=proof(config)
    p["force_qualified"]=True
    with pytest.raises(ValueError):m.adapt_joint_reset(config,joint,p)
