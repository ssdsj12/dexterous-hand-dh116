"""Actual Tripod B recipe consumes the published clean A diagnostic read-only."""
from pathlib import Path
from scripts.run_dh116_work_packages import Repository

ROOT=Path(__file__).resolve().parents[1]
A="teacher.tripod-ppo-fresh-smoke8-v11"
B="teacher.tripod-ppo-resume-smoke8-v11"


def test_tripod_resume_recipe_binds_real_A_without_source_or_budget_drift():
    repo=Repository(ROOT,evidence_dir=Path.home()/".dh116-native-control-11-17")
    registry=repo.registry()["packages"]
    producer,resume=registry[A],registry[B]
    assert resume["command_spec"]==producer["command_spec"]
    a=repo.native_recipe(producer)
    b=repo.native_recipe(resume)
    assert b["spec"]==a["spec"]
    launch=repo.actual_launch(resume,recipe=b)
    assert launch["role"]=="b_resume" and launch["provider"]["entrypoint"]=="run_b"
    assert launch["checkpoint_origin"]=={
        "producer_package":A,
        "receipt_sha256":"b49afa36539549d35d731315f6821339d1dc5cadbc70c2141e3e0442d6d53eb9"}
    origin=repo.resolve_actual_checkpoint_origin(launch,resume_spec=b["spec"],package_name=B)
    assert origin["producer_package"]==A
    assert origin["checkpoint_sha256"]=="35231d09bb947775301743d1c715889ff076eca1c7dedacf93149097c7167cec"
    assert origin["loaded_updates"]==origin["loaded_policy_version"]==1
    assert origin["loaded_environment_steps"]==8
    assert origin["loaded_accepted_minibatch_updates"]>=1
    assert origin["diagnostic_verified"] is True
    assert origin["native_qualification"] is origin["teacher_admission"] is origin["grasp_qualified"] is False
    assert b["spec"]["teacher_contract"]["style"]=="tripod"
    assert b["spec"]["teacher_contract"]["limits"]["controls"]==8
    assert b["spec"]["teacher_contract"]["limits"]["max_internal_solver_substeps"]==768
    assert b["spec"]["teacher_contract"]["limits"]["inner_wall_timeout_s"]==180
    assert b["spec"]["budget"]["physics_steps"]==66
    assert b["spec"]["budget"]["timeout_seconds"]==240
    assert b["spec"]["budget"]["updates"]==b["spec"]["budget"]["environments"]==1
    assert b["spec"]["environment"]=={"DH116_NUM_THREADS":"4"}
    identity=repo.native_execution_identity(resume,recipe=b)
    assert identity["core_identity"]==repo.native_execution_identity(producer,recipe=a)["core_identity"]
    assert identity["attempt_fingerprint"]!=repo.native_execution_identity(producer,recipe=a)["attempt_fingerprint"]
