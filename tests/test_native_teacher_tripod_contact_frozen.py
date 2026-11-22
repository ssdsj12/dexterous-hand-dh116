"""Audit the executable frozen Contact Tripod recipe, not a copied mock inventory."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_dh116_work_packages import Repository
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile, checked_file, digest
from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import load_reset_manifest
from go2_pvcnn.control.dh116.tabletop_native_teacher_worker_artifacts import _cpu_reference
from go2_pvcnn.control.dh116.tabletop_native_teacher_worker_contract import worker_training_source_contract, worker_training_target_contract

ROOT=Path(__file__).resolve().parents[1]
PACKAGE="teacher.tripod-contact-fresh8-v1"


def test_frozen_contact_recipe_binds_real_source_reset_model_and_separate_prime(tmp_path):
    repo=Repository(ROOT,evidence_dir=tmp_path/"private")
    package=repo.registry()["packages"][PACKAGE]
    collector = ROOT / "src/go2_pvcnn/control/dh116/tabletop_native_teacher_collector.py"
    historical_collector_sha = "9acc6939bc284365488d6a5fa511120c4f39946f79f335c1c7152fb1aae89707"
    if hashlib.sha256(collector.read_bytes()).hexdigest() != historical_collector_sha:
        # The completed old run stays bound to its original Collector bytes.
        assert package["native_runs"] == 1
        assert package["last_result"]["tested_head"] == "bbaacaad5cf960f0bee63c0b1d4993d6e435b653"
        with pytest.raises(ValueError, match="source inventory SHA differs"):
            repo.native_recipe(package)
        return
    recipe=repo.native_recipe(package)
    spec=recipe["spec"]
    launch=repo.actual_launch(package,recipe=recipe)
    pin=spec["teacher_contract"]
    training=pin["training_config"]
    cfg=json.loads(checked_file(training["path"],training["sha256"]))
    reset=load_reset_manifest(cfg["preflight"],project=ROOT)
    backend=BackendProfile.load(ROOT)
    assert reset.sha256==pin["reset_sha256"]
    assert reset.sha256=="c123ccbed1161471e2e34dc6941d7e06fcae93cb5ff87499bb8a5440eab0a6ad"
    assert pin["target_sha256"]==digest(worker_training_target_contract(backend,reset))
    assert pin["style"]==cfg["preflight"]["style"]==cfg["model_config"]["grasp"]=="tripod"
    assert pin["profile"]==cfg["preflight"]["profile"]=="approach_contact_prelift_v1"
    assert cfg["preflight"]["r2_enabled"] is cfg["preflight"]["force_proxy_enabled"] is False
    assert cfg["preflight"]["authorization"] is cfg["preflight"]["resource_lock"] is None
    assert cfg["rollout_steps"]==pin["limits"]["controls"]==8
    assert pin["limits"]["setup_sdk_steps"]==2 and pin["limits"]["warmup_sdk_steps"]==1
    assert pin["limits"]["measured_sdk_steps"]==64
    assert pin["limits"]["max_solver_calls"]==65 and pin["limits"]["max_internal_solver_substeps"]==780
    assert spec["budget"]["physics_steps"]==67
    assert spec["budget"]["updates"]==spec["budget"]["environments"]==1
    assert pin["limits"]["inner_wall_timeout_s"]==180 and spec["budget"]["timeout_seconds"]==240
    assert spec["budget"]["gpu_index"]==launch["device_plan"]["physical_gpu_index"]==3
    assert launch["role"]=="a_fresh" and launch["checkpoint_origin"] is None
    assert spec["environment"]=={"DH116_NUM_THREADS":"4"}
    context=dict(package=package,source_contract=worker_training_source_contract(repo,package,spec))
    with _cpu_reference(spec,repo,context) as (trainer,contract):
        identity=trainer.model.model_identity
        assert identity["checkpoint_namespace"]=="native_contact_teacher_tripod_cpu_v1"
        assert identity["context_mode"]==pin["profile"]
        assert trainer.counters=={"updates":0,"environment_steps":0}
    execution=repo.native_execution_identity(package,recipe=recipe)
    assert execution["actual_identity"]["core_recipe_identity"]==execution["core_identity"]
    assert package["executor"]=="native_bounded"
    assert package["retry_policy"]["max_attempts_per_code_config"]==1


def test_consumed_old_smoke_recipe_is_not_rebound_to_contact_sources(tmp_path):
    repo=Repository(ROOT,evidence_dir=tmp_path/"private")
    package=repo.registry()["packages"]["teacher.tripod-ppo-fresh-smoke8-v11"]
    assert package["native_runs"]==1
    with pytest.raises(ValueError,match="source inventory SHA differs"):
        repo.native_recipe(package)
