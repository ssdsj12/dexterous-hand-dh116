"""Frozen Tripod launch consumes the actual schema/profile and current source closure."""
from pathlib import Path
import json

from scripts.run_dh116_work_packages import Repository
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile, digest
from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import load_reset_manifest
from go2_pvcnn.control.dh116.tabletop_native_teacher_worker_contract import worker_training_target_contract

ROOT = Path(__file__).resolve().parents[1]
NAME = "teacher.tripod-ppo-fresh-smoke8-v2"


def test_frozen_tripod_worker_retains_real_tripod_reset_and_finite_native_budget(tmp_path):
    repo = Repository(ROOT, evidence_dir=tmp_path / "private")
    package = repo.registry()["packages"][NAME]
    recipe = repo.native_recipe(package)
    launch = repo.actual_launch(package, recipe=recipe)
    pin = recipe["spec"]["teacher_contract"]
    config = json.loads(Path(pin["training_config"]["path"]).read_bytes())
    manifest = load_reset_manifest(config["preflight"], project=ROOT)
    profile = BackendProfile.load(ROOT)
    target = worker_training_target_contract(profile, manifest)
    assert pin["style"] == config["preflight"]["style"] == config["model_config"]["grasp"] == "tripod"
    assert config["model_config"] == json.loads((ROOT / "configs/dh116_training/privileged_deformable_teacher_tripod_v1.json").read_bytes())
    assert manifest.sha256 == pin["reset_sha256"]
    assert manifest.data()["style"] == "tripod"
    assert digest(target) == pin["target_sha256"]
    assert launch["role"] == "a_fresh" and launch["checkpoint_origin"] is None
    assert config["preflight"]["r2_enabled"] is config["preflight"]["force_proxy_enabled"] is False
    assert config["preflight"]["authorization"] is config["preflight"]["resource_lock"] is None
    assert config["rollout_steps"] == pin["limits"]["controls"] == 8
    assert pin["limits"]["max_solver_calls"] == 64
    assert pin["limits"]["max_internal_solver_substeps"] == 768
    assert recipe["spec"]["budget"]["physics_steps"] == 66
    assert recipe["spec"]["budget"]["updates"] == recipe["spec"]["budget"]["environments"] == 1
    assert recipe["spec"]["budget"]["timeout_seconds"] == 240
    assert pin["limits"]["inner_wall_timeout_s"] == 180
    # Structural parsing and current source hashes grant neither a native lease nor a grasp.
    assert config["model_config"]["native_ready"] is False
