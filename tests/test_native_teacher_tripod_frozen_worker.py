"""Spent Tripod recipe retains its native contract but cannot launch changed code."""
from pathlib import Path
import json
import os
import subprocess
import sys
import pytest

from scripts.run_dh116_work_packages import Repository
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile, checked_file, digest
from go2_pvcnn.control.dh116.tabletop_native_teacher_preflight import load_reset_manifest
from go2_pvcnn.control.dh116.tabletop_native_teacher_worker_contract import worker_training_target_contract, validate_teacher_recipe_v2

ROOT = Path(__file__).resolve().parents[1]
NAME = "teacher.tripod-ppo-fresh-smoke8-v10"


def test_spent_tripod_worker_retains_budget_and_rejects_changed_source(tmp_path):
    repo = Repository(ROOT, evidence_dir=tmp_path / "private")
    package = repo.registry()["packages"][NAME]
    # This attempt is spent. Preserve its frozen bytes, not refresh its SHA
    # whenever training code changes. The scheduler must reject that mismatch.
    companion=package["actual_launch"]
    launch=json.loads(checked_file(ROOT/companion["path"],companion["sha256"]))
    core_sha=launch["core_recipe_identity"]["command_spec_sha256"]
    spec=json.loads(checked_file(ROOT/package["command_spec"],core_sha))
    validate_teacher_recipe_v2(spec)
    with pytest.raises(ValueError,match="source inventory SHA differs"):
        repo.native_recipe(package)
    pin = spec["teacher_contract"]
    config = json.loads(checked_file(pin["training_config"]["path"],pin["training_config"]["sha256"]))
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
    assert spec["budget"]["physics_steps"] == 66
    assert spec["budget"]["updates"] == spec["budget"]["environments"] == 1
    assert spec["budget"]["timeout_seconds"] == 240
    assert pin["limits"]["inner_wall_timeout_s"] == 180
    # Historical structure and verified Reset data grant neither a current native lease nor a grasp.
    assert config["model_config"]["native_ready"] is False
    assert spec["environment"] == {"DH116_NUM_THREADS": "1"}
    # Exercise the real existing launcher with deliberately conflicting inherited
    # numeric settings. CUDA is masked; this imports no Isaac app or solver.
    probe = "import os,torch,json;print(json.dumps(dict(torch_threads=torch.get_num_threads()," \
        "omp=os.environ.get('OMP_NUM_THREADS'),openblas=os.environ.get('OPENBLAS_NUM_THREADS')," \
        "mkl=os.environ.get('MKL_NUM_THREADS'),cuda=os.environ.get('CUDA_VISIBLE_DEVICES'))))"
    result = subprocess.run(["bash", str(ROOT/"scripts/python.sh"), "-c", probe],
        cwd=ROOT,env={**os.environ, **spec["environment"],
            "DH116_PYTHON":sys.executable,"CUDA_VISIBLE_DEVICES":"",
            "OMP_NUM_THREADS":"17","OPENBLAS_NUM_THREADS":"17","MKL_NUM_THREADS":"17"},
        capture_output=True,text=True,check=True,timeout=60)
    assert json.loads(result.stdout.splitlines()[-1]) == dict(
        torch_threads=1,omp="1",openblas="1",mkl="1",cuda="")
