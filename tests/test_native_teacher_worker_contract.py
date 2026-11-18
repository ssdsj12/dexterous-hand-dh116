"""Synthetic CPU contracts; no fixture is an execution authorization."""
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import run_dh116_work_packages as scheduler

MODULE = "go2_pvcnn.control.dh116.tabletop_native_teacher_worker_contract"

def contract_module():
    assert importlib.util.find_spec(MODULE) is not None, "C06 finite Teacher recipe contract is missing"
    return importlib.import_module(MODULE)


def recipe(tmp_path):
    config = tmp_path / "training.json"
    config.write_text("{}")
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    return dict(version=2, entrypoint="scripts/train_dh116_native_teacher_worker.py",
        argv=["--config",str(config),"--config-sha256",config_sha,"--additional-updates","1","--output","{output}"],
        source_files=sorted(contract_module().SOURCE_FILES),
        inputs={str(config):config_sha},
        budget=dict(gpu_index=3,timeout_seconds=240,min_gpu_free_mib=22000,
            min_disk_free_bytes=1000000000,physics_steps=258,environments=1,updates=1),
        teacher_contract=dict(schema="dh116-native-teacher-recipe-v2",profile="approach_wiring_smoke_v1",style="pinch",
            training_config=dict(path=str(config),sha256=config_sha),
            semantic_sha256="1"*64,source_sha256="2"*64,backend_sha256="3"*64,
            target_sha256="4"*64,reset_sha256="5"*64,reward_sha256="6"*64,
            journal_contract=dict(raw_file="training_events.jsonl",raw_preservation="all_native_journal_rows_original_sequence",
                derived_files={"physics.jsonl":"physical","transitions.jsonl":"transition_recorded",
                    "ownership.jsonl":"owner_and_lifecycle_events"},retain_raw_sequence_references=True),
            limits=dict(controls=32,sdk_steps_per_control=8,setup_sdk_steps=2,measured_sdk_steps=256,
                warmup_sdk_steps=0,other_sdk_steps=0,max_resets=1,max_episode_count=1,max_solver_calls=256,
                internal_substeps_per_solver_call=12,max_internal_solver_substeps=3072,max_processes=1,
                inner_wall_timeout_s=180,jsonl_limits={"physics.jsonl":256,"transitions.jsonl":32,
                    "ownership.jsonl":256,"training_events.jsonl":2048})),
        validator=dict(result_file="training_result.json",equals={"schema":"dh116-native-teacher-worker-result-v1"},
            artifacts=["training_result.json","physics.jsonl","transitions.jsonl","ownership.jsonl",
                "training_events.jsonl","shutdown_complete.json","teacher_clean.pt"],
            line_counts={"physics.jsonl":256,"transitions.jsonl":32}))



def pin_source_closure(root,spec,tests=()):
    names={"scripts/python.sh",spec["entrypoint"],*spec["source_files"],*tests}
    if (root/"scripts/run_dh116_work_packages.py").is_file():names.add("scripts/run_dh116_work_packages.py")
    inventory={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in sorted(names)}
    spec["teacher_contract"]["source_sha256"]=hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()
    return spec["teacher_contract"]["source_sha256"]

def test_exact_finite_v2_is_structural_only(tmp_path):
    spec=recipe(tmp_path)
    got=contract_module().validate_teacher_recipe_v2(spec)
    assert got == spec
    assert got is not spec
    assert "approved" not in got and "native_ready" not in got


@pytest.mark.parametrize("section,key,value",[
    ("budget","physics_steps",240),("budget","physics_steps",259),
    ("budget","environments",2),("budget","updates",2),("budget","updates",True),
    ("limits","controls",31),("limits","setup_sdk_steps",3),("limits","measured_sdk_steps",255),
    ("limits","max_resets",2),("limits","max_episode_count",2),("limits","max_solver_calls",257),
    ("limits","internal_substeps_per_solver_call",8),("limits","max_internal_solver_substeps",3073),
    ("limits","max_processes",2),("limits","inner_wall_timeout_s",float("inf")),
    ("teacher_contract","profile","flexible_train_r1_full"),("teacher_contract","style","tripod"),
    ("teacher_contract","semantic_sha256","bad"),
])
def test_v2_rejects_scope_expansion(tmp_path,section,key,value):
    spec=recipe(tmp_path)
    target=spec[section] if section != "limits" else spec["teacher_contract"]["limits"]
    target[key]=value
    with pytest.raises(ValueError):contract_module().validate_teacher_recipe_v2(spec)


@pytest.mark.parametrize("damage",["extra_budget","missing_limit","extra_limit","rows","outer_rows",
    "argv","config_sha","unbound_input","source_closure","entrypoint","version_bool","unknown_validator"])
def test_v2_rejects_unbound_contracts(tmp_path,damage):
    spec=recipe(tmp_path)
    if damage=="extra_budget":spec["budget"]["approved"]=True
    elif damage=="missing_limit":del spec["teacher_contract"]["limits"]["max_resets"]
    elif damage=="extra_limit":spec["teacher_contract"]["limits"]["unlimited"]=True
    elif damage=="rows":spec["validator"]["line_counts"]["physics.jsonl"]=258
    elif damage=="outer_rows":spec["teacher_contract"]["limits"]["jsonl_limits"]["ownership.jsonl"]=100000
    elif damage=="argv":spec["argv"] += ["--additional-updates=2"]
    elif damage=="config_sha":spec["argv"][3]="9"*64
    elif damage=="unbound_input":spec["inputs"]={}
    elif damage=="source_closure":spec["source_files"]=[]
    elif damage=="entrypoint":spec["entrypoint"]="scripts/arbitrary.py"
    elif damage=="version_bool":spec["version"]=True
    else:spec["validator"]["trust_me"]=True
    with pytest.raises(ValueError):contract_module().validate_teacher_recipe_v2(spec)


def test_scheduler_dispatch_shares_source_fingerprint_and_keeps_v1_240_cap(tmp_path):
    spec=recipe(tmp_path);root=tmp_path/"repo";root.mkdir()
    for name in [spec["entrypoint"],"scripts/python.sh","scripts/run_dh116_work_packages.py",*spec["source_files"]]:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("# CPU structural fixture only\n")
    config_name="configs/dh116_execution/teacher-v2.json"
    path=root/config_name;path.parent.mkdir(parents=True);pin_source_closure(root,spec);path.write_text(json.dumps(spec))
    repo=SimpleNamespace(root=root);package=dict(command_spec=config_name,dependencies=[],tests=[])
    out=scheduler.native_recipe(repo,package)
    assert out["spec"]["budget"]["physics_steps"]==258
    first=out["code_config_sha256"]
    (root/spec["source_files"][0]).write_text("# modified CPU fixture source\n")
    with pytest.raises(ValueError,match="source"):
        scheduler.native_recipe(repo,package)
    pin_source_closure(root,spec);path.write_text(json.dumps(spec))
    assert scheduler.native_recipe(repo,package)["code_config_sha256"] != first
    spec["version"]=1;spec.pop("teacher_contract");path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match="budget"):scheduler.native_recipe(repo,package)


def test_v1_fingerprint_formula_is_unchanged(tmp_path):
    root=tmp_path/"repo";root.mkdir()
    source_names=["scripts/python.sh","scripts/legacy.py","scripts/run_dh116_work_packages.py","tests/legacy.py"]
    identity={}
    for i,name in enumerate(source_names):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(f"# legacy {i}\n".encode())
        identity[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    spec=dict(version=1,entrypoint="scripts/legacy.py",argv=["--output","{output}"],source_files=[],
        budget=dict(gpu_index=3,timeout_seconds=240,min_gpu_free_mib=1,min_disk_free_bytes=1,
            physics_steps=240,environments=1,updates=0),
        validator=dict(result_file="result.json",equals={"ok":True},artifacts=["result.json","rows.jsonl"],
            line_counts={"rows.jsonl":240}))
    name="configs/dh116_execution/legacy.json";path=root/name;path.parent.mkdir(parents=True)
    raw=json.dumps(spec).encode();path.write_bytes(raw)
    expected_spec=hashlib.sha256(raw).hexdigest()
    expected=hashlib.sha256(json.dumps(dict(spec=expected_spec,source=identity),sort_keys=True).encode()).hexdigest()
    got=scheduler.native_recipe(SimpleNamespace(root=root),dict(command_spec=name,dependencies=[],tests=["tests/legacy.py"]))
    assert got["command_spec_sha256"]==expected_spec
    assert got["code_config_sha256"]==expected


@pytest.mark.parametrize('value',[[{}],[[]],[True]])
def test_v2_malformed_source_list_is_a_value_error(tmp_path,value):
    spec=recipe(tmp_path);spec['source_files']=value
    with pytest.raises(ValueError):contract_module().validate_teacher_recipe_v2(spec)


def test_caller_equal_source_sha_cannot_replace_actual_source_bytes(tmp_path):
    spec=recipe(tmp_path);root=tmp_path/"source-repo";root.mkdir()
    for name in [spec["entrypoint"],"scripts/python.sh","scripts/run_dh116_work_packages.py",*spec["source_files"]]:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("# synthetic source\n")
    name="configs/dh116_execution/forged-source.json";p=root/name;p.parent.mkdir(parents=True);p.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match="source"):
        scheduler.native_recipe(SimpleNamespace(root=root),dict(command_spec=name,dependencies=[],tests=[]))


def test_full_native_journal_bound_preserves_actual_c04_event_matrix(tmp_path):
    module=contract_module()
    # Actual C04 durable operations: four SDK rows and three control rows.
    # The original complete journal must survive; derived views add rows.
    sdk_ops={name:256 for name in ("delivered_exchange","physical","budget_consumed","policy_pinch_distance")}
    control_ops={name:32 for name in ("decision","settlement","transition_recorded")}
    required=sum(sdk_ops.values())+sum(control_ops.values())
    assert required==1120
    assert module.JSONL_LIMITS["training_events.jsonl"] >= required+8, "original full NativeJournal cannot fit"
    budget=module.teacher_journal_row_budget()
    assert budget==dict(per_sdk_operations=sdk_ops,per_control_operations=control_ops,
        required_measurement_rows=1120,setup_reset_origin_optimizer_cleanup_failure_allowance=928,
        max_raw_journal_rows=2048)
    spec=recipe(tmp_path)
    assert module.validate_teacher_recipe_v2(spec)["teacher_contract"]["journal_contract"]==dict(
        raw_file="training_events.jsonl",raw_preservation="all_native_journal_rows_original_sequence",
        derived_files={"physics.jsonl":"physical","transitions.jsonl":"transition_recorded",
            "ownership.jsonl":"owner_and_lifecycle_events"},retain_raw_sequence_references=True)


def test_v2_source_closure_must_pin_the_new_artifact_auditor(tmp_path):
    required="src/go2_pvcnn/control/dh116/tabletop_native_teacher_worker_artifacts.py"
    assert required in contract_module().SOURCE_FILES,"artifact audit implementation missing from required source closure"


def test_shared_source_inventory_and_training_contract_use_actual_files(tmp_path):
    assert callable(getattr(scheduler,"native_recipe_sources",None)),"shared actual source inventory API missing"
    spec=recipe(tmp_path);root=tmp_path/"actual-files";root.mkdir()
    names=[spec["entrypoint"],"scripts/python.sh","scripts/run_dh116_work_packages.py",*spec["source_files"],"tests/package.py"]
    for name in names:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text("# isolated source bytes: "+name)
    repo=SimpleNamespace(root=root);package=dict(tests=["tests/package.py"])
    expected={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sorted(names)}
    actual=scheduler.native_recipe_sources(repo,package,spec)
    assert actual==expected
    expected_sha=pin_source_closure(root,spec,tests=package["tests"])
    assert contract_module().worker_training_source_contract(repo,package,spec)=={"worker_source_closure":expected_sha}
    (root/"tests/package.py").write_text("# changed package test dependency")
    with pytest.raises(ValueError,match="source"):
        contract_module().worker_training_source_contract(repo,package,spec)
