"""Independent source-derived manifest for the real supported17 Tripod Reset.

Historical receipt integrity and current physical compatibility are separate:
consumer/training code may evolve, but the Reset implementation, backend, assets,
SDFs, numeric state and side identities must still match their measured sources.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .tabletop_native_teacher_profile import (
    BackendProfile, ROOT, TARGET_SCHEMA_VERSION, checked_file, digest, profile_reader_scope,
)

SCHEMA = "dh116-native-teacher-resolved-tripod17-reset-v1"
SCOPE = "native_reset_only_no_contact_or_grasp_admission"
FILES = {
    "reset_readback": "mesh_reset_state.json",
    "numeric_reset": "mesh_reset_numeric_state.json",
    "acquisition": "reset_capture.json",
    "common": "pre_action_common.json",
    "geometry": "reset_geometry.json",
}
# Source consumers/optimizers are not part of this acquisition implementation.
# The plan and backend profile independently add every original grid/material/
# parameter source. No physical source is accepted from a caller exemption list.
RESET_IMPLEMENTATIONS = (
    "scripts/record_dh116_tripod_supported_reset.py",
    "scripts/record_dh116_teacher_numeric_reset.py",
    "scripts/diagnose_dh116_teacher_stat_cache_release.py",
    "scripts/diagnose_dh116_teacher_initialization.py",
    "scripts/dh116_fault_safe_lifecycle_v3.py",
    "scripts/dh116_pinch_acquisition_v2.py",
    "src/go2_pvcnn/control/dh116/tabletop_fresh_supported_builder.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_dual_hand_episode.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_physical_exchange.py",
    "src/go2_pvcnn/control/dh116/tabletop_fresh_supported_exchange.py",
    "src/go2_pvcnn/control/dh116/tabletop_dual_hand_scene_preparation.py",
    "src/go2_pvcnn/control/dh116/tabletop_dual_pinch_builder.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_rod_boundary.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_live_geometry.py",
    "src/go2_pvcnn/control/dh116/tabletop_teacher_mounted_wrist_source.py",
    "src/go2_pvcnn/control/dh116/tabletop_simulation_owner_cleanup.py",
)


def _require(value, message):
    if not value:
        raise ValueError("Tripod Reset manifest: "+message)


def _binding(value):
    _require(isinstance(value,dict) and set(value)=={"path","sha256"}, "explicit pinned source required")
    path = Path(value["path"])
    _require(path.is_absolute() and path.resolve(strict=True)==path, "canonical absolute source path required")
    return path, checked_file(path,value["sha256"])


def _derive(run_dir, outer_binding, project):
    """Read raw source evidence; never accept caller-provided state/qualification."""
    from scripts.run_dh116_work_packages import debugger_status
    from .tabletop_deformable_r0_replay import encode_owned
    from .tabletop_fast_owned_codec import decode_owned_json
    from .tabletop_supported_mesh_reset import validate_supported_readback
    from .tabletop_mesh_reset_numeric_state import validate_numeric_reset_audit
    from .tabletop_teacher_mounted_wrist_source import validate_mounted_observation
    from .tabletop_power_supported_task import guard_hand_supports, prepare_initial_state, TRIPOD_SCOPE

    project = Path(project).resolve(strict=True)
    _require(project==ROOT, "current source-controlled project required")
    out = Path(run_dir).resolve(strict=True)
    outer_path, raw = _binding(outer_binding)
    _require(outer_path==out/"outer.json", "same run outer receipt required")
    outer = json.loads(raw)
    _require(all(type(outer.get(k)) is int and outer[k]==0 for k in
                 ("exit_code","debugger_inferior_exit_code")) and outer.get("forced_exit") is False
             and outer.get("debugger_inferior_signal") is None
             and outer.get("debugger_status_verified") is True, "natural debugger/outer exit0 required")
    run = out/"native"
    inventory = outer["artifact_sha256"]
    required = {*FILES.values(),"frozen_plan.json","result.json","episode_reset.json",
                "cleanup_stages.json","app_close_returned.json","stop_commit.json"}
    _require(required<=set(inventory), "native inventory incomplete")
    _require(all(isinstance(k,str) and Path(k).name==k for k in inventory), "native artifact path escapes run")
    # Parse the checksum-verified bytes from this call, never a later pathname
    # reread. Each independent manifest load still verifies the full inventory.
    documents = {}
    for name, sha in inventory.items():
        artifact_raw = checked_file(run/name,sha)
        if name in required and name!="stop_commit.json":
            documents[name] = artifact_raw
    del artifact_raw
    _require(hashlib.sha256((out/"native.log").read_bytes()).hexdigest()==outer["native_log_sha256"],
             "native log SHA differs")
    _require(all(outer[k]==v for k,v in debugger_status(out/"native.log").items()), "debugger log status differs")
    plan = json.loads(documents.pop("frozen_plan.json"))
    _require(plan==json.loads((out/"plan.json").read_bytes()) and digest(plan)==outer["frozen_sha256"]
             and plan["output"]==str(run) and plan["version"]=="dh116-fresh-supported-tripod-reset-capture-v1"
             and plan["style"]=="tripod", "native recipe identity differs")
    budget = plan["budget"]
    _require(budget["sdk_steps"]==budget["sdk_setup_steps"]==2 and budget["episode_resets"]==1
             and budget["num_envs"]==1
             and all(type(budget[k]) is int and budget[k]==0 for k in
                     ("control_steps","solver_steps","teacher_queries","ppo_updates","auto_retries")),
             "two-step Reset-only native budget required")
    _require(0<outer["elapsed_s"]<=budget["soft_wall_seconds"]==180
             and outer["exceeded_budget"] is False, "native wall budget exceeded")
    profile = BackendProfile.load(project)
    _require(profile.digest==plan["backend_profile_sha256"], "current physical backend profile differs")
    current_pins = {str(Path(item.path).resolve(strict=True)):item.sha256 for item in profile.inventory}
    for rel in RESET_IMPLEMENTATIONS:
        path = project/rel
        current_pins[str(path)] = plan["frozen_files"][str(path)]
    for path,sha in current_pins.items():
        _require(plan["frozen_files"].get(path)==sha, "physical source absent from frozen recipe: "+path)
        checked_file(path,sha)
    lifecycle = json.loads(documents.pop("result.json"))
    _require(lifecycle["failure"] is None and lifecycle["close_mode"]=="normal"
             and lifecycle["lifetime_released_before_close"] is True
             and lifecycle["physics_steps"]==2
             and all(lifecycle[k]==0 for k in ("measurement_steps","rod_steps","ppo_updates")),
             "original Reset lifecycle failed or advanced")
    stages = json.loads(documents.pop("cleanup_stages.json"))
    _require(stages["failure"] is None and all(not row["attempted"] or
        (row["returned"] is True and row["error"] is None) for row in stages["stages"].values()),
        "native cleanup stages incomplete")
    _require(json.loads(documents.pop("app_close_returned.json"))["app_close_returned"] is True,
             "App.close did not return normally")
    saved = decode_owned_json(documents.pop(FILES["reset_readback"]))
    readback = saved["readback"]
    receipt = dict(readback.pop("completed_reset_receipt"),mesh_state_readback=readback)
    event = decode_owned_json(documents.pop("episode_reset.json"))
    from .tabletop_mesh_reset_order import _digest
    _require(_digest(event["reset"])==_digest(receipt), "durable Reset and readback receipts differ")
    with profile_reader_scope(profile):
        validate_supported_readback(readback,reset_receipt=receipt,mesh_plan=saved["mesh_plan"])
        audit = decode_owned_json(documents.pop(FILES["numeric_reset"]))
        validate_numeric_reset_audit(audit,reset_receipt=receipt)
    common = decode_owned_json(documents.pop(FILES["common"]))
    status = validate_mounted_observation(common)
    _require(status["valid_channels"]==66 and status["unknown_channels"]==305,
             "actual common66/unknown305 contract differs")
    _require(all(common["identity"][k]==v for k,v in receipt["after_clock"].items()),
             "same-clock native common state required")
    _require(event["clearance_floor_m"]==.0021 and event["clearance_passed"] is True
             and (np.asarray(event["clearance_m"])>=.0021).all(), "original table protection failed")
    task_path,task_raw = _binding(plan["task_config"])
    task = json.loads(task_raw)
    seed,_,_,_ = prepare_initial_state(task,saved["mesh_plan"]["original_plan"]["prepared_scene"]["table"],
                                      expected_scope=TRIPOD_SCOPE)
    _require(seed==receipt["seed"], "task and native Reset seed differ")
    geometry = decode_owned_json(documents.pop(FILES["geometry"]))
    _require([row["side"] for row in geometry]==["left","right"]
             and all(row["evidence"]["initial_geometry_passed"] is True for row in geometry),
             "independent left/right geometry required")
    hulls = {f"{row['side']}.{group}.{name}":shape["world_vertices"] for row in geometry
             for group in ("source_colliders","locked_pads") for name,shape in row["evidence"][group].items()}
    _require(len(hulls)==44, "complete34 source/10 cooked shapes required")
    guard_hand_supports(hulls,task["support_scene"])
    capture = json.loads(documents.pop(FILES["acquisition"]))
    _require(capture==lifecycle["acquisition"] and capture["mesh_count"]==17 and capture["pair_capacity"]==8364
             and capture["reset_seed_sha256"]==seed["sha256"]
             and capture["mesh_plan_sha256"]==saved["mesh_plan"]["sha256"],
             "native acquisition identity differs")
    refs = dict(original_receipt=outer_binding)
    refs.update({key:dict(path=str(run/name),sha256=inventory[name]) for key,name in FILES.items()})
    return dict(schema=SCHEMA,style="tripod",scope=SCOPE,pose_preview_only=False,
        target_schema=TARGET_SCHEMA_VERSION,target_material_ids=list(range(41)),
        source_refs=refs,native_run_path=str(out),native_fingerprint=outer["frozen_sha256"],
        reset_seed=seed,rod_state={name:audit["samples"][1]["state"][name].tolist()
                                  for name in ("positions","velocities","quaternions")},
        mesh_bindings_sha256=digest(encode_owned(receipt["rod_reset"]["mesh_bindings"])),
        task_config=dict(path=str(task_path),sha256=plan["task_config"]["sha256"]),
        extra_sources=plan["extra_sources"],current_reset_implementation_sha256=current_pins,
        teacher_admission=False,grasp_qualified=False)


@dataclass(frozen=True, slots=True, init=False)
class TripodResetManifest:
    path: str
    sha256: str
    raw: bytes

    def __init__(self,*args,**kwargs):
        raise TypeError("Tripod Reset manifest must be loaded from pinned native sources")

    def data(self):
        return json.loads(self.raw)

    def verify(self,*,project=ROOT):
        config=dict(style="tripod",resolved_reset_manifest=dict(path=self.path,sha256=self.sha256))
        _require(load_tripod_reset_manifest(config,project=project)==self, "current manifest/provenance differs")
        return self


def load_tripod_reset_manifest(config,*,project=ROOT):
    """Derive every exposed field from the pinned native evidence on each load."""
    try:
        _require(config["style"]=="tripod", "independent Tripod style required")
        path,raw = _binding(config["resolved_reset_manifest"])
        data = json.loads(raw)
        _require(data.get("schema")==SCHEMA and data.get("style")=="tripod" and data.get("scope")==SCOPE,
                 "independent Tripod17 schema/style/scope required")
        actual = _derive(data["native_run_path"],data["source_refs"]["original_receipt"],project)
        _require(data==actual, "candidate fields differ from actual native Tripod Reset")
        result = object.__new__(TripodResetManifest)
        for key,value in dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),raw=raw).items():
            object.__setattr__(result,key,value)
        return result
    except (KeyError,TypeError,OSError,ValueError) as error:
        raise ValueError("Tripod resolved Reset source missing: "+str(error)) from error


def seal_manifest(run_dir,*,receipt,expected_sha256,destination,project=ROOT):
    """Save a new manifest; original evidence and checkpoints remain read-only."""
    from scripts.diagnose_dh116_teacher_initialization import write_new
    data = _derive(run_dir,dict(path=str(Path(receipt).resolve(strict=True)),sha256=expected_sha256),project)
    path = Path(destination).resolve()
    _require(not path.is_relative_to(Path(project).resolve()), "raw manifest must remain outside Git project")
    write_new(path,data)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
