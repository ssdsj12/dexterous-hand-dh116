#!/usr/bin/env python3
"""Single bounded GPU worker for one dual Pinch SDK/rod interval."""

import argparse
from copy import deepcopy
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
from go2_pvcnn.control.dh116.tabletop_simulation_owner_cleanup import simulation_source_contract
from scripts import prepare_dh116_dual_pinch_interval as interval
from scripts import verify_dh116_pinch_candidate_v3 as prior
from scripts.probe_dh116_physx_lifetime import write_json

ENTRY = "scripts/verify_dh116_dual_pinch_interval.py"
WRAPPER = "scripts/run_dh116_dual_pinch_work_packages.py"
VERSION = interval.VERSION
WORKER_ENV = "DH116_DUAL_PINCH_INTERVAL_WORKER_V1"
LEFT_RECEIPT = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/native-attempts/"
    "flex.f4-pinch-left-v4-native/d7f87b161b3283a31ffe8a7befbbce0050f7d6bcc4b1473020e2e117067847cc/"
    "attempt-wezd3fyw/receipt.json")
LEFT_SHA = "5731ca089fa1b06afa0d41d27ac70dcbb9853c86a6e48982e068764a48c83e44"
LEFT_AUDIT_SHA = "ee88bbdb22dc8fa2164f4ad2ce086400e99652d721f44bb32b40c09ae3663519"
FILES = (ENTRY, WRAPPER, "scripts/prepare_dh116_dual_pinch_interval.py",
    "scripts/validate_dh116_dual_pinch_interval.py", "scripts/verify_dh116_pinch_candidate_v4.py",
    "tests/test_dh116_dual_pinch_worker.py",
    "tests/test_dh116_dual_pinch_entry.py", "tests/test_dh116_dual_pinch_postexit.py",
    "src/go2_pvcnn/control/dh116/tabletop_dual_pinch_builder.py",
    "src/go2_pvcnn/control/dh116/tabletop_dual_hand_scene_preparation.py",
    "src/go2_pvcnn/control/dh116/tabletop_dual_hand_reset.py",
    "src/go2_pvcnn/control/dh116/tabletop_dual_hand_rod_runtime.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_dual_hand_episode.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_dual_evidence.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_r0_replay.py",
    "src/go2_pvcnn/control/dh116/tabletop_bimanual_table_contacts.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_physical_exchange.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_bimanual_step.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_live_geometry.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_rod_boundary.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_solver_clock.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_wrist_projection.py",
    "src/go2_pvcnn/control/dh116/tabletop_teacher_mounted_wrist_source.py",
    "src/go2_pvcnn/control/dh116/tabletop_teacher_preaction_readback.py",
    "src/go2_pvcnn/control/dh116/tabletop_joint_target_readback.py",
    "src/go2_pvcnn/control/dh116/tabletop_palm_mount.py",
    "src/go2_pvcnn/control/dh116/tabletop_vec_isaac.py",
    "src/go2_pvcnn/control/dh116/tabletop_simulation_owner_cleanup.py",
    "configs/dh116_tasks/flexible_dual_hand_grasp_v1.json")


def read(path):
    return json.loads(Path(path).read_bytes())


def preflight(project):
    """Lock the paired seed, qualified backend and previous scoped receipts."""
    contract, cfg = prior.preflight(project)
    pair = builder.load_pair_proposal()
    sources = deepcopy(contract["sources"])
    local, external = sources["local_sources"], sources["external_sources"]
    for name in FILES:
        local[name] = prior.sha(project/name)
    for path, expected in ((builder.PAIR, builder.PAIR_SHA), (builder.BACKEND, builder.BACKEND_SHA),
            (LEFT_RECEIPT, LEFT_SHA), (LEFT_RECEIPT.with_name("pinch-v4-post-exit-audit.json"), LEFT_AUDIT_SHA)):
        if prior.sha(path) != expected:
            raise ValueError("dual Pinch required native/seed evidence differs")
        external[str(path)] = expected
    left = read(LEFT_RECEIPT)
    audit = read(LEFT_RECEIPT.with_name("pinch-v4-post-exit-audit.json"))["evidence"]
    if (left["native_qualification"] is not True or left["outer_exit_code"] != 0
            or audit["contact_observation_qualified"] is not True or audit["teacher_admission"] is not False):
        raise ValueError("single-left observed contact prerequisite not qualified")
    # Reuse the immutable lock summary without importing the geometry reader before AppLauncher.
    from scripts.verify_dh116_deformx_pinch_contact import default_arguments
    from scripts.verify_dh116_cooked_convex_hulls_native import _sources
    from go2_pvcnn.control.dh116.deformx_dh116_colliders import DISTAL_PAD_LINKS
    args = default_arguments(project)
    lock = read(project/"docs/contracts/dh116_deformx_pad_sdf_evidence_lock.json")
    sdf = read(args.sdf_dir/"result.json")
    for side, asset in _sources(project).items():
        local[str(Path(asset["path"]).relative_to(project))] = asset["sha256"]
        for pad in DISTAL_PAD_LINKS:
            key = side+"."+pad
            for path, expected in ((args.sdf_dir/(key+".npz"), sdf["pads"][key]["sdf_file_sha256"]),
                    (args.capture_dir/"gpu"/(side+"_"+pad+".json"), lock["gpu_rows"][key])):
                if prior.sha(path) != expected:
                    raise ValueError("dual hand pad evidence differs")
                external[str(path)] = expected
    if pair["rod_seed"]["gravity_world_m_s2"] != cfg["rod"]["gravity_world_m_s2"]:
        raise ValueError("paired gravity differs")
    cleanup = simulation_source_contract()
    for row in cleanup["sources"].values():
        external[row["path"]] = row["sha256"]
    code = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return dict(version=VERSION, code_config_sha256=code, sources=sources,
        paired_reset_seed_sha256=builder.SEED_SHA, simulation_clear_source_contract=cleanup,
        native_execution_performed=False, teacher_admission=False, physics_steps=10)


def require_worker(args, project):
    expected = dict(version=VERSION, case="explicit", output=str(args.output.resolve()),
        entry_sha256=prior.sha(project/ENTRY), wrapper_sha256=prior.sha(project/WRAPPER))
    try:
        value = json.loads(os.environ[WORKER_ENV])
    except (KeyError, ValueError):
        raise ValueError("dedicated dual Pinch worker required") from None
    if value != expected or os.environ.get("CUDA_VISIBLE_DEVICES") != "3" or os.environ.get("DH116_PHYSX_LIFETIME_GDB") != "1":
        raise ValueError("dual worker, GPU or stopped debugger identity differs")


def diagnostic_reset_owners():
    scopes = {
        "controller": "zero-action diagnostic has no stateful policy or MPC controller",
        "filter": "zero-action diagnostic does not instantiate a task filter",
        "contact": "fresh diagnostic contact source is bound after the one physical reset",
        "camera_history": "no camera or history owner is instantiated",
    }
    def absent(name, context):
        return dict(owner=name, status="not_present", episode_uid=context["episode_uid"], source=scopes[name])
    return {name:partial(absent,name) for name in scopes}


def execute(args, project, contract):
    from go2_pvcnn.control.dh116.tabletop_stereo_native import native_instance_lock
    from scripts.verify_dh116_pinch_candidate_v4 import _clear_sim
    write_json(args.output/"before_app_launcher.json", prior.assert_clean_import_boundary())
    from isaaclab.app import AppLauncher
    with native_instance_lock():
        sys.path.insert(0,str(builder.BACKEND.parent.parent))
        from ropesim import _ropesim
        if Path(_ropesim.__file__).resolve() != builder.BACKEND.resolve() or prior.sha(builder.BACKEND) != builder.BACKEND_SHA:
            raise ValueError("actual dual solver differs")
        app = AppLauncher(headless=True, device="cuda:0", enable_cameras=False, fast_shutdown=False,
            experience="/mnt/mydisk/xk/projects/IsaacLab/apps/isaacsim_4_5/isaaclab.python.headless.kit").app
        import faulthandler
        import torch
        import carb.settings
        import isaaclab.sim as sim_utils
        from isaacsim.core.simulation_manager import SimulationManager

        faulthandler.enable(file=sys.__stderr__,all_threads=True)
        settings = carb.settings.get_settings()
        def freeze():
            settings.set_bool("/app/player/playSimulations",False)
            return settings.get("/app/player/playSimulations")
        def stop():
            import omni.timeline
            omni.timeline.get_timeline_interface().stop()
        run = "dual-pinch-" + uuid.uuid4().hex
        interval.run_existing_app(argparse.Namespace(update=app.update,
            close=lambda:app.close(wait_for_replicator=False)),
            make_sim=lambda:sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/480,
                device="cuda:0",render_interval=1,gravity=(0.,0.,-9.81),
                physx=sim_utils.PhysxCfg(solver_type=0))),
            synchronize=lambda:torch.cuda.synchronize("cuda:0"),
            clear_sim=partial(_clear_sim,output=args.output,contract=contract["simulation_clear_source_contract"]),
            stop=stop, manager_is_none=lambda:SimulationManager.get_physics_sim_view() is None,
            freeze=freeze,output=args.output,project=project,prepared=None,
            owner_resets=diagnostic_reset_owners(),run_uuid=run,episode_uid=run+":one",
            code_config_sha256=contract["code_config_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup",choices=("explicit",),required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--steps",type=int,default=10)
    parser.add_argument("--execute-native",action="store_true")
    parser.add_argument("--approved-code-config-sha256")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.steps != 10:
        parser.error("exact 2 setup + 8 measured steps required")
    if args.execute_native:
        if args.output is None or not args.headless or args.device != "cuda:0":
            parser.error("fresh output/headless/cuda:0 required")
        require_worker(args,project)
        prior.assert_clean_import_boundary()
    contract = preflight(project)
    if not args.execute_native:
        print(json.dumps(contract,indent=2))
        return
    if args.approved_code_config_sha256 != contract["code_config_sha256"]:
        parser.error("frozen worker code/config SHA differs")
    args.output.mkdir(parents=True,exist_ok=False)
    execute(args,project,contract)


if __name__ == "__main__":
    main()
