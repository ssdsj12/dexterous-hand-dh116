#!/usr/bin/env python3
"""Independent bounded zero-PPO physical preview with private WebRTC and RGB.

Freeze the current sources first, review the resulting contract and authorize a
separate native preview before using --execute-native. No native work occurs
during --freeze. This entry is not a Teacher worker or a qualification run.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from go2_pvcnn.control.dh116.tabletop_preview_contract import freeze, verify, BUDGET, RenderBudget, create_preview_camera

EXPERIENCE = Path("/mnt/mydisk/xk/projects/IsaacLab/apps/isaacsim_4_5/isaaclab.python.headless.rendering.kit")
REQUIRED = (
    "src/go2_pvcnn/control/dh116/live_deformx_display.py",
    "src/go2_pvcnn/control/dh116/tabletop_preview_warp_readback.py",
    "src/go2_pvcnn/control/dh116/tabletop_tiled_stereo_capture.py",
    "src/go2_pvcnn/control/dh116/tabletop_tiled_stereo_native.py",
    "src/go2_pvcnn/control/dh116/tabletop_tiled_stereo_resources.py",
    "configs/dh116_execution/tiled_camera_api_lock_v1.json",
    "src/go2_pvcnn/control/dh116/tabletop_preview_diagnostics.py",
    "src/go2_pvcnn/control/dh116/tabletop_preview_debugger.py",
    "src/go2_pvcnn/control/dh116/tabletop_preview_launcher.py",
    "src/go2_pvcnn/control/dh116/tabletop_preview_device.py",
    "src/go2_pvcnn/control/dh116/tabletop_stereo_rgb_bridge.py",
    "src/go2_pvcnn/control/dh116/tabletop_stereo_rgb_owned_native.py",
    "src/go2_pvcnn/control/dh116/tabletop_stereo_rgb_resources.py",
    "src/go2_pvcnn/control/dh116/tabletop_stereo_camera.py",
    "src/go2_pvcnn/control/dh116/tabletop_stereo_calibration.py",
    "src/go2_pvcnn/control/dh116/tabletop_stereo_validation.py",
    "src/go2_pvcnn/control/dh116/tabletop_vision_camera.py",
    "src/go2_pvcnn/control/dh116/tabletop_preview_presentation.py",
    "src/go2_pvcnn/control/dh116/tabletop_physical_preview.py",
    "src/go2_pvcnn/control/dh116/wire_mesh_publisher.py",
    "src/go2_pvcnn/control/fixed_dual_panda_o6/wire_visual.py",
    "src/go2_pvcnn/control/dh116/tabletop_dual_pinch_builder.py",
    "src/go2_pvcnn/control/dh116/tabletop_native_parameter_endpoints.py",
    "scripts/dh116_fault_safe_lifecycle_v3.py",
    "scripts/run_dh116_work_packages.py",
    "scripts/prepare_dh116_dual_pinch_interval.py",
    "scripts/verify_dh116_dual_pinch_interval.py",
    "scripts/dh116_pinch_acquisition_v2.py",
    "scripts/dh116_pinch_startup_v3.py",
    "scripts/probe_dh116_physx_lifetime_frame_v2.py",
    "src/go2_pvcnn/control/dh116/tabletop_stereo_native.py",
    "src/go2_pvcnn/control/dh116/tabletop_native_teacher_profile.py",
    "src/go2_pvcnn/control/dh116/tabletop_native_teacher_actual_launch.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_dual_evidence.py",
    "src/go2_pvcnn/control/dh116/tabletop_deformable_r0_replay.py",
    "src/go2_pvcnn/control/dh116/tabletop_simulation_owner_cleanup.py",
    "configs/dh116_training/stereo_camera_front_640x480_v3.json",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(contract, output):
    """Only called after CPU contract verification; lock spans the full App."""
    from go2_pvcnn.control.dh116.tabletop_stereo_native import native_instance_lock
    from scripts.run_dh116_work_packages import NATIVE_LOCK
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile, profile_reader_scope
    profile = BackendProfile.load(ROOT)
    profile.verify()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(contract["physical_gpu_index"]):
        raise ValueError("preview requires the frozen physical GPU mask")
    deadline = time.monotonic() + BUDGET["soft_deadline_s"]
    with native_instance_lock(NATIVE_LOCK):
        # Explicit backend must be imported only after the lock and before App.
        binary = Path(profile.binary_path)
        sys.path.insert(0, str(binary.parents[1]))
        from ropesim import _ropesim as backend
        if Path(backend.__file__).resolve() != binary or _sha(binary) != profile.binary_sha256:
            raise ValueError("actual preview backend differs from verified profile")
        from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary
        assert_clean_import_boundary()
        from go2_pvcnn.control.dh116.tabletop_preview_device import probe_device
        probe_device(contract["physical_gpu_index"])
        from isaaclab.app import AppLauncher
        from go2_pvcnn.control.dh116.tabletop_preview_launcher import launch_managed_preview_app
        app = launch_managed_preview_app(AppLauncher, contract)
        entered = False
        try:
            import torch
            import carb.settings
            import isaaclab.sim as sim_utils
            from isaacsim.core.simulation_manager import SimulationManager
            from scripts import dh116_pinch_acquisition_v2 as lifecycle_bindings
            from scripts.dh116_fault_safe_lifecycle_v3 import run_lifecycle
            from scripts.dh116_pinch_startup_v3 import StartupTrace, snapshot_for_startup
            from scripts.probe_dh116_physx_lifetime_frame_v2 import bindings
            from scripts.verify_dh116_dual_pinch_interval import diagnostic_reset_owners
            from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
            from go2_pvcnn.control.dh116.tabletop_native_parameter_endpoints import NativeParameterEndpoints
            from go2_pvcnn.control.dh116.tabletop_deformable_dual_evidence import DualHandEvidenceJournal
            from go2_pvcnn.control.dh116.tabletop_physical_preview import run_owned_scene
            from go2_pvcnn.control.dh116.tabletop_simulation_owner_cleanup import simulation_source_contract, clear_known_simulation_instance
            from types import SimpleNamespace
            settings = carb.settings.get_settings()
            cleanup_contract = simulation_source_contract()

            def freeze_player():
                settings.set_bool("/app/player/playSimulations", False)
                return settings.get("/app/player/playSimulations")

            def stop():
                import omni.timeline
                omni.timeline.get_timeline_interface().stop()

            def make_sim():
                sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
                    dt=1/480, device="cuda:0", render_interval=1,
                    gravity=(0.,0.,-9.81), physx=sim_utils.PhysxCfg(solver_type=0)))
                sim.set_camera_view(eye=(1.35,-1.35,1.05), target=(.36,0.,.32))
                light=sim_utils.DomeLightCfg(intensity=2000.)
                light.func("/World/PreviewLight",light)
                return sim

            def acquire(sim, owners, trace, cleanup):
                camera = None
                render_sim = RenderBudget(sim, BUDGET["max_render_calls"])
                journal = DualHandEvidenceJournal(output/"physical.jsonl",
                    physics_budget_end=BUDGET["total_sdk_steps"], solver_substeps=12)
                holder = {}
                try:
                    render_sim.install()
                    def present(pre, exchange):
                        nonlocal camera
                        from pxr import UsdGeom, Gf
                        from go2_pvcnn.control.dh116.live_deformx_display import LiveDeformxDisplay, capture_snapshot, capture_reset_snapshot
                        from go2_pvcnn.control.dh116.tabletop_preview_presentation import PreviewPresentation
                        from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
                        rod=owners.resources["runtime"]
                        packet=exchange["material_director"]
                        if exchange["preview_phase"] == "reset":
                            snapshot=capture_reset_snapshot(exchange["rod"], packet, source_instance_id=id(rod.rope),
                                read_clock=rod.clock, backend_profile=profile)
                        else:
                            snapshot=capture_snapshot(exchange, source_instance_id=id(rod.rope),
                                read_clock=rod.clock, backend_profile=profile)
                        holder["snapshot"],holder["exchange"]=snapshot,exchange
                        if camera is None:
                            mesh=UsdGeom.Mesh.Define(sim.stage,"/World/CosseratWire")
                            mesh.CreateSubdivisionSchemeAttr().Set("none")
                            mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(1.,.2,.035)])
                            display=LiveDeformxDisplay.for_mesh(mesh,snapshot["radius_m"],
                                source_instance_id=id(rod.rope),backend_profile=profile)
                            def identity():
                                current=holder["exchange"]; clock=rod.clock()
                                return dict(run_uuid=clock["run_uuid"],env_id=clock["env_id"],
                                    episode_uid=clock["episode_uid"],reset_id=clock["reset_id"],
                                    policy_tick=current["preview_policy_tick"],
                                    physics_tick_start=current["preview_physics_start"],
                                    physics_tick_end=clock["physics_step"],
                                    rod_step=clock["rod_tick"], state_sha256=_digest(dict(
                                        rod=current["rod"],material_director=current["material_director"],
                                        action_readback=current.get("action_readback"))),
                                    simulation_timestamp=clock["timestamp_s"],
                                    calibration_sha256=holder["camera"].calibration.sha256,
                                    sdk_step=clock["physics_step"],sdk_timestamp_s=clock["timestamp_s"],
                                    rod_tick=clock["rod_tick"],solver_time_s=clock["solver_time_s"],
                                    material_reference_uid=current["material_director"]["material_reference_uid"],
                                    rod_uid=clock["rod_uid"])
                            camera=create_preview_camera(ROOT,sim,contract,identity)
                            holder["camera"]=camera
                            owners.keep("preview_rgb",camera)
                            trace.watch("preview_rgb",camera,owner="preview RGB",release="close before SDK STOP")
                            (output/"capture_calibration.json").write_text(json.dumps(dict(
                                **camera.calibration.to_dict(),
                                calibration_sha256=camera.calibration.sha256,
                                resolution_mode="native_lens_authoring_not_image_resize",
                                camera_backend=contract["camera_backend"]),indent=2)+"\n")
                            holder["display"]=display
                        before=rod.clock()
                        before_time=float(sim.current_time)
                        index=holder.get("frame",0)
                        def sink(images, owned_snapshot, terminal):
                            from PIL import Image
                            import numpy as np
                            for side in ("left","right"):
                                rgb=images[side+"_rgb"][0].permute(1,2,0).detach().cpu().numpy()
                                Image.fromarray(np.clip(np.rint(rgb*255),0,255).astype("uint8")).save(output/(f"{index:03d}_{side}.png"))
                            metadata=dict(phase=exchange["preview_phase"],terminal_reason=terminal,
                                identity=images["identity"],render_timing=images["render_timing"],
                                camera_backend=contract["camera_backend"],
                                capture_source_scope=images.get("source_scope","legacy_bridge_unqualified"),
                                resolution_wh=[camera.calibration.width,camera.calibration.height],
                                calibration_sha256=images["calibration_sha256"],
                                K_left=images["K_left"].tolist(),K_right=images["K_right"].tolist(),
                                native_qualification=False,teacher_admission=False)
                            (output/(f"{index:03d}_frame.json")).write_text(json.dumps(metadata,indent=2)+"\n")
                        presenter=PreviewPresentation(capture_snapshot=lambda row:snapshot,
                            display=holder["display"],camera=camera,read_clock=rod.clock,sink=sink)
                        presenter(pre,exchange)
                        holder["frame"]=index+1
                        # Give the private WebRTC client time to connect while
                        # pumping a frozen first frame; never advance physics.
                        if index == 0:
                            original=settings.get("/app/player/playSimulations")
                            if type(original) is not bool: raise ValueError("player setting unavailable")
                            try:
                                settings.set_bool("/app/player/playSimulations",False)
                                if settings.get("/app/player/playSimulations") is not False:
                                    raise RuntimeError("preview freeze readback failed")
                                for _ in range(24):
                                    if time.monotonic()>=deadline: raise RuntimeError("preview wall budget expired")
                                    render_sim.render()
                                    if (rod.clock()!=before or sim.current_time_step_index!=before["physics_step"]
                                            or float(sim.current_time)!=before_time):
                                        raise RuntimeError("preview hold advanced physics")
                                    time.sleep(.5)
                            finally:
                                settings.set_bool("/app/player/playSimulations",original)
                                if settings.get("/app/player/playSimulations") is not original:
                                    raise RuntimeError("preview freeze restoration failed")
                    def actions():
                        for _ in range(BUDGET["control_intervals"]):
                            if time.monotonic()>=deadline: raise RuntimeError("preview wall budget expired")
                            yield torch.zeros((1,24),dtype=torch.float64)
                    run="preview-"+uuid.uuid4().hex
                    result=run_owned_scene(sim,owners,trace,builder=builder.build_reset_owned,
                        reader_factory=NativeParameterEndpoints,backend=backend,journal=journal,
                        observer=present,build_kwargs=dict(project=ROOT,prepared=None,ownership=trace,
                            owner_resets=diagnostic_reset_owners(),run_uuid=run,episode_uid=run+":one",
                            physics_budget_end=BUDGET["total_sdk_steps"]),
                        device="cuda:0",actions=actions())
                    result["render_calls"] = render_sim.count
                    return result
                finally:
                    try:
                        if camera is not None: camera.close()
                    finally:
                        try: render_sim.close()
                        finally:
                            holder.clear()
                            journal.close()
            names=tuple(side+"_"+suffix for side in ("left","right") for suffix in
                ("hand","view","asset_sim_view","permanent_composer","instantaneous_composer"))+(
                "runtime","rope","episode","material_director_reader","preview_rgb")
            output.mkdir(parents=True,exist_ok=False)
            with profile_reader_scope(profile), bindings((builder,"BACKEND",binary),
                (builder,"BACKEND_SHA",profile.binary_sha256),
                (lifecycle_bindings,"OWNED_NAMES",names),
                (lifecycle_bindings,"LifetimeTrace",StartupTrace),
                (lifecycle_bindings,"snapshot_weakrefs",snapshot_for_startup)):
                from go2_pvcnn.control.dh116.tabletop_preview_diagnostics import StageJournal
                with StageJournal(output/"preview_stages.jsonl") as stage_journal:
                    entered=True
                    return run_lifecycle(SimpleNamespace(update=app.update, close=lambda:app.close(wait_for_replicator=False)),make_sim=make_sim,acquire=acquire,
                        synchronize=lambda:torch.cuda.synchronize("cuda:0"),
                        clear_sim=lambda sim:clear_known_simulation_instance(sim,contract=cleanup_contract),
                        stop=stop,manager_is_none=lambda:SimulationManager.get_physics_sim_view() is None,
                        freeze=freeze_player,output=output,cleanup="explicit",stage_observer=stage_journal.emit)
        finally:
            if not entered: app.close(wait_for_replicator=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze",type=Path)
    group.add_argument("--execute-native",action="store_true")
    parser.add_argument("--physical-gpu",type=int,default=3,help="physical GPU to bind when freezing a contract")
    parser.add_argument("--resolution",help="WIDTHxHEIGHT at freeze time; both cameras use this native size")
    parser.add_argument("--camera-backend",choices=("legacy_view","tiled_step"),
                        help="freeze-only independent capture backend; legacy_view remains default")
    parser.add_argument("--contract",type=Path)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    if args.camera_backend is not None and not args.freeze:
        parser.error("camera backend is freeze-only; use the authorized contract")
    if args.resolution is not None and not args.freeze:
        parser.error("resolution is freeze-only; native execution uses the authorized contract")
    width,height=640,480
    if args.resolution is not None:
        match=re.fullmatch(r"([1-9][0-9]*)[xX]([1-9][0-9]*)",args.resolution)
        if match is None:parser.error("resolution must be positive WIDTHxHEIGHT")
        width,height=map(int,match.groups())
    if args.freeze:
        if args.freeze.resolve().is_relative_to(ROOT):
            parser.error("freeze contract outside the source inventory")
        missing=[name for name in REQUIRED if not (ROOT/name).is_file()]
        if missing: parser.error("missing required integrated modules: "+", ".join(missing))
        try:
            contract=freeze(ROOT,EXPERIENCE,physical_gpu_index=args.physical_gpu,width=width,height=height,
                            camera_backend=args.camera_backend or "legacy_view")
        except ValueError as error:
            parser.error(str(error))
        payload=json.dumps(contract,sort_keys=True,indent=2)+"\n"
        with args.freeze.open("x") as stream: stream.write(payload)
        print("contract_sha256="+_sha(args.freeze))
        return 0
    if not args.contract or not args.contract_sha256 or not args.output:
        parser.error("native execution requires contract, exact SHA and fresh output")
    if _sha(args.contract)!=args.contract_sha256:
        parser.error("contract SHA differs")
    contract=verify(ROOT,json.loads(args.contract.read_text()))
    execute(contract,args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
