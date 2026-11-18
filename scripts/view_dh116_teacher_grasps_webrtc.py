"""Bounded live display of three frozen Teachers; one button cycles views."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]


def _read_frozen_checkpoints(checkpoint_root, config):
    models = {}
    for mode, path in config["checkpoints"].items():
        data = (checkpoint_root/path).read_bytes()
        if hashlib.sha256(data).hexdigest() != config["checkpoint_sha256"][mode]:
            raise ValueError("frozen Teacher checkpoint changed: "+mode)
        models[mode] = data
    return models


def _publish_audited_view(audit, viewport, output, control_step):
    position = float(audit["max_position_error_m"])
    rotation = float(audit["max_rotation_error_rad"])
    qualified = 0. <= position <= 1e-4 and 0. <= rotation <= 1e-3
    if viewport is not None and viewport.visible != qualified:
        viewport.visible = qualified
    if qualified:
        return True
    sdk_step, sdk_time = audit["physics_clock"]
    record = dict(control_step=control_step, sdk_step=sdk_step, sdk_time_s=sdk_time,
                  updated_links=audit["updated_links"], display_qualified=False,
                  max_position_error_m=position if math.isfinite(position) else None,
                  max_rotation_error_rad=rotation if math.isfinite(rotation) else None)
    with (output/"display_rejections.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"TEACHER_WEBRTC_DISPLAY_REJECTED: control={control_step} sdk={sdk_step} "
          f"position_m={position} rotation_rad={rotation}", flush=True)
    return False


def _shutdown_preview(output, steps, close_app):
    with (output/"shutdown_phases.jsonl").open("x") as ledger:
        def receipt(name, event, **fields):
            row = dict(phase=name, event=event, monotonic_ns=time.monotonic_ns(),
                       process_exit_verified=False, **fields)
            ledger.write(json.dumps(row, allow_nan=False)+"\n")
            ledger.flush()
            os.fsync(ledger.fileno())

        def run(name, operation):
            receipt(name, "started")
            try:
                result = operation()
            except BaseException as error:
                receipt(name, "error", error_type=type(error).__name__, message=str(error))
                raise
            receipt(name, "returned", result=result)

        try:
            for name, operation in steps:
                run(name, operation)
        finally:
            run("app_close", close_app)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu",type=int,default=2)
    parser.add_argument("--duration",type=float,default=1800.)
    parser.add_argument("--max-control-steps",type=int,default=6000)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--checkpoint-root",type=Path,default=ROOT)
    args=parser.parse_args()
    if not 0<args.duration<=1800 or not 1<=args.max_control_steps<=6000:
        parser.error("bounded 1800s / 6000 control-step viewer required")
    cfg=json.loads((ROOT/"configs/dh116_execution/teacher_three_grasp_preview_v1.json").read_text())
    source=ROOT/"scripts/train_dh116_gpu_teacher_ppo.py"
    if hashlib.sha256(source.read_bytes()).hexdigest()!=cfg["source_sha256"]:
        raise ValueError("Teacher source changed; freeze/review new source before preview")
    checkpoint_bytes=_read_frozen_checkpoints(args.checkpoint_root,cfg)
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA","YES")
    os.environ.pop("CUDA_VISIBLE_DEVICES",None)
    sys.argv=[sys.argv[0], "--info"]
    from isaaclab.app import AppLauncher
    from go2_pvcnn.control.dh116.tabletop_preview_launcher import preview_launcher_type
    def capture_startup_error():
        import traceback
        error="".join(traceback.format_exception(*sys.exc_info()))
        with (args.output/"startup_error.txt").open("w") as stream:
            stream.write(error);stream.flush();os.fsync(stream.fileno())
        print(error,file=sys.__stderr__,flush=True)
    launcher=preview_launcher_type(AppLauncher,toolbar_getter=lambda: None,
        partial_stop_presentation=capture_startup_error)(headless=True,livestream=2,
        active_gpu=args.gpu,physics_gpu=args.gpu,multi_gpu=False,
        width=1280,height=720,
        device="cuda:"+str(args.gpu),fast_shutdown=False,
        experience="/mnt/mydisk/xk/projects/IsaacLab/apps/isaacsim_4_5/isaaclab.python.headless.rendering.kit")
    app=launcher.app
    rt=None;display=None
    try:
        import omni.ui as ui
        viewport=ui.Workspace.get_window("Viewport")
        if viewport is None:
            raise RuntimeError("Viewport required before physical scene creation")
        viewport.visible=False
        import torch,io
        import faulthandler,signal
        faulthandler.register(signal.SIGUSR1,file=sys.__stderr__,all_threads=False)
        # Let normal cancellation enter this viewer's ordered finally cleanup.
        signal.signal(signal.SIGINT,signal.default_int_handler)
        from go2_pvcnn.control.dh116.tabletop_preview_launcher import install_fatal_defaults
        install_fatal_defaults(signal)
        torch.cuda.set_device(args.gpu)
        import numpy as np
        import omni.ui as ui
        import carb.settings
        from pxr import UsdGeom,Gf,Vt
        from go2_pvcnn.control.dh116.tabletop_teacher_grasp_preview_runtime import create_runtime,MODES
        runtime_args=SimpleNamespace(mode="pinch",num_envs=3,horizon=480,updates=0,
            eval_only=True,gpu=args.gpu,device="cuda:"+str(args.gpu))
        settings=carb.settings.get_settings()
        rt=create_runtime(ROOT,runtime_args,app,cfg["source_sha256"])
        from go2_pvcnn.control.dh116.tabletop_teacher_read_only_display import ReadOnlyHandDisplay
        bindings=[dict(env_id=i,hand=side,source_root=f"/World/envs/env_{i}/{side}Hand",
            body_names=list(hand.body_names))
            for i in range(3) for side,hand in (("left",rt.left_hands),("right",rt.right_hands))]
        display=ReadOnlyHandDisplay(rt.sim.stage,bindings)
        (args.output/"display_no_physics.json").write_text(json.dumps(display.audit_no_physics()))
        if rt.sim.current_time_step_index>2:
            raise RuntimeError("initialization exceeded 2 SDK physics steps")
        device=runtime_args.device
        module=sys.modules.get("_dh116_preview_teacher")
        spec=importlib.util.spec_from_file_location("_dh116_preview_model",source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        models=[]
        for mode in MODES:
            payload=torch.load(io.BytesIO(checkpoint_bytes[mode]),map_location=device,weights_only=False)
            model=module.build_actor_critic(67,24,init_std=.008,device=device)
            model.load_state_dict(payload["model"]);model.eval()
            models.append(model)
        carb.settings.get_settings().set_bool("/app/player/playSimulations",False)
        light=rt.sim_utils.DomeLightCfg(intensity=650.)
        light.func("/World/TeacherPreviewLight",light)
        curves=[]
        for i in range(3):
            curve=UsdGeom.BasisCurves.Define(rt.sim.stage,f"/World/TeacherWire_{i}")
            curve.CreateTypeAttr("linear");curve.CreateWrapAttr("nonperiodic")
            curve.CreateCurveVertexCountsAttr([41])
            curve.CreateWidthsAttr([float(rt.r_cfg["radius_m"])*2]);curve.SetWidthsInterpolation("constant")
            curve.CreateDisplayColorAttr([Gf.Vec3f(.95,.3,.025)])
            curves.append(curve)
        state=dict(selected=0,pending=0,control=0,cycles=0)
        names=("Pinch (2 fingers)","Tripod (3 fingers)","Power (whole hand)")
        def next_grasp():
            state["pending"]=(state["selected"]+1)%3
        window=ui.Window("DH116 Teacher - Three Grasps",width=440,height=190)
        with window.frame:
            with ui.VStack(spacing=12):
                title=ui.Label("Current: "+names[0])
                ui.Button("Next grasp: Pinch > Tripod > Power",clicked_fn=next_grasp,height=46)
                status=ui.Label("Loading frozen Teacher...")
                ui.Label("Frozen model playback | 0 PPO updates")
        def select_view(index):
            state["selected"]=index
            origin=rt.env_origins_np[index]
            rt.sim.set_camera_view(eye=(origin+np.array([1.0,-.85,.8])).tolist(),
                target=(origin+np.array([.43,0.,.34])).tolist())
            title.text="Current: "+names[index]
        rt.reset_all()
        select_view(0)
        viewport=ui.Workspace.get_window("Viewport")
        if viewport is not None:
            viewport.width=1280;viewport.height=720;viewport.visible=False
        window.position_x=20;window.position_y=35
        control_limit=min(runtime_args.horizon,args.max_control_steps)
        ready_record=dict(
            version=cfg["version"],checkpoints=cfg["checkpoint_sha256"],gpu=args.gpu,
            modes=list(MODES),ppo_updates=0,server="172.31.179.148",
            duration_limit_s=args.duration,max_control_steps=args.max_control_steps)
        def physical_snapshot():
            return dict(display_pose_audit=display_audit if display_qualified else None,
                display_qualified=display_qualified,
                left_q=rt.left_hands.data.joint_pos.detach().cpu().tolist(),
                right_q=rt.right_hands.data.joint_pos.detach().cpu().tolist(),
                left_body_poses=rt.left_hands.data.body_link_pose_w.detach().cpu().tolist(),
                right_body_poses=rt.right_hands.data.body_link_pose_w.detach().cpu().tolist())
        display_audit=None
        display_qualified=False
        first_verified_control=None
        rejected_frames=0
        (args.output/"initial_physics.json").write_text(json.dumps(physical_snapshot()))
        print("TEACHER_WEBRTC_LOADING: validating display-only body poses",flush=True)
        start=time.monotonic();total=0
        obs=rt.get_obs()
        while app.is_running() and time.monotonic()-start<args.duration and not (args.output/"stop_requested").exists():
            if state["pending"]!=state["selected"]:
                select_view(state["pending"]);state["cycles"]+=1
                with (args.output/"switches.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(mode=MODES[state["selected"]],control_step=state["control"]))+"\n")
            if state["control"]<control_limit:
                if rt.sim.current_time_step_index+8>3842:
                    raise RuntimeError("viewer exceeded authorized SDK physics-step budget")
                with torch.inference_mode():
                    action=torch.cat([model(obs[i:i+1])[0] for i,model in enumerate(models)],dim=0)
                    obs,_,med,minimum,_,_=rt.step_env(action)
                state["control"]+=1
                measured={(i,side):hand.data.body_link_pose_w[i]
                    for i in range(3) for side,hand in (("left",rt.left_hands),("right",rt.right_hands))}
                display_audit=display.sync(measured,read_clock=lambda:
                    (rt.sim.current_time_step_index,float(rt.sim.current_time)))
                display_qualified=_publish_audited_view(display_audit,viewport,args.output,state["control"])
                if not display_qualified:
                    rejected_frames+=1
                elif first_verified_control is None:
                    first_verified_control=state["control"]
                    (args.output/"first_display_audit.json").write_text(json.dumps(physical_snapshot()))
                    ready_record["first_verified_control_step"]=first_verified_control
                    (args.output/"ready.json").write_text(json.dumps(ready_record,indent=2))
                    print("TEACHER_WEBRTC_READY: measured poses validated; Next grasp cycles Pinch/Tripod/Power",flush=True)
                if state["control"]==control_limit:
                    (args.output/"final_physics.json").write_text(json.dumps(physical_snapshot()))
            points=rt.rope_pos_view.detach().cpu().numpy()
            for i,curve in enumerate(curves):
                curve.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points[i]))
            index=state["selected"];total+=1
            if display_qualified:
                status.text=f"Lift min: {minimum[index].item()*1000:.2f} mm | step {state['control']}/{control_limit}"
            else:
                status.text=f"View hidden: pose check failed | step {state['control']}/{control_limit}"
            if state["control"]>=control_limit:
                settings.set_bool("/app/player/playSimulations",False)
                app.update()
                time.sleep(1/30)
            else:
                rt.sim.render()
            if total%60==0:
                with (args.output/"progress.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(mode=MODES[index],control_step=state["control"],viewer_frame=total,
                        lift_min_mm=(minimum.detach().cpu()*1000).tolist(),ppo_updates=0,
                        physical_control_step=state["control"],
                        display_qualified=display_qualified,
                        rejected_display_frames=rejected_frames,
                        display_position_error_m=display_audit["max_position_error_m"] if display_qualified else None,
                        display_rotation_error_rad=display_audit["max_rotation_error_rad"] if display_qualified else None,
                        timeline_playing=rt.sim.is_playing()))+"\n")
        (args.output/"viewer_complete.json").write_text(json.dumps(dict(
            control_steps=state["control"], sdk_step=rt.sim.current_time_step_index,
            first_verified_control_step=first_verified_control, rejected_display_frames=rejected_frames,
            final_display_qualified=display_qualified, ppo_updates=0, process_exit_verified=False),indent=2))
        print("TEACHER_WEBRTC_FINISHED",flush=True)
    except BaseException:
        import traceback
        failure=dict(error=traceback.format_exc(),display_qualified=False)
        with (args.output/"runtime_error.json").open("w") as stream:
            json.dump(failure,stream);stream.flush();os.fsync(stream.fileno())
        print(failure["error"],file=sys.__stderr__,flush=True)
        raise
    finally:
        # These loop locals may hold a view even after its runtime namespace is released.
        measured = points = obs = action = med = minimum = None
        steps = []
        if rt is not None:
            import omni.timeline
            from isaacsim.core.simulation_manager import SimulationManager
            from go2_pvcnn.control.dh116.tabletop_simulation_owner_cleanup import (
                clear_known_simulation_instance, simulation_source_contract)

            def freeze_physics():
                settings.set_bool("/app/player/playSimulations",False)
                if settings.get("/app/player/playSimulations") is not False:
                    raise RuntimeError("preview physics freeze failed before shutdown")
                return dict(sdk_step=rt.sim.current_time_step_index, cuda_device=runtime_args.device)

            def release_runtime():
                nonlocal rt, display
                if display is not None:
                    display.close()
                    display = None
                sim = rt.sim
                rt.__dict__.clear()
                rt = None
                return clear_known_simulation_instance(sim, contract=simulation_source_contract())

            def stop_physics():
                omni.timeline.get_timeline_interface().stop()
                app.update()
                released = SimulationManager.get_physics_sim_view() is None
                if not released:
                    raise RuntimeError("SimulationManager still owns its physics view after STOP")
                return dict(physics_view_released=True)

            steps = [
                ("freeze_physics", freeze_physics),
                ("cuda_synchronize", lambda: torch.cuda.synchronize(runtime_args.device)),
                ("release_runtime", release_runtime),
                ("stop_physics", stop_physics),
            ]
        _shutdown_preview(args.output, steps, lambda: app.close(wait_for_replicator=False))



if __name__=="__main__":
    with open("/tmp/dh116-isaac-single-instance.lock","a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
