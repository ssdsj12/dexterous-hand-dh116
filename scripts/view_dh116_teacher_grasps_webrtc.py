"""Bounded live display of three frozen Teachers; one button cycles views."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu",type=int,default=2)
    parser.add_argument("--duration",type=float,default=1800.)
    parser.add_argument("--max-control-steps",type=int,default=6000)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not 0<args.duration<=1800 or not 1<=args.max_control_steps<=6000:
        parser.error("bounded 1800s / 6000 control-step viewer required")
    cfg=json.loads((ROOT/"configs/dh116_execution/teacher_three_grasp_preview_v1.json").read_text())
    source=ROOT/"scripts/train_dh116_gpu_teacher_ppo.py"
    if hashlib.sha256(source.read_bytes()).hexdigest()!=cfg["source_sha256"]:
        raise ValueError("Teacher source changed; freeze/review new source before preview")
    checkpoint_bytes={}
    for mode,path in cfg["checkpoints"].items():
        checkpoint_bytes[mode]=(ROOT/path).read_bytes()
        if hashlib.sha256(checkpoint_bytes[mode]).hexdigest()!=cfg["checkpoint_sha256"][mode]:
            raise ValueError("frozen Teacher checkpoint changed: "+mode)
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
        ready_record=dict(
            version=cfg["version"],checkpoints=cfg["checkpoint_sha256"],gpu=args.gpu,
            modes=list(MODES),ppo_updates=0,server="172.31.179.148",
            duration_limit_s=args.duration,max_control_steps=args.max_control_steps)
        def physical_snapshot():
            return dict(display_pose_audit=display_audit,
                left_q=rt.left_hands.data.joint_pos.detach().cpu().tolist(),
                right_q=rt.right_hands.data.joint_pos.detach().cpu().tolist(),
                left_body_poses=rt.left_hands.data.body_link_pose_w.detach().cpu().tolist(),
                right_body_poses=rt.right_hands.data.body_link_pose_w.detach().cpu().tolist())
        display_audit=None
        (args.output/"initial_physics.json").write_text(json.dumps(physical_snapshot()))
        print("TEACHER_WEBRTC_LOADING: validating display-only body poses",flush=True)
        start=time.monotonic();total=0
        obs=rt.get_obs()
        while app.is_running() and time.monotonic()-start<args.duration and not (args.output/"stop_requested").exists():
            if state["pending"]!=state["selected"]:
                select_view(state["pending"]);state["cycles"]+=1
                with (args.output/"switches.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(mode=MODES[state["selected"]],control_step=state["control"]))+"\n")
            if state["control"]<min(runtime_args.horizon,args.max_control_steps):
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
                if (display_audit["max_position_error_m"]>1e-4
                        or display_audit["max_rotation_error_rad"]>1e-3):
                    if viewport is not None:viewport.visible=False
                    error=dict(error="display-only copy differs from measured SDK pose",
                        control_step=state["control"],audit=display_audit)
                    with (args.output/"failure.json").open("w") as stream:
                        json.dump(error,stream);stream.flush();os.fsync(stream.fileno())
                    raise RuntimeError(error["error"])
                if state["control"]==1:
                    (args.output/"first_display_audit.json").write_text(json.dumps(physical_snapshot()))
                    if viewport is not None:viewport.visible=True
                    (args.output/"ready.json").write_text(json.dumps(ready_record,indent=2))
                    print("TEACHER_WEBRTC_READY: measured poses validated; Next grasp cycles Pinch/Tripod/Power",flush=True)
                if state["control"]==runtime_args.horizon:
                    (args.output/"final_physics.json").write_text(json.dumps(physical_snapshot()))
            points=rt.rope_pos_view.detach().cpu().numpy()
            for i,curve in enumerate(curves):
                curve.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points[i]))
            index=state["selected"];total+=1
            status.text=f"Lift min: {minimum[index].item()*1000:.2f} mm | step {state['control']}/480"
            if state["control"]>=min(runtime_args.horizon,args.max_control_steps):
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
                        display_position_error_m=display_audit["max_position_error_m"],
                        display_rotation_error_rad=display_audit["max_rotation_error_rad"],
                        timeline_playing=rt.sim.is_playing()))+"\n")
        print("TEACHER_WEBRTC_FINISHED",flush=True)
    except BaseException:
        import traceback
        failure=dict(error=traceback.format_exc(),display_qualified=False)
        with (args.output/"runtime_error.json").open("w") as stream:
            json.dump(failure,stream);stream.flush();os.fsync(stream.fileno())
        print(failure["error"],file=sys.__stderr__,flush=True)
        raise
    finally:
        try:
            if display is not None:display.close()
            if rt is not None:
                # SDK's own reset guard also prevents STOP waiting for a new PLAY
                # while the application owner is deliberately shutting down.
                rt.sim._disable_app_control_on_stop_handle=True
                rt.sim.stop()
        finally:
            app.close(wait_for_replicator=False)



if __name__=="__main__":
    with open("/tmp/dh116-isaac-single-instance.lock","a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
