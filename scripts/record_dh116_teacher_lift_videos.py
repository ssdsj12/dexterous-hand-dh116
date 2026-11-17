"""Bounded end-supported native Teacher recording and read-only video replay."""
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
    parser.add_argument("--max-control-steps",type=int,default=480)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not 0<args.duration<=1800 or not 1<=args.max_control_steps<=480:
        parser.error("bounded 1800s / 480 control-step viewer required")
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
        from go2_pvcnn.control.dh116.tabletop_teacher_grasp_preview_runtime import MODES
        from go2_pvcnn.control.dh116.tabletop_teacher_supported_runtime import create_supported_runtime
        runtime_args=SimpleNamespace(mode="pinch",num_envs=3,horizon=480,updates=0,
            eval_only=True,gpu=args.gpu,device="cuda:"+str(args.gpu))
        settings=carb.settings.get_settings()
        rt=create_supported_runtime(ROOT,runtime_args,app,cfg["source_sha256"])
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
        from go2_pvcnn.control.dh116.tabletop_teacher_preview_recording import NativePreviewRecorder
        from go2_pvcnn.control.dh116.tabletop_teacher_video import frame_indices,capture_frame,encode_frames
        from go2_pvcnn.control.dh116.tabletop_teacher_video_delivery import measured_annotation
        from go2_pvcnn.control.dh116.tabletop_deformx_end_supported_scene import support_load_records,evaluate_support_clearance
        from omni.kit.viewport.utility import get_active_viewport,capture_viewport_to_file
        from PIL import Image,ImageDraw
        import imageio_ffmpeg
        names=("Pinch candidate","Tripod candidate","Power candidate - palm qualification pending")
        window=ui.Window("DH116 - Native recording",width=540,height=130)
        with window.frame:
            with ui.VStack(spacing=8):
                title=ui.Label("Recording actual native physics")
                status=ui.Label("Preparing two real collision supports")
                ui.Label("Candidate demonstration | full grasp success NOT qualified")
        viewport=ui.Workspace.get_window("Viewport")
        viewport.width=1280;viewport.height=720;viewport.visible=False
        budget=dict(control_steps=0,render_updates=0,max_control_steps=540,
            max_sdk_steps=4322,max_render_updates=6000,solver_substeps_per_physics_step=12,
            wall_limit_s=args.duration,ppo_updates=0)
        start=time.monotonic()
        def write_json(path,value):
            with path.open("w") as stream:
                json.dump(value,stream,indent=2,default=lambda a:a.tolist())
                stream.flush();os.fsync(stream.fileno())
        def clock():return (rt.sim.current_time_step_index,float(rt.sim.current_time))
        def check_budget():
            if time.monotonic()-start>args.duration:raise RuntimeError("native/video wall budget exhausted")
            if clock()[0]>budget["max_sdk_steps"]:raise RuntimeError("native physics budget exhausted")
            if budget["render_updates"]>=budget["max_render_updates"]:raise RuntimeError("render budget exhausted")
        def render_only():
            check_budget()
            settings.set_bool("/app/player/playSimulations",False)
            before=clock();app.update();budget["render_updates"]+=1
            if clock()!=before:raise RuntimeError("render advanced frozen physical clock")
        def step(action):
            check_budget()
            if budget["control_steps"]>=540 or clock()[0]+8>4322:
                raise RuntimeError("physical action budget exhausted")
            with torch.inference_mode():result=rt.step_env(action)
            budget["control_steps"]+=1
            return result
        def sync(poses,nodes):
            audit=display.sync({(i,side):poses[i,hi]
                for i in range(3) for hi,side in enumerate(("left","right"))},read_clock=clock)
            if audit["max_position_error_m"]>1e-4 or audit["max_rotation_error_rad"]>1e-3:
                viewport.visible=False;raise RuntimeError("measured/display link mismatch")
            for i,curve in enumerate(curves):
                curve.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(nodes[i].astype(np.float32)))
            return audit
        def measured():
            return np.stack([h.data.body_link_pose_w.detach().cpu().numpy()
                for h in (rt.left_hands,rt.right_hands)],axis=1),rt.rope_pos_view.detach().cpu().numpy().copy()
        def select_view(index):
            origin=rt.env_origins_np[index]
            rt.sim.set_camera_view(eye=(origin+np.array([1.0,-.85,.85])).tolist(),
                target=(origin+np.array([.43,0.,.36+rt.supported_scene_plan["scene"]["dz_m"]])).tolist())
            title.text=names[index]
        rt.reset_all()
        write_json(args.output/"support_bindings.json",rt.support_binding)
        write_json(args.output/"supported_scene.json",rt.supported_scene_plan["scene"])
        zero=torch.zeros((3,24),device=device)
        # Observe true initial settling. No actor, extra force, reset or attachment.
        with (args.output/"settling.jsonl").open("w") as ledger:
            for k in range(60):
                step(zero)
                rows=support_load_records(rt.wrench_view,rt.support_binding,
                    contact_data_valid=True,step_id=clock()[0])
                ledger.write(json.dumps(dict(control_step=k+1,sdk_step=clock()[0],
                    sdk_time=clock()[1],support=rows))+"\n")
            ledger.flush();os.fsync(ledger.fileno())
        baseline=rt.set_settled_lift_reference()
        write_json(args.output/"settled_baseline.json",baseline)
        recorders=[NativePreviewRecorder(max_frames=481) for _ in MODES]
        physical_ledger=(args.output/"native_frames.jsonl").open("w")
        def record(k,action):
            poses,nodes=measured()
            audit=sync(poses,nodes)
            loads=support_load_records(rt.wrench_view,rt.support_binding,
                contact_data_valid=True,step_id=clock()[0])
            clearance=[]
            for i,style in enumerate(MODES):
                lift=float(nodes[i,:,2].min()-baseline["initial_min_z_m"][i])
                clear=evaluate_support_clearance(nodes[i]-rt.env_origins_np[i],
                    rt.r_cfg["radius_m"],rt.supported_scene_plan["scene"],loads[i],
                    initial_min_z_m=baseline["initial_min_z_m"][i]-rt.env_origins_np[i,2],
                    expected_step_id=clock()[0])
                clearance.append(clear)
                recorders[i].append(run_uid=args.output.name,episode_uid=args.output.name+"-"+style,
                    env_id=i,style=style,checkpoint_sha256=cfg["checkpoint_sha256"][style],
                    sdk_step=clock()[0],sdk_time=clock()[1],policy_step=k,
                    body_link_pose_w=poses[i],rod_nodes=nodes[i],action=action[i],
                    net_lift=np.float32(lift),wrench=rt.wrench_view[i],
                    support_ledger={"wrenches_world":np.array(loads[i]["support_wrenches_world"])},
                    validity={"wrench":True,"support_ledger":True,"net_lift_support_clear":bool(clear["eligible"])})
            physical_ledger.write(json.dumps(dict(policy_step=k,sdk_step=clock()[0],sdk_time=clock()[1],
                support_loads=loads,clearance=clearance,display_audit=audit))+"\n")
            if k%30==0:
                physical_ledger.flush();os.fsync(physical_ledger.fileno())
                print("NATIVE_RECORDED",k,[float(nodes[i,:,2].min()-baseline["initial_min_z_m"][i])*1000 for i in range(3)],flush=True)
        record(0,zero)
        viewport.visible=True;select_view(0)
        obs=rt.get_obs()
        for k in range(1,481):
            with torch.inference_mode():action=torch.cat([model(obs[i:i+1])[0] for i,model in enumerate(models)])
            obs,*_=step(action)
            record(k,action)
            status.text=f"Actual native action: {k}/480"
            # Keep WebRTC live; these app updates must never add physical steps.
            render_only()
        physical_ledger.flush();os.fsync(physical_ledger.fileno());physical_ledger.close()
        rows_by_policy={row["policy_step"]:row for row in
            map(json.loads,(args.output/"native_frames.jsonl").read_text().splitlines())}
        write_json(args.output/"recording_fields.json",dict(
            net_lift="raw settled-baseline-relative height; qualified only when net_lift_support_clear validity is true",
            final_video_net_lift="support-cleared height from same-clock native ledger"))
        frozen=clock()
        recordings=[r.finish() for r in recorders]
        for style,rec in zip(MODES,recordings):rec.save(args.output/(style+"_native"))
        write_json(args.output/"native_complete.json",dict(budget=budget,frozen_clock=frozen,
            ppo_updates=0,full_grasp_success_qualified=False,power_palm_qualified=False))
        api=get_active_viewport()
        if api is None:raise RuntimeError("no active viewport capture API")
        api.resolution=(1280,720)
        results={}
        for i,(style,rec) in enumerate(zip(MODES,recordings)):
            directory=args.output/(style+"_frames");directory.mkdir()
            select_view(i)
            source_indices=frame_indices(rec.frame_count,60,30)
            with (directory/"source_frames.jsonl").open("w") as provenance:
                for number,idx in enumerate(source_indices):
                    frame=rec.frame(idx)
                    annotation=measured_annotation(frame,rows_by_policy[idx],i)
                    lift_text="NA (supported or unknown)" if annotation["net_lift_m"] is None else f"{annotation['net_lift_m']*1000:.3f} mm"
                    poses=np.stack([r.frame(idx)["body_link_pose_w"] for r in recordings])
                    nodes=np.stack([r.frame(idx)["rod_nodes"] for r in recordings])
                    sync(poses,nodes)
                    title.text=names[i]+" | recorded native replay"
                    status.text=f"t={frame['policy_step']/60:.3f}s | support-cleared Lift={lift_text}"
                    for _ in range(2):render_only()
                    png=directory/f"frame_{number:05d}.png"
                    capture=capture_viewport_to_file(api,str(png))
                    capture_frame(lambda:capture.wait_for_result(),render_only,clock,png,max_updates=6)
                    if clock()!=frozen:raise RuntimeError("video replay advanced physics")
                    # Text only: scene pixels retain the exact captured native record.
                    image=Image.open(png).convert("RGB");draw=ImageDraw.Draw(image)
                    draw.rectangle((0,image.height-48,image.width,image.height),fill=(15,18,22))
                    draw.text((12,image.height-42),names[i]+" | recorded native replay | NOT full-success qualified",fill="white")
                    draw.text((12,image.height-24),f"t={frame['policy_step']/60:.3f}s  support-cleared Lift={lift_text}  height delta={float(frame['net_lift'])*1000:.3f} mm",fill="white")
                    image.save(png)
                    provenance.write(json.dumps(dict(video_frame=number,source_policy_step=idx,
                        source_sdk_step=frame["sdk_step"],source_sdk_time=frame["sdk_time"],
                        recording_sha256=rec.data_sha256,frozen_sdk_step=frozen[0],
                        height_delta_mm=annotation["height_delta_m"]*1000,
                        net_lift_mm=None if annotation["net_lift_m"] is None else annotation["net_lift_m"]*1000,
                        support_status=annotation["support_status"]))+"\n")
                    if number%30==0:print("VIDEO_CAPTURE",style,number,"/",len(source_indices),flush=True)
                provenance.flush();os.fsync(provenance.fileno())
            mp4=args.output/(style+"_full_lift_candidate.mp4")
            result=encode_frames(directory,mp4,fps=30,expected_count=len(source_indices),
                ffmpeg=imageio_ffmpeg.get_ffmpeg_exe())
            annotations=[measured_annotation(rec.frame(j),rows_by_policy[j],i) for j in range(rec.frame_count)]
            qualified=[a["net_lift_m"]*1000 for a in annotations if a["net_lift_m"] is not None]
            result.update(checkpoint_sha256=cfg["checkpoint_sha256"][style],
                recording_sha256=rec.data_sha256,video_file=mp4.name,
                net_lift_peak_mm=max(qualified) if qualified else None,
                net_lift_final_mm=None if annotations[-1]["net_lift_m"] is None else annotations[-1]["net_lift_m"]*1000,
                height_delta_peak_mm=float(rec.net_lift.max())*1000,
                initial_support_height_not_lift=True,full_success_qualified=False,
                support_source="actual_physx_and_deformx_colliders",
                power_palm_qualified=False if style=="power" else None)
            results[style]=result
            write_json(args.output/(style+"_video.json"),result)
        write_json(args.output/"videos_complete.json",dict(videos=results,
            budget=budget,final_frozen_clock=clock(),source="actual_native_recorded_replay",
            natural_exit_pending=True))
        print("ALL_VIDEOS_SAVED",json.dumps(results),flush=True)
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
