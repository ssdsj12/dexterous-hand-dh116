"""Render SHA-checked native motion caches; no physics, solver or Teacher created."""
import argparse,fcntl,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT)]
def main():
    p=argparse.ArgumentParser();p.add_argument("--prepared",type=Path,required=True);p.add_argument("--gpu",type=int,default=2)
    a=p.parse_args();prepared=a.prepared.resolve()
    manifest=json.loads((prepared/"scene_manifest.json").read_text())
    source=Path(manifest["source_directory"])
    assert hashlib.sha256((source/"native_frames.jsonl").read_bytes()).hexdigest()==manifest["source_ledger_sha256"]
    assert hashlib.sha256((prepared/"recorded_scene.usda").read_bytes()).hexdigest()==manifest["scene_sha256"]
    output=prepared/"videos";output.mkdir(exist_ok=False)
    from go2_pvcnn.control.dh116.tabletop_teacher_cached_replay import audited_link_poses
    from go2_pvcnn.control.dh116.tabletop_teacher_preview_recording import NativePreviewRecording
    from go2_pvcnn.control.dh116.tabletop_teacher_video import capture_frame,frame_indices,encode_frames
    from go2_pvcnn.control.dh116.tabletop_teacher_video_delivery import measured_annotation
    modes=("pinch","tripod","power")
    records=[NativePreviewRecording.load(source/(m+"_native")) for m in modes]
    rows=[json.loads(s) for s in (source/"native_frames.jsonl").read_text().splitlines()]
    bindings,_=audited_link_poses(rows[0])
    for k,row in enumerate(rows):
        audited_link_poses(row,bindings)
        for rec in records:
            f=rec.frame(k)
            assert f["policy_step"]==row["policy_step"] and f["sdk_step"]==row["sdk_step"] and f["sdk_time"]==row["sdk_time"]
    os.environ["OMNI_KIT_ACCEPT_EULA"]="YES";os.environ.pop("CUDA_VISIBLE_DEVICES",None)
    sys.argv=[sys.argv[0],"--info"]
    from isaaclab.app import AppLauncher
    from go2_pvcnn.control.dh116.tabletop_preview_launcher import preview_launcher_type,install_fatal_defaults
    launcher=preview_launcher_type(AppLauncher,toolbar_getter=lambda:None)(
        headless=True,livestream=0,active_gpu=a.gpu,physics_gpu=a.gpu,multi_gpu=False,
        width=1280,height=720,device="cuda:"+str(a.gpu),fast_shutdown=False,enable_cameras=True,video=True,
        experience="/mnt/mydisk/xk/projects/IsaacLab/apps/isaacsim_4_5/isaaclab.python.headless.rendering.kit")
    app=launcher.app
    try:
        import signal
        install_fatal_defaults(signal)
        import omni.usd,omni.timeline,carb.settings
        import omni.kit.app
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.kit.viewport.window",True)
        from omni.kit.viewport.utility import get_active_viewport,capture_viewport_to_file,create_viewport_window
        from pxr import Usd,UsdGeom,UsdPhysics,UsdLux,Gf,Vt
        from PIL import Image,ImageDraw
        import imageio_ffmpeg,numpy as np
        context=omni.usd.get_context()
        assert context.open_stage(str(prepared/"recorded_scene.usda"))
        stage=context.get_stage();timeline=omni.timeline.get_timeline_interface()
        settings=carb.settings.get_settings()
        count=0;start=time.monotonic()
        def timeline_clock():return ("no_SDK_context",float(timeline.get_current_time()))
        def update():
            nonlocal count
            if count>=12000 or time.monotonic()-start>1800:raise RuntimeError("render-only budget exhausted")
            if timeline.is_playing():raise RuntimeError("render-only timeline unexpectedly playing")
            settings.set_bool("/app/player/playSimulations",False)
            app.update();count+=1
        for _ in range(12):update()
        assert not timeline.is_playing()
        for prim in stage.Traverse():
            assert not prim.HasAPI(UsdPhysics.RigidBodyAPI) and not prim.HasAPI(UsdPhysics.CollisionAPI)
            assert not any(str(s).lower().startswith(("physics","physx")) for s in prim.GetAppliedSchemas())
        light=UsdLux.DomeLight.Define(stage,"/World/ReplayLight");light.CreateIntensityAttr(650.)
        api=get_active_viewport()
        if api is None:
            viewport_window=create_viewport_window("Recorded native replay",width=1280,height=720)
            api=viewport_window.viewport_api
        if api is None:raise RuntimeError("render-only viewport unavailable")
        api.resolution=(1280,720)
        camera=UsdGeom.Camera.Define(stage,"/World/RecordedReplayCamera")
        camera.CreateFocalLengthAttr(28.);camera.CreateClippingRangeAttr(Gf.Vec2f(.01,100.))
        op=camera.AddTransformOp();api.camera_path=str(camera.GetPath())
        def place_camera(env):
            origin=np.array([0.,env*3.,0.])
            eye=origin+np.array([1.,-.85,.85]);target=origin+np.array([.43,0.,.41])
            matrix=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse()
            op.Set(matrix)
        def apply_frame(k):
            _,poses=audited_link_poses(rows[k],bindings)
            for b in bindings:
                values=poses[(b["env_id"],b["hand"])]
                for name,value in zip(b["body_names"],values):
                    path=f"/World/TeacherDisplay/env_{b['env_id']}/{b['hand']}Hand/{name}"
                    prim=stage.GetPrimAtPath(path)
                    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*value[:3]))
                    prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(value[3],Gf.Vec3d(*value[4:])))
            for env,rec in enumerate(records):
                nodes=rec.frame(k)["rod_nodes"]
                UsdGeom.BasisCurves(stage.GetPrimAtPath(f"/World/ReplayWire_{env}")).GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(nodes))
        names=("Pinch candidate","Tripod candidate","Power candidate - palm qualification pending")
        results={}
        for env,(mode,rec) in enumerate(zip(modes,records)):
            directory=output/(mode+"_frames");directory.mkdir()
            place_camera(env)
            indices=frame_indices(rec.frame_count,60,30)
            annotations=[measured_annotation(rec.frame(k),rows[k],env) for k in range(rec.frame_count)]
            provenance=[]
            for number,k in enumerate(indices):
                apply_frame(k)
                for _ in range(2):update()
                png=directory/f"frame_{number:05d}.png"
                capture=capture_viewport_to_file(api,str(png))
                capture_frame(lambda:capture.wait_for_result(),update,timeline_clock,png,max_updates=12)
                # File write completes asynchronously after the renderer future.
                # Two further bounded updates allow the PNG writer to finalize.
                for _ in range(2):update()
                image=Image.open(png).convert("RGB")
                ann=annotations[k];lift="NA" if ann["net_lift_m"] is None else f"{ann['net_lift_m']*1000:.3f}mm"
                draw=ImageDraw.Draw(image);draw.rectangle((0,image.height-50,image.width,image.height),fill=(15,18,22))
                draw.text((12,image.height-44),names[env]+" | native motion replay | full-success NOT qualified",fill="white")
                draw.text((12,image.height-25),f"t={k/60:.3f}s  height delta={ann['height_delta_m']*1000:.3f}mm  support-cleared Lift={lift}  support={ann['support_status']}",fill="white")
                image.save(png)
                provenance.append(dict(video_frame=number,source_frame=k,source_sdk_step=rec.frame(k)["sdk_step"],
                    source_sdk_time=rec.frame(k)["sdk_time"],annotation=ann,recording_sha256=rec.data_sha256))
                if number%30==0:print("CACHED_VIDEO_FRAME",mode,number,"/",len(indices),flush=True)
            (directory/"source_frames.json").write_text(json.dumps(provenance,indent=2))
            out=output/(mode+"_完整动作_候选演示.mp4")
            result=encode_frames(directory,out,fps=30,expected_count=len(indices),ffmpeg=imageio_ffmpeg.get_ffmpeg_exe())
            eligible=[ann["net_lift_m"]*1000 for ann in annotations if ann["net_lift_m"] is not None]
            result.update(video_file=out.name,recording_sha256=rec.data_sha256,
                checkpoint_sha256=rec.identity["checkpoint_sha256"],
                measured_height_peak_mm=max(ann["height_delta_m"] for ann in annotations)*1000,
                measured_height_final_mm=annotations[-1]["height_delta_m"]*1000,
                support_cleared_lift_peak_mm=max(eligible) if eligible else None,
                new_physics_steps=0,new_solver_steps=0,teacher_queries=0,power_palm_qualified=False)
            results[mode]=result
            (output/(mode+"_video.json")).write_text(json.dumps(result,indent=2))
        (output/"videos_complete.json").write_text(json.dumps(dict(videos=results,
            render_updates=count,new_physics_steps=0,new_solver_steps=0,teacher_queries=0,
            source_directory=str(source),source_ledger_sha256=manifest["source_ledger_sha256"],
            success_qualification=False),indent=2))
        print("CACHED_VIDEOS_COMPLETE",json.dumps(results),flush=True)
    except BaseException:
        import traceback
        with (output/"render_error.txt").open("w") as f:
            f.write(traceback.format_exc());f.flush();os.fsync(f.fileno())
        raise
    finally:app.close(wait_for_replicator=False)
if __name__=="__main__":
    with open("/tmp/dh116-isaac-single-instance.lock","a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
