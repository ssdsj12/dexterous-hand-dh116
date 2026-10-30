"""Isaac visual replay of a measured grip trajectory; never live physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--recording',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--case',type=int,default=35)
parser.add_argument('--start-time',type=float,default=0.)
parser.add_argument('--paused',action='store_true')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
requested_pause=bool(args.paused);requested_start=float(args.start_time)
args.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(1)
root=Path(__file__).resolve().parents[1]
contract=json.loads((args.recording/'resolved_grip_lift.json').read_text())
initial=json.loads((args.recording/'initial_cases.json').read_text())[args.case]
raw=torch.load(args.recording/'raw_physics.pt',map_location='cpu',weights_only=False)
case_count=raw['object_pose_w'].shape[1]
if not 0<=args.case<case_count:raise ValueError('recording case index out of range')
names=contract['link_body_names'];origin=np.asarray(initial['environment_origin_w'])
poses=raw['full_link_pose_w'][:,args.case].numpy().copy();poses[...,:3]-=origin
object_pose=raw['object_pose_w'][:,args.case].numpy().copy();object_pose[:,:3]-=origin
bottom=raw['bottom'][:,args.case].numpy().copy()
forces=raw['hand_normal_load_n'][:,args.case].numpy().copy()
digits=raw['digit_normal_load_n'][:,args.case].numpy().copy()
program_time=raw['elapsed_s'][:,args.case].numpy().copy()
review_path=args.recording/'teacher_review.json'
review=json.loads(review_path.read_text()) if review_path.exists() else {}
peak=review.get('maximum_closure_penetration_mm')
steps=len(bottom);del raw
receipt=dict(mode='recorded_physics_visual_replay_not_live',source=str(args.recording),case=args.case,
    source_source_identity_lf=contract['source_identity_lf'],physics_steps_executed=0,
    stored_physics_sampling_hz=480,coordinate_translation_only=(-origin).tolist(),
    force_target=None,teacher_eligible=review.get('teacher_eligible'),closure_penetration_peak_mm=peak,pid=os.getpid(),
    requested_pause=requested_pause,requested_start_s=requested_start)
(args.output/'replay_contract.json').write_text(json.dumps(receipt,indent=2)+'\n')
app=AppLauncher(args).app
try:
    import omni.usd
    import omni.timeline
    import omni.ui as ui
    import carb
    from pxr import Gf,Usd,UsdGeom,UsdLux,UsdPhysics
    from omni.kit.viewport.utility import get_active_viewport
    from go2_pvcnn.control.dh116.visual_pose_sync import _find_body_prims
    context=omni.usd.get_context();context.new_stage();stage=context.get_stage()
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
    UsdGeom.Xform.Define(stage,'/World')
    omni.timeline.get_timeline_interface().stop()
    settings=carb.settings.get_settings()
    settings.set('/rtx/rendermode','RaytracedLighting')
    settings.set('/app/window/title','DH116 Recorded Grip-Lift Demo - Isaac Sim')
    light=UsdLux.DomeLight.Define(stage,'/World/Light');light.CreateIntensityAttr(500.)
    sun=UsdLux.DistantLight.Define(stage,'/World/Sun');sun.CreateIntensityAttr(500.)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(315.,30.,0.))
    table=UsdGeom.Cube.Define(stage,'/World/Table');table.CreateSizeAttr(1.)
    table.CreateDisplayColorAttr([Gf.Vec3f(.64,.66,.69)])
    tx=UsdGeom.Xformable(table);tx.AddTranslateOp().Set(Gf.Vec3d(.6,0.,.25));tx.AddScaleOp().Set(Gf.Vec3d(.8,1.2,.1))
    rod=UsdGeom.Cylinder.Define(stage,'/World/Rod');rod.CreateRadiusAttr(.008);rod.CreateHeightAttr(.5)
    rod.CreateAxisAttr(UsdGeom.Tokens.y);rod.CreateDisplayColorAttr([Gf.Vec3f(.45,.24,.07)])
    rod_op=UsdGeom.Xformable(rod).AddTransformOp()
    body_ops=[]
    for h,(side,asset) in enumerate(zip(('left','right'),('d134680ab619_DH116_left.usd','3046ad2ed855_DH116_right.usd'))):
        source=root/'assets/fixed_dual_panda_dh116_v1/dependencies'/asset
        assert hashlib.sha256(source.read_bytes()).hexdigest()==contract['source_asset_sha256'][side]
        path='/World/'+side+'Hand'
        prim=UsdGeom.Xform.Define(stage,path).GetPrim();prim.GetReferences().AddReference(str(source))
        found=_find_body_prims(stage,path,set(names[h]))
        if 'base_link' not in found:raise RuntimeError('native hand reference did not resolve')
        for name,body in found.items():
            if body.IsInstanceProxy():raise RuntimeError('cannot author an instance-proxy link')
            x=UsdGeom.Xformable(body);x.ClearXformOpOrder();op=x.AddTransformOp(opSuffix='recordedReplay');x.SetResetXformStack(True)
            body_ops.append((op,h,names[h].index(name)))
        # A playback stage has no live actuators or contacts. Keep it inert
        # even if the viewport timeline play button is accidentally pressed.
        for body in list(Usd.PrimRange(prim)):
            if body.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(body).CreateRigidBodyEnabledAttr(False)
            if body.IsA(UsdPhysics.Joint):body.SetActive(False)
    camera=UsdGeom.Camera.Define(stage,'/World/DemoCamera');camera.CreateFocalLengthAttr(24.)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01,100.));camera_op=UsdGeom.Xformable(camera).AddTransformOp()
    views=dict(Oblique=((.82,-.72,.72),(.46,0.,.36)),Side=((1.10,-.62,.48),(.45,0.,.34)),Top=((.43,0.,1.25),(.43,0.,.34)))
    def view(name):
        eye,target=views[name];up=(0.,1.,0.) if name=='Top' else (0.,0.,1.)
        camera_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(*up)).GetInverse())
    view('Oblique')
    for _ in range(12):app.update()
    viewport=get_active_viewport()
    if viewport is None:raise RuntimeError('Isaac viewport unavailable')
    viewport.camera_path='/World/DemoCamera'
    viewport.set_texture_resolution((960,640))
    auxiliary=('Content','Property','Stage','Console','Layer','Layers','Render Settings','Semantics Schema Editor','Isaac Sim Assets [Beta]')
    for title in auxiliary:
        window=ui.Workspace.get_window(title)
        if window:window.visible=False
    class Panel:
        def __init__(self):
            self.seconds=requested_start;self.paused=requested_pause;self.loop=True;self.speed=.5
            self.window=ui.Window('DH116 - Recorded Lift',width=340,height=400)
            self.window.position_x=15;self.window.position_y=80
            with self.window.frame:
                with ui.VStack(spacing=7):
                    ui.Label('RECORDED PHYSICS / NOT LIVE',height=26,style={'font_size':19})
                    ui.Label(f'Recorded {case_count}-env trial, case {args.case}. No force target.',height=22)
                    self.info=ui.Label('',height=142,word_wrap=True,style={'font_size':17})
                    with ui.HStack(height=30):
                        ui.Button('Replay',clicked_fn=lambda:self.seek(0.))
                        ui.Button('Pause / Resume',clicked_fn=self.toggle)
                        ui.Button('Final pose',clicked_fn=lambda:self.seek((steps-1)/480,True))
                    with ui.HStack(height=28):
                        for name in views:ui.Button(name,clicked_fn=lambda name=name:view(name))
                    ui.Label('0.5x speed, automatic loop. Tabletop Z = 0.300 m.',height=24)
                    ui.Label((f'Checked closure penetration: {peak:.2f} mm.\n' if peak is not None else 'Geometry review: see trial report.\n')+
                        ('Teacher criteria passed.' if receipt['teacher_eligible'] is True else 'Not a qualified teacher sample.'),height=42,word_wrap=True)
            viewport_window=ui.Workspace.get_window('Viewport')
            if viewport_window:
                try:self.window.dock_in(viewport_window,ui.DockPosition.LEFT,.28)
                except (AttributeError,TypeError,RuntimeError):pass
        def seek(self,seconds,paused=False):self.seconds=seconds;self.paused=paused
        def toggle(self):self.paused=not self.paused
    panel=Panel()
    def matrix(pose):
        q=pose[3:];m=Gf.Matrix4d(1.)
        m.SetRotate(Gf.Quatd(float(q[0]),Gf.Vec3d(*map(float,q[1:]))));m.SetTranslateOnly(Gf.Vec3d(*map(float,pose[:3])))
        return m
    last=time.monotonic();last_status=0.;frames=0;duration=(steps-1)/480;control_stamp=None
    print('REPLAY_READY',json.dumps(dict(receipt,visible_body_links=len(body_ops))),flush=True)
    while app.is_running() and not (args.output/'stop_requested').exists():
        for title in auxiliary:
            window=ui.Workspace.get_window(title)
            if window and window.visible:window.visible=False
        now=time.monotonic();delta=min(now-last,.15);last=now
        control_path=args.output/'control.json'
        if control_path.exists() and control_path.stat().st_mtime_ns!=control_stamp:
            try:
                control=json.loads(control_path.read_text());control_stamp=control_path.stat().st_mtime_ns
                if 'seek_s' in control:panel.seconds=max(0.,min(duration,float(control['seek_s'])))
                if 'paused' in control:panel.paused=bool(control['paused'])
                if control.get('view') in views:view(control['view'])
            except (OSError,ValueError):pass
        if not panel.paused:panel.seconds+=delta*panel.speed
        if panel.seconds>duration+2.:panel.seconds=0.
        index=min(steps-1,max(0,int(panel.seconds*480)))
        for op,h,b in body_ops:op.Set(matrix(poses[index,h,b]))
        rod_op.Set(matrix(object_pose[index]))
        phase='CLOSE' if program_time[index]<2.5 else ('LIFT' if program_time[index]<7.5 else 'HOLD')
        panel.info.text=(f'{"PAUSED" if panel.paused else "PLAYING"} | {phase} | t = {index/480:.2f} s\n'
            f'Object bottom above table: {(bottom[index]-.3)*1000:.2f} mm\n'
            f'LEFT normal load: {forces[index,0]:.2f} N\n'
            f'RIGHT normal load: {forces[index,1]:.2f} N\n'
            f'L thumb / index: {digits[index,0,0]:.2f} / {digits[index,0,1]:.2f} N\n'
            f'R thumb / index: {digits[index,1,0]:.2f} / {digits[index,1,1]:.2f} N')
        if omni.timeline.get_timeline_interface().is_playing():omni.timeline.get_timeline_interface().pause()
        app.update();frames+=1
        if now-last_status>=1.:
            status=dict(mode=receipt['mode'],frame=index,time_s=index/480,actual_bottom_height_mm=float((bottom[index]-.3)*1000),
                actual_normal_n=forces[index].tolist(),render_frames=frames,physics_steps_executed=0,paused=panel.paused,pid=os.getpid())
            (args.output/'status.json').write_text(json.dumps(status,indent=2)+'\n');last_status=now
        time.sleep(.01)
finally:
    print('REPLAY_CLOSED',flush=True)
    app.close()
