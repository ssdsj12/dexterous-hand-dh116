"""Human-controlled Isaac view. No PPO updates or qualified teacher labels."""
import json
import math
import time
import os
from datetime import datetime,timezone
import torch
from .tabletop_control_probe import joint_actions
from .tabletop_training import write_json


class ManualCommands:
    def __init__(self,reference,lower,center,upper,device):
        self.device=device
        self.home=reference.detach().cpu().double().clone()
        self.lower=lower.detach().cpu().double();self.center=center.detach().cpu().double()
        self.upper=upper.detach().cpu().double()
        self.joints=self.home.clone();self.force=torch.full((2,),5.,dtype=torch.float64)
        self.palms=torch.zeros((2,3),dtype=torch.float64)
        self.lift_goal=.01;self.lift_speed=.005;self.lift_current=0.;self.lifting=False
        self.running=False;self.closed=False
        self.physics_wall_s=0.;self.last_step_wall_s=0.
        self.auto_lift_goal=None;self.force_stable_s=0.;self.previous_load=None
        self.pending_reset=False;self.pending_force=False;self.pending_pose=False;self.pending_save=False

    def pause_after_reset(self):
        self.running=False;self.lifting=False;self.lift_current=0.
        self.auto_lift_goal=None;self.force_stable_s=0.;self.previous_load=None

    def start_lift(self):
        self.lifting=True
        self.running=True

    def lift_when_stable(self,grasp_stable):
        if self.auto_lift_goal is None:return False
        if grasp_stable:
            self.lift_goal=self.auto_lift_goal;self.auto_lift_goal=None;self.lifting=True
            return True
        return False

    def set_placement(self,payload):
        if set(payload)-{'palm_offsets_m','force_n'}:
            raise ValueError('manual placement only accepts translations and normal force targets')
        palms=torch.as_tensor(payload['palm_offsets_m'],dtype=torch.float64)
        force=torch.as_tensor(payload.get('force_n',[5.,5.]),dtype=torch.float64)
        if (palms.shape!=(2,3) or force.shape!=(2,) or not torch.isfinite(palms).all()
                or not torch.isfinite(force).all() or ((force<2)|(force>12)).any()
                or (palms<palms.new_tensor([-.003,-.003,-.004])).any()
                or (palms>palms.new_tensor([.003,.003,.05])).any()):
            raise ValueError('invalid manual translation or normal force target')
        self.palms=palms.clone();self.force=force.clone()

    def actions(self,dt):
        if not math.isfinite(dt) or dt<=0:raise ValueError('positive finite control dt required')
        if self.lifting:
            delta=min(.05,max(0.,self.lift_goal))-self.lift_current
            self.lift_current+=max(-min(.01,max(0.,self.lift_speed))*dt,
                                  min(min(.01,max(0.,self.lift_speed))*dt,delta))
        palms=self.palms.clone();palms[:,2]+=self.lift_current
        lo=palms.new_tensor([-.003,-.003,-.004]);hi=palms.new_tensor([.003,.003,.05])
        palms=palms.maximum(lo).minimum(hi)
        normalized=palms/torch.where(palms>=0,hi,-lo)
        joints=self.joints.maximum(self.lower).minimum(self.upper)
        result=torch.cat((joint_actions(joints,self.lower,self.center,self.upper),normalized.flatten(),
                          (self.force.clamp(2.,12.)-2.)/5.-1.))
        return result.clamp(-1.,1.).to(device=self.device,dtype=torch.float32)[None]


class ManualPanel:
    def __init__(self,commands):
        import omni.ui as ui
        from .linear_coupling import ACTIVE
        self.c=commands;self.models=[]
        self.window=ui.Window('DH116 Manual Control',width=520,height=820)
        labels=('Thumb yaw','Thumb flex','Index','Middle','Ring','Little')
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=5):
                    ui.Label('DH116 manual control | 1 environment',height=24)
                    ui.Label(('RUNNING.' if commands.running else 'Starts PAUSED.')+' Force values are per-hand normal contact load.',word_wrap=True,height=36)
                    with ui.HStack(height=28):
                        ui.Button('Run / Pause',clicked_fn=self.toggle)
                        ui.Button('Reset + Pause',clicked_fn=lambda:setattr(commands,'pending_reset',True))
                        ui.Button('Save settings',clicked_fn=lambda:setattr(commands,'pending_save',True))
                    self.status=ui.Label('Initializing...',word_wrap=True,height=380)
                    for h,side in enumerate(('LEFT HAND','RIGHT HAND')):
                        ui.Label(side,height=22)
                        self.drag(ui,'Target force (N)',float(commands.force[h]),2.,12.,.1,
                                  lambda v,h=h:commands.force.__setitem__(h,v))
                        for j,joint in enumerate(ACTIVE):
                            index=6*h+j
                            self.drag(ui,labels[j]+' / '+joint+' (deg)',math.degrees(float(commands.joints[index])),
                                math.degrees(float(commands.lower[index])),math.degrees(float(commands.upper[index])),.2,
                                lambda v,k=index:commands.joints.__setitem__(k,math.radians(v)))
                        for j,axis in enumerate(('X','Y','Z')):
                            self.drag(ui,'Palm '+axis+' offset (mm)',float(commands.palms[h,j]*1000),-4. if j==2 else -3.,50. if j==2 else 3.,.1,
                                lambda v,h=h,j=j:commands.palms.__setitem__((h,j),v/1000.))
                    with ui.HStack(height=28):
                        ui.Button('Apply force reference',clicked_fn=lambda:setattr(commands,'pending_force',True))
                        ui.Button('Apply finger pose',clicked_fn=lambda:setattr(commands,'pending_pose',True))
                    ui.Label('After close, reference/pose stay latched until explicitly applied.',word_wrap=True,height=32)
                    self.drag(ui,'Common lift height (mm)',commands.lift_goal*1000,0.,50.,.5,lambda v:setattr(commands,'lift_goal',v/1000.))
                    self.drag(ui,'Lift speed (mm/s, simulation)',5.,0.,10.,.1,lambda v:setattr(commands,'lift_speed',v/1000.))
                    with ui.HStack(height=28):
                        ui.Button('Lift to height',clicked_fn=commands.start_lift)
                        ui.Button('Stop lift',clicked_fn=lambda:setattr(commands,'lifting',False))
                        ui.Button('Lower to start',clicked_fn=self.lower)
                    ui.Label('Lift moves the wrists. Actual object height is measured separately.',word_wrap=True,height=32)
                    ui.Label('Physical guards remain active. Failure pauses motion; Reset starts again.',word_wrap=True,height=32)
                    ui.Button('Close manual Isaac',height=28,clicked_fn=lambda:setattr(commands,'closed',True))

    def drag(self,ui,label,value,minimum,maximum,step,callback):
        with ui.HStack(height=23):
            ui.Label(label,width=275)
            model=ui.SimpleFloatModel(value)
            ui.FloatDrag(model=model,min=minimum,max=maximum,step=step,format='%.2f')
            model.add_value_changed_fn(lambda m:callback(m.as_float))
            self.models.append(model)

    def toggle(self):self.c.running=not self.c.running
    def lower(self):self.c.lift_goal=0.;self.c.start_lift()


def manual_status(env,c,steps,notice):
    t=env.task;m=t.m
    valid=m['contact_valid'][0]
    positions=m['contact_center_relative_m'][0].detach().cpu().tolist()
    for h in range(2):
        if not bool(valid[h]):positions[h]=None
    return dict(status='manual_running' if c.running else 'manual_paused',control_steps=steps,
        phase=int(t.phase[0]),requested_force_n=t.f_requested[0].cpu().tolist(),
        measured_force_n=m['hand_object_loads'][0].cpu().tolist(),
        normal_force_z_n=m['vertical_hand_force_n'][0].cpu().tolist(),
        below_axis=m['contact_all_below_centerline'][0].cpu().tolist(),
        contact_valid=valid.cpu().tolist(),contact_relative_m=positions,
        finger_actual_deg=(m['q'][0].cpu()*180/math.pi).tolist(),
        finger_target_deg=(c.joints*180/math.pi).tolist(),
        object_lift_m=float(m['bottom'][0]-.30),wrist_requested_lift_m=c.lift_current,
        wrist_measured_offset_m=m['adaptive_palm_offset'][0].cpu().tolist(),
        lift_speed_m_s=c.lift_speed,lift_criterion_met=bool(t.manual_lift_qualified[0]),
        notice=notice,training=False,teacher_qualified=False,
        manual_fast_force=getattr(t,'manual_fast_force',False),effort_limit_nm=t.parameters.effort_limit_nm,
        force_feedback_offset_rad=t.f_offset[0].cpu().tolist(),
        pressure_limited=t.pressure_limited[0].cpu().tolist(),
        max_mimic_error_rad=t.m['mimic_errors'].abs().amax(1).cpu().tolist(),
        palm_motion_soft_paused=t.paused.cpu().tolist(),
        last_control_step_wall_s=c.last_step_wall_s,
        physics_realtime_factor=steps*env.world.dt*env.world.decimation/max(c.physics_wall_s,1e-9))


def run_manual_control(env,args,contract):
    import omni.kit.app
    t=env.task;t.manual_control=True
    t.manual_lift_qualified=torch.zeros(t.n,device=t.device,dtype=torch.bool)
    t.manual_lightweight=True
    t.last_observation=t.observe().clone()
    c=ManualCommands(t.reference[0],t.lower,t.center,t.upper,env.device)
    if getattr(args,'manual_placement',None):
        c.set_placement(json.loads(args.manual_placement.read_text()))
    t.manual_fast_force=bool(getattr(args,'manual_fast_force',False))
    t.manual_stable_grip_ascent=t.manual_fast_force
    digit_contract=None
    if t.manual_fast_force:
        from .tabletop_manual_digit_force_balance import enable,CONTRACT
        enable(t);digit_contract=dict(CONTRACT)
    if getattr(args,'manual_auto_lift_mm',None) is not None:
        c.auto_lift_goal=args.manual_auto_lift_mm/1000.
        c.lift_goal=c.auto_lift_goal
    if t.manual_fast_force:
        env.world.render_interval_steps=max(1,round(1/(15*env.world.dt)))
    env.world.manual_render_wall_interval_s=.2
    resolution=None
    try:
        from omni.kit.viewport.utility import get_active_viewport
        viewport=get_active_viewport()
        if viewport is not None and hasattr(viewport,'set_texture_resolution'):
            viewport.set_texture_resolution((960,540));resolution=[960,540]
        elif viewport is not None and hasattr(viewport,'texture_resolution'):
            viewport.texture_resolution=(960,540);resolution=[960,540]
    except (ImportError,AttributeError,RuntimeError):
        pass
    c.running=bool(getattr(args,'manual_run',False))
    t.f_requested[0]=c.force.to(t.f_requested);t.f_target[0]=t.f_requested[0]
    panel=ManualPanel(c)
    from .tabletop_manual_digit_contacts import collect,Visuals
    contact_visuals=Visuals()
    from .tabletop_manual_thumb_stability import ThumbStability,CONTRACT as THUMB_STABILITY
    thumb_stability=ThumbStability()
    dt=env.world.dt*env.world.decimation
    steps=0;last_status=0.;notice='Ready. Set controls, then click Run / Pause.'
    kit=omni.kit.app.get_app()
    write_json(args.output/'manual_contract.json',dict(contract,manual_control=True,
        learning_deadlines_disabled=True,manual_target_changes_unqualified=True,
        lightweight_diagnostics=True,training_reward_computation=False,
        digit_force_balance=digit_contract,
        pinch_geometry='experimental_compensation_disabled_after_failed_physical_validation',
        thumb_force_stability=dict(THUMB_STABILITY),
        render_wall_interval_s=.2,viewport_resolution=resolution))
    profiler=None;profile_steps=0
    if os.environ.get('DH116_MANUAL_PROFILE')=='1':
        import cProfile
        profiler=cProfile.Profile()
    try:
        while not c.closed and panel.window.visible and kit.is_running():
            start=time.monotonic()
            if (args.output/'manual_stop.request').exists():break
            if c.pending_reset:
                c.pending_reset=False;c.pause_after_reset()
                thumb_stability.reset()
                env.reset();notice='Reset complete. Paused.'
                t.f_requested[0]=c.force.to(t.f_requested);t.f_target[0]=t.f_requested[0]
            hold=t.policy_lift_hold
            if c.pending_force:
                c.pending_force=False
                thumb_stability.reset()
                reference=c.force.to(t.f_requested).clamp(2.,12.)
                t.f_requested[0]=reference;t.f_target[0]=reference
                if bool(hold.active[0]):
                    hold.force_target[0]=reference;hold.force_integral[0]=0.
                    hold.force_filtered[0]=t.m['hand_object_loads'][0]
                notice='New force reference applied by user.'
            if c.pending_pose:
                c.pending_pose=False
                thumb_stability.reset()
                if bool(hold.active[0]):
                    hold.command[0]=c.joints.to(t.command).maximum(t.low).minimum(t.high)
                    hold.closure[0]=0.;hold.force_integral[0]=0.
                notice='Finger targets applied through bounded drives.'
            if c.running:
                if not env.world.w.sim.is_playing():env.world.w.sim.play()
                try:
                    step_started=time.monotonic()
                    if profiler is not None:profiler.enable()
                    _,_,done,extra=env.step(c.actions(dt));steps+=1
                    if profiler is not None:
                        profiler.disable();profile_steps+=1
                        if profile_steps>=12:
                            import pstats
                            with (args.output/'manual_profile.txt').open('w') as stream:
                                pstats.Stats(profiler,stream=stream).sort_stats('cumulative').print_stats(35)
                            profiler=None
                    c.last_step_wall_s=time.monotonic()-step_started;c.physics_wall_s+=c.last_step_wall_s
                    if not bool(done.any()):
                        thumb_stability.update(t.m['grouped_object_loads'][0,:,:2].detach().cpu().tolist(),dt)
                    if bool(done.any()):
                        c.pause_after_reset()
                        thumb_stability.reset()
                        from .tabletop_tensor_task import FAILURE_CODES
                        code=int(extra['failure'][0])
                        reason=next((name for name,value in FAILURE_CODES.items() if value==code),str(code))
                        notice='Episode reset: '+reason
                        t.f_requested[0]=c.force.to(t.f_requested);t.f_target[0]=t.f_requested[0]
                        if hasattr(env,'manual_last_failure'):
                            write_json(args.output/'manual_last_failure.json',env.manual_last_failure)
                except RuntimeError as error:
                    if 'tensor physical safety failure' not in str(error):raise
                    c.pause_after_reset()
                    thumb_stability.reset()
                    notice='Physics guard paused motion. Reset before continuing.'
                    write_json(args.output/'manual_guard.json',dict(error=str(error),control_step=steps))
                    env.reset()
            else:
                if env.world.w.sim.is_playing():env.world.w.sim.pause()
                env.world.w.sim.render()
            now=time.monotonic()
            if now-last_status>.1 or c.pending_save:
                row=manual_status(env,c,steps,notice)
                try:
                    digits,center,axis=collect(env)
                    row['digit_contacts']=digits
                    contact_visuals.update(digits,center,axis)
                except (ValueError,RuntimeError) as error:
                    row['digit_contacts']=dict(all_four_digits_below_axis=False,error=str(error))
                row['last_failure']=getattr(env,'manual_last_failure',None)
                row['thumb_stability']=thumb_stability.result
                balance_text=''
                if getattr(t,'manual_digit_force_balance',False):
                    from .tabletop_manual_digit_force_balance import summary
                    row['digit_force_balance']=summary(t)
                    balance=row['digit_force_balance']
                    ratios=['no paired load' if r is None else f'{(r-1)*100:.1f}%' for r in balance['measured_ratio']]
                    balance_text=f"Thumb/index spread L/R: {ratios[0]} / {ratios[1]}; max 50%\n"
                row['grasp_stable']=(all(thumb_stability.result['thumb_force_stable'])
                    and all(row.get('digit_force_balance',{}).get('balanced',[False,False]))
                    and row['digit_contacts']['all_four_digits_below_axis'])
                if c.running and c.lift_when_stable(row['grasp_stable']):
                    notice='Both measured thumbs stable and digit spread <=50%; starting requested lift.'
                    row['notice']=notice
                actual=[(m[2]-float(c.palms[h,2]))*1000 for h,m in enumerate(row['wrist_measured_offset_m'])]
                digit_text=[]
                for name,d in row['digit_contacts'].get('digits',{}).items():
                    z=d['max_radial_z_mm'];depth=d['max_penetration_mm']
                    digit_text.append(f"{name}: {d['load_n']:.2f} N; max Z {z:.2f} mm; depth {depth:.3f} mm" if z is not None and depth is not None else f"{name}: no measured loaded contact")
                load=row['measured_force_n'];target=row['requested_force_n']
                failure_text=''
                if row['last_failure']:
                    last=row['last_failure']
                    failure_text=(f"Last reset: mimic {max(last['max_mimic_error_rad']):.5f} rad; "
                                  f"load L/R {last['hand_load_n'][0][0]:.2f}/{last['hand_load_n'][0][1]:.2f} N; "
                                  f"object {last['object_lift_m'][0]*1000:.3f} mm\n")
                panel.status.text=(f"{row['status']} | steps {steps}\n"
                    f"LEFT target {target[0]:.2f} N / measured {load[0]:.2f} N\n"
                    f"RIGHT target {target[1]:.2f} N / measured {load[1]:.2f} N\n"
                    f"Wrist request {c.lift_current*1000:.1f} mm / actual L/R {actual[0]:.2f}/{actual[1]:.2f} mm\n"
                    f"Object actual {row['object_lift_m']*1000:.3f} mm\n"+
                    balance_text+'\n'.join(digit_text)+'\n'+
                    f"All FOUR digits below: {row['digit_contacts']['all_four_digits_below_axis']}\n"
                    f"Thumb stable L/R: {row['thumb_stability']['thumb_force_stable']}; grasp stable: {row['grasp_stable']}\n"
                    f"Lift criterion: {row['lift_criterion_met']}\n"+failure_text+notice)
                write_json(args.output/'manual_status.json',row)
                if c.pending_save:
                    c.pending_save=False
                    snapshot=dict(row,utc=datetime.now(timezone.utc).isoformat(),
                        settings=dict(joint_target_rad=c.joints.tolist(),force_n=c.force.tolist(),
                                      palms_m=c.palms.tolist(),lift_height_m=c.lift_goal,lift_speed_m_s=c.lift_speed))
                    with (args.output/'manual_snapshots.jsonl').open('a') as stream:
                        stream.write(json.dumps(snapshot,allow_nan=False)+'\n')
                    notice='Settings saved to manual_snapshots.jsonl (unqualified candidate).'
                last_status=now
            time.sleep(max(.001,dt-(time.monotonic()-start)))
    finally:
        panel.window.destroy()
    return dict(status='manual_closed',manual_control=True,control_steps=steps,grasp_qualified=False)
