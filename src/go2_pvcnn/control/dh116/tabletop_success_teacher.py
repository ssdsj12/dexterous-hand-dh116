"""Measured-contact teacher search through the unchanged twenty policy actions."""
import json
import os
import time
import torch
from .tabletop_control_probe import joint_actions, force_action, ProbeRecorder
from .tabletop_training import write_json
from .tabletop_demonstrations import SuccessEpisodeRecorder


class TeacherController:
    def __init__(self,cases,reference,lower,center,upper,device):
        self.cases=cases
        if not cases or len({c['name'] for c in cases})!=len(cases):
            raise ValueError('teacher cases need unique names')
        self.base=torch.zeros((len(cases),20),device=device,dtype=torch.float32)
        depths=[]
        for i,c in enumerate(cases):
            z=float(c['contact_z_m']); depths.append(z)
            if not -.004<=z<0:raise ValueError('contact target outside downward search')
            offset=torch.as_tensor(c.get('joint_offset_rad',[0.]*12),device=device,dtype=lower.dtype)
            if offset.shape!=(12,) or not torch.isfinite(offset).all():raise ValueError('invalid joint offsets')
            target=(reference.to(device)+offset).clamp(lower,upper)
            self.base[i,:12]=joint_actions(target,lower,center,upper)
            forces=c['force_n']
            if len(forces)!=2:raise ValueError('two force targets required')
            self.base[i,18:]=torch.tensor([force_action(float(f),12.) for f in forces],device=device)
            xy=torch.as_tensor(c.get('xy_m',[[0.,0.],[0.,0.]]),device=device)
            if xy.shape!=(2,2) or not torch.isfinite(xy).all() or (xy.abs()>.000125001).any():
                raise ValueError('XY target exceeds weighted policy envelope')
            self.base[i,12:14]=xy[0]/.003;self.base[i,15:17]=xy[1]/.003
        self.target=torch.tensor(depths,device=device,dtype=torch.float32)
        self.initial=self.target.clone();self.z=self.initial.clone()

    def reset(self,done):
        self.z[done]=self.initial[done]

    def actions(self,measurement):
        if 'contact_valid' in measurement and 'contact_center_relative_m' in measurement:
            centers=measurement['contact_center_relative_m'][...,2].to(self.z.dtype)
            valid=measurement['contact_valid'].bool()&torch.isfinite(centers)
            error=torch.where(valid,self.target[:,None]-centers,0.).sum(1)/valid.sum(1).clamp_min(1)
            self.z=(self.z+error.clamp(-.002/60.,.002/60.)).clamp(-.004,0.)
        actions=self.base.clone()
        actions[:,14]=self.z/.004;actions[:,17]=actions[:,14]
        return actions


class TeacherRecorder(ProbeRecorder):
    """Probe physics evidence plus last live sample (before automatic reset)."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.evidence={}

    def after_step(self,live,previous_phase):
        super().after_step(live,previous_phase)
        t=self.env.task;m=t.m;w=self.env.world
        valid=m['contact_valid'].bool()
        center=m['contact_center_relative_m'];normal=m['contact_normal_w']
        valid=valid&torch.isfinite(center).all(-1)&torch.isfinite(normal).all(-1)
        values=dict(contact_center_relative_m=torch.where(valid[...,None],center,0.),
            contact_normal=torch.where(valid[...,None],normal,0.),contact_valid=valid,
            normal_vertical_force_n=w.vertical_components[:,:,0].sum(1),
            lift_vertical_force_n=w.vertical_components.sum((1,2)),object_bottom_m=m['bottom'])
        if hasattr(t,'joint_requested_target'):
            values.update(joint_request_rad=t.joint_requested_target,
                joint_governed_rad=t.joint_governed_target,
                joint_submitted_rad=w.commands.flatten(1).float(),
                joint_measured_rad=m['q'],force_target_n=t.f_target,
                force_tracking_limited=getattr(t,'force_tracking_limited',t.f_blocked),
                contact_effort_nm=getattr(w,'effort_feedforward',torch.zeros_like(w.commands)))
        if hasattr(t,'close_grasp_hold'):
            values.update(close_hand_holding=t.close_grasp_hold.active,
                close_hand_ready=t.close_grasp_hold.ready,
                close_held_force_target=t.close_grasp_hold.force_target,
                close_held_palm_offset=t.close_grasp_hold.palm_offset)
        for k,v in values.items():
            if k not in self.evidence:self.evidence[k]=v.clone()
            else:self.evidence[k][live]=v[live]


def run_success_teacher(env,args,contract,obs):
    from .tabletop_teacher_motion import motion_snapshot,TeacherMotionView
    cases=json.loads(args.teacher_search_cases.read_text())['cases']
    if len(cases)!=env.n:raise ValueError('teacher case count must equal num_envs')
    reference=json.loads(args.grasp_initialization.read_text())['grasp_command']
    controller_type=TeacherController
    if getattr(env.task,'force_region',False):
        from .tabletop_force_teacher import ForceTeacherController
        controller_type=ForceTeacherController
    controller=controller_type(cases,torch.tensor(reference,device=env.device,dtype=torch.float64),
        env.task.lower,env.task.center,env.task.upper,env.device)
    trace=TeacherRecorder(env,cases,args.output,episode_limit=args.teacher_episodes)
    env.control_probe=trace
    recorder=SuccessEpisodeRecorder(env.n,observation_dim=obs['policy'].shape[1],action_dim=20,
                                   max_steps=args.teacher_control_steps)
    write_json(args.output/'tensor_contract.json',contract)
    write_json(args.output/'teacher_cases.json',dict(cases=cases,normalized_base_actions=controller.base.cpu().tolist()))
    current=obs['policy'];started=time.time();status='error';steps=0
    view=TeacherMotionView() if not getattr(args,'headless',True) else None
    profiler=None
    if os.environ.get('DH116_TEACHER_PROFILE')=='1':
        import cProfile
        profiler=cProfile.Profile()
        profiler.enable()
    try:
        for steps in range(1,args.teacher_control_steps+1):
            actions=(controller.actions(env.task.m,env.task) if getattr(env.task,'force_region',False)
                     else controller.actions(env.task.m))
            next_obs,rewards,dones,extras=env.step(actions)
            extras=dict(extras,physical_evidence=trace.evidence)
            recorder.append(current,actions,rewards,dones,extras)
            controller.reset(dones);current=next_obs
            if view is not None:view.update(env,motion_snapshot(env))
            if steps%30==0 or (env.episodes>=args.teacher_episodes).all():
                progress=dict(status='teacher_search',control_steps=steps,elapsed_s=time.time()-started,
                    episodes=int(env.episodes.sum()),successes=int(env.successes.sum()),
                    max_bottom_m=float(trace.peak_bottom.max()),velocity_diagnostics=env.velocity_diagnostics(),
                    motion=motion_snapshot(env),phases=torch.bincount(env.task.phase.long(),minlength=5).tolist())
                write_json(args.output/'monitor.json',progress)
                print(json.dumps(progress),flush=True)
            # A polled request exits outside Kit callbacks and saves once.
            if (args.output/'teacher_stop.request').exists():
                status='user_stopped'
                break
            if (env.episodes>=args.teacher_episodes).all():break
        if status!='user_stopped':status='completed'
    finally:
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(str(args.output/'performance.prof'))
        summary=trace.save(status)
        write_json(args.output/'teacher_search.json',dict(status=status,control_steps=steps,
            episodes=env.episodes.cpu().tolist(),successes=env.successes.cpu().tolist(),
            recorder=recorder.summary(),velocity_diagnostics=env.velocity_diagnostics(),motion=motion_snapshot(env),
            physics_summary=summary,qualification=dict(qualified=False)))
        if recorder.episodes:
            recorder.save(args.output/'teacher_success_pending.pt',contract=contract,
                qualification=dict(qualified=False,physical_validation=False,fresh_resets=False,successful_replays=0))
        if view is not None:view.close()
    return dict(status=status,teacher_search=True,teacher_successes=int(env.successes.sum()),
                teacher_control_steps=steps,student_training_started=False)
