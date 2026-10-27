"""Inference-only controlled grasp experiments using the production tensor loop."""
import json
import math
import time
import numpy as np
import torch
from .tabletop_training import write_json


FORCE_DIRECTION_TOTAL_TARGET_N=10.
FORCE_DIRECTION_HAND_TARGET_N=FORCE_DIRECTION_TOTAL_TARGET_N/2.


def joint_actions(target,lower,center,upper):
    if not torch.isfinite(target).all() or ((target<lower)|(target>upper)).any():
        raise ValueError('fixed grasp target outside joint envelope')
    delta=target-center
    scale=torch.where(delta>=0,upper-center,center-lower)
    if (scale<=0).any():raise ValueError('degenerate joint envelope')
    return delta/scale


def force_action(force,maximum):
    if not math.isfinite(force) or not .1<=force<=maximum or maximum<=.1:
        raise ValueError('force target outside configured limits')
    return 2*(force-.1)/(maximum-.1)-1


def teacher_palm_xy_candidates():
    """Six bounded palm translations for teacher-position screening."""
    return [(-.0015, 0.), (.0015, 0.), (0., -.0015),
            (0., .0015), (-.0015, .0015), (.0015, -.0015)]


def teacher_palm_z_down_candidates():
    """Downward-only palm offsets for the below-axis pinch diagnostic."""
    return [-.004, -.003, -.002, -.0015, -.001, -.0005]


def teacher_contact_z_candidates():
    return [-.004, -.003, -.002, -.0015, -.001, -.0005]


def check_probe_contract(saved,current,*,allow_controller_change=False):
    left=dict(saved);right=dict(current)
    left.pop('num_envs',None);right.pop('num_envs',None)
    if allow_controller_change:
        for key in ('grasp_controller','palm_reposition','force_direction_reposition',
                    'training_failure_handling','close_downward_penalty',
                    'minibatches','palm_exploration'):
            left.pop(key,None);right.pop(key,None)
    if left!=right:
        changed=[k for k in left.keys()|right.keys() if left.get(k)!=right.get(k)]
        raise ValueError('probe contract mismatch: '+str(changed))


def longest_dwell(mask,dt):
    best=current=0
    for value in mask:
        current=current+1 if value else 0
        best=max(best,current)
    return best*dt


class ProbeRecorder:
    def __init__(self,env,cases,output,episode_limit=3):
        self.env=env;self.cases=cases;self.output=output
        self.episode_limit=episode_limit
        self.peak_bottom=torch.full((env.n,),.30,device=env.device,dtype=torch.float64)
        self.frames=[];self.columns=[];self.before={}

    def before_step(self):
        t=self.env.task
        from .linear_coupling import ACTIVE,MIMIC
        recovery=[]
        for hand in range(2):
            for j,(_,master,ratio,_) in enumerate(MIMIC):
                k=6*hand+ACTIVE.index(master)
                recovery.append((t.m['q'][:,k]+t.m['mimic_errors'][:,5*hand+j]/ratio).clamp(t.low[k],t.high[k]))
        threshold=torch.full((t.n,1),.004,device=t.device,dtype=torch.float64)
        if hasattr(t,'coordinated_control'):
            from .tabletop_coordinated_grasp import CONTRACT
            threshold[t.coordinated_control]=CONTRACT['severe_coupling_rad']
        self.before=dict(pre_load=t.m['hand_object_loads'].clone(),
            pre_target=t.f_target.clone(),pre_mimic=t.m['mimic_errors'].abs().amax(1),
            pre_tracking=(t.command-t.m['q']).abs().amax(1),pre_slip=t.slip.clone(),
            pre_offset=t.offset.clone(),pre_signed_mimic=t.m['mimic_errors'].clone(),
            recovery_active=t.m['mimic_errors'].abs()>=threshold,
            recovery_target=torch.stack(recovery,1))

    def after_step(self,live,previous_phase):
        e=self.env;t=e.task;w=e.world;m=t.m
        eligible=live&(e.episodes<self.episode_limit)
        self.peak_bottom=torch.maximum(self.peak_bottom,torch.where(eligible,m['bottom'],self.peak_bottom))
        fields=dict(eligible=eligible,episode=e.episodes,
            phase_before=previous_phase,phase_after=t.phase,elapsed=t.elapsed,
            failure=t.failure,success=t.success,bottom=m['bottom'],
            speed=m['object_velocity'][:,:3].norm(dim=1),slip=t.slip,
            weight=w.object_weight_n,force=w.vertical_components.sum((1,2)),
            vertical_components=w.vertical_components,load=m['hand_object_loads'],
            contact_center_relative_m=m['contact_center_relative_m'],
            contact_normal=m.get('contact_normal_w',torch.full_like(m['contact_center_relative_m'],float('nan'))),
            contact_valid=m['contact_valid'],
            contact_all_below_centerline=m['contact_all_below_centerline'],
            contact_count=m.get('contact_count',torch.zeros_like(m['contact_valid'],dtype=torch.long)),
            target=t.f_target,force_offset=t.f_offset,offset=t.offset-w.initial_offset,
            palms=w.palms,palm_target=t.palm_targets,palm_clipped=t.palm_clipped,
            policy_palm=t.actions[:,12:18],command=t.command,q=m['q'],qd=m['qd'],
            signed_mimic=m['mimic_errors'],
            hand_clearance=m['hand_clearance'],pad_relative=m['pad_relative'],
            object_velocity=m['object_velocity'],ready=t.grip_force_ready(m['hand_object_loads']),
            gates=t.lift_diagnostic_masks,**self.before)
        # These are forces on the OBJECT. Contact geometry comes from the PhysX
        # contact report; link centers remain separate kinematic telemetry.
        sensors=w.w.sensors
        fields['link_normal_z']=-torch.stack([s.data.force_matrix_w[:,0,0,2] for s in sensors],1)
        fields['link_friction_z']=-torch.stack([s.data.friction_forces_w[:,0,0,2] for s in sensors],1)
        fields['body_z']=w.poses[...,2]-w.origins[:,None,None,2]
        fields['palm_vz']=torch.stack([h.data.body_lin_vel_w[:,h.body_names.index('base_link'),2] for h in w.hands],1)
        fields['palm_z']=w.roots[:,:,2]-w.origins[:,None,2]
        if hasattr(t,'palm_reposition'):
            fields['reposition_state']=t.palm_reposition.state
            fields['reposition_attempts']=t.palm_reposition.attempts
            fields['reposition_aborted']=t.palm_reposition.aborted
            fields['reposition_goal_clipped']=t.palm_reposition.goal_clipped
            for name in ('force_before','force_release','force_move','force_reclose'):
                if hasattr(t.palm_reposition,name):
                    fields['reposition_'+name]=getattr(t.palm_reposition,name)
        if getattr(w.w,'mounts',None) is not None:
            fields['wrist_q']=torch.stack([h.data.joint_pos[:,mount.ids] for h,mount in zip(w.hands,w.w.mounts)],1)
            fields['wrist_target']=torch.stack([h.data.joint_pos_target[:,mount.ids] for h,mount in zip(w.hands,w.w.mounts)],1)
            fields['wrist_effort']=torch.stack([h.data.applied_torque[:,mount.ids] for h,mount in zip(w.hands,w.w.mounts)],1)
            fields['wrist_anchor']=torch.stack([h.data.root_pose_w for h in w.hands],1)
            fields['wrist_pose_target']=w.root_targets

        fields['body_vz']=torch.stack([h.data.body_lin_vel_w[...,2] for h in w.hands],1)
        values=[];columns=[]
        for name,value in fields.items():
            flat=value.reshape(e.n,-1)
            columns.extend([name] if flat.shape[1]==1 else [f'{name}:{i}' for i in range(flat.shape[1])])
            values.append(flat.double())
        if self.columns and self.columns!=columns:raise ValueError('probe telemetry shape changed')
        self.columns=columns
        self.frames.append(torch.cat(values,1).cpu().numpy())

    def save(self,status):
        if not self.frames:return dict(status=status,episodes=[])
        from .tabletop_grasp_shaping import LIFT_GATE_NAMES
        from .linear_coupling import MIMIC
        data=np.stack(self.frames);index={k:i for i,k in enumerate(self.columns)}
        np.savez_compressed(self.output/'control_trace.npz',data=data,columns=np.array(self.columns))
        result=[];dt=self.env.world.dt
        for env_id,case in enumerate(self.cases):
            for ep in range(self.episode_limit):
                rows=data[:,env_id];rows=rows[(rows[:,index['eligible']]>0)&(rows[:,index['episode']]==ep)]
                if not len(rows):continue
                c=lambda key:rows[:,index[key]]
                lifting=c('phase_before')==3
                stable=lifting&(c('ready')>0)&(c('speed')<.03)&(c('slip')<=.02)
                gate_counts={key:float(c(f'gates:{i}').sum()/max(1,lifting.sum()))
                             for i,key in enumerate(LIFT_GATE_NAMES) if i}
                early=longest_dwell(stable&(c('bottom')>=.303),dt)
                terminal=bool(c('failure')[-1] or c('success')[-1])
                result.append(dict(env=env_id,case=case,episode=ep,completed=terminal,
                    elapsed_s=float(c('elapsed')[-1]),failure_code=int(c('failure')[-1]),
                    success=bool(c('success')[-1]),max_clearance_m=float(c('bottom').max()-.30),
                    early_3mm_dwell_s=early,early_3mm_pass=early>=.1-1e-12,
                    lift_steps=int(lifting.sum()),lift_gate_fractions=gate_counts,
                    longest_allowed_lift_s=longest_dwell(c('gates:1')>0,dt),
                    max_command_lift_m=float(c('offset').max()),
                    mean_force_n=float(c('force').mean()),peak_up_force_n=float(c('force').max()),
                    above_weight_fraction=float((c('force')>c('weight')).mean()),
                    mean_load_n=[float(c(f'load:{i}').mean()) for i in range(2)],
                    max_load_n=[float(c(f'load:{i}').max()) for i in range(2)],
                    palm_z_range_m=[[float(c(f'palms:{i}').min()),float(c(f'palms:{i}').max())] for i in (2,5)]))
        report=dict(status=status,episodes=result,physics_dt_s=dt,
            position_semantics='PhysX contact positions and force-weighted normals; pad_relative remains a proxy',
            recovery_semantics='recovery_target is the projected center, NOT final governed/slewed command; active uses legacy .004 or coordinated severe .012',
            force_semantics='signed hand-to-object normal+friction; load is summed normal magnitudes',
            repeat_semantics='same reset, repeatability only, not randomized generalization',
            gate_timing='pressure uses pre_load with updated target (not pre_target); pre_mimic/pre_tracking precede control; load/force are post-physics',
            completed_slots='continue guarded simulation; only requested episodes analyzed',
            episode_limit=self.episode_limit,
            body_names=[list(h.body_names) for h in self.env.world.hands],
            mimic_joint_order=[row[0] for row in MIMIC],
            sensor_paths=[list(s.body_physx_view.prim_paths) for s in self.env.world.w.sensors],
            gates=list(LIFT_GATE_NAMES),cases=self.cases)
        write_json(self.output/'control_probe.json',report)
        return report


def run_control_probe(env,algorithm,args,contract,obs):
    from .tabletop_tensor_training import VERSION,observations
    saved=torch.load(args.control_probe_checkpoint,map_location=env.device,weights_only=False)
    if saved['version']!=VERSION:raise ValueError('not a tensor checkpoint')
    mode=getattr(args,'control_probe_mode','grasp')
    check_probe_contract(saved['contract'],contract,
                         allow_controller_change=mode in ('force_direction','grasp_teacher_contact_z_sweep'))
    algorithm.policy.load_state_dict(saved['policy']);algorithm.policy.eval()
    if not all(torch.isfinite(v).all() for v in algorithm.policy.state_dict().values()):
        raise ValueError('nonfinite probe policy')
    write_json(args.output/'tensor_contract.json',contract)
    write_json(args.output/'probe_source.json',dict(checkpoint=str(args.control_probe_checkpoint),
        iteration=saved['iteration'],training_envs=saved['contract']['num_envs'],probe_envs=env.n,
        optimizer_updates=0,changed_contract_fields=['num_envs']))
    t=env.task
    authored=torch.as_tensor(json.loads(args.grasp_initialization.read_text())['grasp_command'],device=env.device)
    joints=joint_actions(authored,t.lower,t.center,t.upper)
    fixed=torch.zeros((env.n,20),device=env.device,dtype=torch.float64)
    teacher_z_offsets=None
    episode_limit=getattr(args,'control_probe_episodes',3)
    original_control=None
    if mode=='wrist_motion':
        cases=[f'wrist_free_motion_{i}' for i in range(env.n)]
        original_control=t.control
        t.command.copy_(t.m['q'])
        t.lift_diagnostic_masks=torch.zeros((env.n,8),device=env.device,dtype=torch.bool)
        def free_motion(actions,live,dt):
            t.offset[live]=(t.offset[live]+.01*dt).clamp_max(env.world.initial_offset+.03)
            t.palm_actions.zero_();t.palm_targets.zero_();t.lift_diagnostic_masks.zero_()
            return live
        t.control=free_motion
        write_json(args.output/'diagnostic_override.json',dict(mode=mode,
            scope='free wrist translation with fingers held at reset pose',speed_m_s=.01,
            hard_guards_changed=False,optimizer_updates=0))
    elif mode=='loaded_palm':
        from .tabletop_loaded_palm_motion import LoadedPalmReposition,CONTRACT
        if not getattr(t,'coordinated_all',False):raise ValueError('loaded palm probe requires coordinated controller')
        mask=torch.tensor([False,True]*4,device=env.device)
        if getattr(args,'control_probe_swap',False):mask=~mask
        t.palm_reposition=LoadedPalmReposition(mask)
        cases=[]
        for i in range(8):
            name=('fixed_4N','fixed_8N','ppo_mean_rep0','ppo_mean_rep1')[i//2]
            cases.append(name+'_candidate'+str(int(mask[i])))
            if i<4:
                fixed[i,:12]=joints
                fixed[i,18:]=force_action(4. if i<2 else 8.,t.parameters.max_force_n)
        write_json(args.output/'diagnostic_override.json',dict(mode=mode,
            selected_environments=mask.cpu().tolist(),palm_reposition=CONTRACT,
            hard_guards_changed=False,rewards_changed=False,optimizer_updates=0))
    elif mode=='force_direction':
        from .tabletop_force_direction_reposition import ForceDirectionReposition,CONTRACT
        if not getattr(t,'coordinated_all',False):raise ValueError('force direction probe requires coordinated controller')
        mask=torch.tensor([False,True]*(env.n//2),device=env.device)
        if getattr(args,'control_probe_swap',False):mask=~mask
        t.palm_reposition=ForceDirectionReposition(mask)
        t.palm_reposition.goal_table_clearance_m=.0003
        t.diagnostic_palm_descent_clearance_m=.0003
        t.force_direction_reposition=True
        fixed[:,:12]=joints
        fixed[:,18:]=force_action(FORCE_DIRECTION_HAND_TARGET_N,t.parameters.max_force_n)
        cases=[f'force_direction_candidate{int(mask[i])}' for i in range(env.n)]
        write_json(args.output/'diagnostic_override.json',dict(mode=mode,
            selected_environments=mask.cpu().tolist(),palm_reposition=CONTRACT,
            diagnostic_goal_table_clearance_m=.0003,
            fixed_force_target_n=FORCE_DIRECTION_HAND_TARGET_N,
            fixed_force_total_target_n=FORCE_DIRECTION_TOTAL_TARGET_N,
            hard_guards_changed=False,rewards_changed=False,optimizer_updates=0))
    elif mode=='grasp_teacher_contact_z_sweep':
        if not getattr(t,'coordinated_all',False):
            raise ValueError('measured contact Z probe requires coordinated controller')
        teacher_z_offsets=torch.tensor(teacher_contact_z_candidates(),device=env.device,dtype=torch.float64)
        t.contact_target_z_m=teacher_z_offsets[:,None].expand(-1,2).clone()
        t.contact_z_step_m=.0005
        fixed=fixed[:len(teacher_z_offsets)]
        fixed[:,:12]=joints
        fixed[:,18:]=force_action(FORCE_DIRECTION_HAND_TARGET_N,t.parameters.max_force_n)
        cases=[f'contact_z{float(z):g}m' for z in teacher_z_offsets]
        write_json(args.output/'diagnostic_override.json',dict(mode=mode,
            contact_target_z_m=teacher_z_offsets.cpu().tolist(),
            contact_target_semantics='force-weighted PhysX contact center relative to object center',
            valid_contact_gate='all valid contacts below centerline for both hands',
            fixed_force_target_n=FORCE_DIRECTION_HAND_TARGET_N,
            fixed_force_total_target_n=FORCE_DIRECTION_TOTAL_TARGET_N,
            max_correction_m=t.contact_z_step_m,
            hard_guards_changed=False,rewards_changed=False,optimizer_updates=0))
    elif mode=='coordinated':
        from .tabletop_coordinated_grasp import CONTRACT
        settings=[(f,enabled) for _ in range(2) for f in (4.,8.) for enabled in (False,True)]
        mask=torch.tensor([s[1] for s in settings],device=env.device)
        t.enable_coordinated_control(mask)
        cases=[]
        for i,(force,enabled) in enumerate(settings):
            fixed[i,:12]=joints;fixed[i,18:]=force_action(force,t.parameters.max_force_n)
            cases.append(f'fixed_{force:g}N_coordinated{int(enabled)}_rep{i//4}')
        write_json(args.output/'diagnostic_override.json',dict(mode=mode,
            selected_environments=mask.cpu().tolist(),controller=CONTRACT,
            hard_guards_changed=False,rewards_changed=False))
    elif mode=='bounded_thumb':
        settings=[(f,cap) for _ in range(2) for f in (4.,8.) for cap in (0.,.008)]
        t.diagnostic_thumb_position_lead=torch.tensor([s[1] for s in settings],device=env.device,dtype=torch.float64)
        cases=[]
        for i,(force,cap) in enumerate(settings):
            fixed[i,:12]=joints;fixed[i,18:]=force_action(force,t.parameters.max_force_n)
            cases.append(f'fixed_{force:g}N_thumb_lead{cap:g}_rep{i//4}')
        write_json(args.output/'diagnostic_override.json',dict(mode=mode,
            thumb_position_lead_rad=t.diagnostic_thumb_position_lead.cpu().tolist(),
            active_joint_indices=[1,7],lift_mimic_threshold_rad=.004,
            priority='joint envelope and command slew precede position-lead bound',
            hard_guards_changed=False,rewards_changed=False))
    elif mode=='lift_gate':
        settings=[(f,threshold) for _ in range(2) for f in (4.,8.) for threshold in (.004,.008)]
        t.diagnostic_lift_mimic_threshold=torch.tensor([s[1] for s in settings],device=env.device,dtype=torch.float64)
        cases=[]
        for i,(force,threshold) in enumerate(settings):
            fixed[i,:12]=joints;fixed[i,18:]=force_action(force,t.parameters.max_force_n)
            cases.append(f'fixed_{force:g}N_lift_mimic{threshold:g}_rep{i//4}')
        write_json(args.output/'diagnostic_override.json',dict(mode=mode,
            lift_mimic_threshold_rad=t.diagnostic_lift_mimic_threshold.cpu().tolist(),
            joint_recovery_threshold_rad=.004,hard_guards_changed=False,rewards_changed=False))
    elif mode in ('grasp','grasp_fixed','grasp_z_sweep','grasp_teacher_z_sweep','grasp_teacher_z_fine','grasp_teacher_z_confirm','grasp_teacher_z_down_sweep','grasp_teacher_contact_z_sweep','grasp_teacher_xy_sweep'):
        if mode == 'grasp_teacher_z_down_sweep':
            t.diagnostic_palm_descent_clearance_m=.0003
            write_json(args.output/'diagnostic_override.json',dict(mode=mode,
                palm_descent_clearance_m=.0003,
                scope='probe-only downward Z exploration; ordinary training/demo floor unchanged',
                hard_guards_changed=False,rewards_changed=False))
        cases=['ppo_mean_0','ppo_mean_1']
        settings=[(f,z) for z in (0.,-.001) for f in (4.,8.,12.)]
        if mode=='grasp':
            for i,(force,z) in enumerate(settings,2):
                fixed[i,:12]=joints;fixed[i,18:]=force_action(force,t.parameters.max_force_n)
                fixed[i,14]=fixed[i,17]=z/.004
                cases.append(f'fixed_{force:g}N_z{z:g}m')
        elif mode=='grasp_z_sweep':
            settings=[(8.,z) for z in (-.002,-.0015,-.001,-.0005,0.,.0005)]
            fixed=fixed[:len(settings)]
            for i,(force,z) in enumerate(settings):
                fixed[i,:12]=joints;fixed[i,18:]=force_action(force,t.parameters.max_force_n)
                fixed[i,14]=fixed[i,17]=z/.004
            cases=[f'fixed_{force:g}N_z{z:g}m' for force,z in settings]
        elif mode in ('grasp_teacher_z_sweep','grasp_teacher_z_fine','grasp_teacher_z_confirm'):
            values=(-.002,-.0015,-.001,-.0005,0.,.0005) if mode=='grasp_teacher_z_sweep' else ((-.00125,-.00075,-.00025,.00025,.00075,.00125) if mode=='grasp_teacher_z_fine' else (.00075,)*6)
            teacher_z_offsets=torch.tensor(values,device=env.device,dtype=torch.float64)
            fixed=fixed[:len(teacher_z_offsets)]
            cases=[f'teacher_z{float(z):g}m' for z in teacher_z_offsets]
        elif mode == 'grasp_teacher_z_down_sweep':
            teacher_z_offsets=torch.tensor(teacher_palm_z_down_candidates(),device=env.device,dtype=torch.float64)
            fixed=fixed[:len(teacher_z_offsets)]
            cases=[f'teacher_z{float(z):g}m' for z in teacher_z_offsets]
        elif mode == 'grasp_teacher_xy_sweep':
            teacher_xy_offsets=torch.tensor(teacher_palm_xy_candidates(),device=env.device,dtype=torch.float64)
            fixed=fixed[:len(teacher_xy_offsets)]
            cases=[f'teacher_xy{x:+.4f}_{y:+.4f}m' for x,y in teacher_palm_xy_candidates()]
        else:
            fixed=fixed[:len(settings)]
            for i,(force,z) in enumerate(settings):
                fixed[i,:12]=joints;fixed[i,18:]=force_action(force,t.parameters.max_force_n)
                fixed[i,14]=fixed[i,17]=z/.004
            cases=[f'fixed_{force:g}N_z{z:g}m' for force,z in settings]
    if getattr(args,'palm_transport','fixed_pose')=='driven_carriage':
        from .tabletop_palm_mount import CONTRACT as MOUNT_CONTRACT
        write_json(args.output/'palm_transport_override.json',MOUNT_CONTRACT)
    recorder=ProbeRecorder(env,cases,args.output,episode_limit);env.control_probe=recorder
    started=time.monotonic();status='control_probe_incomplete'
    try:
        with torch.inference_mode():
            # Force-direction mode validates the close-time release/move/reclose
            # state machine.  Stop before the long lift tail, whose existing
            # coupled-joint guard is diagnosed by the training run itself.
            tick_limit = (240 if mode == 'force_direction' else
                          (180 if mode=='wrist_motion' else (15*episode_limit+2)*60))
            for tick in range(tick_limit):
                action=fixed.clone()
                if mode=='grasp':action[:2]=algorithm.policy.act_inference(obs)[:2].clamp(-1.,1.)
                if mode in ('grasp_teacher_z_sweep','grasp_teacher_z_fine','grasp_teacher_z_confirm','grasp_teacher_z_down_sweep'):
                    action[:]=algorithm.policy.act_inference(obs).clamp(-1.,1.)
                    action[:,14]=teacher_z_offsets/.004
                    action[:,17]=teacher_z_offsets/.004
                if mode=='grasp_teacher_contact_z_sweep':
                    action=algorithm.policy.act_inference(obs).clamp(-1.,1.)
                    action[:,:12]=fixed[:,:12]
                    action[:,12:18]=0.
                    action[:,18:]=fixed[:,18:]
                if mode == 'grasp_teacher_xy_sweep':
                    action[:]=algorithm.policy.act_inference(obs).clamp(-1.,1.)
                    action[:,12]=teacher_xy_offsets[:,0]/.003
                    action[:,13]=teacher_xy_offsets[:,1]/.003
                    action[:,15]=teacher_xy_offsets[:,0]/.003
                    action[:,16]=teacher_xy_offsets[:,1]/.003
                if mode=='loaded_palm':action[4:]=algorithm.policy.act_inference(obs)[4:].clamp(-1.,1.)
                value,_,_,_=env.step(action)
                obs=observations(value)
                if tick%60==0:
                    progress=dict(status='control_probe',control_ticks=tick+1,
                                  episodes=env.episodes.cpu().tolist(),wall_seconds=time.monotonic()-started,
                                  max_clearance_mm=((recorder.peak_bottom-.30)*1000).cpu().tolist())
                    write_json(args.output/'monitor.json',progress)
                    print('CONTROL_PROBE_PROGRESS',json.dumps(progress),flush=True)
                if bool((env.episodes>=episode_limit).all()):
                    status='control_probe_complete';break
            else:
                if mode == 'force_direction':
                    status='force_direction_probe_complete'
    finally:
        if mode=='wrist_motion' and tick==179:status='wrist_motion_probe_complete'
        report=recorder.save(status)
        if original_control is not None:t.control=original_control
        del env.control_probe
        if hasattr(t,'diagnostic_lift_mimic_threshold'):del t.diagnostic_lift_mimic_threshold
        if hasattr(t,'diagnostic_thumb_position_lead'):del t.diagnostic_thumb_position_lead
        if hasattr(t,'diagnostic_palm_descent_clearance_m'):del t.diagnostic_palm_descent_clearance_m
        if hasattr(t,'contact_target_z_m'):t.contact_target_z_m=None
        if hasattr(t,'contact_z_step_m'):del t.contact_z_step_m
        if hasattr(t,'force_direction_reposition'):del t.force_direction_reposition
        if hasattr(t,'palm_reposition') and hasattr(t.palm_reposition,'goal_table_clearance_m'):
            del t.palm_reposition
        if hasattr(t,'coordinated_control'):
            del t.coordinated_control;del t.coordinated_all;del t.control_pad_anchor
    return dict(status=status,control_probe=report,grasp_qualified=any(x['success'] for x in report['episodes']))
