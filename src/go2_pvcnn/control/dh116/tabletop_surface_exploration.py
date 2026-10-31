"""Free fingers/wrists with no fixed Newton goal and surface-based sharing."""
import torch
from .tabletop_surface_force_sharing import surface_targets,CONTRACT as SHARING

MEASUREMENT_KEYS=('effective_pad_loads_n','effective_pad_vectors_w','effective_pad_points_w',
    'effective_pad_valid','rejected_contact_loads_n','effective_pad_depth_m','pad_reference_points_w')
REWARD_NAMES=('surface_between_faces','surface_within_faces','surface_missing_opposition','ordinary_collision')
STABLE_HOLD_REWARD_CONTRACT=dict(version='surface_stable_hold_reward_v2',
    lift_reward_contact_gate='opposed_effective_pads;ordinary_fraction<=0.05;depth<=0.0008m',
    hold_reward='120*(continuous_native_hold_s/5)^2_points_per_s',
    duration_source='last_completed_native_control_window',
    slip_reward='-160*clamp((measured_fixed_anchor_drift_m/0.0005)^2,0,4)_points_per_s',
    slip_tracking_minimum_lift_m=.003,
    slip_tracking='separate_read_only_native_window;events_never_count_as_success',
    success_hold_s=5.,success_slip_m=.0005,motion_gate=False)


def enable_stable_hold_rewards(t):
    """Observe early real slip; keep the actual 30 mm/5 s success independent."""
    from .tabletop_surface_success import SurfaceLiftSuccess
    t.surface_stable_hold_rewards=True
    t.surface_slip_tracking=SurfaceLiftSuccess(t.n,t.device,t.surface_reference_bottom_m,
        dtype=t.command.dtype,minimum_lift_m=.003)
    for name in ('surface_hold_progress','surface_slip_window'):
        t.terms[name]=torch.zeros_like(t.reward)


def clear_reset_contacts(data,ids):
    """Native manifolds are stale until the next step after a selected reset."""
    for name in ('effective_pad_loads_n','effective_pad_vectors_w','effective_pad_points_w',
                 'effective_pad_valid','rejected_contact_loads_n','effective_pad_depth_m'):
        data[name][ids]=0
    if 'effective_pad_body_ids' in data:data['effective_pad_body_ids'][ids]=-1


def validate_options(args):
    if (getattr(args,'surface_contact_history',False)
            and not getattr(args,'surface_force_sharing',False)):
        raise ValueError('causal contact history requires surface mode')
    if (getattr(args,'surface_noise_mode','legacy')!='legacy'
            and not getattr(args,'surface_force_sharing',False)):
        raise ValueError('surface noise configuration requires surface mode')
    imitation=getattr(args,'surface_imitation_mode','legacy')
    if imitation!='legacy' and (not getattr(args,'surface_force_sharing',False)
            or not getattr(args,'surface_demonstrations',None)):
        raise ValueError('stable hold imitation requires native demonstrations and surface mode')
    evaluation=getattr(args,'surface_student_eval_steps',0)
    if not isinstance(evaluation,int) or not 0<=evaluation<=3600:
        raise ValueError('surface student evaluation budget must be 0..3600 control steps')
    if getattr(args,'surface_mpc',False) and not evaluation:
        raise ValueError('surface MPC is currently inference-only; requires student evaluation')
    if evaluation and (not getattr(args,'surface_force_sharing',False) or
                       not (getattr(args,'surface_student_actor',None) or getattr(args,'resume',None))):
        raise ValueError('surface evaluation requires a real surface student actor or PPO checkpoint')
    surface_features=tuple(getattr(args,name,None) for name in
        ('surface_teacher_replay','surface_student_actor','surface_demonstrations'))
    if any(surface_features) and not getattr(args,'surface_force_sharing',False):
        raise ValueError('surface teacher/student options require --surface-force-sharing')
    if surface_features[0] and (any(surface_features[1:]) or getattr(args,'resume',None) or evaluation):
        raise ValueError('teacher replay and student optimization are distinct runs')
    if surface_features[1] and getattr(args,'resume',None):
        raise ValueError('student actor initialization and PPO resume are distinct sources')
    if getattr(args,'native_sensor_batch',False) and not getattr(args,'surface_force_sharing',False):
        raise ValueError('--native-sensor-batch requires --surface-force-sharing')
    if not getattr(args,'surface_force_sharing',False):return
    if not getattr(args,'free_wrist_grasp',False):
        raise ValueError('--surface-force-sharing requires --free-wrist-grasp')
    if getattr(args,'initialize_from',None) or getattr(args,'actor_initialization',None):
        raise ValueError('surface 24D mode requires its own trained contract; no silent legacy transfer')


def validate_startup_measurements(data,n):
    shapes=dict(effective_pad_loads_n=(n,2,5),effective_pad_vectors_w=(n,2,5,3),
        effective_pad_points_w=(n,2,5,3),effective_pad_valid=(n,2,5),
        effective_pad_body_ids=(n,2,5),effective_pad_depth_m=(n,2,5),
        pad_reference_points_w=(n,2,5,3),rejected_contact_loads_n=(n,2),effective_pad_bad_env=(n,))
    report={}
    for key,shape in shapes.items():
        value=data[key]
        if not isinstance(value,torch.Tensor) or value.shape!=shape or not torch.isfinite(value).all():
            raise ValueError('invalid native pad startup measurement: '+key)
        if key in ('effective_pad_valid','effective_pad_bad_env'):
            if value.dtype!=torch.bool:raise ValueError('boolean native pad mask required: '+key)
        elif key=='effective_pad_body_ids':
            if value.dtype not in (torch.long,torch.int):raise ValueError('integer pad body ids required')
        elif not value.is_floating_point():raise ValueError('floating native pad data required: '+key)
        if key in ('effective_pad_loads_n','effective_pad_depth_m','rejected_contact_loads_n') and (value<0).any():
            raise ValueError('negative native pad force or depth: '+key)
        report[key]=dict(shape=list(shape),dtype=str(value.dtype),finite=True,
            evidence='FK_geometry_not_contact' if key=='pad_reference_points_w' else 'native_manifold_or_acceptance_mask')
    if data['effective_pad_bad_env'].any():raise ValueError('invalid pad environment at startup')
    if ((data['effective_pad_loads_n']>0)!=data['effective_pad_valid']).any():
        raise ValueError('native effective pad validity must correspond to measured positive load')
    if ((data['effective_pad_body_ids']>=0)!=data['effective_pad_valid']).any():
        raise ValueError('native pad body id must correspond to accepted contact')
    return report


def enable(t,w,*,redistribution=False):
    if not getattr(t,'free_wrist_grasp',False):raise ValueError('surface sharing requires free wrists')
    t.surface_force_sharing=True;t.action_size=24;t.surface_world=w
    t.surface_policy_owned_targets=True
    t.episode_timeout_s=30.
    t.actions=torch.zeros((t.n,24),device=t.device,dtype=t.command.dtype)
    t.previous_actions=torch.zeros_like(t.actions)
    for name in ('actions','previous_actions'):
        t._seed[name]=getattr(t,name)[0].clone()
    for name in ('f_target','f_requested','f_offset','f_boost'):
        getattr(t,name).zero_();t._seed[name]=getattr(t,name)[0].clone()
    t.surface_reference_bottom_m=torch.as_tensor(getattr(w,'lift_reference_bottom_m',.30),device=t.device,dtype=t.command.dtype)
    t.surface_success_hold_s=5.
    from .tabletop_surface_success import SurfaceLiftSuccess
    t.surface_lift_success=SurfaceLiftSuccess(t.n,t.device,t.surface_reference_bottom_m,
        dtype=t.command.dtype)
    measured=w.measure()
    for key in MEASUREMENT_KEYS:
        if key not in measured:
            # Optional geometry/depth additions in synthetic, non-native tests.
            if redistribution is not False:raise ValueError('missing effective-pad measurement '+key)
            shape=(t.n,2,5,3) if key=='pad_reference_points_w' else (t.n,2,5)
            measured[key]=torch.zeros(shape,device=t.device,dtype=t.command.dtype)
        t.m[key]=measured[key].clone()
    t.keys=tuple(dict.fromkeys((*t.keys,*MEASUREMENT_KEYS)))
    for name in REWARD_NAMES:t.terms[name]=torch.zeros_like(t.reward)
    if redistribution is None:redistribution=SurfaceRedistribution(w)
    t.surface_redistribution=redistribution
    t.last_observation=t.observe().clone()


def allocation(t):
    m=t.m
    return surface_targets(m['effective_pad_loads_n'],m['effective_pad_vectors_w'],valid=m['effective_pad_valid'])


def verified_support(t,whole_hand_lift_force):
    """Normal on the rod from accepted pads; friction is not digit-resolved.

    Whole-hand friction is attributable only when that hand has no rejected
    object contacts. Mixed pad/back contacts conservatively get no friction
    credit, while their actual physics and full-force diagnostics remain live.
    """
    m=t.m
    normal=torch.minimum(-m['effective_pad_vectors_w'][...,2].sum(-1),m['vertical_hand_force_n'])
    friction=whole_hand_lift_force-m['vertical_hand_force_n']
    attributable=m['rejected_contact_loads_n']<=1e-6
    return normal,normal+torch.where(attributable,friction,0.)


def qualified_pad_contact(t):
    m=t.m;a=allocation(t)
    load=m['effective_pad_loads_n'].sum((1,2));other=m['rejected_contact_loads_n'].sum(-1)
    return (a['paired'].all(-1)&(other/(load+other).clamp_min(1e-12)<=.05)
            &(m['effective_pad_depth_m'].amax((1,2))<=.0008))


def observation(t):
    m=t.m;a=allocation(t)
    paired=a['paired'].to(t.command.dtype)
    center=m['object_relative']+m['wrist_pose'][:,:,:3].mean(1)
    references=(m['pad_reference_points_w']-center[:,None,None])/.05
    # Every digit is visible, independently of the historical thumb/index FK.
    return torch.cat((torch.log1p(m['effective_pad_loads_n']).flatten(1),
        torch.stack((a['groups']==0,a['groups']==1),-1).to(t.command.dtype).flatten(1),
        torch.log1p(a['face_loads']).flatten(1),a['face_counts'].to(t.command.dtype).flatten(1)/5,
        a['axis'].flatten(1),paired,a['face_imbalance'],a['within_imbalance'],
        m['effective_pad_valid'].to(t.command.dtype).flatten(1),
        torch.log1p(m['rejected_contact_loads_n']),references.flatten(1)),1)


def reward_rates(t):
    m=t.m;a=allocation(t);loads=m['effective_pad_loads_n']
    paired=a['paired'];gate=paired.all(-1).to(loads.dtype)
    stable_hold=getattr(t,'surface_stable_hold_rewards',False)
    if stable_hold:gate=qualified_pad_contact(t).to(loads.dtype)
    height=(m['bottom']-t.surface_reference_bottom_m-.0005).clamp_min(0)
    # Reference points are geometry only. Their distance never counts as contact.
    reach=torch.zeros_like(t.reward)
    if 'pad_reference_points_w' in m:
        center=m['object_relative']+m['wrist_pose'][:,:,:3].mean(1)
        rel=m['pad_reference_points_w']-center[:,None,None]
        q=m['object_quat'];q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        w,x,y,z=q.unbind(-1)
        axis=torch.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),-1)
        along=(rel*axis[:,None,None]).sum(-1)
        radial=rel-along[...,None]*axis[:,None,None]
        distance=(radial.norm(dim=-1)-.008).abs()+(along.abs()-.25).clamp_min(0)
        near=torch.exp(-distance/.008)
        reach=3*near.topk(2,dim=-1).values.mean((1,2))+near.mean((1,2))
    bearing=(loads>.05)&m['effective_pad_valid']
    contact=2*bearing.to(loads.dtype).mean((1,2))+2*paired.to(loads.dtype).mean(-1)
    accepted=loads.sum(-1);rejected=m['rejected_contact_loads_n']
    collision=-(rejected/(accepted+rejected).clamp_min(1e-12)).mean(-1)
    action_rate=-.5*(((t.actions-t.previous_actions)/2).clamp(-1,1).square().mean(-1))
    rates=dict(dex_reach=reach,dex_contact=contact,dex_lift=60*(height/.03).clamp(0,1)*gate,
        dex_force=torch.zeros_like(t.reward),dex_balance=torch.zeros_like(t.reward),dex_action_rate=action_rate,
        surface_between_faces=-2*a['face_imbalance'].mean(-1),
        surface_within_faces=-2*a['within_imbalance'].mean(-1),
        surface_missing_opposition=-.5*((accepted>.05)&~paired).to(loads.dtype).mean(-1),
        ordinary_collision=collision)
    if stable_hold:
        duration=t.surface_lift_success.duration.clamp(0,5.)
        rates['surface_hold_progress']=120.*(duration/5.).square()*gate
        tracker=getattr(t,'surface_slip_tracking',None)
        drift=tracker.measured_drift_m if tracker is not None else torch.zeros_like(t.reward)
        airborne=(m['bottom']>=t.surface_reference_bottom_m+.003).to(loads.dtype)
        rates['surface_slip_window']=-160.*(drift/.0005).square().clamp(0,4)*gate*airborne
    return rates


def apply_contract(c,t):
    if getattr(t,'surface_stable_hold_rewards',False):
        c['surface_training_objective']=dict(STABLE_HOLD_REWARD_CONTRACT)
    c.update(surface_force_sharing=dict(SHARING,action_size=24,stage_flow=None,pose_latch=False,
        redistribution=('soft_reward_only;policy_owns_joint_targets' if t.surface_redistribution is False
                        else 'bounded_joint_position_feedback;preserve_current_measured_total'),
        effective_contact='CAD_local_pad_region_and_native_normal;ordinary_collisions_separate',
        lift_reference_bottom_m=t.surface_reference_bottom_m.cpu().tolist(),success_hold_s=5.),
        policy_command_size=24,action='24D_free_fingers_worldXYZ_intrinsicXYZ_surface_sharing_v1',
        observation_version='dh116-surface-sharing-lift-objective-v2',observation_size=t.observe().shape[-1],
        initial_force_action_mean=None,initial_force_target_n=None,force_action_range_n=None,
        force_action_semantics=dict(quantity=None,range=None,reason='pressure_emerges_from_joint_pose_exploration'))
    if getattr(t,'surface_contact_history',None) is not None:
        from .tabletop_surface_contact_history import CONTRACT as HISTORY,OBSERVATION_VERSION as HISTORY_OBSERVATION
        c.update(surface_contact_history=dict(HISTORY),observation_version=HISTORY_OBSERVATION)
    c.update(success_bottom_m=(t.surface_reference_bottom_m+.03).cpu().tolist(),
        success_dwell_s=5.,success_slip_m=.0005)
    c['force_region_control']=dict(control=False,force_target_range_n=None,friction_excluded=True,
        use='signed_vertical_force_rewards_only;no_fixed_force_tracking')
    if 'phase_force_reward' in c:
        c['phase_force_reward'].update(grasp_load_band_n='effective opposed contact;no_fixed_band',
            lift_load_band_n='effective opposed contact;no_fixed_band',force_target=None,
            contact_region='any_actual_effective_opposed_pads;no_fixed_finger_or_axis_height',
            normal_source='accepted_pad_manifold_on_hand_negated',
            friction_credit='whole_hand_friction_only_when_that_hand_has_no_rejected_contacts')
    c['surface_force_sharing'].update(ordinary_contact_success_fraction_max=.05,
        accepted_native_depth_success_max_m=.0008,
        positive_force_credit='verified_pad_normal;friction_only_when_whole_hand_attributable',
        downward_penalty='all_object_contacts_normal;unchanged')
    c['surface_force_sharing'].update(success_force_precision_gate=False,success_force_balance_gate=False,
        contact_basis='native_dt_weighted_control_interval_mean;all_raw_physical_checks',
        success_slip_basis='rod_three_points_in_each_actual_wrist;fixed_anchor_in_hold_window',
        software_mimic_master_override=False,physical_native_mimic_retained=True)
    for key in ('policy_control','grasp_controller','free_wrist_grasp'):
        c[key]=dict(c.get(key,{}),action_size=24,normal_force_range_n=None,
            normal_force_parameters=0,policy_command_size=24,ordinary_contact_pause=False)
    c['dexhands_feedback']=dict(c.get('dexhands_feedback',{}),
        requested_force_range_n=None,force_tracking=False,balance='opposing_surfaces_then_digits_in_each_surface')
    for key in ('policy_control','grasp_controller','free_wrist_grasp'):
        c[key].update(version='surface_sharing_free_wrist_v1',
            action_order='joint_velocity12_world_translation_velocity6_intrinsic_XYZ_velocity6')
    c['force_feedback']=dict(enabled=True,mode=('measured_observation_and_soft_balance_reward'
        if t.surface_redistribution is False else 'zero_total_increment_surface_redistribution'),
        fixed_newton_target=None,measured_load_guaranteed=False,friction_excluded=True)
    if 'free_grasp_lift' in c:
        c['free_grasp_lift'].update(actions=c['policy_control']['action_order'],
            success='net_bottom_lift30mm_from_initial_support;effective_opposed_surfaces;5s;slip<=0.5mm',
            total_episode_timeout_s=t.episode_timeout_s)
    return c


class SurfaceRedistribution:
    """Small correction to the policy path; never a total-pressure servo."""
    def __init__(self,w):
        from .linear_coupling import ACTIVE,MIMIC
        self.w=w;self.tick=0;self.rate=w.t(torch.zeros(w.n,2,6));self.folds=[]
        self.axis=w.t(torch.zeros(w.n,2,3));self.history=[];self.model_trace=[]
        self.body_compliance=None
        for hand in w.hands:
            fold=w.t(torch.zeros(len(hand.joint_names),6))
            for j,name in enumerate(ACTIVE):fold[hand.joint_names.index(name),j]=1.
            for follower,master,ratio,_ in MIMIC:fold[hand.joint_names.index(follower),ACTIVE.index(master)]=ratio
            self.folds.append(fold)

    def reset(self,ids):
        self.rate[ids]=0;self.axis[ids]=0;self.history=[]

    def apply(self,target,command,m,dt,live,low,high):
        from .tabletop_manual_joint_contact import point_jacobian
        from .tabletop_multifinger_control import force_joint_rates,compliance_from_physx_material,table_safe_rates
        w=self.w
        if self.tick%w.decimation==0:
            loads=m['effective_pad_loads_n'];vectors=m['effective_pad_vectors_w']
            a=surface_targets(loads,vectors,valid=m['effective_pad_valid'],previous_axis=self.axis)
            self.axis=torch.where(a['paired'][...,None],a['axis'],self.axis)
            n=loads.shape[0];rows=torch.arange(n,device=loads.device)
            jac=loads.new_zeros(n,2,5,3,6);compliance=loads.new_zeros(n,2,5)
            for h,hand in enumerate(w.hands):
                native=hand.root_physx_view.get_jacobians().double()
                for d,name in enumerate(('finger14_link','finger23_link','finger33_link','finger43_link','finger53_link')):
                    b=hand.body_names.index(name)
                    jac[:,h,d]=point_jacobian(native[:,max(b-1,0)],m['effective_pad_points_w'][:,h,d],
                        hand.data.body_com_pos_w[:,b].double(),self.folds[h])
                    sensor=w.w.sensors[h*len(w.w.sensor_links)+w.w.sensor_links.index(name)]
                    prop=sensor.body_physx_view.get_material_properties().to(loads.device).double()
                    compliance[:,h,d]=compliance_from_physx_material(prop).reshape(n,-1).amax(-1)
            direction=vectors/vectors.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            normal_jac=torch.einsum('nhdc,nhdck->nhdk',-direction,jac)
            active=(a['groups']>=0)&a['paired'][...,None]
            q=command.reshape(n,2,6)
            lower=((low.reshape(1,2,6)-q)*60).clamp(-.03,.03)
            upper=((high.reshape(1,2,6)-q)*60).clamp(-.03,.03)
            self.rate=force_joint_rates(normal_jac,loads,a['desired'],active,lower,upper,1/60,
                seeking=torch.zeros_like(active),retreat=m['effective_pad_depth_m']>.0008,
                contact_compliance_m_n=compliance)
            self.last=dict(**a,normal_jac=normal_jac,point_jac=jac,
                points_w=m['effective_pad_points_w'],outward_normals_w=direction,
                effective_pad_loads_n=loads,rate=self.rate.clone())
            if getattr(w,'surface_trace_limit',0)>0 and len(self.model_trace)<w.surface_trace_limit//w.decimation+1:
                record=dict(**self.last,object_pose_w=torch.cat((w.wp,w.wq),-1),
                    object_weight_n=w.object_weight_n,physics_tick=loads.new_tensor(self.tick))
                self.model_trace.append({k:v.detach().cpu().clone() for k,v in record.items()})
        self.tick+=1
        result=target+self.rate.flatten(1)*dt
        # Main controller applies the final original .3 rad/s total path cap.
        return torch.where(live[:,None],result,target).maximum(low).minimum(high)
