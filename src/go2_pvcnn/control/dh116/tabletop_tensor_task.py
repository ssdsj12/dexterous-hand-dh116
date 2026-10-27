"""Persistent device-resident rigid full-hand task; physics is supplied externally."""
import math
import torch
from .linear_coupling import ACTIVE, MIMIC, active_limits
from .tabletop_control import PHASES
from .tabletop_velocity_limits import CLOSE_COMMAND_RATE_RAD_S

REWARD_VERSION = 'phase_potential_v1_10n_5'
LIFT_PRESSURE_TIMING_CONTRACT=dict(version='current_target_lift_pressure_v1',
    gate='same_pressure_limited_mask_as_joint_and_palm_control',threshold_unchanged=True)
CLOSE_DOWNWARD_PENALTY_CONTRACT=dict(version='close_downward_impulse_reset_v3',
    event='valid_nonfailure_close_to_lift_physics_substep',
    condition='signed_sum_hand_to_object_vertical_force_n_less_than_zero',points=-3.,
    frequency='once_per_episode_phase_transition',dt_scaled=False,
    continuous_downward_penalty='close_contact_and_post_close',observation_unchanged=True,
    impulse_limit_n_s=2.,close_timeout_s=4.,lift_admission='vertical_force_n >= 0')
MEASURED_READINESS_CONTRACT=dict(version='measured_hand_load_v1',min_hand_load_n=1.,
    max_hand_load='1.25 * configured_max_force_n + deadband_n',
    close_dwell_s=.2,hold_dwell_s=.3,max_speed_m_s=.02,max_slip_m=.003,
    force_goal_tracking=False,scope='tensor_rigid_full_hand',
    interpretation='trial_lift_admission_not_grasp_success')
FAILURE_NAMES = dict(enumerate((None, 'invalid_action', 'nonfinite_state', 'joint_limit',
    'mimic_error', 'table_penetration', 'excess_force', 'object_dropped', 'timeout', 'grasp_not_established', 'nonfinite_joint_state')))
FAILURE_CODES = {v:k for k,v in FAILURE_NAMES.items()}

PALM_XY_SEARCH_LIMIT_M = 0.0005
PALM_Z_SEARCH_LIMIT_M = 0.004
PALM_XY_EXPLORATION_WEIGHT = 0.25


def measured_readiness_contract(skip_hold=False):
    if skip_hold:
        return dict(MEASURED_READINESS_CONTRACT,version='measured_close_direct_lift_v1',
                    hold_dwell_s=0.,phase_flow='close_to_lift',phase_ids_preserved=True)
    return dict(MEASURED_READINESS_CONTRACT)


def constrain_palm_exploration(actions, phase, mode):
    """Restrict policy palm exploration to the below-axis Z direction."""
    if mode not in ('xyz','z_down'):
        raise ValueError('unsupported palm exploration mode')
    if actions.ndim != 3 or actions.shape[-1] != 3 or phase.shape != actions.shape[:1]:
        raise ValueError('palm actions must be (N, 2, 3) and phase must be (N,)')
    result=actions.clone()
    if mode=='z_down':
        close=phase<3
        xy_limit = PALM_XY_SEARCH_LIMIT_M / .003 * PALM_XY_EXPLORATION_WEIGHT
        result[close,:,:2]=result[close,:,:2].clamp(-xy_limit, xy_limit)
        result[close,:,2]=result[close,:,2].clamp(-PALM_Z_SEARCH_LIMIT_M / .004, 0.)
        result[~close]=0.
    return result


def apply_measured_contact_z_target(palm_actions, contact_center_relative_m,
                                    target_z_m, valid, *, max_correction_m=.0005,
                                    palm_limit_z=.004):
    """Apply a bounded Z-only correction from measured contact geometry.

    ``contact_center_relative_m`` is relative to the object center.  Invalid
    contact reports leave the policy action unchanged; XY is never modified.
    """
    if palm_actions.ndim != 3 or palm_actions.shape[-1] != 3:
        raise ValueError('palm actions must be N x 2 x 3')
    if contact_center_relative_m.shape != palm_actions.shape:
        raise ValueError('contact centers must be N x 2 x 3')
    if target_z_m.shape != palm_actions.shape[:2] or valid.shape != target_z_m.shape:
        raise ValueError('contact target and validity must be N x 2')
    if max_correction_m < 0 or palm_limit_z <= 0:
        raise ValueError('correction and palm limits must be positive')
    result=palm_actions.clone()
    error=target_z_m-contact_center_relative_m[:,:,2]
    correction=error.clamp(-max_correction_m,max_correction_m)/palm_limit_z
    result[:,:,2]=torch.where(valid,result[:,:,2]+correction,result[:,:,2])
    return result


def phase_potential(phase, contact_time):
    """Continuous readiness potential: close 0..1, hold 1..3, lift 3."""
    return torch.where(phase == 1, torch.clamp(contact_time/.2,0.,1.),
        torch.where(phase == 2, 1.+2.*torch.clamp(contact_time/.3,0.,1.), torch.full_like(contact_time,3.)))


class TensorTask:
    keys=('q','qd','forces','clearance','object_relative','object_quat','object_velocity',
          'bottom','approach_error','mimic_errors','hand_clearance','adaptive_palm_offset',
          'grip_stiffness','pad_relative','vertical_hand_force_n','object_link_loads',
          'grouped_object_loads','hand_object_loads','joint_failure')

    def __init__(self, template_env, num_envs, device, *, pressure_relief=False,contact_reward_window=None,grasp_readiness=None,vertical_force_reward=False,vertical_force_reward_mode='window',force_reward_start='episode_start',skip_hold=False,close_downward_penalty=False,grasp_shaping=False,palm_exploration_axis='xyz'):
        e=template_env
        if e.force_controller is None or e.stage!='full' or e.closed or not e.full_hand_contact:
            raise ValueError('open full-hand full-stage force-feedback template required')
        self.n=int(num_envs); self.device=torch.device(device)
        self.pressure_relief_enabled=bool(pressure_relief)
        if palm_exploration_axis not in ('xyz','z_down'):
            raise ValueError('unsupported palm exploration mode')
        self.palm_exploration_axis=palm_exploration_axis
        self.palm_descent_clearance_m=.0003 if palm_exploration_axis=='z_down' else .0021
        if grasp_readiness not in (None,'measured_load'):
            raise ValueError('unsupported grasp readiness mode')
        self.grasp_readiness=grasp_readiness
        self.skip_hold=bool(skip_hold)
        self.close_downward_penalty=bool(close_downward_penalty)
        if self.close_downward_penalty and (not self.skip_hold or not vertical_force_reward
                or vertical_force_reward_mode!='continuous' or force_reward_start!='downward_after_close'):
            raise ValueError('close_downward_penalty requires skip_hold and continuous downward_after_close force rewards')
        if grasp_shaping and not self.close_downward_penalty:
            raise ValueError('grasp_shaping requires close_downward_penalty configuration')
        from .tabletop_grasp_shaping import GraspShaping
        self.shaping=GraspShaping(self.n,self.device) if grasp_shaping else None
        if self.skip_hold and grasp_readiness!='measured_load':
            raise ValueError('skip_hold requires measured_load readiness')
        self.vertical_force_reward=bool(vertical_force_reward)
        if vertical_force_reward_mode not in ('window','continuous'):
            raise ValueError('unsupported vertical force reward mode')
        if vertical_force_reward_mode=='continuous' and not self.vertical_force_reward:
            raise ValueError('continuous mode requires vertical force reward')
        if force_reward_start not in ('episode_start','after_close','downward_after_close'):
            raise ValueError('unsupported force reward start')
        if force_reward_start!='episode_start' and (not self.vertical_force_reward or vertical_force_reward_mode!='continuous'):
            raise ValueError('after_close requires continuous vertical force reward')
        self.force_reward_start=force_reward_start
        self.vertical_force_reward_mode=vertical_force_reward_mode
        if self.vertical_force_reward and contact_reward_window!='close_start':
            raise ValueError('vertical force reward requires close_start window')
        self.contact_reward_window=contact_reward_window
        if contact_reward_window is not None:
            from .tabletop_contact_window import reward_contract
            reward_contract(contact_reward_window)
        self.pressure_limited=torch.zeros((self.n,2),device=self.device,dtype=torch.bool)
        self.full_hand_contact=True; self.parameters=e.force_controller.parameters
        def tensor(x): return torch.as_tensor(x,device=self.device,dtype=torch.float64).clone()
        def repeat(x):
            t=tensor(x); return t.expand(self.n,*t.shape).clone()
        self.low,self.high=(tensor(list(v)*2) for v in active_limits())
        from .tabletop_adaptive_backend import PALM_LIMIT_M
        self.palm_limit=tensor(PALM_LIMIT_M)
        from .tabletop_object_contact import link_groups
        self.link_groups=torch.as_tensor(link_groups(e.backend.object_contact_links),device=self.device)
        self.envelope=e.joint_envelope
        if self.envelope is not None:
            from .tabletop_joint_envelope import bounds
            self.lower,self.center,self.upper=map(tensor,bounds(self.envelope))
        self.m={k:repeat(e.measurement[k]) for k in self.keys
                if k!='joint_failure' and k in e.measurement}
        self.m['contact_center_relative_m']=repeat(e.measurement.get(
            'contact_center_relative_m',[[float('nan')]*3]*2))
        self.m['contact_normal_w']=repeat(e.measurement.get(
            'contact_normal_w',[[float('nan')]*3]*2))
        self.m['contact_count']=repeat(e.measurement.get('contact_count',[0,0]))
        self.m['contact_valid']=repeat(e.measurement.get('contact_valid',[False,False])).bool()
        self.m['contact_all_below_centerline']=repeat(e.measurement.get(
            'contact_all_below_centerline',[False,False])).bool()
        self.contact_target_z_m=None
        if 'vertical_hand_force_n' not in self.m:
            self.m['vertical_hand_force_n']=repeat([0.,0.])
        self.m['joint_failure']=torch.full((self.n,),FAILURE_CODES[e.measurement['joint_failure']],device=self.device,dtype=torch.long)
        for name in ('reference','command','previous','filtered_residual'):
            setattr(self,name,repeat(getattr(e,name)))
        for name in ('offset','close_progress'):
            setattr(self,name,repeat(getattr(e,name)))
        self.descent_stop=(None,'pad_contact','table_margin').index(e.descent_stop)
        self.direction=repeat(e.force_direction)
        self.compensation=repeat([[0.]*3]*2 if e.force_palm_compensation is None else e.force_palm_compensation)
        self.has_compensation=torch.full((self.n,),e.force_palm_compensation is not None,device=self.device,dtype=torch.bool)
        self.palm_targets=repeat(e.pressure_palm_target)
        self.palm_clipped=repeat(e.pressure_palm_clipped).bool()
        self.palm_actions=torch.zeros((self.n,2,3),device=self.device,dtype=torch.float64)
        self.phase=torch.full((self.n,),PHASES.index(e.task.phase),device=self.device,dtype=torch.long)
        for name in ('elapsed','contact_time','success_time'): setattr(self,name,repeat(getattr(e.task,name)))
        self.success=torch.full((self.n,),bool(e.task.success),device=self.device,dtype=torch.bool)
        self.failure=torch.full((self.n,),FAILURE_CODES[e.task.failure],device=self.device,dtype=torch.long)
        self.grip=repeat([0.]*3 if e.grip_relative is None else e.grip_relative)
        self.grip_valid=torch.full((self.n,),e.grip_relative is not None,device=self.device,dtype=torch.bool)
        self.slip=repeat(e.metrics['slip_m'])
        for name in ('offset','filtered','target','requested','boost','last_slip','blocked'):
            value=repeat(getattr(e.force_controller,name))
            setattr(self,'f_'+name,value.bool() if name=='blocked' else value)
        self.reward=repeat(0.); self.used=repeat(0.)
        self.close_downward_impulse=repeat(0.)
        self.paused=torch.zeros_like(self.success); self.nonfinite=torch.zeros_like(self.success)
        self.contact_reward_clock=repeat(0. if contact_reward_window=='close_start' else -1.)
        self.contact_reward_paid=repeat(0.)
        self.grasp_ready_awarded=torch.zeros_like(self.success)
        self.force_scoring_started=torch.full_like(self.success,force_reward_start=='episode_start')
        self.terms={k:repeat(0.) for k in ('approach','contact','lift','mpc_tracking','slip','smoothness','force','time','success','failure','stage','contact_dwell','lift_dwell','grasp_ready')}
        if self.vertical_force_reward: self.terms['downward_force']=repeat(0.)
        if self.close_downward_penalty: self.terms['close_downward']=repeat(0.)
        if self.shaping is not None:
            self.terms['force_progress']=repeat(0.);self.terms['lift_milestone']=repeat(0.)
        self.old_command=self.command.clone()
        self._seed={k:v[0].clone() for k,v in vars(self).items() if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==self.n and k not in ('low','high','lower','center','upper','palm_limit','link_groups')}
        self.last_observation=self.observe().clone()

    @property
    def done(self): return (self.failure!=0)|self.success

    def reset(self, ids, measurement):
        ids=torch.as_tensor(ids,device=self.device,dtype=torch.long)
        if hasattr(self,'palm_reposition'):self.palm_reposition.reset(ids)
        if self.shaping is not None:self.shaping.reset(ids)
        for key,value in self._seed.items(): getattr(self,key)[ids]=value
        for key in self.keys:
            value=torch.as_tensor(measurement[key],device=self.device,dtype=self.m[key].dtype)
            self.m[key][ids]=value[ids] if value.shape==self.m[key].shape else value
        for key in ('contact_center_relative_m','contact_normal_w','contact_count',
                    'contact_valid','contact_all_below_centerline'):
            if key in measurement:
                value=torch.as_tensor(measurement[key],device=self.device,dtype=self.m[key].dtype)
                value=value[ids] if value.shape==self.m[key].shape else value
                self.m[key][ids]=value.bool() if key in ('contact_valid','contact_all_below_centerline') else value
        for value in self.terms.values(): value[ids]=0.
        if hasattr(self,'coordinated_control'):
            from .tabletop_coordinated_grasp import midpoint_coordinates
            self.control_pad_anchor[ids]=midpoint_coordinates(self.m)[ids]
        self.last_observation[ids]=self.observe()[ids]

    def begin_step(self, actions):
        actions=torch.as_tensor(actions,device=self.device,dtype=torch.float64)
        if actions.shape!=(self.n,20): raise ValueError('expected N x 20 actions')
        if self.shaping is not None:self.shaping.begin(~self.done)
        self.reward.zero_(); self.used.zero_(); self.nonfinite.zero_()
        for value in self.terms.values(): value.zero_()
        self.old_command.copy_(self.command)
        invalid=(~torch.isfinite(actions).all(dim=1)|(abs(actions)>1.).any(dim=1))&~self.done
        self.failure[invalid]=FAILURE_CODES['invalid_action']
        self.reward[invalid]=-25.; self.terms['failure'][invalid]=-25.
        self.actions=torch.where(invalid[:,None],torch.zeros_like(actions),actions)
        return self.actions

    def grip_force_ready(self,loads,*,lifting=False):
        if self.grasp_readiness=='measured_load':
            return ((loads>=MEASURED_READINESS_CONTRACT['min_hand_load_n']).all(dim=1)
                    &(loads<=1.25*self.parameters.max_force_n+self.parameters.deadband_n).all(dim=1))
        ready=(loads>=.8*self.f_requested).all(dim=1)
        return ready if lifting else ready&(loads<=1.25*self.f_requested+self.parameters.deadband_n).all(dim=1)

    def enable_coordinated_control(self,mask):
        from .tabletop_coordinated_grasp import midpoint_coordinates
        mask=torch.as_tensor(mask,device=self.device,dtype=torch.bool)
        if mask.shape!=(self.n,):raise ValueError('one coordinated control selection per environment')
        self.coordinated_control=mask.clone()
        self.coordinated_all=bool(mask.all())
        if self.shaping is not None:
            self.lift_diagnostic_masks=torch.zeros((self.n,8),device=self.device,dtype=torch.bool)
        self.control_pad_anchor=midpoint_coordinates(self.m).clone()

    def control(self,actions,live,dt):
        if not hasattr(self,'coordinated_control'):
            return self._legacy_control(actions,live,dt)
        from .tabletop_coordinated_grasp import control
        mask=self.coordinated_control
        if self.coordinated_all:
            if self.shaping is not None:self.lift_diagnostic_masks.zero_()
            return control(self,actions,live,dt)
        legacy=self._legacy_control(actions,live&~mask,dt)
        return legacy|control(self,actions,live&mask,dt)

    def _legacy_control(self,actions,live,dt):
        m=self.m
        self.paused=abs(m['mimic_errors']).amax(dim=1)>=.004
        # Inference-only ablation. Joint recovery and all other guards retain
        # their original thresholds; normal training has no override.
        lift_paused=abs(m['mimic_errors']).amax(dim=1)>=getattr(self,'diagnostic_lift_mimic_threshold',.004)
        tracking=abs(self.command-m['q']).amax(dim=1)<.03
        loads=m['hand_object_loads'] if self.full_hand_contact else m['forces'].reshape(self.n,2,2).sum(dim=2)
        slipping=(self.slip>.002)&((self.slip-self.f_last_slip)/dt>.002)
        lift=(self.phase==3)&~lift_paused&tracking&self.grip_force_ready(loads,lifting=True)&~slipping&live
        filtered=self.f_filtered+(1-math.exp(-dt/.05))*(loads-self.f_filtered)
        maximum=self.parameters.max_force_n
        deadband=self.parameters.deadband_n
        rate=self.parameters.force_rate_n_s
        nominal=.1+(actions[:,18:]+1)*.5*(maximum-.1)
        boost=self.f_boost+torch.where(((self.phase==3)&slipping)[:,None]&(loads>.02),.25*dt,0.)
        boost=torch.minimum(torch.clamp(boost,min=0.),maximum-nominal)
        requested=torch.clamp(nominal+boost,.1,maximum)
        target=(requested.clone() if rate is None else
                self.f_target+torch.clamp(requested-self.f_target,-rate*dt,rate*dt))
        error=target-filtered
        error=torch.where(loads>target+deadband,target-loads,error)
        error=torch.where(abs(error)<=deadband,0.,error)
        blocked=abs(m['mimic_errors'].reshape(self.n,2,5)).amax(dim=2)>=.004
        movement=torch.clamp(.1*error,-.03,.03)*dt
        movement=torch.where(blocked&(movement>0),0.,movement)
        for name,value in (('filtered',filtered),('boost',boost),('requested',requested),
                ('target',target),('offset',torch.clamp(self.f_offset+movement,-.08,.08)),('blocked',blocked)):
            getattr(self,'f_'+name)[live]=value[live]
        self.f_last_slip[live]=self.slip[live]
        self.palm_actions=constrain_palm_exploration(
            actions[:,12:18].reshape(self.n,2,3),self.phase,self.palm_exploration_axis)
        palm_target=self.palm_actions*self.palm_limit+self.f_offset[:,:,None]*self.compensation
        compensated=torch.clamp(palm_target/self.palm_limit,-1.,1.)
        clipped=abs(palm_target)>self.palm_limit
        selected=live&self.has_compensation
        self.palm_actions[selected]=compensated[selected]
        self.palm_clipped[selected]=clipped[selected]
        self.palm_targets[selected]=self.palm_actions[selected]*self.palm_limit
        target=(torch.clamp(self.reference+.08*actions[:,:12],self.low,self.high)
                if self.envelope is None else self.center+actions[:,:12]*torch.where(actions[:,:12]>=0,self.upper-self.center,self.center-self.lower))
        adjusted=target+(self.f_offset[:,:,None]*self.direction).reshape(self.n,12)
        target=(torch.clamp(self.reference+torch.clamp(adjusted-self.reference,-.15,.15),self.low,self.high)
                if self.envelope is None else torch.clamp(adjusted,self.lower,self.upper))
        if self.pressure_relief_enabled:
            from .tabletop_pressure_relief import pressure_limited_target
            target,self.pressure_limited=pressure_limited_target(
                self.command.reshape(self.n,2,6),target.reshape(self.n,2,6),
                loads,self.f_target,self.direction,dt,deadband)
            target=target.reshape(self.n,12)
            lift &= ~self.pressure_limited.any(dim=1)
        if self.shaping is not None:
            lifting=(self.phase==3)&live
            self.lift_diagnostic_masks=torch.stack((lifting,lift&(self.offset<.05),
                lifting&lift_paused,lifting&~tracking,lifting&~self.grip_force_ready(loads,lifting=True),
                lifting&slipping,lifting&self.pressure_limited.any(dim=1),lifting&(self.offset>=.05)),dim=1)
        self.offset[lift]=torch.clamp(self.offset[lift]+.01*dt,max=.05)
        for j,(_,master,ratio,_) in enumerate(MIMIC):
            for hand in range(2):
                idx=6*hand+ACTIVE.index(master);err=m['mimic_errors'][:,5*hand+j]
                target[:,idx]=torch.where(abs(err)>=.004,
                    torch.clamp(m['q'][:,idx]+err/ratio,self.low[idx],self.high[idx]),target[:,idx])
        # Inference-only candidate validated first on an isolated thumb fixture.
        # Bounds and the original slew cap retain priority over position lead.
        if hasattr(self,'diagnostic_thumb_position_lead'):
            cap=self.diagnostic_thumb_position_lead[:,None]
            ids=[1,7]
            limited=torch.maximum(torch.minimum(target[:,ids],m['q'][:,ids]+cap),m['q'][:,ids]-cap)
            target[:,ids]=torch.where(cap>0,limited,target[:,ids])
        if self.envelope is not None:
            target=torch.clamp(target,self.lower,self.upper)
        else:
            target=torch.clamp(target,self.low,self.high)
        self.command[live]+=torch.clamp(target-self.command,-CLOSE_COMMAND_RATE_RAD_S*dt,
            CLOSE_COMMAND_RATE_RAD_S*dt)[live]
        return tracking&~self.paused&live&~self.pressure_limited.any(dim=1)

    def observe(self):
        m=self.m
        joints=lambda v: 2*(v-self.low)/(self.high-self.low)-1
        one=torch.nn.functional.one_hot(self.phase,4).to(torch.float64)
        stop=torch.nn.functional.one_hot(torch.full_like(self.phase,self.descent_stop),3)
        maximum=self.parameters.max_force_n
        parts=(joints(m['q']),m['qd']/.3,joints(self.previous),joints(self.reference),
            torch.log1p(torch.clamp(m['forces'],min=0)/.1),one,
            torch.stack((self.elapsed,self.contact_time,self.success_time),dim=1)/15.,
            self.offset[:,None]/.05,self.close_progress[:,None],stop,m['clearance'][:,None]/.05,
            m['object_relative']/.05,m['object_quat'],m['object_velocity']/m['object_velocity'].new_tensor([.1,.1,.1,1.,1.,1.]),
            self.grip/.05,self.grip_valid[:,None],m['mimic_errors']/.02,self.filtered_residual/.1,
            (m['adaptive_palm_offset']/m['q'].new_tensor([.003,.003,.004])).flatten(1),
            (m['grip_stiffness']-10.)/10.,(m['bottom'][:,None]-.30)/.03,m['pad_relative'].flatten(1)/.05,
            self.f_offset/.08,self.f_filtered/maximum,self.f_target/maximum,self.f_boost/maximum,
            self.f_requested/maximum,self.f_last_slip[:,None]/.02,m['grouped_object_loads'].flatten(1)/10.)
        if self.contact_reward_window is not None:
            parts+= (self.contact_reward_clock[:,None],self.grasp_ready_awarded[:,None])
        if self.force_reward_start!='episode_start':
            parts+=(self.force_scoring_started[:,None],)
        if self.shaping is not None:parts+=(self.shaping.observe(),)
        return torch.cat(parts,dim=1).float()

    def consume(self, measurement, live, dt, *, vertical_force_n=None,object_weight_n=None):
        new={k:torch.as_tensor(measurement[k],device=self.device,dtype=self.m[k].dtype) for k in self.keys}
        if self.vertical_force_reward:
            if vertical_force_n is None:
                raise ValueError('measured signed vertical force is required')
            vertical_force_n=torch.as_tensor(vertical_force_n,device=self.device,dtype=torch.float64)
            if vertical_force_n.shape!=(self.n,):
                raise ValueError('vertical force must contain one signed sum per environment')
        if self.shaping is not None:
            if object_weight_n is None:raise ValueError('runtime object weight required for grasp shaping')
            object_weight_n=torch.as_tensor(object_weight_n,device=self.device,dtype=torch.float64)
            if object_weight_n.shape!=(self.n,):raise ValueError('one object weight per environment required')
        old_bottom=self.m['bottom'].clone()
        old_contact=torch.clamp(self.m['hand_object_loads'].sum(dim=1)/10.,0.,1.)*5.
        previous_phase=self.phase.clone()
        old_progress=phase_potential(self.phase,self.contact_time)
        old_success_time=self.success_time.clone()
        finite=torch.ones_like(live)
        for key in self.keys:
            if key!='joint_failure': finite &= torch.isfinite(new[key]).reshape(self.n,-1).all(dim=1)
        if self.vertical_force_reward: finite &= torch.isfinite(vertical_force_n)
        if self.shaping is not None:finite &= torch.isfinite(object_weight_n)&(object_weight_n>0)
        # The full contact contract uses nonnegative loads and additive hand sums.
        for key in ('object_link_loads','grouped_object_loads','hand_object_loads'):
            finite &= (new[key]>=0).reshape(self.n,-1).all(dim=1)
        finite &= torch.isclose(new['object_link_loads'].sum(dim=2),new['hand_object_loads'],rtol=1e-6,atol=1e-8).all(dim=1)
        expected=torch.stack([new['object_link_loads'][...,self.link_groups==g].sum(dim=2) for g in range(6)],dim=2)
        finite &= torch.isclose(expected,new['grouped_object_loads'],rtol=1e-6,atol=1e-9).flatten(1).all(dim=1)
        before=self.m
        self.m=new
        observation_finite=torch.isfinite(self.observe()).all(dim=1)
        self.m=before
        finite &= observation_finite
        finite &= torch.isfinite(torch.linalg.vector_norm(new['object_velocity'][:,:3],dim=1))
        finite &= ~self.grip_valid | torch.isfinite(torch.linalg.vector_norm(new['object_relative']-self.grip,dim=1))
        bad=live&~finite
        self.failure[bad]=FAILURE_CODES['nonfinite_state']; self.nonfinite|=bad
        self.reward[bad]-=25.; self.terms['failure'][bad]-=25.
        valid=live&finite
        self.force_scoring_started |= valid&(previous_phase>=2)
        for key in self.keys: self.m[key][valid]=new[key][valid]
        for key in ('contact_center_relative_m','contact_normal_w','contact_count',
                    'contact_valid','contact_all_below_centerline'):
            if key in measurement:
                value=torch.as_tensor(measurement[key],device=self.device,dtype=self.m[key].dtype)
                self.m[key][valid]=value[valid].bool() if key in ('contact_valid','contact_all_below_centerline') else value[valid]
        m=self.m
        slip=torch.where(self.grip_valid,torch.linalg.vector_norm(m['object_relative']-self.grip,dim=1),0.)
        speed=torch.linalg.vector_norm(m['object_velocity'][:,:3],dim=1)
        self.slip[valid]=slip[valid]
        jointbad=valid&(m['joint_failure']!=0)
        self.failure[jointbad]=m['joint_failure'][jointbad]
        active=valid&~jointbad
        self.elapsed[active]+=dt
        maximum=m['object_link_loads'].flatten(1).amax(dim=1)
        for reason,condition in (('table_penetration',m['clearance']<-.001),('excess_force',maximum>30.),
                ('object_dropped',m['bottom']<.29),('timeout',self.elapsed>15.)):
            self.failure[active&(self.failure==0)&condition]=FAILURE_CODES[reason]
        active &= self.failure==0
        close=active&(self.phase<3)
        if hasattr(self,'palm_reposition'):
            reposition_failed=close&self.palm_reposition.aborted.any(1)
            self.failure[reposition_failed]=FAILURE_CODES['grasp_not_established']
            active &= self.failure==0
            close=active&(self.phase<3)
        expired=close&(self.elapsed>CLOSE_DOWNWARD_PENALTY_CONTRACT['close_timeout_s'])
        self.failure[expired]=FAILURE_CODES['grasp_not_established']; close &= ~expired
        loads=m['hand_object_loads']
        ready=(speed<.02)&(self.slip<=.003)&self.grip_force_ready(loads)
        close_contact=loads.sum(1)>.2
        if self.close_downward_penalty:
            close_live=active&(self.phase==1)&close_contact
            self.close_downward_impulse[close_live]+=torch.clamp(-vertical_force_n[close_live],min=0.)*dt
            over_limit=close_live&(self.close_downward_impulse>=CLOSE_DOWNWARD_PENALTY_CONTRACT['impulse_limit_n_s']-1e-12)
            self.failure[over_limit]=FAILURE_CODES['grasp_not_established']
            active &= self.failure==0
            close=active&(self.phase<3)
            ready &= ~over_limit
        if self.vertical_force_reward and self.skip_hold:
            ready &= vertical_force_n>=0.
        if hasattr(self,'palm_reposition'):ready &= ~self.palm_reposition.active.any(1)
        if hasattr(self,'palm_reposition'):ready &= ~self.palm_reposition.aborted.any(1)
        if hasattr(self,'coordinated_control'):
            from .tabletop_coordinated_grasp import CONTRACT
            ready &= ~self.coordinated_control | (m['mimic_errors'].abs().amax(1)<CONTRACT['coupling_budget_rad'])
        if self.contact_target_z_m is not None:
            ready &= (m['contact_valid'] & m['contact_all_below_centerline']).all(dim=1)
        self.phase[close&~ready]=1; self.contact_time[close&~ready]=0.
        self.contact_time[close&ready]+=dt
        transition=close&ready&(self.contact_time>=torch.where(self.phase==1,torch.full_like(self.contact_time,.2),torch.full_like(self.contact_time,.3))-1e-12)
        self.phase[transition]=3 if self.skip_hold else self.phase[transition]+1
        self.contact_time[transition]=0.
        lift=active&(previous_phase==3)
        stable=(m['bottom']>=.33)&(speed<.03)&(self.slip<=.02)
        self.success_time[lift]=torch.where(stable,self.success_time+dt,0.)[lift]
        event=lift&(self.success_time>=.5-1e-12)
        self.success[event]=True
        changed=valid&(previous_phase!=self.phase)
        establish=changed&(self.phase>=2)
        self.grip[establish]=m['object_relative'][establish]; self.grip_valid[establish]=True
        self.grip_valid[changed&(self.phase==1)]=False; self.grip[changed&(self.phase==1)]=0.
        progress=phase_potential(self.phase,self.contact_time)-old_progress
        contact_delta=5.*torch.clamp(loads.sum(dim=1)/10.,0.,1.)-old_contact
        pieces=dict(contact=torch.where(previous_phase==1,contact_delta,
                torch.where(previous_phase==2,torch.clamp(contact_delta,max=0.),0.)),
            lift=torch.where(previous_phase==3,5.*(torch.clamp((m['bottom']-.30)/.03,0.,1.)-torch.clamp((old_bottom-.30)/.03,0.,1.)),0.),
            contact_dwell=torch.where(previous_phase<3,progress,0.),
            lift_dwell=torch.where(previous_phase==3,2.*(torch.clamp(self.success_time/.5,0.,1.)-torch.clamp(old_success_time/.5,0.,1.)),0.),
            slip=torch.where(previous_phase==2,-torch.clamp((torch.clamp(self.slip-.003,min=0.)/.003)**2,max=25)*dt,
                torch.where(m['bottom']>.305,-torch.clamp((torch.clamp(self.slip-.02,min=0.)/.02)**2,max=25)*dt,0.)),
            smoothness=-.05*torch.clamp((((self.command-self.old_command)/.1)**2).mean(dim=1),max=25)*dt,
            force=-.1*torch.clamp(torch.clamp(maximum-15.,min=0.)**2/225,max=25)*dt,
            time=torch.full_like(self.elapsed,-.1*dt),success=torch.where(event,25.,0.),failure=torch.where(self.failure!=0,-25.,0.))
        if self.contact_reward_window is not None:
            from .tabletop_contact_window import window_credit
            eligible=valid&(self.failure==0)
            clock,credit=window_credit(self.contact_reward_clock,self.contact_reward_paid,
                torch.where(finite,vertical_force_n.clamp_min(0.),0.) if self.vertical_force_reward else loads.sum(dim=1),
                eligible,dt,self.contact_reward_window)
            self.contact_reward_clock.copy_(clock)
            if self.vertical_force_reward_mode=='continuous':
                # Keep the legacy clock only for observation shape compatibility.
                # Actual positive credit has no window or episode budget.
                upward_enabled=self.force_scoring_started | (self.force_reward_start=='downward_after_close')
                credit=torch.where(eligible&upward_enabled,.5*vertical_force_n.clamp(0.,10.)*dt,0.)
                self.contact_reward_paid.add_(credit)
            else:
                self.contact_reward_paid.add_(credit).clamp_(max=5.)
            ready_event=eligible&(previous_phase<3)&(self.phase==3)&~self.grasp_ready_awarded
            if self.vertical_force_reward:
                # Downward squeezing is not upward support. Delay the one-off
                # bonus until a still-ready grasp actually pushes upward.
                ready_event=eligible&(self.phase==3)&ready&(vertical_force_n>0)&~self.grasp_ready_awarded
            self.grasp_ready_awarded|=ready_event
            pieces['contact']=credit
            pieces['contact_dwell']=torch.zeros_like(credit)
            pieces['grasp_ready']=ready_event.to(torch.float64)*3.
        if self.vertical_force_reward:
            # Negative force on the object is downward. Once scoring starts,
            # include terminal samples and never cap accumulated cost.
            downward_enabled=self.force_scoring_started
            if self.close_downward_penalty:
                downward_enabled |= (previous_phase==1)&close_contact
            pieces['downward_force']=torch.where(downward_enabled,
                -.5*torch.clamp(-vertical_force_n,min=0.)*dt,0.)
        if self.close_downward_penalty:
            # Score the force at the exact close completion sample, once.
            # Phase remains lift until reset; later force changes cannot repay it.
            downward_close=valid&(self.failure==0)&(previous_phase==1)&(self.phase==3)&(vertical_force_n<0.)
            pieces['close_downward']=torch.where(downward_close,-3.,0.)
        if self.shaping is not None:
            pieces['lift_milestone']=self.shaping.update(vertical_force_n,object_weight_n,m['bottom'],
                previous_phase==3,(speed<.03)&(self.slip<=.02)&self.grip_force_ready(loads),
                valid&(self.failure==0),dt)
        for name,value in pieces.items():
            self.terms[name][valid]+=value[valid]; self.reward[valid]+=value[valid]
        self.used[live]+=dt
        self.filtered_residual[valid]=self.command[valid]-self.reference[valid]

    def end_step(self):
        if self.shaping is not None:
            progress=self.shaping.end(self.done)
            self.reward+=progress;self.terms['force_progress']+=progress
        self.previous.copy_(self.command)
        observation=self.observe()
        observation=torch.where(self.nonfinite[:,None],self.last_observation,observation)
        self.last_observation.copy_(observation)
        extras={'phase':self.phase.clone(),'success':self.success.clone(),'failure':self.failure.clone(),
                'terms':{k:v.clone() for k,v in self.terms.items()},'used_time':self.used.clone(),
                'pressure_limited':self.pressure_limited.clone()}
        return observation,self.reward.clone(),self.done.clone(),extras
