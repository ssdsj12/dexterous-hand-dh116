"""Vectorized equivalent of force-feedback TabletopEnv physics-substep logic.

Host NumPy arrays batch all environments. Physics remains exclusively in the
supplied advance callback. Episode recording uses the original environment.
"""
import numpy as np
from .linear_coupling import active_limits, ACTIVE, MIMIC
from .tabletop_control import PHASES
from .tabletop_contact_reward import contact_potential


class BatchTask:
    keys=('q','qd','forces','clearance','object_relative','object_quat','object_velocity',
          'bottom','approach_error','mimic_errors','hand_clearance','adaptive_palm_offset',
          'grip_stiffness','pad_relative')

    contact_keys=('object_link_loads','grouped_object_loads','hand_object_loads')

    optional_shapes = {'material_relative': (2, 3), 'material_speed': (2,)}

    @classmethod
    def stack_measurements(cls,rows):
        result={k:np.asarray([m[k] for m in rows],dtype=float) for k in cls.keys}
        if any('object_link_loads' in m for m in rows):
            for key in cls.contact_keys:
                if any(key not in m for m in rows):raise ValueError('incomplete full-hand contact batch')
                result[key]=np.asarray([m[key] for m in rows],dtype=float)
        for key, shape in cls.optional_shapes.items():
            if any(key in m for m in rows):
                result[key] = np.zeros((len(rows), *shape))
                result['_has_' + key] = np.array([key in m for m in rows])
                for i, m in enumerate(rows):
                    if key in m:
                        value = np.asarray(m[key], dtype=float)
                        result[key][i] = value if value.shape == shape else np.nan
        result['joint_failure']=np.asarray([m['joint_failure'] for m in rows],dtype=object)
        return result

    def optional_measurements(self, measurements):
        if self.full_hand_contact:
            from .tabletop_object_contact import validate_object_contacts
            arrays,valid=validate_object_contacts(measurements,self.envs[0].backend.object_contact_links,batch_size=self.n,return_validity=True)
            measurements={**measurements, **arrays, '_valid_object_contacts':valid}
        # Presence masks allow rigid and flexible backends in the same batch.
        result = dict(measurements)
        for key, shape in self.optional_shapes.items():
            present = key in result
            result['_has_' + key] = np.asarray(result.get('_has_' + key,
                np.full(self.n, present)), dtype=bool)
            value = np.asarray(result.get(key, np.zeros((self.n, *shape))), dtype=float)
            result[key] = value if value.shape == (self.n, *shape) else np.full((self.n, *shape), np.nan)
        return result

    def measured_motion(self, m):
        slip = np.where(self.grip_valid, np.linalg.norm(m['object_relative']-self.grip, axis=1), 0.)
        regions = self.region_valid & m['_has_material_relative']
        slip = np.where(regions, np.linalg.norm(m['material_relative']-self.regions, axis=2).max(axis=1), slip)
        speed = np.where(m['_has_material_speed'], m['material_speed'].max(axis=1),
                         np.linalg.norm(m['object_velocity'][:, :3], axis=1))
        return slip, speed

    def __init__(self,envs):
        self.envs=list(envs);self.n=len(envs)
        if any(e.force_controller is None or e.stage!='full' or e.closed for e in envs):
            raise ValueError('batched task requires open full-stage force-feedback episodes')
        self.full_hand_contact=bool(envs[0].full_hand_contact)
        if any(e.full_hand_contact != self.full_hand_contact for e in envs):raise ValueError('mixed contact contracts')
        if self.full_hand_contact and any(tuple(e.backend.object_contact_links)!=tuple(envs[0].backend.object_contact_links) for e in envs):raise ValueError('mixed contact link order')
        self.m=self.optional_measurements(self.stack_measurements([e.measurement for e in envs]))
        self.reference=np.stack([e.reference for e in envs])
        self.command=np.stack([e.command for e in envs])
        self.old_command=self.command.copy()
        self.offset=np.array([e.offset for e in envs])
        self.phase=np.array([PHASES.index(e.task.phase) for e in envs])
        self.elapsed=np.array([e.task.elapsed for e in envs])
        self.contact_time=np.array([e.task.contact_time for e in envs])
        self.success_time=np.array([e.task.success_time for e in envs])
        self.success=np.array([e.task.success for e in envs])
        self.failure=np.array([e.task.failure for e in envs],dtype=object)
        self.grip_valid=np.array([e.grip_relative is not None for e in envs])
        self.grip=np.array([np.zeros(3) if e.grip_relative is None else e.grip_relative for e in envs])
        self.region_valid=np.array([e.grip_regions is not None for e in envs])
        self.regions=np.array([np.zeros((2,3)) if e.grip_regions is None else e.grip_regions for e in envs])
        self.slip=np.array([e.metrics['slip_m'] for e in envs])
        self.direction=np.stack([e.force_direction for e in envs])
        self.parameters=envs[0].force_controller.parameters
        if any(e.force_controller.parameters != self.parameters for e in envs):
            raise ValueError('batched environments must share load parameters')
        self.compensation=np.stack([np.zeros((2,3)) if e.force_palm_compensation is None
                                   else e.force_palm_compensation for e in envs])
        self.has_compensation=np.array([e.force_palm_compensation is not None for e in envs])
        self.palm_actions=np.zeros((self.n,2,3))
        self.palm_targets=np.stack([e.pressure_palm_target for e in envs])
        self.palm_clipped=np.stack([e.pressure_palm_clipped for e in envs])
        for name in ('offset','filtered','target','requested','boost','last_slip','blocked'):
            setattr(self,'f_'+name,np.asarray([getattr(e.force_controller,name) for e in envs]).copy())
        self.low,self.high=(np.tile(v,2) for v in active_limits())
        self.seen=np.zeros((self.n,4),dtype=bool)
        self.nonfinite=np.zeros(self.n,dtype=bool)
        self.paused=np.zeros(self.n,dtype=bool)
        self.used=np.zeros(self.n)
        self.reward=np.zeros(self.n)
        self.terms={k:np.zeros(self.n) for k in ('approach','contact','lift','mpc_tracking',
            'slip','smoothness','force','time','success','failure','stage','contact_dwell','lift_dwell')}

    def control(self,actions,live,dt):
        m=self.m
        self.paused=np.max(abs(m['mimic_errors']),axis=1)>=.004
        tracking=np.max(abs(self.command-m['q']),axis=1)<.03
        loads=m['hand_object_loads'] if self.full_hand_contact else m['forces'].reshape(self.n,2,2).sum(axis=2)
        slipping=(self.slip>.002)&((self.slip-self.f_last_slip)/dt>.002)
        lift=(self.phase==3)&~self.paused&tracking&np.all(loads>=.8*self.f_requested,axis=1)&~slipping&live
        self.offset[lift]=np.minimum(.05,self.offset[lift]+.01*dt)
        filtered=self.f_filtered+(1-np.exp(-dt/.05))*(loads-self.f_filtered)
        maximum=self.parameters.max_force_n
        deadband=self.parameters.deadband_n
        rate=self.parameters.force_rate_n_s
        from .tabletop_tensor_task import POLICY_MIN_FORCE_N, POLICY_MAX_FORCE_N
        # The GPU PPO contract uses a 2–12 N request range. Keep the legacy
        # pad-load path differential-compatible; full-hand batches are the
        # active PPO deployment path.
        policy_mode = self.full_hand_contact or maximum >= POLICY_MAX_FORCE_N
        policy_min = POLICY_MIN_FORCE_N if policy_mode else .1
        policy_max = POLICY_MAX_FORCE_N if policy_mode else maximum
        nominal=policy_min+(actions[:,18:]+1)*.5*(policy_max-policy_min)
        boost=self.f_boost+np.where(((self.phase==3)&slipping)[:,None]&(loads>.02),.25*dt,0.)
        boost=np.clip(boost,0.,np.maximum(0.,maximum-nominal))
        requested=np.clip(nominal+boost,policy_min,maximum)
        requested=np.minimum(requested,policy_max)
        target=(requested.copy() if rate is None else
                self.f_target+np.clip(requested-self.f_target,-rate*dt,rate*dt))
        error=target-filtered
        error=np.where(loads>target+deadband,target-loads,error)
        error=np.where(abs(error)<=deadband,0.,error)
        blocked=np.max(abs(m['mimic_errors'].reshape(self.n,2,5)),axis=2)>=.004
        movement=np.clip(.1*error,-.03,.03)*dt
        movement=np.where(blocked&(movement>0),0.,movement)
        for name,value in (('filtered',filtered),('boost',boost),('requested',requested),
                ('target',target),('offset',np.clip(self.f_offset+movement,-.08,.08)),('blocked',blocked)):
            getattr(self,'f_'+name)[live]=value[live]
        self.f_last_slip[live]=self.slip[live]
        from .tabletop_pressure_geometry import compensated_palm_action
        from .tabletop_adaptive_backend import PALM_LIMIT_M
        self.palm_actions=actions[:,12:18].reshape(self.n,2,3).copy()
        compensated,clipped=compensated_palm_action(self.palm_actions,self.f_offset,self.compensation)
        selected=live&self.has_compensation
        self.palm_actions[selected]=compensated[selected]
        self.palm_clipped[selected]=clipped[selected]
        self.palm_targets[selected]=self.palm_actions[selected]*PALM_LIMIT_M
        profile=self.envs[0].joint_envelope
        if profile is None:
            target=np.clip(self.reference+.08*actions[:,:12],self.low,self.high)
        else:
            from .tabletop_joint_envelope import targets,clip_targets
            target=targets(profile,actions[:,:12])
        adjusted=target+(self.f_offset[:,:,None]*self.direction).reshape(self.n,12)
        target=(np.clip(self.reference+np.clip(adjusted-self.reference,-.15,.15),self.low,self.high)
                if profile is None else clip_targets(profile,adjusted))
        for j,(_,master,ratio,_) in enumerate(MIMIC):
            for hand in range(2):
                idx=6*hand+ACTIVE.index(master);err=m['mimic_errors'][:,5*hand+j]
                target[:,idx]=np.where(abs(err)>=.004,
                    np.clip(m['q'][:,idx]+err/ratio,self.low[idx],self.high[idx]),target[:,idx])
        if profile is not None:
            target=clip_targets(profile,target)
        self.command[live]+=np.clip(target-self.command,-.05*dt,.05*dt)[live]
        return tracking&~self.paused&live

    def consume(self,new,live,dt):
        new=self.optional_measurements(new)
        old={k:v.copy() for k,v in self.m.items()}
        previous_phase=self.phase.copy()
        finite=np.ones(self.n,dtype=bool)
        if self.full_hand_contact:finite &= new['_valid_object_contacts']
        for k in self.keys:
            a=np.asarray(new[k]);finite&=np.isfinite(a).reshape(self.n,-1).all(axis=1)
        for k in self.optional_shapes:
            finite &= ~new['_has_' + k] | np.isfinite(new[k]).reshape(self.n, -1).all(axis=1)
        finite &= ~new['_has_material_speed'] | (new['material_speed'] >= 0.).all(axis=1)
        # Match the legacy per-substep observation guard, including finite
        # float64 sensor values whose normalized float32 observation overflows.
        with np.errstate(over='ignore',invalid='ignore'):
            scaled=[2*(new['q']-self.low)/(self.high-self.low)-1,new['qd']/.3,
                new['forces']/30.,new['clearance']/.05,new['object_relative']/.05,
                new['object_quat'],new['object_velocity']/np.array([.1,.1,.1,1.,1.,1.]),
                new['mimic_errors']/.02,new['adaptive_palm_offset']/np.array([.003,.003,.004]),
                (new['grip_stiffness']-10.)/10.,(new['bottom']-.30)/.03,new['pad_relative']/.05,
                self.f_offset/.08,self.f_filtered/self.parameters.max_force_n,
                self.f_target/self.parameters.max_force_n,self.f_boost/self.parameters.max_force_n,
                self.f_requested/self.parameters.max_force_n,
                self.f_last_slip/.02,self.grip/.05,
                np.stack([self.elapsed,self.contact_time,self.success_time],axis=1)/15.]
            if self.full_hand_contact:scaled.append(new['grouped_object_loads']/10.)
            for a in scaled:
                finite&=np.all(np.isfinite(a).reshape(self.n,-1)&
                    (abs(a).reshape(self.n,-1)<=np.finfo(np.float32).max),axis=1)
            new_slip,new_speed=self.measured_motion(new)
            finite&=np.isfinite(new_slip)&np.isfinite(new_speed)
        bad=live&~finite
        self.failure[bad]='nonfinite_state';self.nonfinite|=bad
        self.reward[bad]-=25.;self.terms['failure'][bad]-=25.
        valid=live&finite
        for i,diagnostics in new.get('_diagnostics',{}).items():
            if valid[i]:
                self.envs[i].measurement['diagnostics']=diagnostics
                if i in new.get('_lowest_links', {}):
                    self.envs[i].measurement['lowest_link']=new['_lowest_links'][i]
        for k in self.m:
            self.m[k][valid]=new[k][valid]
        m=self.m
        measured_slip,speed=self.measured_motion(m)
        self.slip[valid]=measured_slip[valid]
        jointbad=valid&(m['joint_failure']!=None)
        self.failure[jointbad]=m['joint_failure'][jointbad]
        active=valid&~jointbad
        self.elapsed[active]+=dt
        # TabletopTask's guard precedence is significant for terminal labels.
        for reason,condition in (('table_penetration',m['clearance']<-.001),
                ('excess_force',(m['object_link_loads'].max(axis=(1,2)) if self.full_hand_contact else m['forces'].max(axis=1))>30.),('object_dropped',m['bottom']<.29),
                ('timeout',self.elapsed>15.)):
            take=active&(self.failure==None)&condition
            self.failure[take]=reason
        active&=self.failure==None
        close=active&(self.phase<3)
        expired=close&(self.elapsed>6.)
        self.failure[expired]='grasp_not_established';close&=~expired
        loads=m['hand_object_loads'] if self.full_hand_contact else m['forces'].reshape(self.n,2,2).sum(axis=2)
        ready=(speed<.02)&(self.slip<=.003)&np.all(loads>=.8*self.f_requested,axis=1)&np.all(loads<=1.25*self.f_requested+self.parameters.deadband_n,axis=1)
        self.phase[close&~ready]=1;self.contact_time[close&~ready]=0.
        self.contact_time[close&ready]+=dt
        transition=close&ready&(self.contact_time>=np.where(self.phase==1,.2,.3))
        self.phase[transition]+=1;self.contact_time[transition]=0.
        lift=active&(previous_phase==3)
        stable=(m['bottom']>=.33)&(speed<.03)&(self.slip<=.02)
        self.success_time[lift]=np.where(stable,self.success_time+dt,0.)[lift]
        event=lift&(self.success_time>=.5)
        self.success[event]=True
        changed=valid&(previous_phase!=self.phase)
        self.seen[np.flatnonzero(changed),self.phase[changed]]=True
        establish=changed&(self.phase>=2)
        self.grip[establish]=m['object_relative'][establish];self.grip_valid[establish]=True
        self.grip_valid[changed&(self.phase==1)]=False
        self.grip[changed&(self.phase==1)]=0.
        self.regions[establish]=m['material_relative'][establish]
        self.region_valid[establish]=m['_has_material_relative'][establish]
        self.region_valid[changed&(self.phase==1)]=False
        self.regions[changed&(self.phase==1)]=0.
        support=lambda x:contact_potential(x['hand_object_loads'] if self.full_hand_contact else x['forces'])
        height=lambda x:np.clip((x['bottom']-.30)/.03,0.,1.)
        pieces=dict(contact=np.where(previous_phase<3,support(m)-support(old),0.),
            lift=np.where(previous_phase==3,5*(height(m)-height(old)),0.),
            slip=np.where(m['bottom']>.305,-np.minimum((np.maximum(0.,self.slip-.02)/.02)**2,25)*dt,0.),
            smoothness=-.05*np.minimum(np.mean(((self.command-self.old_command)/.1)**2,axis=1),25)*dt,
            force=-.1*np.minimum(np.maximum(0.,(m['object_link_loads'].max(axis=(1,2)) if self.full_hand_contact else m['forces'].max(axis=1))-15.)**2/225,25)*dt,
            time=np.full(self.n,-.1*dt),success=np.where(event,25.,0.),
            failure=np.where(self.failure!=None,-25.,0.),
            lift_dwell=np.where((previous_phase==3)&(m['bottom']>=.31)&(speed<.03)&(self.slip<=.02),dt,0.))
        for name,value in pieces.items():
            self.terms[name][valid]+=value[valid];self.reward[valid]+=value[valid]

    def step(self,actions,advance,after_substep=None):
        actions=np.asarray(actions,dtype=float)
        if actions.shape!=(self.n,20):raise ValueError('expected N x 20 actions')
        invalid=~np.isfinite(actions).all(axis=1)|np.any(abs(actions)>1,axis=1)
        self.failure[invalid]='invalid_action'
        self.reward[invalid]-=25.;self.terms['failure'][invalid]-=25.
        safe_actions=np.where(invalid[:,None],0.,actions)
        dt=self.envs[0].backend.dt
        for substep in range(1,self.envs[0].backend.decimation+1):
            live=(self.failure==None)&~self.success
            allowed=self.control(safe_actions,live,dt)
            new=advance(self.offset,self.command,allowed,live)
            self.used[live]+=dt
            self.consume(new,live,dt)
            if after_substep is not None:after_substep(self,new,substep)
        results=[]
        for i,e in enumerate(self.envs):
            e.pressure_palm_target=self.palm_targets[i].copy()
            e.pressure_palm_clipped=self.palm_clipped[i].copy()
            e.command=self.command[i].copy()
            if self.used[i]>0:
                e.filtered_residual=e.command-e.reference
            e.offset=float(self.offset[i]);e.task.phase=PHASES[self.phase[i]]
            e.task.elapsed=float(self.elapsed[i]);e.task.contact_time=float(self.contact_time[i])
            e.task.success_time=float(self.success_time[i]);e.task.success=bool(self.success[i])
            e.task.failure=self.failure[i];e.grip_relative=self.grip[i].copy() if self.grip_valid[i] else None
            e.grip_regions=self.regions[i].copy() if self.region_valid[i] else None
            for phase in (1,2,3):
                if self.seen[i,phase] and PHASES[phase] not in e.reached_phases:e.reached_phases.append(PHASES[phase])
            for name in ('offset','filtered','target','requested','boost','last_slip','blocked'):
                value=getattr(self,'f_'+name)[i]
                setattr(e.force_controller,name,value.copy() if isinstance(value,np.ndarray) else float(value))
            e.measurement={**e.measurement,**{k:v[i].copy() if isinstance(v[i],np.ndarray) else v[i]
                for k,v in self.m.items() if not k.startswith('_') and
                (k not in self.optional_shapes or self.m['_has_' + k][i])}}
            for key in self.optional_shapes:
                if not self.m['_has_' + key][i]:
                    e.measurement.pop(key, None)
            m=e.measurement
            e.metrics=dict(approach_error_m=m['approach_error'],pad_forces_n=m['forces'],clearance_m=m['clearance'],
                object_bottom_m=m['bottom'],object_speed_m_s=(float(np.max(m['material_speed']))
                    if 'material_speed' in m else float(np.linalg.norm(m['object_velocity'][:3]))),
                slip_m=float(self.slip[i]),max_force_n=float(np.max(m['object_link_loads'] if self.full_hand_contact else m['forces'])))
            if self.full_hand_contact:e.metrics['hand_object_loads']=m['hand_object_loads'].copy()
            results.append(e.finish_transition(actions[i],float(self.reward[i]),
                {k:float(v[i]) for k,v in self.terms.items()},float(self.used[i]),bool(self.paused[i]),bool(self.nonfinite[i])))
        return results
