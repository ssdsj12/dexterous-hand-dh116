"""Real-contact upward support through bounded finger motor feedforward.

Force components are signed hand-to-object world-Z Newtons. The normal clamp
goal remains separate. No external force is applied to the object.
"""
import math
import torch

CONTRACT=dict(version='measured_contact_support_v2',source='actual_normal_Z_plus_actual_friction_Z',
    target='mg*(1+0.1*tanh((0.003-object_vz)/0.003))',
    tangent_target='desired_total_Fz/2-measured_hand_normal_Z',
    friction_feedback_gain=.5,traction_ratio=.5,maximum_tangent_hand_force_weight_ratio=2.,
    force_application='actual_folded_active_finger_Jacobian_transpose',
    motor_budget='explicit_feedforward_and_implicit_drive_share_original_effort_limit',
    contact='measured_digit_normal_load>0.05N',object_external_force=False,
    physics_materials_changed=False,hardware_calibrated=False)
CONTRACT.update(normal_target='current_explored_2_to12N_per_hand_not_fixed',normal_feedback='continuous_PI',
    tangent_feedback='continuous_PI',servo_traction_ratio=1.,torque_slew_nm_s=2.,
    normal_integral_limit_n=12.,normal_correction_limit_n=12.,
    actual_contact_model='fresh_actual_point_link_and_native_jacobian_every_physics_step')


def support_torques(jac,inward,digit_loads,normal_z,friction_z,weight,object_vz,enabled,*,effort_limit_nm=.2,
                    tangent_integral_n=None,normal_correction_n=None,traction_ratio=.5):
    n=jac.shape[0]
    shapes=((jac,(n,2,2,3,6)),(inward,(n,2,2,3)),(digit_loads,(n,2,2)),
            (normal_z,(n,2)),(friction_z,(n,2)),(weight,(n,)),(object_vz,(n,)),(enabled,(n,)))
    if any(value.shape!=shape for value,shape in shapes):raise ValueError('matching measured support arrays required')
    if not all(torch.isfinite(value).all() for value,_ in shapes):raise ValueError('finite measured support required')
    if any(value.device!=jac.device for value,_ in shapes) or enabled.dtype!=torch.bool:
        raise ValueError('one device and boolean lift request required')
    if (weight<=0).any() or (digit_loads<0).any() or not math.isfinite(effort_limit_nm) or effort_limit_nm<=0:
        raise ValueError('positive weight/effort and nonnegative normal loads required')
    if not math.isfinite(traction_ratio) or traction_ratio<=0:raise ValueError('positive traction ratio required')
    tangent_integral_n=torch.zeros_like(normal_z) if tangent_integral_n is None else tangent_integral_n
    normal_correction_n=torch.zeros_like(digit_loads) if normal_correction_n is None else normal_correction_n
    if (tangent_integral_n.shape!=normal_z.shape or normal_correction_n.shape!=digit_loads.shape
            or not torch.isfinite(tangent_integral_n).all() or not torch.isfinite(normal_correction_n).all()):
        raise ValueError('finite matching force controller corrections required')
    norm=inward.norm(dim=-1,keepdim=True)
    if (norm<=0).any():raise ValueError('nonzero contact normals required')
    normal=inward/norm
    target_total=weight*(1+.1*torch.tanh((.003-object_vz)/.003))
    target_hand=target_total[:,None].expand(-1,2)/2
    desired_tangent=target_hand-normal_z
    requested=desired_tangent+.5*(desired_tangent-friction_z)+tangent_integral_n
    requested=requested.clamp(-2*weight[:,None],2*weight[:,None])
    contact=(digit_loads>.05)&enabled[:,None,None]
    fraction=contact.to(jac.dtype)/contact.sum(-1,keepdim=True).clamp_min(1)
    vertical=jac.new_tensor([0.,0.,1.])
    tangent=vertical-normal*normal[...,2:3]
    vertical_capacity=tangent[...,2]
    usable=contact&(vertical_capacity>.05)
    force=requested[:,:,None,None]*fraction[:,:,:,None]*tangent/vertical_capacity.clamp_min(.05)[:,:,:,None]
    force=torch.where(usable[:,:,:,None],force,0.)
    # Conservative controller traction budget, separate from unchanged PhysX mu.
    traction=traction_ratio*digit_loads
    magnitude=force.norm(dim=-1).clamp_min(1e-12)
    force=force*(traction/magnitude).clamp_max(1.)[:,:,:,None]
    normal_contact=digit_loads>.05
    normal_force=torch.where(normal_contact[:,:,:,None],normal_correction_n[:,:,:,None]*normal,0.)
    raw=torch.einsum('nhdck,nhdc->nhk',jac,force+normal_force)
    torque=raw.clamp(-effort_limit_nm,effort_limit_nm)
    return torque,dict(target_total_fz_n=torch.where(enabled,target_total,0.),
        target_hand_fz_n=torch.where(enabled[:,None],target_hand,0.),
        target_tangent_z_n=torch.where(enabled[:,None],desired_tangent,0.),
        measured_normal_z_n=normal_z.clone(),measured_friction_z_n=friction_z.clone(),
        measured_total_fz_n=(normal_z+friction_z).sum(-1),object_weight_n=weight.clone(),
        total_fz_error_n=torch.where(enabled,target_total-(normal_z+friction_z).sum(-1),0.),
        digit_tangent_force_w_n=force,active_torque_command_nm=torque,
        digit_normal_correction_force_w_n=normal_force,
        requested_tangent_with_integral_n=requested,
        torque_saturated=raw.abs()>effort_limit_nm,traction_saturated=(magnitude>traction)&usable,
        measured_contact=contact,enabled=enabled.clone())


class SupportServo:
    def __init__(self,n,device,effort_limit_nm=.2):
        self.effort_limit_nm=effort_limit_nm
        self.normal_integral_n=torch.zeros(n,2,2,device=device,dtype=torch.float64)
        self.tangent_integral_n=torch.zeros(n,2,device=device,dtype=torch.float64)
        self.previous_torque_nm=torch.zeros(n,2,6,device=device,dtype=torch.float64)
        self.previous_contact=torch.zeros(n,2,2,device=device,dtype=torch.bool)
        self.previous_body_ids=torch.full((n,2,2),-1,device=device,dtype=torch.long)

    def reset(self,ids):
        for value in (self.normal_integral_n,self.tangent_integral_n,self.previous_torque_nm,self.previous_contact):value[ids]=0.
        self.previous_body_ids[ids]=-1

    def step(self,jac,inward,digit_loads,normal_z,friction_z,weight,object_vz,enabled,*,requested_normal_n,dt,hand_normal_n=None,contact_body_ids=None,contact_model_valid=None):
        if not math.isfinite(dt) or not 0<dt<=1/60+1e-12:raise ValueError('positive physics dt required')
        if not all(torch.isfinite(v).all() for v in (jac,inward,digit_loads,normal_z,friction_z,weight,object_vz)):
            raise ValueError('finite support feedback required before controller state update')
        if (requested_normal_n.shape!=normal_z.shape or not torch.isfinite(requested_normal_n).all()
                or ((requested_normal_n<2)|(requested_normal_n>12)).any()):
            raise ValueError('current chosen per-hand normal force within2–12N required')
        hand_normal_n=digit_loads.sum(-1) if hand_normal_n is None else hand_normal_n
        switched=torch.zeros_like(self.previous_contact)
        if contact_body_ids is not None:
            if contact_body_ids.shape!=self.previous_body_ids.shape or contact_body_ids.dtype!=torch.long:
                raise ValueError('matching actual contact link indices required')
            switched=contact_body_ids!=self.previous_body_ids
            self.normal_integral_n.masked_fill_(switched,0.)
            self.tangent_integral_n.masked_fill_(switched.any(-1),0.)
            self.previous_body_ids.copy_(contact_body_ids)
        other=(hand_normal_n-digit_loads.sum(-1)).clamp_min(0.)
        requested_pair=(requested_normal_n-other).clamp_min(0.)[:,:,None]/2
        contact_model_valid=torch.ones_like(digit_loads,dtype=torch.bool) if contact_model_valid is None else contact_model_valid
        if contact_model_valid.shape!=digit_loads.shape or contact_model_valid.dtype!=torch.bool:
            raise ValueError('per-digit actual contact model validity required')
        effective_loads=torch.where(contact_model_valid,digit_loads,0.)
        normal_error=requested_pair-digit_loads;contact=effective_loads>.05
        self.normal_integral_n=torch.where(contact,
            (self.normal_integral_n+1.5*normal_error*dt).clamp(-12,12),0.)
        normal_correction=(.35*normal_error+self.normal_integral_n).clamp(-12,12)
        target=weight*(1+.1*torch.tanh((.003-object_vz)/.003))
        tangent_error=target[:,None]/2-normal_z-friction_z
        self.tangent_integral_n=torch.where(enabled[:,None]&contact.any(-1),
            (self.tangent_integral_n+tangent_error*dt).maximum(-weight[:,None]).minimum(weight[:,None]),0.)
        torque,diagnostic=support_torques(jac,inward,effective_loads,normal_z,friction_z,weight,object_vz,enabled,
            effort_limit_nm=self.effort_limit_nm,tangent_integral_n=self.tangent_integral_n,
            normal_correction_n=normal_correction,traction_ratio=1.)
        # A changed bearing set invalidates the aggregate slew history. Clear it
        # immediately so a lost digit cannot retain its former motor command.
        changed=((contact!=self.previous_contact)|switched).any(-1)
        previous=torch.where(changed[:,:,None],0.,self.previous_torque_nm)
        torque=previous+(torque-previous).clamp(-2*dt,2*dt)
        active=contact.any(-1)
        torque=torch.where(active[:,:,None],torque,0.)
        self.previous_torque_nm.copy_(torque)
        self.previous_contact.copy_(contact)
        diagnostic.update(requested_normal_n=requested_normal_n.clone(),measured_normal_n=hand_normal_n.clone(),
            normal_force_error_n=requested_normal_n-hand_normal_n,
            normal_integral_n=self.normal_integral_n.clone(),tangent_integral_n=self.tangent_integral_n.clone(),
            active_torque_command_nm=torque)
        return torque,diagnostic


def install_support_servo(world,task,controller):
    """Teacher-only wrapper: update real motor efforts at every physics step."""
    from .tabletop_tensor_world import write_contact_effort
    if hasattr(task,'contact_effort'):raise ValueError('one motor feedforward controller required')
    servo=SupportServo(world.n,world.device,world.parameters.effort_limit_nm)
    original_advance,original_reset=world.advance,world.reset
    world.contact_effort_enabled=True
    world.effort_feedforward=torch.zeros(world.n,2,6,device=world.device,dtype=torch.float64)
    world.surface_support_servo=servo
    window_size=int(getattr(world,'decimation',8))
    if window_size<1:raise ValueError('positive feedback window required')
    force_window=torch.zeros(window_size,world.n,2,2,device=world.device,dtype=torch.float64)
    controller.force_feedback_loads=torch.zeros_like(force_window[0])
    window_index=window_count=0

    def advance(t,allowed,live):
        nonlocal window_index,window_count
        m=world.measure();controller.refresh_contact_model(m)
        jac=controller.last['jacobian'][...,6:]
        inward=controller.last['inward'];loads=m['grouped_object_loads'][:,:,:2]
        components=world.vertical_components;weight=world.object_weight_n;velocity=world.wv[:,2]
        requested=t.f_requested
        valid=torch.ones(world.n,device=world.device,dtype=torch.bool)
        for value in (jac,inward,loads,components,weight,velocity,requested,m['hand_object_loads']):
            valid &= torch.isfinite(value).reshape(world.n,-1).all(-1)
        valid &= (weight>0)&(inward.norm(dim=-1)>0).flatten(1).all(-1)
        valid &= (loads>=0).flatten(1).all(-1)&((requested>=2)&(requested<=12)).all(-1)
        active=valid&live&~t.done
        def selected(value,fill=0.):
            return torch.where(active.reshape((world.n,)+(1,)*(value.ndim-1)),value,torch.full_like(value,fill))
        safe_normal=selected(inward)
        safe_normal=torch.where(active[:,None,None,None],safe_normal,inward.new_tensor([1.,0.,0.]))
        torque,diagnostic=servo.step(selected(jac),safe_normal,selected(loads),selected(components[...,0]),
            selected(components[...,1]),selected(weight,1.),selected(velocity),
            active&controller.last['lift_requested'],requested_normal_n=selected(requested,2.),dt=world.dt,
            hand_normal_n=selected(m['hand_object_loads']),contact_body_ids=controller.last.get('body_ids'),
            contact_model_valid=controller.last['contact_model_valid']&active[:,None,None])
        if getattr(controller,'pinch_hold',False):
            # Simple pinch/hold uses the original position drives at their full
            # effort budget. Measured force adjusts finger targets at control
            # rate; no additional contact-model torque competes with the drive.
            torque.zero_();servo.reset(torch.arange(world.n,device=world.device))
            for key in ('target_total_fz_n','target_hand_fz_n','target_tangent_z_n',
                        'digit_tangent_force_w_n','digit_normal_correction_force_w_n',
                        'torque_saturated','traction_saturated'):
                diagnostic[key].zero_()
            diagnostic.update(extra_motor_feedforward_applied=False,
                active_torque_command_nm=torque.clone(),normal_integral_n=servo.normal_integral_n.clone(),
                tangent_integral_n=servo.tangent_integral_n.clone())
        torque=torch.where(active[:,None,None],torque,0.)
        servo.reset((~active).nonzero().flatten())
        world.effort_feedforward.copy_(torque)
        for h,hand in enumerate(world.hands):
            write_contact_effort(hand,world.w.ids[h],torque[:,h],world.parameters.effort_limit_nm)
        diagnostic.update(feedback_valid=valid,active=active,raw_vertical_components_n=components.clone(),
            requested_normal_n=requested.clone(),measured_normal_n=m['hand_object_loads'].clone())
        world.surface_support_diagnostic=diagnostic
        measurement=original_advance(t,allowed,live)
        if getattr(controller,'pinch_hold',False):
            # Filter only the controller input; all raw measurements and
            # teacher acceptance stay untouched at the physics sampling rate.
            force_window[window_index].copy_(measurement['grouped_object_loads'][:,:,:2])
            window_index=(window_index+1)%window_size;window_count=min(window_count+1,window_size)
            controller.force_feedback_loads.copy_(force_window[:window_count].mean(0))
        return measurement

    def reset(ids):
        force_window[:,ids]=0.;controller.force_feedback_loads[ids]=0.
        servo.reset(ids);world.effort_feedforward[ids]=0.
        for h,hand in enumerate(world.hands):
            write_contact_effort(hand,world.w.ids[h],world.effort_feedforward[ids,h],
                world.parameters.effort_limit_nm,env_ids=ids)
        return original_reset(ids)
    world.advance=advance;world.reset=reset
    return servo
