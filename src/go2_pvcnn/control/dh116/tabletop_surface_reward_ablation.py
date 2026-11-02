"""Stageable native reward signals for the DH116 surface Student."""
import math
import torch

PROFILES=('baseline','lift_progress','early_hold','contact_break','slip_relative','support_deficit')
STAGE_TERMS={
    'baseline':(),
    'lift_progress':('surface_lift_progress',),
    'early_hold':('surface_lift_progress','surface_early_hold_rate','surface_hold_milestone'),
    'contact_break':('surface_lift_progress','surface_early_hold_rate','surface_hold_milestone','surface_contact_break'),
    'slip_relative':('surface_lift_progress','surface_early_hold_rate','surface_hold_milestone','surface_contact_break','surface_slip_window','surface_relative_motion'),
    'support_deficit':('surface_lift_progress','surface_early_hold_rate','surface_hold_milestone','surface_contact_break','surface_slip_window','surface_relative_motion','surface_support_deficit'),
}
MILESTONE_SECONDS=(.1,.3,.5,1.,2.,5.)
MILESTONE_POINTS=(1.,2.,2.,3.,4.,5.)


def ablation_contract(profile):
    if profile not in PROFILES: raise ValueError('unknown staged reward profile')
    return dict(version='surface_stable_hold_reward_v3',profile=profile,
        enabled_terms=list(STAGE_TERMS[profile]),stage_order=list(PROFILES),
        height='8*new_qualified_height_ratio_per_episode;no_absolute_height_repetition',
        early_hold='2_points_per_second;qualified_3mm_window;max10_per_episode',
        milestones=dict(seconds=list(MILESTONE_SECONDS),points=list(MILESTONE_POINTS),max_points=17),
        slip='-24*clamp((fixed_anchor_drift/0.0005)^2,0,4)*dt;valid_3mm_window_only',
        contact_break='-4_per_event;two_invalid_control_windows;max2_per_episode',
        relative_motion='-2*clamp((tangent_relative_speed/0.002)^2,0,4)*dt;valid_contact_only',
        support_deficit='-1.5*clamp((mg-Fz)/mg,0,1)*dt;airborne_valid_contact_only',
        success='unchanged_30mm_5s_0.5mm',teacher_servo=False)


class RewardAblation:
    def __init__(self,profile,bottom):
        if profile not in PROFILES: raise ValueError('unknown staged reward profile')
        self.profile=profile;self.reference=bottom.clone();n=len(bottom)
        self.best_ratio=torch.zeros_like(bottom)
        self.milestone_paid=torch.zeros(n,6,dtype=torch.bool,device=bottom.device)
        self.was_stable=torch.zeros(n,dtype=torch.bool,device=bottom.device)
        self.invalid_windows=torch.zeros(n,dtype=torch.long,device=bottom.device)
        self.break_events=torch.zeros(n,dtype=torch.long,device=bottom.device)

    def reset(self,ids,bottom):
        ids=torch.as_tensor(ids,device=self.best_ratio.device,dtype=torch.long)
        self.reference[ids]=bottom[ids]
        self.best_ratio[ids]=0.;self.milestone_paid[ids]=False
        self.was_stable[ids]=False;self.invalid_windows[ids]=0;self.break_events[ids]=0

    @torch.no_grad()
    def update(self,*,bottom,qualified,early_window_good,early_duration,drift,
               relative_speed,support,weight,valid,dt):
        if not math.isfinite(dt) or dt<=0: raise ValueError('positive physical dt required')
        vals=(bottom,early_duration,drift,relative_speed,support,weight)
        if any(v is None or not bool(torch.isfinite(v).all()) for v in vals):
            raise ValueError('finite native reward measurements required')
        if bool((weight<=0).any()): raise ValueError('positive runtime object weight required')
        valid=valid.bool();qualified=qualified.bool()&valid
        result={name:torch.zeros_like(bottom) for name in STAGE_TERMS[self.profile]}
        if self.profile!='baseline':
            ratio=((bottom-self.reference-.0005)/.03).clamp(0,1)
            new_best=torch.where(qualified,torch.maximum(self.best_ratio,ratio),self.best_ratio)
            result['surface_lift_progress']=8.*(new_best-self.best_ratio)
            self.best_ratio.copy_(new_best)
        early=qualified&early_window_good.bool()&(bottom>=self.reference+.003)
        if self.profile in ('early_hold','contact_break','slip_relative','support_deficit'):
            result['surface_early_hold_rate']=2.*early.to(bottom.dtype)*dt
            levels=bottom.new_tensor(MILESTONE_SECONDS)
            reached=early_duration[:,None]>=levels[None,:]
            pulse=reached&~self.milestone_paid&early[:,None]
            result['surface_hold_milestone']=(pulse*bottom.new_tensor(MILESTONE_POINTS)[None,:]).sum(-1)
            self.milestone_paid|=pulse
        if self.profile in ('contact_break','slip_relative','support_deficit'):
            self.invalid_windows=torch.where(early,self.invalid_windows*0,self.invalid_windows+1)
            broke=self.was_stable&(self.invalid_windows>=2)&(self.break_events<2)
            result['surface_contact_break']=torch.where(broke,bottom.new_tensor(-4.),torch.zeros_like(bottom))
            self.break_events+=broke.long()
            self.invalid_windows=torch.where(broke,self.invalid_windows*0,self.invalid_windows)
            self.was_stable|=(early&(early_duration>=.1))
        if self.profile in ('slip_relative','support_deficit'):
            drift_penalty=24.*((drift/.0005).square().clamp_max(4.))*dt
            relative_penalty=2.*((relative_speed/.002).square().clamp_max(4.))*dt
            result['surface_slip_window']=-drift_penalty*early.to(bottom.dtype)
            result['surface_relative_motion']=-relative_penalty*early.to(bottom.dtype)
        if self.profile=='support_deficit':
            deficit=((weight-support).clamp_min(0.)/weight).clamp(0.,1.)
            result['surface_support_deficit']=-1.5*deficit*dt*early.to(bottom.dtype)
        return result


def enable(task):
    from .tabletop_surface_success import SurfaceLiftSuccess
    task.surface_reward_ablation=RewardAblation(task.surface_reward_profile,
        task.surface_reference_bottom_m.expand(task.n))
    task.surface_early_hold=SurfaceLiftSuccess(task.n,task.device,task.surface_reference_bottom_m,
        dtype=task.command.dtype,minimum_lift_m=.003)
    for name in STAGE_TERMS[task.surface_reward_profile]:
        task.terms[name]=torch.zeros_like(task.reward)


def _relative_tangent_speed(task):
    m=task.m;loads=m['effective_pad_loads_n'];valid=m['effective_pad_valid']
    points=m['effective_pad_points_w'];vectors=m['effective_pad_vectors_w']
    center=m['object_relative']+m['wrist_pose'][...,:3].mean(1)
    object_velocity=m['object_velocity'];obj_v=object_velocity[:,:3];obj_w=object_velocity[:,3:]
    wrist=m['wrist_pose'][...,:3];wrist_v=m['wrist_velocity'][...,:3];wrist_w=m['wrist_velocity'][...,3:]
    obj_contact=obj_v[:,None,None]+torch.linalg.cross(obj_w[:,None,None],points-center[:,None,None],dim=-1)
    wrist_contact=wrist_v[:,:,None]+torch.linalg.cross(wrist_w[:,:,None],points-wrist[:,:,None],dim=-1)
    relative=obj_contact-wrist_contact
    normal=(vectors*loads[...,None]).sum(-2);normal=normal/normal.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    tangent=relative-(relative*normal).sum(-1,keepdim=True)*normal
    weights=torch.where(valid,loads,torch.zeros_like(loads))
    return (tangent.norm(dim=-1)*weights).sum((1,2))/weights.sum((1,2)).clamp_min(1e-8)


def apply_reward(task,pieces,valid,dt,hand_lift_force_n,object_weight_n):
    if task.surface_reward_profile=='baseline': return
    from .tabletop_surface_exploration import qualified_pad_contact,verified_support
    if hand_lift_force_n is None or object_weight_n is None: raise ValueError('native support and weight required')
    _,support=verified_support(task,hand_lift_force_n)
    tracker=getattr(task,'surface_early_hold',None)
    if tracker is None: raise ValueError('3mm native early hold tracker required')
    terms=task.surface_reward_ablation.update(bottom=task.m['bottom'],
        qualified=qualified_pad_contact(task),early_window_good=tracker._window_good,
        early_duration=tracker.duration,drift=tracker.measured_drift_m,
        relative_speed=_relative_tangent_speed(task),support=support.sum(-1),
        weight=object_weight_n,valid=valid,dt=dt)
    for name,value in terms.items(): pieces[name]=value
