"""One non-restarting, one-second contact reward window per episode."""
import torch

REWARD_VERSION='contact_window_v2_1s_10n_5max'
OBSERVATION_VERSION='dh116-free-grasp-contact-window-v2'


def reward_contract(start):
    if start not in ('first_contact','close_start'):
        raise ValueError('contact reward window requires first_contact or close_start')
    return dict(version=REWARD_VERSION,start=start,window_s=1.,contact_threshold_n=.02,
        force_full_scale_n=10.,rate_points_per_s=5.,episode_max_points=5.,
        restart_on_contact_loss=False,restart_on_phase_change=False,
        grasp_ready_points=3.,grasp_ready_once_per_episode=True)


def window_credit(clock,paid,force,eligible,dt,start):
    """Clock -1 is unstarted, 0..1 runs and 1 is permanently expired."""
    spec=reward_contract(start)
    started=(clock>=0)|(force>=spec['contact_threshold_n'])
    if start=='close_start':started=torch.ones_like(started)
    elapsed=clock.clamp_min(0.)
    available=torch.clamp(1.-elapsed,min=0.,max=dt)
    amount=5.*torch.clamp(force/10.,0.,1.)*available
    credit=torch.where(eligible&started,torch.minimum(amount,(5.-paid).clamp_min(0.)),0.)
    next_clock=torch.where(eligible&started,torch.clamp(elapsed+dt,max=1.),clock)
    return next_clock,credit
