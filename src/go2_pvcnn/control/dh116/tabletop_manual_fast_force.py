"""Manual-only bounded micro-closing servo; never changes palm orientation."""
import torch

CONTRACT=dict(version='manual_micro_closing_force_v1',gain_rad_n_s=.4,
    rate_rad_s=.12,closure_budget_rad=.08,severe_mimic_rad=.012,
    can_acquire_contact=True,scope='explicit_manual_mode_only',
    outer_guards='mechanical_joint_slew_12N_release_mimic_and_clearance_unchanged')


def adjust(task,target,requested,live,dt):
    if getattr(task,'manual_digit_force_balance',False):
        from .tabletop_manual_digit_force_balance import adjust as digit_adjust
        return digit_adjust(task,target,requested,live,dt)
    loads=task.m['hand_object_loads']
    severe=task.m['mimic_errors'].reshape(-1,2,5).abs().amax(-1)>=.012
    change=(.4*(requested-loads)*dt).clamp(-.12*dt,.12*dt)
    if not getattr(task,'continuous_motion',False):
        change=torch.where(severe&(change>0),0.,change)
    task.f_offset[live]=(task.f_offset+change).clamp(-.08,.08)[live]
    task.f_target[live]=requested[live];task.f_requested[live]=requested[live]
    task.f_filtered[live]=loads[live]
    result=target.clone()
    corrected=target+(task.f_offset[:,:,None]*task.direction).flatten(1)
    result[live]=corrected.to(target)[live]
    return result
