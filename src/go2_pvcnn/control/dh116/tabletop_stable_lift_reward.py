"""Measured lift progress and bounded low-slip holding credit."""
import math
import torch

CONTRACT = dict(version='measured_stable_lift_reward_v1',table_top_m=.30,
    target_height_m=.03,progress_points=5.,progress='once_per_episode_high_water_height',
    replaces='lift_height_difference',hold_min_height_m=.003,
    hold_points_per_weighted_s=4.,hold_weight='clipped_actual_height_over_0.03',
    hold_credit_budget_s=.5,hold_max_points=2.,max_speed_m_s=.03,max_slip_m=.002,
    gate='previous_phase_lift_and_live_finite_nonfailure_and_measured_bilateral_load_ready_and_nonnegative_vertical_force',
    vertical_force='measured_hand_normal_plus_friction_world_z',
    state_reset='selected_episode_reset_only',
    observation_append=['best_rewarded_lift_height_div_0.03','hold_credit_div_0.5'])


class StableLiftReward:
    def __init__(self,n,device):
        self.best_height_m=torch.zeros(n,device=device,dtype=torch.float64)
        self.hold_credit_s=torch.zeros_like(self.best_height_m)

    def reset(self,ids):
        self.best_height_m[ids]=0.
        self.hold_credit_s[ids]=0.

    def update(self,*,bottom,lifting,load_ready,speed,slip,vertical_force,eligible,dt):
        if not math.isfinite(dt) or dt<=0:
            raise ValueError('stable lift reward dt must be finite and positive')
        finite=torch.isfinite(bottom)&torch.isfinite(speed)&torch.isfinite(slip)&torch.isfinite(vertical_force)
        good=(eligible&lifting&load_ready&finite&(speed>=0)&(speed<.03)&
              (slip>=0)&(slip<=.002)&(vertical_force>=0))
        height=torch.where(good,(bottom-.30).clamp(0.,.03),0.)
        best=torch.maximum(self.best_height_m,height)
        progress=5.*(best-self.best_height_m)/.03
        self.best_height_m.copy_(best)
        holding=good&(bottom>=.303)
        weighted_time=torch.where(holding,dt*height/.03,0.)
        credit=torch.minimum(weighted_time,(.5-self.hold_credit_s).clamp_min(0.))
        self.hold_credit_s.add_(credit).clamp_(max=.5)
        return progress,4.*credit

    def observe(self):
        return torch.stack((self.best_height_m/.03,self.hold_credit_s/.5),dim=1)
