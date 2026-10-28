"""Control-step potential shaping and once-per-episode physical lift milestones."""
import math
import torch

LIFT_GATE_NAMES=('lift_substeps','command_advanced','mimic_blocked','tracking_blocked',
                 'load_blocked','slip_blocked','pressure_blocked','height_cap')


def lift_gate_summary(counts):
    values=counts.cpu().tolist();n=values[0]
    return dict(lift_substeps=int(n),scope='live_lift_physics_substeps',blocked_gates_overlap=True,
                **{name+'_fraction':value/n if n else None for name,value in zip(LIFT_GATE_NAMES[1:],values[1:])})


CONTRACT=dict(version='force_progress_lift_milestones_v1',gamma=.995,
    potential='2*tanh(ema_signed_hand_to_object_force_over_runtime_weight)',filter_tau_s=.05,
    timing='once_per_ppo_control_transition',terminal_next_potential=0.,reset_potential=0.,
    rollout_boundary='preserve_potential',milestone_heights_m=[.003,.010],
    milestone_points=[2.,3.],milestone_dwell_s=.1,table_top_m=.30,
    milestone_condition='lift_and_measured_load_ready_and_speed_lt_0.03_and_slip_le_0.02',
    milestone_frequency='once_each_per_episode',
    observation_append=['force_potential_div_2','milestone_dwell_3mm','milestone_dwell_10mm',
                        'milestone_paid_3mm','milestone_paid_10mm'])


class GraspShaping:
    def __init__(self,n,device):
        self.force_ratio=torch.zeros(n,device=device,dtype=torch.float64)
        self.dwell=torch.zeros((n,2),device=device,dtype=torch.float64)
        self.paid=torch.zeros((n,2),device=device,dtype=torch.bool)
        self.before=torch.zeros_like(self.force_ratio)
        self.live=torch.zeros(n,device=device,dtype=torch.bool)
        self.threshold=self.force_ratio.new_tensor([.303,.310])
        self.points=self.force_ratio.new_tensor([2.,3.])
        self.pending=False

    def reset(self,ids):
        self.force_ratio[ids]=0.;self.dwell[ids]=0.;self.paid[ids]=False
        self.before[ids]=0.;self.live[ids]=False

    def begin(self,live):
        self.before.copy_(2*self.force_ratio.tanh());self.live.copy_(live);self.pending=True

    def update(self,force,weight,bottom,lifting,stable,eligible,dt):
        safe_force=torch.where(eligible,force,0.)
        safe_weight=torch.where(eligible,weight,1.)
        ratio=safe_force/safe_weight
        filtered=self.force_ratio+(1-math.exp(-dt/.05))*(ratio-self.force_ratio)
        self.force_ratio.copy_(torch.where(eligible,filtered,self.force_ratio))
        ready=(eligible&lifting&stable)[:,None]&(bottom[:,None]>=self.threshold)
        self.dwell.copy_(torch.where(ready,torch.clamp(self.dwell+dt,max=.1),0.))
        event=ready&(self.dwell>=.1-1e-12)&~self.paid
        self.paid|=event
        return (event*self.points).sum(1)

    def end(self,done):
        if not self.pending:return torch.zeros_like(self.force_ratio)
        self.pending=False
        after=torch.where(done,0.,2*self.force_ratio.tanh())
        return torch.where(self.live,CONTRACT['gamma']*after-self.before,0.)

    def observe(self):
        return torch.cat((self.force_ratio.tanh()[:,None],self.dwell/.1,self.paid),dim=1)
