"""Teacher-only force exploration; PPO retains ownership of its force actions."""
import math
import torch

CONTRACT=dict(version='exploratory_normal_goals_v2_latched_hold',range_n=[2.,12.],
    initial='independent_uniform_per_hand_and_candidate',waypoint_interval_s=1.,
    waypoint='bounded_seeded_random_walk',goal_slew_n_s=1.,fixed_force=False,
    optional_pinch_hold='latch_chosen_force;continuous_feedback_remains_enabled')


class ForceGoalExplorer:
    def __init__(self,n,device,seed=23,initial_max_n=12.):
        if not isinstance(n,int) or n<1:raise ValueError('positive environment count required')
        if not math.isfinite(initial_max_n) or not 2<=initial_max_n<=12:
            raise ValueError('initial force search ceiling within2–12N required')
        self.generator=torch.Generator(device=device);self.generator.manual_seed(seed)
        self.goal=2+(initial_max_n-2)*torch.rand(n,2,device=device,dtype=torch.float64,generator=self.generator)
        self.waypoint=self.goal.clone();self.elapsed=0.;self.next_waypoint=.5
        self.holding=False

    def hold(self):
        """Latch the chosen force; feedback keeps running against this command."""
        self.holding=True;self.waypoint=self.goal.clone()

    def step(self,dt):
        if not math.isfinite(dt) or not 0<dt<=1/60+1e-12:raise ValueError('positive control dt at most1/60 required')
        if self.holding:return self.goal.clone()
        self.elapsed+=dt
        if self.elapsed>=self.next_waypoint:
            self.waypoint=(self.goal+.8*torch.randn(self.goal.shape,device=self.goal.device,
                dtype=self.goal.dtype,generator=self.generator)).clamp(2,12)
            self.next_waypoint+=1.
        self.goal=(self.goal+(self.waypoint-self.goal).clamp(-dt,dt)).clamp(2,12)
        return self.goal.clone()

    def actions(self):return (self.goal-2)/5-1
