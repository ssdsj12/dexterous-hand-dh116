"""Measured release -> reposition -> reclose during close, never during lift."""
import math
import torch

CONTRACT=dict(version='synchronized_release_reposition_reclose_v2',contact_load_n=.1,
    released_load_n=.01,recontact_load_n=.05,released_dwell_s=.05,request_distance_m=.0003,
    arrival_tolerance_m=.0002,arrival_dwell_s=.05,adjust_timeout_s=2.,
    reclose_timeout_s=1.,max_attempts_per_hand=2,release_joints='thumb_and_index',
    release_retreat_below_policy_rad=.03,unloaded_fingers='hold_command',
    goal_table_clearance_m=.0021,
    effective_clearance_floor='per_call_world_floor_legacy_default_when_omitted',
    shared_release='both_idle_hands_join_when_either_requests_loaded_descent',
    motion_timeout='exclude_shared_wait_for_other_hand_release',
    reclose_barrier='all_participating_hands_released_and_arrived',
    completion_barrier='all_participating_hands_reclosed_or_reclose_timeout',
    exhausted_request='abort_loaded_downward_request_after_attempt_limit',
    phases='close_only',policy_downward_positioning=True,
    actual_velocity_guaranteed=False,rewards_changed=False,hard_guards_changed=False)


class LoadedPalmReposition:
    def __init__(self,mask):
        self.mask=mask.to(dtype=torch.bool).clone();n=len(mask)
        self.state=torch.zeros((n,2),device=mask.device,dtype=torch.long)
        self.attempts=torch.zeros_like(self.state);self.aborted=torch.zeros_like(self.state,dtype=torch.bool)
        self.clock=torch.zeros((n,2),device=mask.device,dtype=torch.float64)
        self.dwell=torch.zeros_like(self.clock)
        self.goal=torch.zeros((n,2,3),device=mask.device,dtype=torch.float64)
        self.hold=torch.zeros_like(self.goal)
        self.goal_clipped=torch.zeros_like(self.aborted)
        self.exhausted_reposition=torch.zeros_like(self.aborted)

    @property
    def active(self):return self.state!=0

    def reset(self,ids):
        for name in ('state','attempts','aborted','clock','dwell','goal','hold','goal_clipped','exhausted_reposition'):
            getattr(self,name)[ids]=0

    def apply(self,target,current,loads,phase,live,dt,*,clearance=None,clearance_floor_m=None,held_hands=None):
        floor=CONTRACT['goal_table_clearance_m'] if clearance_floor_m is None else float(clearance_floor_m)
        if not math.isfinite(floor) or floor<0:
            raise ValueError('clearance floor must be finite and nonnegative')
        selected=self.mask[:,None]&live[:,None]&(phase==1)[:,None]
        if held_hands is not None:
            held_hands=torch.as_tensor(held_hands,device=loads.device,dtype=torch.bool)
            if held_hands.shape!=loads.shape:raise ValueError('held_hands must have shape (N, 2)')
            selected = selected & ~held_hands
        eligible_start=(selected&(self.state==0)&~self.aborted
               &(self.attempts<CONTRACT['max_attempts_per_hand']))
        requested_start=(eligible_start
               &(loads>CONTRACT['contact_load_n'])
               &(target[:,:,2]<current[:,:,2]-CONTRACT['request_distance_m']))
        # Both hands share one Z trajectory. An unloaded idle hand must hold
        # its fingers too, rather than start closing during the other's motion.
        start=eligible_start&requested_start.any(1,keepdim=True)&~(selected&(self.state==3)).any(1,keepdim=True)
        self.goal[start]=target[start];self.hold[start]=current[start]
        self.state[start]=1;self.clock[start]=0.;self.dwell[start]=0.;self.attempts[start]+=1
        # A contact regained during motion must be released again before moving.
        recontact=selected&(self.state==2)&(loads>CONTRACT['recontact_load_n'])
        self.state[recontact]=1;self.hold[recontact]=current[recontact];self.dwell[recontact]=0.
        shared_wait=(self.state==2)&(selected&(self.state==1)).any(1,keepdim=True)
        self.clock+=torch.where(selected&self.active&~shared_wait,dt,0.)
        releasing=selected&(self.state==1)
        self.dwell[releasing]=torch.where(loads[releasing]<=CONTRACT['released_load_n'],
                                        self.dwell[releasing]+dt,0.)
        released=releasing&(self.dwell>=CONTRACT['released_dwell_s']-1e-12)
        if clearance is not None:
            safe_z=current[:,:,2]-(clearance-floor).clamp_min(0.)
            clipped=released&(self.goal[:,:,2]<safe_z)
            self.goal_clipped|=clipped
            self.goal[:,:,2]=torch.where(clipped,safe_z,self.goal[:,:,2])
        self.state[released]=2;self.dwell[released]=0.
        moving=selected&(self.state==2)&~released
        arrived=(self.goal-current).norm(dim=2)<=CONTRACT['arrival_tolerance_m']
        self.dwell[moving]=torch.where(arrived[moving]&(loads[moving]<=CONTRACT['recontact_load_n']),
                                     self.dwell[moving]+dt,0.)
        arrived_ready=moving&(self.dwell>=CONTRACT['arrival_dwell_s']-1e-12)
        participating=selected&((self.state==1)|(self.state==2))
        finished=arrived_ready&(~participating|arrived_ready).all(1,keepdim=True)
        timeout=selected&((self.state==1)|(self.state==2))&(self.clock>=CONTRACT['adjust_timeout_s'])
        self.aborted|=timeout;self.goal[timeout]=current[timeout]
        self.state[finished|timeout]=3;self.clock[finished|timeout]=0.;self.dwell[finished|timeout]=0.
        reclosing=selected&(self.state==3)&~(finished|timeout)
        self.dwell[reclosing]=torch.where(loads[reclosing]>CONTRACT['contact_load_n'],
                                        self.dwell[reclosing]+dt,0.)
        reclosed=reclosing&((self.dwell>=.05-1e-12)|(self.clock>=CONTRACT['reclose_timeout_s']))
        complete=reclosed&(~(selected&self.active)|reclosed).all(1,keepdim=True)
        self.state[complete]=0;self.dwell[complete]=0.;self.clock[complete]=0.
        exhausted=(selected&(self.state==0)&~self.aborted
                   &(self.attempts>=CONTRACT['max_attempts_per_hand'])
                   &(loads>CONTRACT['contact_load_n'])
                   &(target[:,:,2]<current[:,:,2]-CONTRACT['request_distance_m']))
        self.exhausted_reposition|=exhausted;self.aborted|=exhausted
        result=target.clone()
        # Small loaded downward requests wait for a release cycle; unloaded
        # positioning remains unrestricted within the existing motion bounds.
        small=selected&(self.state==0)&(loads>CONTRACT['contact_load_n'])
        result[:,:,2]=torch.where(small,torch.maximum(result[:,:,2],current[:,:,2]),result[:,:,2])
        wait=selected&(self.state==1);move=selected&((self.state==2)|(self.state==3))
        result[wait]=self.hold[wait];result[move]=self.goal[move]
        return result,selected&((self.state==1)|(self.state==2))
