"""Behavioral regressions for continuous actuation and isolated state recovery."""
import copy
import pytest
import torch
from test_dh116_tensor_env import World
from test_dh116_tensor_task import template
from go2_pvcnn.control.dh116.tabletop_tensor_env import TensorEnv, physical_guard
from go2_pvcnn.control.dh116.tabletop_tensor_task import TensorTask, FAILURE_CODES


class FaultWorld(World):
    def __init__(self, task, fault, repeat=False, broken_reset=False):
        super().__init__(task)
        self.fault, self.repeat, self.broken_reset = fault, repeat, broken_reset
        self.seed = {k:v.clone() for k,v in self.m.items()}
        self.vertical_components=torch.zeros((2,2,2))
        self.object_weight_n=torch.ones(2)
        self.healthy_steps = 0

    def advance(self, task, allowed, live):
        self.tick += 1
        self.healthy_steps += int(live[1])
        if self.tick == 3 or (self.repeat and self.resets):
            if self.fault == 'pose':
                self.m['object_relative'][0,0] = float('nan')
            elif self.fault == 'velocity':
                self.qd[0,0,0] = float('inf')
            elif self.fault == 'overflow':
                self.m['object_relative'][0,0] = 1e100
            elif self.fault == 'force':
                self.m['object_link_loads'][0,0,0] = float('nan')
            elif self.fault == 'negative':
                self.m['object_link_loads'][0,0,0] = -1.
            elif self.fault == 'sum':
                self.m['hand_object_loads'][0,0] = 3.
            elif self.fault == 'mimic_nan':
                self.m['mimic_errors'][0,1] = float('nan')
                self.errors[0,0,1] = float('nan')
            elif self.fault == 'force_hard':
                self.m['object_link_loads'][0,0,0] = 46.
                self.m['grouped_object_loads'][0,0,5] = 46.
                self.m['hand_object_loads'][0,0] = 46.
            elif self.fault == 'geometry':
                self.m['clearance'][0] = -.002
            elif self.fault == 'components':
                self.vertical_components[0,0,0]=float('nan')
            elif self.fault in ('zero_weight','negative_weight'):
                self.object_weight_n[0]=0. if self.fault=='zero_weight' else -1.
        return self.m

    def reset(self, ids):
        self.resets.append(ids.tolist())
        for key,value in self.seed.items():
            self.m[key][ids] = value[ids]
        self.q[ids] = 0.
        self.qd[ids] = 0.
        self.errors[ids] = 0.
        self.vertical_components[ids]=0.
        if self.broken_reset=='overflow':
            self.m['object_relative'][ids,0]=1e100
        elif self.broken_reset=='sum':
            self.m['hand_object_loads'][ids,0]=3.
        elif self.broken_reset=='components':
            self.vertical_components[ids,0,0]=float('nan')
        elif self.broken_reset:
            self.qd[ids] = float('nan')
        return self.m


def make_env(fault, **options):
    task = TensorTask(template(),2,'cpu')
    task.continuous_motion = True
    world = FaultWorld(task,fault,**options)
    env = TensorEnv(world,task,recovery=True,training_joint_limit=True)
    return task,world,env


@pytest.mark.parametrize('error',[.0121,.02086,.031,-.04])
def test_finite_mimic_error_is_telemetry_without_reset(error):
    task,world,env = make_env('none')
    world.m['mimic_errors'][0,1] = error
    world.errors[0,0,1] = error
    world.m['joint_failure'][0] = FAILURE_CODES['mimic_error']
    obs,reward,done,extra = env.step(torch.zeros((2,20)))
    assert not done.any()
    assert not world.resets
    assert world.tick == 8 and world.healthy_steps == 8
    assert task.m['mimic_errors'][0,1] == pytest.approx(error)
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()


def test_continuous_guard_keeps_real_joint_limits():
    task,world,env = make_env('none')
    world.q[0,0,0] = 2.
    assert not physical_guard(world,world.m)[0][0]


def test_finite_normalized_action_saturates_instead_of_failing():
    task,world,env = make_env('none')
    actions = task.begin_step(torch.full((2,20),1.2))
    assert actions.max() == 1.
    assert not task.done.any()


@pytest.mark.parametrize('fault',['pose','velocity','overflow','force','negative','sum','mimic_nan'])
def test_invalid_state_resets_only_failed_slot_with_finite_transition(fault):
    task,world,env = make_env(fault)
    obs,reward,done,extra = env.step(torch.zeros((2,20)))
    assert done.tolist() == [True,False]
    assert world.resets == [[0]]
    assert world.tick == 8 and world.healthy_steps == 8
    assert extra['failure'][0] in (FAILURE_CODES['nonfinite_state'],FAILURE_CODES['nonfinite_joint_state'])
    assert extra['terms']['failure'].tolist() == [-25.,0.]
    assert extra['recovered'].tolist() == [True,False]
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    assert torch.isfinite(extra['terminal_observation']).all()
    assert env.episodes.tolist() == [1,0]
    assert env.recovered.tolist() == [1,0]
    env.step(torch.zeros((2,20)))
    assert world.resets == [[0]]


def test_nonfinite_state_after_reset_stops_without_repeated_reset():
    task,world,env = make_env('velocity',repeat=True)
    with pytest.raises(RuntimeError,match='post_reset_or_terminal'):
        env.step(torch.zeros((2,20)))
    assert world.resets == [[0]]
    assert world.tick == 4


def test_reset_returning_invalid_state_stops_immediately():
    task,world,env = make_env('velocity',broken_reset=True)
    with pytest.raises(RuntimeError,match='immediate_reset'):
        env.step(torch.zeros((2,20)))
    assert world.resets == [[0]] and world.tick == 3


@pytest.mark.parametrize('fault',['overflow','sum','components'])
def test_all_invalid_measurement_forms_are_checked_after_reset(fault):
    task,world,env=make_env(fault,repeat=True)
    with pytest.raises(RuntimeError,match='post_reset_or_terminal'):
        env.step(torch.zeros((2,20)))
    assert world.resets==[[0]] and world.tick==4


@pytest.mark.parametrize('fault',['overflow','sum','components'])
def test_reset_validates_policy_observation_and_load_consistency(fault):
    task,world,env=make_env('velocity',broken_reset=fault)
    with pytest.raises(RuntimeError,match='immediate_reset'):
        env.step(torch.zeros((2,20)))
    assert world.resets==[[0]]


@pytest.mark.parametrize('fault',['zero_weight','negative_weight'])
def test_unrecoverable_nonpositive_weight_stops_after_one_reset(fault):
    task,world,env=make_env(fault)
    with pytest.raises(RuntimeError,match='immediate_reset'):
        env.step(torch.zeros((2,20)))
    assert world.resets==[[0]] and world.tick==3


def test_continuous_recovery_requires_recovery_enabled():
    task,world,env = make_env('velocity')
    env.recovery = False
    with pytest.raises(RuntimeError):
        env.step(torch.zeros((2,20)))
    assert not world.resets


@pytest.mark.parametrize('fault',['force_hard','geometry'])
def test_other_invalid_physical_states_attempt_one_validated_reset(fault):
    task,world,env = make_env(fault)
    obs,reward,done,extra=env.step(torch.zeros((2,20)))
    assert done.tolist()==[True,False] and world.resets==[[0]]
    assert extra['recovered'].tolist()==[True,False]
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_actuator_saturation_does_not_turn_nonfinite_action_into_command(value):
    task,world,env = make_env('none')
    action=torch.zeros((2,20));action[0,0]=value
    bounded=task.begin_step(action)
    assert task.failure[0]==FAILURE_CODES['invalid_action']
    assert torch.isfinite(bounded).all() and not bounded[0].any()


def test_one_nonfinite_action_resets_failed_slot_before_more_physics():
    task,world,env=make_env('none')
    action=torch.zeros((2,20));action[0,0]=float('nan')
    obs,reward,done,extra=env.step(action)
    assert done.tolist()==[True,False] and world.resets==[[0]]
    assert extra['failure'][0]==FAILURE_CODES['invalid_action']
    assert extra['recovered'].tolist()==[True,False]
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    assert world.healthy_steps==8


def test_nonfinite_action_persisting_after_reset_stops_without_retry_loop():
    task,world,env=make_env('none')
    action=torch.zeros((2,20));action[0,0]=float('nan')
    env.step(action)
    with pytest.raises(RuntimeError,match='post_reset_invalid_action'):
        env.step(action)
    assert world.resets==[[0]]
