"""CPU source-lifecycle tests, no environment, policy, optimizer or SDK step."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_stereo_action_contract_audit import controller_fixture
from go2_pvcnn.control.dh116.tabletop_teacher_policy_owner_source import (
    PolicyOwnerSource, adapt_policy_owner, validate_policy_owner,
)


class Clock:
    def __init__(self):
        self.value=dict(run_uuid="cpu",env_id=8,episode_uid="ep0",reset_id=1,rod_uid="rod",
            physics_step=2,rod_tick=0,policy_step=0,substep=0,timestamp_s=2/480,
            solver_time_s=0.,clock_valid=True,phase="pre_action_before_target_write")

    def __call__(self):
        return deepcopy(self.value)

    def complete(self):
        self.value.update(physics_step=self.value["physics_step"]+8,
            rod_tick=self.value["rod_tick"]+96,policy_step=self.value["policy_step"]+1,
            timestamp_s=self.value["timestamp_s"]+8/480,
            solver_time_s=self.value["solver_time_s"]+8/480)


def fixture():
    t,w=controller_fixture()
    t.previous=t.command.clone()
    t._seed={k:getattr(t,k)[0].clone() for k in ("command","actions","previous_actions","previous")}
    clock=Clock()
    owner=PolicyOwnerSource(t,read_identity=clock,data_env_index=1,expected_env_id=8)
    return t,w,clock,owner


def ready():
    t,w,clock,owner=fixture()
    owner.record_reset()
    initial=owner.capture_pre_action()
    return t,w,clock,owner,initial


def submit(t,owner,value=.25):
    actions=torch.full((t.n,24),value,dtype=torch.float64)
    t.begin_step(actions)  # actual original method, no policy or physics
    owner.record_submission()


def complete(t,clock):
    t.command[1,0]+=.001234567890123
    # CPU lifecycle double: physical update/end_step are not run. Test the
    # observer's required original end_step side effect at a declared clock.
    t.previous.copy_(t.command)
    clock.complete()


def test_reset_requires_explicit_seed_and_yields_only_36_policy_fields():
    t,w,clock,owner=fixture()
    with pytest.raises(ValueError,match="reset"):owner.capture_pre_action()
    owner.record_reset()
    result=adapt_policy_owner(owner.capture_pre_action(),expected_identity=clock())
    assert np.flatnonzero(result["valid"]).tolist()==list(range(203,239))
    assert np.isnan(result["values"][~result["valid"]]).all()
    assert validate_policy_owner(result)["unknown_channels"]==335
    assert result["qualified_for_teacher_ppo"] is False


def test_original_begin_step_and_observe_parity_preserve_float64_command(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_dexhands_feedback
    from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import observe
    t,w,clock,owner,initial=ready()
    submit(t,owner)
    with pytest.raises(ValueError,match="complete"):owner.capture_pre_action()
    complete(t,clock)
    packet=owner.capture_pre_action()
    result=adapt_policy_owner(packet,expected_identity=clock())
    m=dict(q=t.command,wrist_pose=torch.zeros((3,2,7),dtype=torch.float64),
        wrist_velocity=torch.zeros((3,2,6),dtype=torch.float64),
        wrist_joint_angles=torch.zeros((3,2,3),dtype=torch.float64))
    stub=SimpleNamespace(m=m,free_wrist_world=SimpleNamespace(root_seed=torch.zeros((3,2,7)),
        wrist_angles=torch.zeros((3,2,3)),palms=torch.zeros((3,2,3))),
        command=t.command,low=t.low,high=t.high,rotation_targets=torch.zeros((3,2,3)),
        actions=t.actions,surface_force_sharing=False)
    monkeypatch.setattr(tabletop_dexhands_feedback,"geometry_features",lambda m:torch.zeros((3,20)))
    np.testing.assert_array_equal(result["values"][203:239],observe(stub)[1,50:86].numpy())
    assert packet["state"]["command"][0] == t.command[1,0].item()
    assert packet["state"]["command"][0] != t.command[1,0].float().double().item()
    assert packet["submission"]["identity"]["policy_step"]==0
    assert packet["identity"]["policy_step"]==1


def test_two_completed_intervals_and_owned_copies():
    t,w,clock,owner,_=ready()
    for value in (.25,-.4):
        submit(t,owner,value);complete(t,clock)
        packet=owner.capture_pre_action()
        assert packet["state"]["actions"]==[value]*24
    saved=deepcopy(packet)
    t.command[1]=99;t.actions[1]=99
    assert packet==saved
    assert validate_policy_owner(adapt_policy_owner(saved,expected_identity=clock()))["valid_channels"]==36


@pytest.mark.parametrize("edit",[
    lambda t:t._seed.pop("actions"),
    lambda t:setattr(t,"action_size",26),
    lambda t:setattr(t,"command",t.command.float()),
    lambda t:t.low.__setitem__(0,99),
    lambda t:t.actions.__setitem__((1,0),.1),
    lambda t:setattr(t,"surface_force_sharing",False),
])
def test_missing_seed_wrong_contract_dtype_limits_or_unrecorded_action_rejected(edit):
    t,w,clock,owner=fixture();edit(t)
    with pytest.raises(ValueError):owner.record_reset()


def test_no_inferred_default_seed_and_selected_per_environment_seed():
    t,w,clock,owner=fixture()
    t._per_env_reset_seed={"command":t.command.clone()}
    t._per_env_reset_seed["command"][1,0]+=.0001
    with pytest.raises(ValueError,match="seed"):owner.record_reset()
    t.command[1]=t._per_env_reset_seed["command"][1]
    owner.record_reset()
    assert owner.capture_pre_action()["state"]["command"]==t.command[1].tolist()


@pytest.mark.parametrize("edit",[
    lambda t:t.old_command.__setitem__((1,0),.5),
    lambda t:t.previous_actions.__setitem__((1,0),.5),
    lambda t:t.last_raw_action24.__setitem__((1,0),.5),
    lambda t:t.command.__setitem__((1,0),.5),
    lambda t:t.failure.__setitem__(1,1),
])
def test_submission_requires_original_accepted_begin_step_side_effects(edit):
    t,w,clock,owner,_=ready()
    t.begin_step(torch.full((3,24),.25,dtype=torch.float64));edit(t)
    with pytest.raises(ValueError):owner.record_submission()


def test_submission_requires_preaction_capture_and_is_exactly_once_per_interval():
    t,w,clock,owner=fixture();owner.record_reset()
    t.begin_step(torch.zeros((3,24),dtype=torch.float64))
    with pytest.raises(ValueError):owner.record_submission()
    owner.capture_pre_action();submit(t,owner)
    with pytest.raises(ValueError):owner.record_submission()


@pytest.mark.parametrize("edit",[
    lambda c:c.value.update(policy_step=2),
    lambda c:c.value.update(substep=1),
    lambda c:c.value.update(reset_id=2),
    lambda c:c.value.update(rod_uid="wrong"),
    lambda c:c.value.update(env_id=99),
    lambda c:c.value.update(clock_valid=False),
])
def test_skipped_partial_cross_reset_rod_or_invalid_clock_rejected(edit):
    t,w,clock,owner,_=ready();submit(t,owner);complete(t,clock);edit(clock)
    with pytest.raises(ValueError):owner.capture_pre_action()


def test_missing_end_step_action_mutation_and_unrecorded_command_rejected():
    t,w,clock,owner,_=ready()
    t.command[1,0]+=.01
    with pytest.raises(ValueError):owner.capture_pre_action()
    t,w,clock,owner,_=ready();submit(t,owner);complete(t,clock)
    t.previous[1,0]-=.01
    with pytest.raises(ValueError,match="end_step"):owner.capture_pre_action()
    t.previous.copy_(t.command);t.actions[1,0]=-.5
    with pytest.raises(ValueError,match="action"):owner.capture_pre_action()


def test_reset_identity_cannot_be_reused_and_new_reset_clears_submission():
    t,w,clock,owner,_=ready()
    with pytest.raises(ValueError,match="reset"):owner.record_reset()
    submit(t,owner);complete(t,clock);owner.capture_pre_action()
    clock.value.update(reset_id=2,episode_uid="ep1",policy_step=0,rod_tick=0)
    for key,value in t._seed.items():getattr(t,key)[1]=value
    owner.record_reset()
    packet=owner.capture_pre_action()
    assert packet["submission"] is None
    assert packet["state"]["actions"]==[0.]*24


def test_clock_advancing_during_copy_and_mutated_replay_are_rejected():
    t,w,clock,owner,_=ready()
    original=owner.read_identity
    def moving():
        value=original();clock.value["rod_tick"]+=1;return value
    owner.read_identity=moving
    with pytest.raises(ValueError,match="changed"):owner.capture_pre_action()
    t,w,clock,owner,_=ready()
    result=adapt_policy_owner(owner.capture_pre_action(),expected_identity=clock())
    result["values"][203]=99
    with pytest.raises(ValueError,match="reconstruct"):validate_policy_owner(result)
    result=adapt_policy_owner(owner.capture_pre_action(),expected_identity=clock())
    result["valid"][:]=True
    with pytest.raises(ValueError,match="reconstruct"):validate_policy_owner(result)


def test_plain_objects_sdk_target_relabel_and_nonfinite_input_cannot_supply_source():
    t,w,clock,owner=fixture()
    with pytest.raises(ValueError,match="TensorTask"):
        PolicyOwnerSource(SimpleNamespace(**vars(t)),read_identity=clock,data_env_index=1,expected_env_id=8)
    t.command[1,0]=float("nan")
    with pytest.raises(ValueError):owner.record_reset()


def test_replay_submission_clock_cannot_precede_or_skip_epoch_reset_boundary():
    t,w,clock,owner,_=ready();submit(t,owner);complete(t,clock)
    packet=owner.capture_pre_action()
    for target in (packet["identity"],packet["submission"]["identity"]):
        target["physics_step"]+=8
        target["timestamp_s"]+=8/480
    with pytest.raises(ValueError,match="reset"):
        adapt_policy_owner(packet,expected_identity=packet["identity"])


def test_replay_rejects_implementation_action_contract_and_execution_counter_tampering():
    t,w,clock,owner,_=ready()
    packet=owner.capture_pre_action()
    packet["action_contract_sha256"]="0"*64
    with pytest.raises(ValueError,match="source"):
        adapt_policy_owner(packet,expected_identity=clock())
    packet=owner.capture_pre_action()
    packet["implementation_sha256"]["tabletop_tensor_task.py"]="0"*64
    with pytest.raises(ValueError,match="source"):
        adapt_policy_owner(packet,expected_identity=clock())
    result=adapt_policy_owner(owner.capture_pre_action(),expected_identity=clock())
    result["optimizer_updates"]=1
    with pytest.raises(ValueError,match="reconstruction"):validate_policy_owner(result)


def test_actual_reset_and_end_step_methods_preserve_observer_lifecycle():
    t,w,clock,owner=fixture()
    # Only full-observation materialization is stubbed; actual reset/end_step
    # remain unchanged original methods and execute their buffer writes.
    t.keys=tuple(t.m)
    t.last_observation=torch.zeros((t.n,1),dtype=torch.float64)
    t.observe=lambda:torch.zeros_like(t.last_observation)
    t.command[1,0]+=.01;t.actions[1]=.7
    t.reset(torch.tensor([1]),{k:v.clone() for k,v in t.m.items()})
    owner.record_reset();owner.capture_pre_action()
    submit(t,owner)
    t.command[1,0]+=.001234567890123
    assert not torch.equal(t.previous[1],t.command[1])
    t.end_step()
    clock.complete()
    packet=owner.capture_pre_action()
    assert packet["state"]["command"]==t.command[1].tolist()
    assert validate_policy_owner(adapt_policy_owner(packet,expected_identity=clock()))["valid_channels"]==36


def test_accepted_clipped_action_is_copied_without_redecoding_or_rng_change():
    t,w,clock,owner,_=ready()
    rng=torch.get_rng_state().clone()
    before_command=t.command.clone()
    t.begin_step(torch.full((3,24),2.,dtype=torch.float64))
    owner.record_submission()
    assert torch.equal(t.command,before_command)
    complete(t,clock)
    state_command=t.command.clone();state_actions=t.actions.clone()
    packet=owner.capture_pre_action()
    assert packet["state"]["actions"]==[1.]*24  # original begin_step accepted clipping
    assert torch.equal(state_command,t.command) and torch.equal(state_actions,t.actions)
    assert torch.equal(rng,torch.get_rng_state())
