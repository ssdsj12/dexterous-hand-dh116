"""CPU target-authority regression for real Task reset after first terminal.

These fixtures exercise existing target buffers only, never native execution
or fixed-contact/geometry qualification. Stage FSM remains explicit opt-in.
"""
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('stage,hands',[('A','right'),('B','both')])
@pytest.mark.parametrize('closed_offset',[0.,.03])
def test_retired_reset_slot_has_no_previous_episode_transport_authority(stage,hands,closed_offset):
    from go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime import TripodRecoveryRuntime
    runtime=TripodRecoveryRuntime.__new__(TripodRecoveryRuntime)
    runtime.active=True;runtime.stage=stage;runtime.recovery_hands=hands
    runtime.tripod=torch.tensor([True,True,False]);runtime.ended=torch.tensor([True,False,False])
    runtime.fixed_offset=torch.full((3,),.03,dtype=torch.float64)
    runtime.fixed_palms=torch.arange(18,dtype=torch.float64).reshape(3,2,3)/1000
    runtime.fixed_angles=-torch.arange(18,dtype=torch.float64).reshape(3,2,3)/100
    world=SimpleNamespace(palms=torch.full((3,2,3),.002,dtype=torch.float64),
        wrist_angles=torch.full((3,2,3),-.003,dtype=torch.float64),roots=torch.ones(3,2,7))
    task=SimpleNamespace(offset=torch.tensor([closed_offset,.03,0.],dtype=torch.float64),
        palm_targets=world.palms.clone(),rotation_targets=world.wrist_angles.clone())
    original={name:getattr(world,name).clone() for name in ('palms','wrist_angles','roots')}
    task_before={name:getattr(task,name).clone() for name in ('offset','palm_targets','rotation_targets')}
    # TensorEnv has physically reset slot0 and Task.reset makes live true on
    # the next policy tick. runtime.ended continues to retire that first trial.
    live=torch.ones(3,dtype=torch.bool);allowed=live.clone()
    runtime.prepare_transport(world,task,allowed,live)
    for i in (0,2):
        assert torch.equal(world.palms[i],original['palms'][i])
        assert torch.equal(world.wrist_angles[i],original['wrist_angles'][i])
        assert torch.equal(task.palm_targets[i],task_before['palm_targets'][i])
        assert torch.equal(task.rotation_targets[i],task_before['rotation_targets'][i])
    assert torch.equal(world.palms[1],runtime.fixed_palms[1])
    assert torch.equal(task.palm_targets[1],runtime.fixed_palms[1])
    assert torch.equal(world.wrist_angles[1],runtime.fixed_angles[1])
    assert torch.equal(task.rotation_targets[1],runtime.fixed_angles[1])
    assert torch.equal(task.offset,task_before['offset']) and torch.equal(world.roots,original['roots'])
    assert live.all() and allowed.all()
