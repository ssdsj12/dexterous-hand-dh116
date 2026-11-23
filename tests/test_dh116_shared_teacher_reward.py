"""Shared-source scalar endpoints go through the unchanged exactly-once R0."""
import pytest
import numpy as np
from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import ROOT
from go2_pvcnn.control.dh116.tabletop_training_reward import load_config,TrainingReward,SubmittedRequest


@pytest.mark.parametrize("grasp",["pinch","tripod","power"])
def test_original_r0_settles_eight_actual_intervals_once(grasp):
    from go2_pvcnn.control.dh116.tabletop_shared_teacher_reward import endpoint
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
    config=load_config(ROOT/f"configs/dh116_training/{grasp}_finger_goal_reward_v1.json",ppo_gamma=.995)
    hands={side:dict(joint_names=ACTIVE,q=np.asarray([target])) for side,target in
           zip(("left","right"),config.finger_goal.target_rad)}
    def sample(t):
        clock=dict(run_uuid="fixture",env_id=0,episode_uid="fixture-ep",reset_id=1,
                   rod_uid="fixture-rod",physics_step=t,rod_tick=t*12,
                   timestamp_s=t/480,solver_time_s=t/480,clock_valid=True,policy_step=0,substep=t%8)
        return endpoint(grasp,clock,distance_m=.02-t*.0001,health=0,guard=None,
                        hands=hands,finger_goal=config.finger_goal,scope="cpu_fixture")
    pre=sample(2)
    reward=TrainingReward(config)
    request=lambda token:SubmittedRequest(token,"policy:0",pre.stamp,(0.,)*24,
                                         "normalized_submitted_command24_preprojection")
    reward.reset_from_verified_state(pre,previous_request=request("reset"))
    reward.begin(request("window"),remaining_substeps=8)
    for tick in range(3,11):
        reward.observe(sample(tick))
    value=reward.settle_once("window",buffer_full=True)
    assert value.duration_s==8/480 and len(value.substep_terms)==8
    assert np.isfinite(value.reward) and not value.terminated
    with pytest.raises(RuntimeError):
        reward.settle_once("window")
