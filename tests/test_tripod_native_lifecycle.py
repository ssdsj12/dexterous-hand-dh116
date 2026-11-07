"""Execute original environment boundaries with CPU SDK fixtures, not Isaac."""
from types import SimpleNamespace as NS
import pytest
import torch
from test_dh116_tensor_env import World, TerminalWorld, template, TensorTask, TensorEnv
from go2_pvcnn.control.dh116.tabletop_tensor_world import TensorWorld


def test_native_fatal_evidence_is_captured_before_exception_and_before_any_reset():
    task=TensorTask(template(),2,'cpu'); world=World(task)
    env=TensorEnv(world,task,recovery=False); rows=[]
    env.control_probe=NS(before_step=lambda:None,after_step=lambda *a:None,
        on_physics_failure=lambda mask,m,context:rows.append((world.tick,context,
            m['object_link_loads'].clone(),list(world.resets))))
    with pytest.raises(RuntimeError,match='independent_guard'):
        env.step(torch.zeros(2,20))
    assert len(rows)==1 and rows[0][:2]==(3,'independent_guard')
    assert rows[0][2][0,0,0]==31 and not rows[0][3]


def test_terminal_probe_observes_final_physics_before_immediate_reset():
    task=TensorTask(template(),2,'cpu');world=TerminalWorld(task);env=TensorEnv(world,task)
    task.elapsed[0]=4.01;rows=[]
    env.control_probe=NS(before_step=lambda:None,
        after_step=lambda *a:rows.append((world.tick,task.done.clone(),world.m['bottom'].clone(),
                                        list(world.resets))))
    env.step(torch.zeros(2,20))
    assert rows[0][0]==1 and rows[0][1][0] and rows[0][2][0]==pytest.approx(.34)
    assert not rows[0][3] and world.reset_ticks==[1]


def reset_world():
    def noop(*args,**kwargs):pass
    hands=[NS(reset=noop,write_joint_state_to_sim=noop,write_root_velocity_to_sim=noop,
        set_joint_position_target=noop,write_joint_stiffness_to_sim=noop,write_data_to_sim=noop,
        actuators={'active':NS(stiffness=torch.zeros(2,6))}) for _ in range(2)]
    return NS(n=2,device='cpu',hands=hands,seed_q=[torch.zeros(2,6)]*2,
        seed_qd=[torch.zeros(2,6)]*2,root_seed=torch.zeros(2,2,7),initial_offset=0.,write_root_pose=noop,
        w=NS(ids=[list(range(6))]*2,sensors=[NS(reset=noop)],
            sim=NS(physics_sim_view=NS(update_articulations_kinematic=noop))),
        root_targets=torch.zeros(2,2,7),commands=torch.zeros(2,2,6,dtype=torch.float64),
        palms=torch.zeros(2,2,3),gains=torch.ones(2,2),
        wire=NS(reset=noop,write_root_pose_to_sim=noop,write_root_velocity_to_sim=noop),
        seed_wire=torch.zeros(2,7),seed_velocity=torch.zeros(2,6),
        link_loads=torch.ones(2,2,17),forces=torch.ones(2,4),vertical_components=torch.ones(2,2,2),
        contact_center_w=torch.ones(2,2,3),contact_normal_w=torch.ones(2,2,3),
        contact_normal_load_n=torch.ones(2,2),contact_valid=torch.ones(2,2,dtype=torch.bool),
        contact_all_below_centerline=torch.ones(2,2,dtype=torch.bool),contact_count=torch.ones(2,2,dtype=torch.long),
        refresh=lambda sensors=False:{'fresh_pose':True},_physics_steps=7,
        tripod_contact_snapshot=dict(complete=True,valid_envs=torch.ones(2,dtype=torch.bool)))


def test_actual_reset_notifies_runtime_after_fresh_read_and_snapshot_invalidation():
    world=reset_world();rows=[]
    def observed(ids,m):
        rows.append((ids.clone(),m,world.commands.clone(),world.tripod_contact_snapshot))
    world.tripod_recovery_runtime=NS(after_physical_reset=observed)
    TensorWorld.reset(world,torch.tensor([1]))
    assert len(rows)==1 and rows[0][0].tolist()==[1] and rows[0][1]['fresh_pose']
    assert rows[0][3]['valid_envs'].tolist()==[True,False]


def test_empty_reset_does_not_replenish_any_runtime_budget():
    world=reset_world()
    world.measure=lambda:{}
    world.tripod_recovery_runtime=NS(after_physical_reset=lambda *a:pytest.fail('no physical reset'))
    TensorWorld.reset(world,torch.tensor([],dtype=torch.long))
