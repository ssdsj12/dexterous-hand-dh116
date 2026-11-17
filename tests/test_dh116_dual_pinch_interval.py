"""One real builder/episode control interval, with explicit CPU SDK execution."""
import gc
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from test_dh116_dual_pinch_builder import setup,prepared,geometry
from go2_pvcnn.control.dh116 import tabletop_dual_pinch_builder as builder
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_runtime import run_owned_call
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE,JOINTS,expand
from go2_pvcnn.control.dh116.tabletop_teacher_mounted_wrist_source import require_mounted_teacher_common

ROOT=Path(__file__).resolve().parents[1]
DT=1/480


def install_sdk_execution(sim,rod,bindings,events,*,fault=None):
    """Extend published builder doubles at the actual post-reset boundary."""
    for binding in bindings:
        hand=binding.hand
        hand._cpu_link_velocity=torch.zeros((1,23,6))
        hand.root_physx_view.get_link_velocities=lambda h=hand:h._cpu_link_velocity
        def update(dt,h=hand):
            events.append(h.side+":interval_cache")
            h.data.joint_pos=h.measured.clone();h.data.joint_vel=h.velocity.clone()
            h.data.body_link_pose_w=h.link_poses[..., [0,1,2,6,3,4,5]].clone()
            h.data.body_lin_vel_w=h._cpu_link_velocity[...,:3].clone()
            h.data.body_ang_vel_w=h._cpu_link_velocity[...,3:].clone()
        hand.update=update
        hand.update(DT)
        composer=hand.permanent_wrench_composer
        def write_wrench(forces,torques,body_ids=None,*,c=composer,**kwargs):
            if body_ids is None:
                c.composed_force_as_torch[:]=forces;c.composed_torque_as_torch[:]=torques
            else:
                c.composed_force_as_torch[:,body_ids]=forces
                c.composed_torque_as_torch[:,body_ids]=torques
            c.active=True
        composer.set_forces_and_torques=write_wrench
        def flush(h=hand):
            events.append(h.side+":interval_write")
            h.backend[:]=h.data.joint_pos_target
            h.actuation[:]=h.effort_target
            h.instantaneous_wrench_composer.reset()
        hand.write_data_to_sim=flush

    def sdk_step(*,render):
        assert render is False
        events.append("sdk_step")
        old_poses=[b.hand.link_poses.clone() for b in bindings]
        for binding in bindings:
            hand=binding.hand
            old=hand.measured.clone()
            active=[hand.joint_names.index(n) for n in ACTIVE]
            full=expand(hand.backend[0,active].numpy())
            # Simulated passive mimic response; commands remain original 17DOF.
            hand.measured[:]=hand.backend
            for i,name in enumerate(JOINTS):
                hand.measured[0,hand.joint_names.index(name)]=float(full[i])
            hand.velocity[:]=(hand.measured-old)/DT
        sim.current_time_step_index+=1
        sim.current_time=sim.current_time_step_index*DT
        sim.physics_sim_view.update_articulations_kinematic()
        for binding,old in zip(bindings,old_poses):
            hand=binding.hand
            hand._cpu_link_velocity[...,:3]=(hand.link_poses[...,:3]-old[...,:3])/DT
            old_rotation=Rotation.from_quat(old[0,:,3:].numpy())
            new_rotation=Rotation.from_quat(hand.link_poses[0,:,3:].numpy())
            hand._cpu_link_velocity[0,:,3:]=torch.tensor(
                (new_rotation*old_rotation.inv()).as_rotvec()/DT,dtype=torch.float32)
    sim.step=sdk_step

    def pairs():
        rope=rod.rope
        ids=np.array([[s,0,slot] for s in range(12) for slot in (0,6)],dtype=np.int64)
        vectors=np.zeros((24,12));vectors[:,5]=1.;vectors[:,8]=np.tile([1.,2.],12)
        if fault=="pairs":ids[-1]=ids[0]
        return dict(substeps=12,mesh_count=11,capacity=41*11*12,attempted=len(ids),overflow=False,
            indices=ids,vectors=vectors,node_forces_on_rod_world_n=rope.force[0].copy(),
            mesh_wrenches_on_mesh_world=rope.wrenches[0].copy())
    rod.rope.contact_pair_samples=pairs


def interval(monkeypatch,tmp_path,prepared,geometry,*,fault=None,stats=None):
    sim,trace,reset_owners,sync,record,rows,events,refs,meta,drop=setup(
        monkeypatch,tmp_path,prepared,geometry)
    stats={} if stats is None else stats
    def operation(owners):
        episode=rod=bindings=None
        try:
            setup_result=builder.build_reset_owned(sim,owners,project=ROOT,prepared=prepared,
                ownership=trace,record=record,owner_resets=reset_owners,
                run_uuid="cpu-interval",episode_uid="cpu-interval:paired",physics_budget_end=10)
            episode=owners.resources["dual_episode"];rod=owners.resources["runtime"]
            bindings=owners.resources["hand_bindings"]
            install_sdk_execution(sim,rod,bindings,events,fault=fault)
            if fault=="dof":bindings[1].hand.joint_names[0]="not_original"
            pre=episode.pre_action_state()
            assert pre["common"]["valid"].sum()==66
            with pytest.raises(ValueError,match="305"):
                require_mounted_teacher_common(pre["common"])
            action=torch.zeros((1,24),dtype=torch.float64)
            # Small original finger-rate command, no new wrist motion or safety floor.
            action[0,1]=action[0,7]=.001
            try:
                result=episode.control(action)
            except Exception:
                with pytest.raises(RuntimeError,match="active"):episode.control(action)
                stats["retry_rejected"]=True
                raise
            post=episode.pre_action_state()
            with pytest.raises(RuntimeError,match="budget"):episode.control(action)
            return dict(setup=setup_result,pre=pre,post=post,result=result,
                sdk_step=sim.current_time_step_index,rod_clock=rod.clock(),
                solver_steps=rod.rope.steps,joint_names=[list(b.hand.joint_names) for b in bindings])
        finally:
            stats.update(sdk_step=sim.current_time_step_index,
                         solver_steps=0 if rod is None else rod.rope.steps,
                         physical_rows=sum(r.get("kind")=="physical" for r in rows))
            episode=rod=bindings=None
    try:
        result=run_owned_call(operation,synchronize=sync,ownership=trace)
    finally:
        # The SDK double itself owns the hand list and step closure, unlike the
        # production owner that clears those references during STOP.
        drop();sim.step=None
        # CPU test-only cycle collection after releasing the explicit fixture
        # owners; production ownership cleanup is unchanged.
        gc.collect()
        stats["live_aliases"]=[name for name,ref in refs.items() if ref() is not None]
    return result,rows,events,trace,refs


def test_complete_eight_tick_interval_uses_real_builder_reset_readback_and_control(monkeypatch,tmp_path,prepared,geometry):
    stats={}
    result,rows,events,trace,refs=interval(monkeypatch,tmp_path,prepared,geometry,stats=stats)
    assert stats==dict(sdk_step=10,solver_steps=8,physical_rows=8,live_aliases=[])
    assert result["setup"]["setup_steps"]==2
    assert result["result"]["physical_samples"]==8 and result["result"]["termination_reason"] is None
    assert result["result"]["training_reward"] is None and result["result"]["teacher_admission"] is False
    assert result["sdk_step"]==10 and result["solver_steps"]==8
    assert result["rod_clock"]["rod_tick"]==96
    assert result["rod_clock"]["solver_time_s"]==pytest.approx(8/480)
    assert all(len(names)==17 for names in result["joint_names"])
    assert result["pre"]["common"]["valid"].sum()==result["post"]["common"]["valid"].sum()==66
    valid=result["pre"]["common"]["valid"]
    assert np.any(result["pre"]["common"]["values"][valid]!=result["post"]["common"]["values"][valid])
    assert np.isnan(result["post"]["common"]["values"][~result["post"]["common"]["valid"]]).all()
    physical=[row for row in rows if row.get("kind")=="physical"]
    assert len(physical)==8
    for index,row in enumerate(physical):
        exchange=row["exchange"];rod=exchange["rod"];step=index+3
        assert exchange["physics_step"]==rod["physics_step"]==step
        assert rod["rod_tick"]==12*(index+1)
        assert row["rod_boundary"]["rod_snapshot"]["physics_step"]==step
        assert exchange["reaction_source_physics_step"]==(None if index==0 else step-1)
        assert rod["rod_mesh_contact_pairs"]["physx_contact_pairs"] is None
        assert len(rod["rod_mesh_contact_pairs"]["pairs"])==24
        assert all(r["mesh_index"] in (0,6) for r in rod["rod_mesh_contact_pairs"]["pairs"])
        assert row["teacher_admission"] is False
        for side,amount in (("left",1.),("right",2.)):
            issued=exchange["issued_composer_wrench"][side]
            assert np.sum(issued["forces"][...,2])==pytest.approx(0. if index==0 else -amount)
    assert events.count("sdk_step")==8
    assert result["setup"]["clearance_floor_m"]==.0021


@pytest.mark.parametrize("fault",["dof","pairs"])
def test_interval_fault_never_replays_or_claims_complete_state(monkeypatch,tmp_path,prepared,geometry,fault):
    stats={}
    with pytest.raises(RuntimeError):
        interval(monkeypatch,tmp_path,prepared,geometry,fault=fault,stats=stats)
    assert stats["sdk_step"]==(2 if fault=="dof" else 3)
    assert stats["solver_steps"]==(0 if fault=="dof" else 1)
    assert stats["physical_rows"]==0 and stats["live_aliases"]==[]
    assert stats.get("retry_rejected",False)==(fault=="pairs")
