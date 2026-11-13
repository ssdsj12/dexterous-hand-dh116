"""Archived native reset + installed SDK Python getters; no app or solver.

Only the CPU tensor frontend/backend, timestamp containers and target flush are
doubles. Unrecorded link COM velocity is explicitly synthetic zero.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace,MethodType

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from go2_pvcnn.control.dh116.tabletop_deformable_bimanual_step import BimanualStepConnection,HandBinding,DT
from go2_pvcnn.control.dh116.tabletop_palm_mount import MountView
from go2_pvcnn.control.dh116.tabletop_teacher_mounted_wrist_source import (
    capture_mounted_hand,adapt_mounted_hands,require_mounted_teacher_common,SEED_SOURCE,SEED_VERSION)
from go2_pvcnn.control.dh116.tabletop_teacher_preaction_readback import PHASE

SDK=Path("/mnt/mydisk/xk/projects/IsaacLab/source/isaaclab/isaaclab")
API=Path("/mnt/mydisk/xk/envs/dh116/lib/python3.10/site-packages/isaacsim/extsPhysics/omni.physics.tensors/omni/physics/tensors/impl/api.py")
LEDGER=Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/approved-dual-pinch-zero-v3-20261113-aYoTfF/native-runs/flex.f7-dual-pinch-zero-interval-v3-native/attempt-n2pwu0f3/native/physical.jsonl")
PINS={
    LEDGER:"36cff198922a48525cedfcbf5bd68c3fff7d2cdbbe44eec2ca1ba9c948bb0168",
    API:"bb5e684dce25e0ad7f3e03ec395ec4729d55203f3b3a58bfa824b8d106c0e54b",
    SDK/"assets/articulation/articulation_data.py":"d69495c66c409fb20823e4e467f9d90e63163b2050b77d2ac8e433221d1d208a",
    SDK/"assets/articulation/articulation.py":"fb179aacdd848906367d01105020e1d36412bd0f1da7187be09f0781ff589309",
    SDK/"utils/math.py":"881ab27758816475e26d6721bab22c934fd70d0c064392266a90b06f5bbe4149",
}

def extract(path,names,scope,*,class_name=None):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==PINS[path]
    tree=ast.parse(path.read_text())
    nodes=[deepcopy(n) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    if class_name:
        nodes=[ast.ClassDef(name=class_name,bases=[],keywords=[],body=nodes,decorator_list=[])]
    future=ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future,*nodes],type_ignores=[])),str(path),"exec"),scope)
    return scope[class_name] if class_name else scope

@pytest.fixture
def native_boundary():
    assert hashlib.sha256(LEDGER.read_bytes()).hexdigest()==PINS[LEDGER]
    rows=[json.loads(line) for line in LEDGER.read_text().splitlines()]
    reset=next(decode_owned(row["payload"])["reset"] for row in rows if row.get("operation")=="reset")
    assert reset["status"]=="complete" and reset["new_sdk_steps"]==reset["new_solver_steps"]==0
    clock=dict(reset["after_clock"],phase=PHASE,policy_step=0,substep=0)
    seed=dict(version=SEED_VERSION,source=SEED_SOURCE,identity=deepcopy(clock),
        authored_hand_roots_w=reset["seed"]["hand_roots"],environment_origin_w=reset["seed"]["origin"],
        root_xy=reset["seed"]["root_xy"],root_seed_w=reset["seed"]["authored_root_seed_w"])
    math_scope=extract(SDK/"utils/math.py",{"convert_quat"},dict(torch=torch,np=np))
    names={"joint_pos","joint_vel","joint_acc","update","body_link_pose_w","body_com_vel_w",
           "body_com_lin_vel_w","body_com_ang_vel_w","body_lin_vel_w","body_ang_vel_w",
           "root_pose_w","root_link_pose_w"}
    Data=extract(SDK/"assets/articulation/articulation_data.py",names,
        dict(torch=torch,math_utils=SimpleNamespace(convert_quat=math_scope["convert_quat"])),class_name="Data")
    getters={"get_dof_positions","get_dof_velocities","get_dof_position_targets","get_dof_types",
             "get_link_transforms","get_link_velocities","get_root_transforms","get_dof_limits"}
    View=extract(API,getters,dict(float32=torch.float32,uint8=torch.uint8),class_name="View")
    setter=extract(SDK/"assets/articulation/articulation.py",{"set_joint_position_target"},dict(torch=torch))["set_joint_position_target"]
    hands=[];bindings=[];events=[]
    for side in ("left","right"):
        r=reset["hands"][side];view=View();view.count=1;view.max_dofs=17;view.max_links=23
        view.prim_paths=r["articulation_paths"];view.link_paths=r["link_paths"]
        values={key:torch.as_tensor(value,dtype=torch.float32).clone() for key,value in dict(
            dof_positions=r["q"],dof_velocities=r["qd"],dof_position_targets=r["backend_targets"],
            link_transforms=r["body_poses_xyzw"],root_transforms=r["root_pose_xyzw"]).items()}
        values["dof_types"]=torch.as_tensor(r["dof_types"],dtype=torch.uint8).clone()
        # These two fields were not recorded by the native reset. They are CPU
        # API-shape fixtures, not reconstructed native velocity/limit evidence.
        values["link_velocities"]=torch.zeros((1,23,6),dtype=torch.float32)
        values["dof_limits"]=torch.tensor([[[-1.,1.]]*17],dtype=torch.float32)
        view.values=values
        def create(shape,dtype,*args):
            tensor=torch.zeros(shape,dtype=dtype);return tensor,tensor
        view._frontend=SimpleNamespace(create_tensor=create)
        backend=SimpleNamespace()
        for key in values:
            def get(target,k=key,v=values):
                target.copy_(v[k]);return True
            setattr(backend,"get_"+key,get)
        view._backend=backend
        data=Data();data._root_physx_view=view;data._sim_timestamp=0.
        data._physics_sim_view=SimpleNamespace(update_articulations_kinematic=lambda:events.append("kinematics_only"))
        for key in ("joint_pos","joint_vel","joint_acc","body_link_pose_w","body_com_vel_w","root_link_pose_w"):
            setattr(data,"_"+key,SimpleNamespace(data=None,timestamp=-1.))
        data._previous_joint_vel=values["dof_velocities"].clone()
        data.joint_pos_target=torch.as_tensor(r["command_cache_targets"],dtype=torch.float32).clone()
        hand=SimpleNamespace(side=side,joint_names=r["joint_names"],body_names=r["body_names"],
            root_physx_view=view,data=data,_data=data)
        hand.set_joint_position_target=MethodType(setter,hand)
        def flush(h=hand):
            events.append(h.side+":flush")
            h.root_physx_view.values["dof_position_targets"].copy_(h.data.joint_pos_target)
        hand.write_data_to_sim=flush
        data.update(DT)
        hands.append(hand);bindings.append(HandBinding(side,hand,{0:view.prim_paths[0]},MountView(hand)))
    return SimpleNamespace(hands=hands,bindings=bindings,clock=clock,seed=seed,events=events,reset=reset)

def capture(state):
    packets=[]
    for binding in state.bindings:
        path=binding.env_paths[0]
        packets.append(capture_mounted_hand(binding.hand,binding.mount,side=binding.side,
            read_identity=lambda:deepcopy(state.clock),data_env_index=0,
            expected_articulation_path=path,expected_hand_container_path=path.rsplit("/",1)[0],
            reset_seed=state.seed))
    return adapt_mounted_hands(packets,expected_identity=state.clock)

def test_installed_cache_and_raw_getters_accept_archived_17dof_mount(native_boundary):
    s=native_boundary;raw=[h.root_physx_view.values["link_transforms"].clone() for h in s.hands]
    observation=capture(s)
    assert observation["valid"].sum()==66 and np.isnan(observation["values"][~observation["valid"]]).all()
    with pytest.raises(ValueError,match="305"):require_mounted_teacher_common(observation)
    assert observation["qualified_for_teacher_ppo"] is False
    for h,before in zip(s.hands,raw):
        assert h.root_physx_view.get_dof_limits().shape==(1,17,2)
        assert h.root_physx_view.get_dof_types().shape==(1,17)
        assert torch.equal(h.root_physx_view.values["link_transforms"],before)
    assert s.clock["physics_step"]==2 and s.clock["rod_tick"]==0

@pytest.mark.parametrize("amount",[0.,.001,-.001])
def test_original_nonzero_action_boundary_distinguishes_target_from_measured_motion(native_boundary,amount):
    s=native_boundary;capture(s)
    clock=lambda:[deepcopy(s.clock)]
    connection=BimanualStepConnection(env_ids=(0,),solver_substeps=12)
    action=torch.zeros((1,24),dtype=torch.float64);action[0,1]=action[0,7]=amount
    q=[h.root_physx_view.get_dof_positions().clone() for h in s.hands]
    request=connection.begin(s.bindings,action,read_clock=clock,palms=torch.zeros((1,2,3)),angles=torch.zeros((1,2,3)))
    roots=torch.tensor(np.stack([h.data.root_pose_w.numpy() for h in s.hands],axis=1),dtype=torch.float64)
    connection.submit(s.bindings,read_clock=clock,projected_root_poses=roots,
        projected_angles=torch.zeros((1,2,3)),projection_source="CPU unchanged native reset anchor; no geometry qualification")
    assert s.events.count("left:flush")==s.events.count("right:flush")==1
    for h,before in zip(s.hands,q):assert torch.equal(h.root_physx_view.get_dof_positions(),before)
    # Counter changes below are a simulated completed boundary, never an SDK step.
    s.clock.update(physics_step=3,rod_tick=12,timestamp_s=3*DT,solver_time_s=DT)
    result=connection.finish(s.bindings,read_clock=clock)
    assert result["new_physics_steps_executed_by_adapter"]==0 and result["qualified_for_teacher_ppo"] is False
    assert result["target_readback_is_not_measured_joint_tracking"] is True
    if amount:
        for h in s.hands:
            idx=h.joint_names.index("finger12")
            assert h.root_physx_view.get_dof_position_targets()[0,idx]!=h.root_physx_view.get_dof_positions()[0,idx]
    assert torch.equal(request["action24"],action)

def test_stale_cloned_link_pose_cache_rejects_without_any_target_write(native_boundary):
    s=native_boundary;capture(s)
    s.hands[0].root_physx_view.values["link_transforms"][0,6,0]+=.001
    with pytest.raises(ValueError,match="cache differs"):capture(s)
    assert not any("flush" in event for event in s.events)

@pytest.mark.parametrize("fault",["invalid_type","body_path","nonfinite_velocity"])
def test_actual_source_faults_fail_closed(native_boundary,fault):
    s=native_boundary
    if fault=="invalid_type":s.hands[0].root_physx_view.values["dof_types"][0,0]=255
    elif fault=="body_path":s.hands[0].root_physx_view.link_paths[0][6]="/World/other/base_link"
    else:s.hands[0].root_physx_view.values["link_velocities"][0,6,0]=float("nan")
    with pytest.raises(ValueError):capture(s)
    assert not any("flush" in event for event in s.events)

def test_action_range_failure_precedes_native_target_writes(native_boundary):
    s=native_boundary;action=torch.zeros((1,24));action[0,1]=1.001
    connection=BimanualStepConnection(env_ids=(0,),solver_substeps=12)
    with pytest.raises(ValueError,match="bounded"):
        connection.begin(s.bindings,action,read_clock=lambda:[deepcopy(s.clock)],
            palms=torch.zeros((1,2,3)),angles=torch.zeros((1,2,3)))
    assert not any("flush" in event for event in s.events)
