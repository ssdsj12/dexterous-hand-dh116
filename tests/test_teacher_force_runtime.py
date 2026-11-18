"""CPU execution/source wiring only; never native contact qualification."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"scripts/train_dh116_gpu_teacher_ppo.py"

def module():
    from go2_pvcnn.control.dh116 import tabletop_teacher_force_runtime as m
    return m

def packet(loaded=True, substeps=12):
    ids=[];values=[]
    if loaded:
        for s in range(substeps):
            for mesh,direction in ((0,1),(1,-1),(6,1),(7,-1)):
                ids.append([s,0,mesh])
                values.append([0.,0.,0., direction,0.,0., .25*direction,0.,0.,0.,0.,0.])
    indices=np.asarray(ids,dtype=np.int64).reshape(-1,3)
    vectors=np.asarray(values,dtype=float).reshape(-1,12)
    nf=np.zeros((41,3));mw=np.zeros((13,6))
    for (_,node,mesh),v in zip(indices,vectors):
        nf[node]+=v[6:9]/substeps
        mw[mesh,:3]-=v[6:9]/substeps
    return dict(substeps=substeps,mesh_count=13,capacity=256,attempted=len(ids),
        overflow=False,indices=indices,vectors=vectors,
        node_forces_on_rod_world_n=nf,mesh_wrenches_on_mesh_world=mw)

def state():
    def hand():
        data=NS(body_link_pos_w=torch.zeros(1,1,3),body_link_quat_w=torch.tensor([[[1.,0.,0.,0.]]]),
            joint_pos=torch.zeros(1,17),joint_vel=torch.zeros(1,17),
            joint_pos_limits=torch.tensor([[[-2.,2.]]*17]),applied_torque=torch.zeros(1,17))
        return NS(data=data,joint_names=[str(i) for i in range(17)])
    class Rod:
        time=0.
        current_packet=packet(False)
        def contact_pair_samples(self):return self.current_packet
    d=dict(N=1,args=NS(mode="pinch"),sim=NS(current_time_step_index=2,current_time=2/480),
        rope=Rod(),rope_pos_view=torch.zeros(1,41,3),left_hands=hand(),right_hands=hand(),
        l_base_id=0,r_base_id=0,l_wrist_ids=list(range(11,17)),r_wrist_ids=list(range(11,17)),
        l_wrist_pos=torch.zeros(1,3),r_wrist_pos=torch.zeros(1,3),
        l_wrist_ang=torch.zeros(1,3),r_wrist_ang=torch.zeros(1,3),
        l_q=torch.zeros(1,6),r_q=torch.zeros(1,6),l_loc=torch.zeros(1,6),r_loc=torch.zeros(1,6))
    return d

def advance(d,hook,loaded=True):
    raw=torch.ones(1,24)
    hook.begin_control(raw,d)
    issued=hook.before_substep(raw,d)
    d["sim"].current_time_step_index+=1
    d["sim"].current_time+=1/480
    d["rope"].time+=1/480
    d["rope"].current_packet=packet(loaded)
    hook.after_substep(issued,d)
    return issued

def test_source_hooks_cover_every_physics_substep_and_single_reset():
    m=module();src=SOURCE.read_text()
    tree=m.compile_force_runtime(src,hashlib.sha256(src.encode()).hexdigest())
    text=ast.unparse(tree)
    step=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="step_env")
    loop=next(n for n in step.body if isinstance(n,ast.For))
    calls=[n for n in ast.walk(loop) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert sum(n.func.attr=="before_substep" for n in calls)==1
    assert sum(n.func.attr=="after_substep" for n in calls)==1
    assert "enable_contact_pair_capture" in text
    assert "reset_episode" in text and "begin_control" in text
    assert "GetCollisionEnabledAttr().Set(False)" not in text
    assert "GetCollisionEnabledAttr().Set(True)" in text
    assert "0.3 * DT" in text and "0.01 * DT" in text
    compile(tree,"force-source-wiring","exec")

def test_factory_preflight_is_before_native_imports():
    m=module();src=SOURCE.read_text()
    tree=m.compile_force_runtime(src,hashlib.sha256(src.encode()).hexdigest())
    factory=tree.body[0]
    assert ast.unparse(factory.body[0]).startswith("force_hook.validate_arguments(")
    with pytest.raises(ValueError,match="source changed"):
        m.compile_force_runtime(src,"0"*64)

def test_no_force_freezes_both_wrists_but_preserves_finger_commands():
    rows=[];h=module().ForceContactHook(grasp="pinch",run_uid="cpu",emit=rows.append)
    d=state();h.reset_episode(d)
    issued=advance(d,h,False)
    assert torch.equal(issued[:,:12],torch.ones(1,12))
    assert not issued[:,12:].any()
    assert rows[-1]["native_success"] is False
    assert rows[-1]["phase"]=="Close"
    assert rows[-1]["source_action"]==[[1.]*24]
    assert rows[-1]["issued_action"]==issued.tolist()
    assert rows[-1]["closure"]["bilateral"] is False
    assert rows[-1]["policy_step"]==1
    assert rows[-1]["physics_step"]==3
    assert rows[-1]["contact_slip_m"] is None

def test_force_hold_is_elapsed_endpoints_not_number_of_samples_and_loss_blocks_next_step():
    rows=[];h=module().ForceContactHook(grasp="pinch",run_uid="cpu",emit=rows.append)
    d=state();h.reset_episode(d)
    for _ in range(121):advance(d,h,True)
    # First endpoint lacks measured velocity; second starts the .25s candidate clock.
    assert rows[-1]["phase"]=="PreliftHold"
    advance(d,h,True)
    assert rows[-1]["phase"]=="Lift"
    issued=advance(d,h,False)
    assert issued[0,14]==1
    assert rows[-1]["phase"]=="Close"
    assert not advance(d,h,True)[:,12:].any()
    assert not any(r["native_success"] for r in rows)

def test_force_loss_cancels_pending_target_lead_using_actual_pose_once():
    h=module().ForceContactHook(grasp="pinch",run_uid="cpu",emit=lambda row:None)
    d=state();h.reset_episode(d)
    for _ in range(122):advance(d,h,True)
    advance(d,h,False)
    d["l_wrist_pos"].fill_(.08)
    d["left_hands"].data.body_link_pos_w.fill_(.002)
    advance(d,h,False)
    assert torch.all(d["l_wrist_pos"]==.002)
    d["left_hands"].data.body_link_pos_w.fill_(.003)
    advance(d,h,False)
    assert torch.all(d["l_wrist_pos"]==.002)  # latch, do not chase drift

def test_reset_allocates_unique_episode_and_never_reuses_contact():
    rows=[];h=module().ForceContactHook(grasp="pinch",run_uid="cpu",emit=rows.append)
    d=state();h.reset_episode(d);advance(d,h)
    old=rows[-1]["episode_uid"]
    h.reset_episode(d);assert not advance(d,h,False)[:,12:].any()
    assert rows[-1]["episode_uid"]!=old

def test_unknown_packet_aborts_instead_of_fabricating_no_contact_or_success():
    rows=[];h=module().ForceContactHook(grasp="pinch",run_uid="cpu",emit=rows.append)
    d=state();h.reset_episode(d)
    h.begin_control(torch.ones(1,24),d);a=h.before_substep(torch.ones(1,24),d)
    d["sim"].current_time_step_index+=1;d["sim"].current_time+=1/480;d["rope"].time+=1/480
    d["rope"].current_packet["overflow"]=True
    with pytest.raises(ValueError):h.after_substep(a,d)
    assert rows[-1]["terminal"] is True
    assert rows[-1]["failure"]=="invalid_contact_packet"
    with pytest.raises(RuntimeError,match="faulted"):h.before_substep(a,d)

def test_clock_gap_and_joint_terminal_are_preserved():
    rows=[];h=module().ForceContactHook(grasp="pinch",run_uid="cpu",emit=rows.append)
    d=state();h.reset_episode(d);h.begin_control(torch.ones(1,24),d)
    a=h.before_substep(torch.ones(1,24),d)
    d["sim"].current_time_step_index+=2
    with pytest.raises(RuntimeError,match="clock"):h.after_substep(a,d)
    assert rows[-1]["terminal"]
    rows=[];h=module().ForceContactHook(grasp="pinch",run_uid="cpu",emit=rows.append)
    d=state();h.reset_episode(d)
    d["right_hands"].data.joint_pos[0,1]=-2.021
    with pytest.raises(RuntimeError,match="joint"):advance(d,h)
    assert rows[-1]["hands"]["right"]["joint_position_rad"][0][1]==pytest.approx(-2.021)
    assert rows[-1]["failure"]=="joint_position_limit"
    assert rows[-1]["physics_step"]==3

def test_preflight_rejects_training_multi_env_and_invalid_observer():
    h=module().ForceContactHook(grasp="pinch",run_uid="cpu",emit=lambda r:None)
    for args in (NS(num_envs=3,updates=0,eval_only=True,mode="pinch"),
                 NS(num_envs=1,updates=1,eval_only=True,mode="pinch"),
                 NS(num_envs=1,updates=0,eval_only=False,mode="pinch"),
                 NS(num_envs=1,updates=0,eval_only=True,mode="tripod")):
        with pytest.raises(ValueError):h.validate_arguments(args)
    with pytest.raises(ValueError):module().ForceContactHook(grasp="pinch",run_uid="cpu",emit=None)


def test_extracted_real_control_loop_calls_force_gate_all_eight_times():
    """Execute actual source integration and SDK-target calls on CPU test doubles."""
    m=module();d=state();rows=[]
    hook=m.ForceContactHook(grasp="pinch",run_uid="cpu-loop",emit=rows.append)
    d["left_hands"].data.body_com_pos_w=torch.zeros(1,1,3)
    d["right_hands"].data.body_com_pos_w=torch.zeros(1,1,3)
    def decorate(hand):
        hand.permanent_wrench_composer=NS(reset=lambda:None,set_forces_and_torques=lambda **k:None)
        hand.targets=[]
        def target(value,*,joint_ids):
            hand.targets.append((list(joint_ids),value.clone()))
        hand.set_joint_position_target=target
        hand.write_data_to_sim=lambda:None
        hand.update=lambda dt:None
    decorate(d["left_hands"]);decorate(d["right_hands"])
    class Sim:
        current_time_step_index=2
        current_time=2/480
        def step(self,render=False):
            self.current_time_step_index+=1;self.current_time+=1/480
    d["sim"]=Sim()
    d["rope"].last_error=None
    d["rope"].set_mesh_poses_all=lambda *args:None
    def free(dt,n):
        assert n==12
        d["rope"].time+=dt
        d["rope"].current_packet=packet(False)
    d["rope"].step_free=free
    d.update(force_hook=hook,force_state=d.copy(),
        DT=1/480,SUBSTEPS_PER_CTRL=8,torch=torch,
        q_lo=torch.full((6,),-2.),q_hi=torch.full((6,),2.),
        l_pad_ids=[0]*5,r_pad_ids=[0]*5,l_all_ids=list(range(11)),
        r_all_ids=list(range(11)),wrench_view=torch.zeros(1,64,6),
        mesh_pos=torch.zeros(1,13,3),mesh_quat=torch.zeros(1,13,4),
        tele_buf=torch.zeros(1,dtype=torch.bool),
        local_translation=lambda p,root:p,
        _expand_11=lambda q:torch.cat((q,torch.zeros(1,5)),dim=-1))
    d["left_hands"].data.root_pose_w=torch.zeros(1,7)
    d["right_hands"].data.root_pose_w=torch.zeros(1,7)
    src=SOURCE.read_text()
    tree=m.compile_force_runtime(src,hashlib.sha256(src.encode()).hexdigest())
    step=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="step_env")
    end=next(i for i,n in enumerate(step.body) if isinstance(n,ast.For))
    step.body=step.body[:end+1]+ast.parse("return act").body
    nonlocals=[n.id for n in ast.walk(step.body[0]) if isinstance(n,ast.Name)]
    # Nonlocal declaration stores string names, not Name nodes.
    nonlocals=step.body[0].names
    for n in nonlocals:
        if n not in d:d[n]=torch.zeros(1,24)
    outer=ast.FunctionDef(name="build",args=ast.arguments(posonlyargs=[],args=[
        ast.arg(arg="context")],kwonlyargs=[],kw_defaults=[],defaults=[]),
        body=[ast.Assign(targets=[ast.Name(id=k,ctx=ast.Store())],
            value=ast.Subscript(value=ast.Name(id="context",ctx=ast.Load()),
                slice=ast.Constant(k),ctx=ast.Load())) for k in d if k.isidentifier()]
            +[step,ast.Return(value=ast.Name(id="step_env",ctx=ast.Load()))],
        decorator_list=[])
    code=ast.fix_missing_locations(ast.Module(body=[outer],type_ignores=[]))
    ns={};exec(compile(code,"actual-eight-substep-source","exec"),ns)
    hook.reset_episode(d)
    act=ns["build"](d)(torch.ones(1,24))
    physical=[r for r in rows if r["event"]=="physical_sample"]
    assert len(physical)==8
    assert [r["physics_step"] for r in physical]==list(range(3,11))
    assert [r["policy_substep"] for r in physical]==list(range(1,9))
    assert {r["policy_step"] for r in physical}=={1}
    assert all(not np.asarray(r["issued_action"])[0,12:].any() for r in physical)
    assert all(np.array_equal(np.asarray(r["issued_action"])[0,:12],np.ones(12)) for r in physical)
    # Both wrists retain actual zero pose while finger targets integrate 8 steps.
    assert len(d["right_hands"].targets)==16
    assert d["right_hands"].targets[-1][1][0,:3].tolist()==[0.,0.,0.]
    assert d["right_hands"].targets[-2][1][0,0].item()==pytest.approx(.005)

def test_nonfinite_geometry_cannot_establish_prelift_qualification():
    rows=[];h=module().ForceContactHook(grasp="pinch",run_uid="cpu",emit=rows.append)
    d=state();h.reset_episode(d);d["rope_pos_view"][0,0,0]=float("nan")
    with pytest.raises(RuntimeError,match="geometry"):advance(d,h)
    assert rows[-1]["terminal"] and rows[-1]["failure"]=="invalid_physical_geometry"


def test_solver_error_is_logged_before_legacy_raise_without_invented_force():
    rows=[];h=module().ForceContactHook(grasp="pinch",run_uid="cpu",emit=rows.append)
    d=state();h.reset_episode(d)
    h.begin_control(torch.ones(1,24),d)
    a=h.before_substep(torch.ones(1,24),d)
    d["sim"].current_time_step_index+=1;d["sim"].current_time+=1/480;d["rope"].time+=1/480
    d["rope"].last_error="actual solver failure"
    with pytest.raises(RuntimeError,match="solver"):h.after_substep(a,d)
    assert rows[-1]["terminal"] and rows[-1]["failure"]=="solver_error"
    assert "closure" not in rows[-1]
    assert rows[-1]["hands"]["right"]["joint_position_rad"]==[[0.]*17]

def test_physics_exception_is_logged_once_without_advancing_clock():
    rows=[];h=module().ForceContactHook(grasp="pinch",run_uid="cpu",emit=rows.append)
    d=state();h.reset_episode(d)
    h.begin_control(torch.ones(1,24),d);h.before_substep(torch.ones(1,24),d)
    h.fail_substep(RuntimeError("SDK failure"),d)
    h.fail_substep(RuntimeError("outer duplicate"),d)
    assert len([r for r in rows if r.get("terminal")])==1
    assert rows[-1]["physics_step"]==2
    assert rows[-1]["failure"]=="physical_step_error"
    assert rows[-1]["contact_valid"] is False
    with pytest.raises(RuntimeError,match="faulted"):h.before_substep(torch.ones(1,24),d)

def test_durable_exclusive_ledger_flushes_terminal_and_preserves_nonfinite(tmp_path):
    import json
    path=tmp_path/"physical.jsonl"
    with module().ForceDiagnosticLedger(path) as writer:
        writer(dict(terminal=True,position=[float("nan"),float("inf")],native_success=False))
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        assert rows[0]["position"]==["NaN","+Infinity"]
        assert rows[0]["nonfinite_encoding"]=="tagged_non_measurements"
        assert writer.rows_written==1
        with pytest.raises(FileExistsError):module().ForceDiagnosticLedger(path)
    assert len(path.read_text().splitlines())==1


def test_original_unclamped_action_is_distinct_from_bounded_issued_action():
    rows=[];h=module().ForceContactHook(grasp="pinch",run_uid="cpu",emit=rows.append)
    d=state();h.reset_episode(d)
    raw=torch.ones(1,24)*2
    h.begin_control(raw,d)
    issued=h.before_substep(raw.clamp(-1,1),d)
    d["sim"].current_time_step_index+=1;d["sim"].current_time+=1/480;d["rope"].time+=1/480
    h.after_substep(issued,d)
    assert rows[-1]["source_action"]==[[2.]*24]
    assert rows[-1]["clipped_source_action"]==[[1.]*24]
    assert rows[-1]["action_clipping_occurred"] is True
    assert issued[:,:12].eq(1).all() and issued[:,12:].eq(0).all()


@pytest.mark.parametrize("clock",["sdk","rod"])
def test_time_changes_without_tick_are_rejected_before_any_command(clock):
    h=module().ForceContactHook(grasp="pinch",run_uid="cpu",emit=lambda r:None)
    d=state();h.reset_episode(d);h.begin_control(torch.ones(1,24),d)
    if clock=="sdk":d["sim"].current_time+=1/480
    else:d["rope"].time+=1/480
    d["l_wrist_pos"].fill_(.08)
    with pytest.raises(RuntimeError,match="clock"):h.before_substep(torch.ones(1,24),d)
    assert d["l_wrist_pos"].eq(.08).all()

def test_hook_cannot_change_native_environment_identity_inside_episode_or_reset():
    h=module().ForceContactHook(grasp="pinch",run_uid="cpu",emit=lambda r:None)
    d=state();h.reset_episode(d);h.begin_control(torch.ones(1,24),d)
    replaced=dict(d);replaced["right_hands"]=state()["right_hands"]
    with pytest.raises(ValueError,match="identity"):h.before_substep(torch.ones(1,24),replaced)
    with pytest.raises(ValueError,match="identity"):h.reset_episode(replaced)


def test_failed_next_control_never_claims_previous_action_was_issued():
    rows=[];h=module().ForceContactHook(grasp="pinch",run_uid="cpu",emit=rows.append)
    d=state();h.reset_episode(d);advance(d,h)
    h.begin_control(torch.zeros(1,24),d)
    d["sim"].current_time+=1/480
    try:h.before_substep(torch.zeros(1,24),d)
    except RuntimeError as error:h.fail_substep(error,d)
    assert rows[-1]["issued_action"] is None
    assert rows[-1]["projected_action"] is None
    assert rows[-1]["source_action"]==[[0.]*24]
    assert rows[-1]["execution_status"]=="not_confirmed_complete"
