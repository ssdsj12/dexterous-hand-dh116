"""Explicit CPU measurements: health is not motion or grasp success."""
from copy import deepcopy
import importlib
import numpy as np
import pytest
from test_dh116_deformable_control_step import inputs
from test_dh116_deformable_grasp_task import snapshot,signals,measurement
from test_dh116_r0_reward_domain import no_bilateral
from test_dh116_r0_applicability import reset_measurement
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import RESET_HOOKS
from go2_pvcnn.control.dh116.linear_coupling import JOINTS,ACTIVE,active_limits,expand
from go2_pvcnn.control.dh116.tabletop_joint_target_readback import WRIST_LINEAR,WRIST_ANGULAR
from go2_pvcnn.control.dh116 import tabletop_deformable_clearance_guard as clearance

MODULE="go2_pvcnn.control.dh116.tabletop_r0_health_separation"


def api():return importlib.import_module(MODULE)


def frame(task,step,*,dz=0.):
    sample=snapshot(task,step,dz=dz)
    rod=sample["rod_snapshot"]
    nodes=np.zeros((1,41,3),np.float32);nodes[0,:,1]=np.linspace(0,1,41);nodes[:,:,2]=.31+dz
    rod.update(node_positions=nodes,rest_lengths_m=np.full((1,40),.025,np.float32))
    return sample


def exchange(task,step,*,speed=.1,dz=0.):
    sample=frame(task,step,dz=dz);nodes=sample["rod_snapshot"]["node_positions"]
    clock=dict(run_uuid="cpu-health",env_id=0,episode_uid="ep-0",reset_id=0,rod_uid="rod0",
        physics_step=step,rod_tick=step*12,policy_step=(step-3)//8,substep=(step-3)%8,
        timestamp_s=step/480,solver_time_s=step/480,clock_valid=True)
    rawclock={k:v for k,v in clock.items() if k not in ("policy_step","substep")}
    names=list(JOINTS+WRIST_LINEAR+WRIST_ANGULAR)
    low,high=active_limits();q=np.zeros((1,17),np.float32);q[0,:11]=expand((low+high)/2)
    hands={side:dict(joint_names=names,q=q.copy(),qd=np.zeros_like(q),backend_targets=q.copy(),
        active_indices=[names.index(n) for n in ACTIVE],
        measured_source="physx_backend_get_dof_positions/get_dof_velocities") for side in ("left","right")}
    pre=dict(clock,physics_step=step-1,rod_tick=(step-1)*12,timestamp_s=(step-1)/480,solver_time_s=(step-1)/480)
    row=dict(physics_step=step,policy_step=clock["policy_step"],substep=clock["substep"],
        rod=dict(rawclock,node_positions=nodes.copy(),velocities=np.full_like(nodes,speed/np.sqrt(3)),
            hand_wrenches_com_world={side:dict(forces=np.zeros((1,5,3)),torques=np.zeros((1,5,3))) for side in hands},
            distributed_contact=np.zeros((1,41,3)),
            source="RopeBatch.positions/velocities/contact_forces/mesh_wrenches"),
        action_readback=dict(pre=dict(identity=[pre]),post=dict(identity=[clock],hands=hands)),
        post_step_clearance=dict(version=clearance.VERSION,floor_m=.0021,source=clearance.GEOMETRY_SOURCE,
            source_sha256=clearance._pins(),identity=clock,geometry_phase="sdk_complete_rod_pending",
            physics_step=step,rod_source_physics_step=step-1,sdk_time_s=step/480,
            clearance_m=[[.003,.003]],lowest_shape_index=[0],failed_sides=[],passed=True,
            minimum_clearance_m=.003,native_clearance_qualified=False,hand_table_contacts=None,teacher_admission=False),
        guard_failure=None,original_guard_failure=None)
    return sample,row


def solver(row,value=0):
    return api().make_solver_status(row["rod"],value,source_scope="cpu_fixture",
        source_instance_id=1,backend_sha256=api().BACKEND_SHA)


def measured(task,step,*,error=0,speed=.1,motion=None,bilateral=False,dz=0.):
    snap,x=exchange(task,step,speed=speed,dz=dz)
    packet=api().observe_health(x,solver_status=None if error is None else solver(x,error))
    m=signals(task,step=step) if bilateral else no_bilateral(step,stable=None)
    m["stable"]=measurement(None)
    m["health_observation"]=packet
    if motion is not None:m["hold_motion"]=measurement(motion,"cpu_fixture:explicit_motion_contract")
    return snap,m


def make(*,reducer=False,weights=True):
    t,b,w=inputs()
    cls=api().HealthSeparatedR0ControlStep if reducer else api().HealthSeparatedR0Episode
    kwargs=dict(env_id=0,diagnostic_coefficients=w if weights else None)
    if reducer:kwargs["budget_config"]=b
    owner=cls(t,"pinch",**kwargs)
    owner.reset({k:lambda:None for k in RESET_HOOKS},
        lambda:(frame(t,2),measurement(.005)),reset_approach=reset_measurement())
    return owner,t,b,w


def test_normal_fast_approach_settles_eight_steps_without_motion_or_success():
    r,t,_,_=make(reducer=True)
    r.begin(control_index=0,recipe_sha256=r.recipe_sha256,action=np.zeros(24))
    for step in range(3,11):
        snap,m=measured(t,step,speed=.2)
        assert r.push(snap,m,control_index=0,recipe_sha256=r.recipe_sha256) is (step==10)
    row=r.settle()
    assert row["physical_samples"]==8 and row["terminated"] is False
    assert row["diagnostic_reward"] is not None and row["blocked_terms"]==[]
    assert row["terms"]["instability"]["raw_sum"]==0
    assert row["terms"]["hold_progress"]["raw_sum"] is None
    assert row["terms"]["hold_progress"]["contribution_sum"]==0
    assert row["training_reward"] is None and row["native_qualification"] is False


@pytest.mark.parametrize("health,motion,expected",[(0,None,0.),(0,False,0.),(3,True,1.),(None,True,None)])
def test_health_and_motion_are_independent_and_statistics_are_replaced(health,motion,expected):
    e,t,_,_=make()
    snap,m=measured(t,3,error=health,motion=motion)
    row=e.step(snap,m)
    assert row["terms"]["instability"]["raw"]==expected
    stats=row["term_stats"]["instability"]
    assert stats["validity_count"]==(0 if expected is None else 1)
    assert stats["episode_sum"]==expected
    assert (row["diagnostic_reward"] is None)==(health is None)
    assert row["health_separation"]["hold_motion"]["value"] is motion
    assert row["success_candidate"] is False
    assert row["training_reward"] is None


def test_cumulative_stats_follow_health_not_the_legacy_motion_value():
    e,t,_,_=make()
    for step,error,motion in [(3,0,False),(4,2,True),(5,None,False),(6,0,None)]:
        snap,m=measured(t,step,error=error,motion=motion)
        row=e.step(snap,m)
    stats=row["term_stats"]["instability"]
    assert stats["validity_count"]==3 and stats["episode_sum"]==1
    assert stats["mean"]==pytest.approx(1/3) and stats["std"]==pytest.approx(np.std([0,1,0]))
    assert stats["min"]==0 and stats["max"]==1


def test_known_solver_failure_is_first_sample_terminal_and_no_future_step():
    r,t,_,_=make(reducer=True)
    r.begin(control_index=0,recipe_sha256=r.recipe_sha256,action=np.zeros(24))
    snap,m=measured(t,3,error=7,motion=True)
    assert r.push(snap,m,control_index=0,recipe_sha256=r.recipe_sha256)
    result=r.settle()
    assert result["terminated"] and not result["truncated"]
    assert result["terminal_reasons"]==["instability"] and result["physical_samples"]==1
    assert result["partial_interval_due_to_terminal"]
    with pytest.raises(RuntimeError,match="closed"):
        r.begin(control_index=1,recipe_sha256=r.recipe_sha256,action=np.zeros(24))


def test_missing_last_error_is_unknown_even_with_finite_state_and_no_guard():
    e,t,_,_=make()
    snap,m=measured(t,3,error=None)
    assert m["health_observation"]["numerical_health"]["value"] is None
    row=e.step(snap,m)
    assert row["terms"]["instability"]["raw"] is None
    assert row["blocked_terms"]==["instability"] and row["diagnostic_reward"] is None


def test_motion_false_only_breaks_hold_and_never_causes_instability():
    e,t,_,_=make()
    snap,m=measured(t,3,motion=True,bilateral=True,dz=.035)
    first=e.step(snap,m)
    snap,m=measured(t,4,motion=False,bilateral=True,dz=.035)
    row=e.step(snap,m)
    assert not row["success_candidate"] and row["hold_s"]==0
    assert row["terms"]["hold_progress"]["raw"]==0
    assert row["legacy_instability"]["raw"]==1 and row["terms"]["instability"]["raw"]==0


def test_unqualified_motion_cannot_become_native_stable():
    e,t,_,_=make()
    snap,m=measured(t,3,motion=True,error=None)
    snap["rod_snapshot"]["source_scope"]="solver_readback_unqualified"
    with pytest.raises(ValueError,match="motion"):e.step(snap,m)
    assert not e.active


def test_null_training_coefficients_remain_null():
    e,t,_,_=make(weights=False)
    snap,m=measured(t,3)
    row=e.step(snap,m)
    assert len(row["blocked_terms"])==10 and row["diagnostic_reward"] is None
    assert all(value is None for value in t["reward"]["coefficients"].values())


def test_legacy_stable_false_still_terminates():
    from go2_pvcnn.control.dh116.tabletop_r0_hold_necessary_domain import NecessaryHoldR0ControlStep
    t,b,w=inputs();r=NecessaryHoldR0ControlStep(t,"pinch",env_id=0,budget_config=b,diagnostic_coefficients=w)
    r.reset({k:lambda:None for k in RESET_HOOKS},lambda:(frame(t,2),measurement(.005)))
    r.begin(control_index=0,recipe_sha256=r.recipe_sha256,action=np.zeros(24))
    assert r.push(frame(t,3),no_bilateral(3,stable=False),control_index=0,recipe_sha256=r.recipe_sha256)
    assert r.settle()["terminal_reasons"]==["instability"]


@pytest.mark.parametrize("failure",["solver","constraint"])
def test_unhealthy_or_constraint_failure_cannot_begin_hold(failure):
    e,t,_,_=make()
    snap,m=measured(t,3,error=4 if failure=="solver" else 0,motion=True,bilateral=True,dz=.035)
    if failure=="constraint":
        _,x=exchange(t,3,dz=.035)
        x["post_step_clearance"].update(clearance_m=[[.002,.003]],minimum_clearance_m=.002,
            failed_sides=["left"],passed=False)
        x["original_guard_failure"]=x["guard_failure"]="post_step_source_clearance:left"
        m["health_observation"]=api().observe_health(x,solver_status=solver(x,0))
    row=e.step(snap,m)
    assert row["hold_start_physics_step"] is None and row["hold_s"]==0
    assert row["success_candidate"] is False


@pytest.mark.parametrize("source,reason",[("geometry","closure_geometry_source_invalid"),
    ("table","closure_table_source_invalid"),("director","material_director_source_invalid")])
def test_final_source_failure_is_terminal_and_reward_unknown_not_instability(source,reason):
    r,t,_,_=make(reducer=True)
    r.begin(control_index=0,recipe_sha256=r.recipe_sha256,action=np.zeros(24))
    snap,x=exchange(t,3)
    x["guard_failure"]=reason
    x[source+"_capture_error"]="CPU injected missing source"
    m=no_bilateral(3,stable=None)
    m["health_observation"]=api().observe_health(x,solver_status=solver(x,0))
    m["hold_motion"]=measurement(True,"cpu_fixture:explicit_motion_contract")
    assert m["health_observation"]["numerical_health"]["value"] is True
    assert r.push(snap,m,control_index=0,recipe_sha256=r.recipe_sha256)
    result=r.settle()
    assert result["terminated"] and result["physical_samples"]==1
    assert result["terminal_reasons"]==["physical_terminal:"+reason]
    assert result["terms"]["instability"]["raw_sum"]==0
    assert result["diagnostic_reward"] is None
    assert "physical_source_validity" in result["blocked_terms"]
    with pytest.raises(RuntimeError,match="closed"):
        r.begin(control_index=1,recipe_sha256=r.recipe_sha256,action=np.zeros(24))


def test_final_reason_cannot_erase_original_constraint():
    _,t,_,_=make()
    _,x=exchange(t,3)
    x["post_step_clearance"].update(clearance_m=[[.002,.003]],minimum_clearance_m=.002,
        failed_sides=["left"],passed=False)
    x["original_guard_failure"]="post_step_source_clearance:left"
    with pytest.raises(ValueError,match="terminal"):
        api().observe_health(x,solver_status=solver(x,0))


def test_measured_constraint_failure_is_first_sample_terminal_without_numerical_penalty():
    r,t,_,_=make(reducer=True)
    r.begin(control_index=0,recipe_sha256=r.recipe_sha256,action=np.zeros(24))
    snap,x=exchange(t,3)
    x["post_step_clearance"].update(clearance_m=[[.002,.003]],minimum_clearance_m=.002,
        failed_sides=["left"],passed=False)
    x["original_guard_failure"]=x["guard_failure"]="post_step_source_clearance:left"
    m=no_bilateral(3,stable=None)
    m["health_observation"]=api().observe_health(x,solver_status=solver(x,0))
    assert r.push(snap,m,control_index=0,recipe_sha256=r.recipe_sha256)
    row=r.settle()
    assert row["terminated"] and not row["truncated"] and row["physical_samples"]==1
    assert row["terminal_reasons"]==["physical_terminal:post_step_source_clearance:left"]
    assert row["terms"]["instability"]["raw_sum"]==0
    assert not row["success_candidate"]
    with pytest.raises(RuntimeError,match="closed"):
        r.begin(control_index=1,recipe_sha256=r.recipe_sha256,action=np.zeros(24))
