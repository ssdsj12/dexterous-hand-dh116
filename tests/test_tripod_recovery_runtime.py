"""CPU execution ledger integration; no native success is implied."""
import importlib
from types import SimpleNamespace as NS
import io
import pytest
import torch
from test_tripod_contact_recovery import api,observe,retry,reserve


def test_original_pressure_prefix_records_actual_control_issued_action_without_execution_changes():
    m=importlib.import_module('go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime')
    r=m.TripodRecoveryRuntime.__new__(m.TripodRecoveryRuntime)
    r.active=False;r.acquisition_origins=torch.zeros(1,3,dtype=torch.bool)
    r.control_issued_actions=torch.zeros(1,24)
    pressure=torch.linspace(-.2,.2,24).reshape(1,24)
    issued=r.refine(pressure)
    assert torch.equal(issued,pressure)
    assert torch.equal(r.control_issued_actions,pressure) and torch.equal(r.requested_actions,pressure)


def prepared():
    f=api();observe(f);retry(f);ticket=reserve(f)
    old=torch.full((1,12),.123456789012,dtype=torch.float64)
    target=old.clone();target[0,6]+=.0001
    pending=dict(episode_ids=f.episode_ids.clone(),reservation_id=ticket['reservation_id'],
        execution_id='native-submit-1',physics_step=1,
        approved_approach_m=ticket['allowed_distance_m'].clone(),
        executed_mask=ticket['allowed_distance_m']>0,requested_action24=torch.zeros(1,24),
        final_target_rad=target,approved_action24=torch.zeros(1,24))
    world=NS(commands=target.reshape(1,2,6).clone(),_physics_steps=0)
    return f,pending,world


def commit(f,p,w):
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime').commit_submitted_targets(f,p,w)


def test_candidates_and_reservations_do_not_charge_until_confirmed_native_submission():
    f,p,w=prepared();assert f.executed_approach_m.sum()==0
    commit(f,p,w)
    torch.testing.assert_close(f.executed_approach_m,p['approved_approach_m'])
    commit(f,p,w)
    torch.testing.assert_close(f.executed_approach_m,p['approved_approach_m'])


@pytest.mark.parametrize('case',['wrong_step','logical_target_changed','encoded_target_changed'])
def test_submit_mismatch_fails_before_ledger_charge(case):
    f,p,w=prepared()
    if case=='wrong_step':w._physics_steps=3
    elif case=='logical_target_changed':w.commands[0,1,0]+=1e-12
    else:w.commands[0,1,0]+=.001
    with pytest.raises(ValueError,match='submission'):
        commit(f,p,w)
    assert f.executed_approach_m.sum()==0


def test_submission_records_actual_inactive_reset_targets_without_confusing_live_commands():
    f=api(2);observe(f);retry(f);ticket=reserve(f)
    expected=torch.ones(2,12,dtype=torch.float64)
    actual=expected.clone();actual[1]=.25
    amount=ticket['allowed_distance_m'].clone();amount[1]=0.
    pending=dict(episode_ids=f.episode_ids.clone(),reservation_id=ticket['reservation_id'],
        execution_id='with_inactive_reset',physics_step=1,approved_approach_m=amount,
        executed_mask=amount>0,requested_action24=torch.zeros(2,24),final_target_rad=expected,
        approved_action24=torch.zeros(2,24),native_live_mask=torch.tensor([True,False]))
    w=NS(commands=actual.reshape(2,2,6),_physics_steps=0)
    commit(f,pending,w)
    assert f.executed_approach_m[0].sum()>0 and not f.executed_approach_m[1].any()
    record=next(row for row in f.events if row['kind']=='execution_committed' and row['env']==1)
    assert record['final_target_rad'].eq(.25).all()


def test_stage_options_cannot_accidentally_start_ppo_or_later_stage():
    m=importlib.import_module('go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime')
    good=NS(tripod_recovery_stage='A',vision_teacher_source='source.pt',vision_teacher_mpc=True,
        vision_teacher_mpc_profile='pressure_reference',vision_student=True,
        vision_action_distribution='tanh_gaussian',vision_camera_mode='fixed',num_envs=8,
        vision_teacher_max_control_steps=560,resume=None,vision_evaluation_checkpoint=None,
        surface_mpc=False,vision_lift_reward_experiment='R0')
    m.validate_staged_options(good)
    for key,value in [('tripod_recovery_stage','D'),('resume','old.pt'),
                      ('vision_teacher_mpc_profile','pressure_geometry_constrained'),
                      ('vision_teacher_max_control_steps',10000)]:
        changed=NS(**vars(good));setattr(changed,key,value)
        with pytest.raises(ValueError):m.validate_staged_options(changed)


def test_acquisition_velocity_cannot_continue_unbudgeted_after_real_contact_within_control():
    m=importlib.import_module('go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime')
    f=api();observe(f);retry(f)
    t=NS(actions=torch.zeros(1,24,dtype=torch.float64))
    t.actions[0,8]=.1
    origins=torch.zeros(1,3,dtype=torch.bool);origins[0,1]=True
    # Next physical sample establishes real contact while the same 60Hz action
    # remains held. Protected bearing targets cannot be cancelled downstream.
    observe(f,contact=True)
    m.cancel_inapplicable_substep_actions(t,f,torch.tensor([True]),origins)
    assert t.actions[0,8]==0
    assert not f.executed_approach_m.any()


def test_lineage_cancellation_preserves_pinch_and_original_bearing_pressure_relief():
    m=importlib.import_module('go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime')
    f=api(2);observe(f);retry(f)
    t=NS(actions=torch.full((2,24),-.1,dtype=torch.float64))
    before=t.actions.clone();origins=torch.zeros(2,3,dtype=torch.bool)
    # Genuine pressure origin is not an acquire action and cannot be cancelled
    # merely because its effective pad currently bears load.
    observe(f,contact=True)
    m.cancel_inapplicable_substep_actions(t,f,torch.tensor([False,True]),origins)
    assert torch.equal(t.actions,before)


def test_old_held_pressure_velocity_stops_on_actual_loss_and_waits_for_budgeted_acquisition():
    m=importlib.import_module('go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime')
    f=api();observe(f,contact=True);observe(f,contact=False)
    t=NS(actions=torch.full((1,24),.1,dtype=torch.float64))
    m.cancel_inapplicable_substep_actions(t,f,torch.tensor([True]),torch.zeros(1,3,dtype=torch.bool))
    assert not t.actions[0,6:10].any()
    assert torch.equal(t.actions[0,:6],torch.full((6,),.1,dtype=torch.float64))


def test_before_physics_switches_acquired_pad_to_fresh_pressure_without_budget_reset():
    from test_tripod_pressure_hold import inputs
    m=importlib.import_module('go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime')
    hold,a=inputs();f=api();observe(f,contact=True)
    r=m.TripodRecoveryRuntime.__new__(m.TripodRecoveryRuntime)
    r.active=True;r.pending=None;r.device='cpu';r.n=1;r.tick=437
    r.ledger=f;r.tripod=torch.tensor([True]);r.ended=torch.tensor([False])
    r.acquisition_origins=torch.ones(1,3,dtype=torch.bool)
    r.t=NS(done=torch.tensor([False]),actions=torch.full((1,24),.1,dtype=torch.float64),
        command=torch.as_tensor(a['current_command']).reshape(1,12).clone())
    pads={key:torch.as_tensor(a[name])[None] for key,name in (
        ('effective_pad_loads_n','effective_pad_loads_n'),
        ('effective_pad_vectors_w','effective_pad_vectors_w'),
        ('effective_pad_points_w','effective_pad_points_w_m'),
        ('effective_pad_valid','effective_pad_valid'),
        ('effective_pad_depth_m','effective_pad_depth_m'),
        ('rejected_contact_loads_n','rejected_contact_loads_n'))}
    r.w=NS(pad_measurements=pads,_physics_steps=0)
    r.env=NS(substep=2);r.episode_ids=f.episode_ids.clone();r.decisions=io.StringIO()
    r.pressure_controllers={0:hold};r.upstream_pressure_actions=torch.zeros(1,24)
    r._settled=lambda _:True;r._planner=lambda _:hold.planner
    r._contexts=lambda _:{k:a[k] for k in ('snapshots','full_qd','snapshot_identity','bearing_mask','bearing_provenance')}
    r.geometry_blocked={};distance=f.executed_approach_m.clone();time=f.attempt_time_s.clone()
    # Match the synthetic native-loaded index; do not call missing pads bearing.
    f.measured_contact[0,1,:3]=torch.tensor([False,True,False])
    f.contact_known[0,1,:3]=True
    r.before_step()
    assert r.pressure_step_proposals=={0}
    assert r.t.actions[0,6:8].eq(0).all() and r.t.actions[0,9]==0
    assert r.substep_action_origin[0,1]==2
    assert torch.equal(r.requested_actions,r.t.actions)
    assert torch.equal(f.executed_approach_m,distance) and torch.equal(f.attempt_time_s,time)


def test_ended_tripod_reset_left_target_does_not_abort_sibling_first_episode():
    m=importlib.import_module('go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime')
    r=m.TripodRecoveryRuntime.__new__(m.TripodRecoveryRuntime)
    r.active=True;r.device='cpu';r.n=2;r.tripod=torch.tensor([True,True])
    r.ended=torch.tensor([True,False]);r.fixed_left_commands=torch.ones(2,6,dtype=torch.float64)
    r.ledger=api(2);r.pressure_step_proposals=set()
    command=torch.ones(2,12,dtype=torch.float64);command[0,:6]=.25
    packet=dict(command_rad=command,governed_target_rad=command.clone(),
        finger_mask=torch.zeros(2,12,dtype=torch.bool),live=torch.tensor([False,True]))
    assert torch.equal(r.filter_post_guard_targets(packet),command)


def test_real_partial_reset_disables_only_completed_trial_before_sibling_reservation():
    m=importlib.import_module('go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime')
    f=api(2);observe(f);retry(f)
    p=__import__('test_tripod_contact_recovery').packet(f)
    p['terminal'][0]=True;f.observe_substep(**p)
    ids=torch.tensor([0]);episodes=f.episode_ids.clone();episodes[0]+=1
    f.reset(ids,episode_ids=episodes,source_sha='a'*64)
    m.retire_completed_first_episode(f,ids,torch.tensor([True,False]))
    assert not f.enabled_pad_mask[0].any() and f.enabled_pad_mask[1,1,:3].all()
    retry(f);ticket=reserve(f)
    assert ticket['settled'][0] and not ticket['settled'][1]
    amount=ticket['allowed_distance_m'];assert not amount[0].any() and amount[1].sum()>0
    target=torch.zeros(2,12,dtype=torch.float64)
    payload=dict(episode_ids=episodes,reservation_id=ticket['reservation_id'],execution_id='sibling_after_reset',
        physics_step=2,approved_approach_m=amount,executed_mask=amount>0,
        requested_action24=torch.zeros(2,24),approved_action24=torch.zeros(2,24),final_target_rad=target,
        native_live_mask=torch.tensor([False,True]))
    m.commit_submitted_targets(f,payload,NS(commands=target.reshape(2,2,6),_physics_steps=1))
    assert f.executed_approach_m[1].sum()>0 and not f.executed_approach_m[0].any()


def test_prior_fresh_action_is_revoked_when_next_substep_cannot_rebuild_model():
    m=importlib.import_module('go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime')
    r=m.TripodRecoveryRuntime.__new__(m.TripodRecoveryRuntime)
    r.active=True;r.pending=None;r.n=1;r.device='cpu';r.tick=437;r.last_before_step_tick=437
    r.ledger=api();observe(r.ledger,contact=True)
    r.tripod=torch.tensor([True]);r.ended=torch.tensor([False]);r.acquisition_origins=torch.zeros(1,3,dtype=torch.bool)
    r.substep_action_origin=torch.tensor([[0,2,0]])
    r.pressure_mandatory_mask=torch.zeros(1,3,dtype=torch.bool)
    r.t=NS(done=torch.tensor([False]),actions=torch.zeros(1,24,dtype=torch.float64))
    r.t.actions[0,8]=-.004746512
    r.w=NS(pad_measurements={});r._settled=lambda _:False
    r.before_step()
    assert r.t.actions[0,8]==0 and r.substep_action_origin[0,1]==3
    assert not r.pressure_step_proposals and not r.requested_actions[0,8]


def test_prior_mandatory_relief_cannot_be_silently_replaced_with_zero_outside_model_domain():
    m=importlib.import_module('go2_pvcnn.control.dh116.tabletop_tripod_recovery_runtime')
    r=m.TripodRecoveryRuntime.__new__(m.TripodRecoveryRuntime)
    r.active=True;r.pending=None;r.n=1;r.device='cpu';r.tick=437;r.last_before_step_tick=437
    r.ledger=api();observe(r.ledger,contact=True)
    r.tripod=torch.tensor([True]);r.ended=torch.tensor([False]);r.acquisition_origins=torch.zeros(1,3,dtype=torch.bool)
    r.substep_action_origin=torch.tensor([[0,2,0]])
    r.pressure_mandatory_mask=torch.tensor([[False,True,False]])
    r.t=NS(done=torch.tensor([False]),actions=torch.zeros(1,24,dtype=torch.float64))
    r.t.actions[0,8]=-.01;r.w=NS(pad_measurements={});r._settled=lambda _:False
    with pytest.raises(RuntimeError,match='mandatory pressure relief lost model admission'):
        r.before_step()
