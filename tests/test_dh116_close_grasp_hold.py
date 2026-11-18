"""Current per-hand close admission and bounded retained-grasp feedback."""
import copy
from types import SimpleNamespace
import pytest
import torch
from test_dh116_coordinated_grasp import enabled
from test_dh116_tensor_contact_window import loads
from go2_pvcnn.control.dh116.tabletop_loaded_palm_motion import LoadedPalmReposition

DT=1/480


def measured(t,left=2.,right=.2,normal=(0.,0.)):
    m=loads(t,4.)
    for h,value in enumerate((left,right)):
        for k in ('hand_object_loads','object_link_loads','grouped_object_loads'):
            m[k][:,h]*=value/2
    m['vertical_hand_force_n'][:]=torch.tensor(normal,dtype=torch.float64)
    m['commanded_palm_offset_m']=torch.full((t.n,2,3),.0001,dtype=torch.float64)
    m['commanded_palm_offset_m'][:,:,2]=torch.tensor([-.00035,.0002],dtype=torch.float64)
    return m


def consume(t,m,live=None,force=None):
    t.consume(m,~t.done if live is None else live,DT,
              vertical_force_n=m['vertical_hand_force_n'].sum(1) if force is None else force,
              object_weight_n=torch.ones(t.n))


def held(side=0,n=2):
    t=enabled(n);m=measured(t,*( (2.,.2) if side==0 else (.2,2.)))
    t.command.copy_(m['q']);t.f_target.fill_(2.);t.f_requested.fill_(2.);t.f_filtered.fill_(2.)
    consume(t,m)
    return t


@pytest.mark.parametrize('side',[0,1])
def test_first_ready_hand_holds_immediately_without_partner_or_dwell(side):
    t=held(side)
    assert t.phase.tolist()==[1,1]
    assert t.close_grasp_hold.active[:,side].all()
    assert not t.close_grasp_hold.active[:,1-side].any()
    assert torch.equal(t.close_grasp_hold.ready,t.close_grasp_hold.active)
    torch.testing.assert_close(t.close_grasp_hold.command[:,6*side:6*side+6],t.command[:,6*side:6*side+6])
    torch.testing.assert_close(t.close_grasp_hold.palm_offset[:,side,2],torch.full((2,),[-.00035,.0002][side],dtype=torch.float64))
    assert not t.contact_time.any()


@pytest.mark.parametrize('skip_hold',[False,True])
@pytest.mark.parametrize('normal',[(0.,0.),(.1,.1)])
def test_both_current_sides_enter_lift_on_first_physics_substep(skip_hold,normal):
    t=enabled();t.skip_hold=skip_hold;t.close_downward_penalty=skip_hold
    consume(t,measured(t,2.,2.,normal))
    assert t.phase.tolist()==[3,3]
    assert t.lift_grasp_hold.active.all()
    assert not t.contact_time.any()


@pytest.mark.parametrize('normal',[(-.1,1.),(1.,-.1)])
def test_own_negative_normal_blocks_even_with_positive_total_or_friction(normal):
    t=enabled();m=measured(t,2.,2.,normal)
    m['vertical_friction_force_n']=torch.full((2,2),10.)
    consume(t,m,force=torch.ones(2))
    assert (t.phase==1).all()
    assert t.close_grasp_hold.ready[0].tolist()==[v>=0 for v in normal]


@pytest.mark.parametrize('weak',['load','normal'])
def test_stale_held_hand_cannot_qualify_partner_until_current_readiness_restored(weak):
    t=held();m=measured(t,.2 if weak=='load' else 2.,2.,(-.1,1.) if weak=='normal' else (0.,0.))
    consume(t,m)
    assert (t.phase==1).all()
    assert t.close_grasp_hold.active.all()
    assert not t.close_grasp_hold.ready[:,0].any()
    consume(t,measured(t,2.,2.))
    assert (t.phase==3).all()


@pytest.mark.parametrize('side',[0,1])
def test_policy_cannot_change_held_grip_force_or_palm_while_partner_explores(side):
    a=held(side);b=copy.deepcopy(a);anchor=a.close_grasp_hold.palm_offset.clone();integral=a.f_offset.clone();lift=a.offset.clone()
    for _ in range(8):
        a.control(-torch.ones((2,20)),~a.done,DT)
        b.control(torch.ones((2,20)),~b.done,DT)
    s=slice(6*side,6*side+6)
    torch.testing.assert_close(a.command[:,s],b.command[:,s])
    torch.testing.assert_close(a.palm_targets[:,side],anchor[:,side])
    torch.testing.assert_close(b.palm_targets[:,side],anchor[:,side])
    torch.testing.assert_close(a.f_target[:,side],b.f_target[:,side])
    torch.testing.assert_close(a.f_offset[:,side],integral[:,side])
    assert not torch.equal(a.command[:,6*(1-side):6*(1-side)+6],b.command[:,6*(1-side):6*(1-side)+6])
    assert not torch.equal(a.palm_targets[:,1-side],b.palm_targets[:,1-side])
    torch.testing.assert_close(a.offset,lift)


def test_held_pressure_feedback_moves_last_command_and_respects_governors():
    t=held();t.f_filtered[:,0]=1.;t.m['hand_object_loads'][:,0]=.2
    before=t.command.clone();integral=t.f_offset.clone()
    t.control(-torch.ones((2,20)),~t.done,DT)
    delta=(t.command-before).reshape(2,2,6)
    assert ((delta[:,0]*t.direction[:,0])>0).any()
    assert (delta.abs()<=.20*DT+1e-12).all()
    torch.testing.assert_close(t.f_offset[:,0],integral[:,0])
    t.m['mimic_errors'][:,1]=-.013;before=t.command.clone()
    t.control(torch.ones((2,20)),~t.done,DT)
    assert ((t.command-before).abs()<=.20*DT+1e-12).all()
    assert (t.command>=t.low).all() and (t.command<=t.high).all()


def test_contact_loss_retains_pose_without_recapturing_snapshot():
    t=held();anchor=t.close_grasp_hold.command.clone();m=measured(t,.01,.2)
    for _ in range(9):consume(t,m)
    assert t.close_grasp_hold.active[:,0].all()
    torch.testing.assert_close(t.close_grasp_hold.command,anchor)
    consume(t,m)
    assert t.close_grasp_hold.active[:,0].all()
    t.command+=.001;consume(t,measured(t))
    assert t.close_grasp_hold.active[:,0].all()
    torch.testing.assert_close(t.close_grasp_hold.command[:,:6],anchor[:,:6])


def test_reclose_side_can_capture_without_partner_barrier_and_keeps_partner_attempts():
    t=enabled();c=LoadedPalmReposition(torch.ones(2,dtype=torch.bool));t.palm_reposition=c
    c.state[:]=torch.tensor([3,1]);c.attempts[:]=torch.tensor([1,2]);c.clock[:]=.3;c.dwell[:]=.01
    consume(t,measured(t))
    assert t.close_grasp_hold.active.tolist()==[[True,False]]*2
    assert c.state.tolist()==[[0,1]]*2
    assert c.attempts.tolist()==[[1,2]]*2
    assert not c.clock[:,0].any() and (c.clock[:,1]==.3).all()


@pytest.mark.parametrize('state',[1,2])
def test_own_release_or_motion_blocks_capture(state):
    t=enabled();t.palm_reposition=LoadedPalmReposition(torch.ones(2,dtype=torch.bool))
    t.palm_reposition.state[:,0]=state;consume(t,measured(t))
    assert not t.close_grasp_hold.active.any()


def test_partner_descent_cannot_reopen_or_move_held_hand():
    t=held();t.palm_reposition=LoadedPalmReposition(torch.ones(2,dtype=torch.bool))
    t.m['adaptive_palm_offset'].copy_(t.close_grasp_hold.palm_offset)
    t.m['hand_object_loads'][:,1]=2.
    before=t.command[:,:6].clone();anchor=t.close_grasp_hold.palm_offset[:,0].clone()
    action=torch.zeros((2,20));action[:,14]=action[:,17]=-1.
    t.control(action,~t.done,DT)
    assert t.palm_reposition.state.tolist()==[[0,1]]*2
    torch.testing.assert_close(t.palm_targets[:,0],anchor)
    torch.testing.assert_close(t.command[:,:6],before)


def test_selected_reset_and_inactive_consume_preserve_other_rows():
    t=held();before=copy.deepcopy(t.close_grasp_hold)
    consume(t,measured(t,.01,.01),live=torch.tensor([True,False]))
    for k,v in vars(before).items():torch.testing.assert_close(getattr(t.close_grasp_hold,k)[1],v[1])
    t.reset([0],measured(t))
    for k,v in vars(before).items():
        assert not getattr(t.close_grasp_hold,k)[0].any()
        torch.testing.assert_close(getattr(t.close_grasp_hold,k)[1],v[1])


@pytest.mark.parametrize('bad',['hard','nonfinite','timeout'])
def test_failure_clears_existing_close_hold(bad):
    t=held();m=measured(t)
    if bad=='hard':m['clearance'][:]=-.002
    elif bad=='nonfinite':m['vertical_hand_force_n'][:]=float('nan')
    else:t.elapsed[:]=4.01
    consume(t,m,force=torch.ones(2))
    assert t.done.all() and not t.close_grasp_hold.active.any()
    assert not t.close_grasp_hold.ready.any()


def test_held_close_world_preserves_unequal_palms_and_per_hand_clearance():
    from test_dh116_tensor_world import palm_world
    w,_=palm_world();t=held(n=3)
    w.palms.copy_(t.close_grasp_hold.palm_offset)
    t.palm_actions.copy_(w.palms/t.palm_limit)
    w.roots=w.root_seed.clone();w.roots[:,:,:3]+=w.palms;w.roots[:,:,2]+=t.offset[:,None]
    w.root_targets=w.roots.clone()
    w.clearances[:]=.004
    w.advance(t,torch.ones(3,dtype=torch.bool),torch.ones(3,dtype=torch.bool))
    torch.testing.assert_close(w.palms,t.close_grasp_hold.palm_offset)
    w.clearances[:,1]=.0011
    w.advance(t,torch.zeros(3,dtype=torch.bool),torch.ones(3,dtype=torch.bool))
    torch.testing.assert_close(w.palms[:,0],t.close_grasp_hold.palm_offset[:,0])
    assert (w.palms[:,1,2]>t.close_grasp_hold.palm_offset[:,1,2]).all()


@pytest.mark.parametrize('axis',['xyz','z_down'])
def test_partial_hold_keeps_unheld_z_request_independent_of_held_policy_action(axis):
    a=held();b=copy.deepcopy(a)
    for t in (a,b):t.palm_exploration_axis=axis
    x=torch.zeros((2,20));x[:,17]=-.5;y=x.clone();y[:,14]=-1.
    a.control(x,~a.done,DT);b.control(y,~b.done,DT)
    torch.testing.assert_close(a.palm_targets[:,1],b.palm_targets[:,1])
    assert (a.palm_targets[:,1,2]<0).all()


def test_force_direction_reposition_excludes_held_side_even_with_new_negative_normal():
    from go2_pvcnn.control.dh116.tabletop_force_direction_reposition import ForceDirectionReposition
    t=held();t.force_direction_reposition=True
    t.palm_reposition=ForceDirectionReposition(torch.ones(2,dtype=torch.bool))
    t.m['vertical_hand_force_n'][:]=-.3;t.m['hand_object_loads'][:]=2.
    anchor=t.close_grasp_hold.palm_offset[:,0].clone()
    t.control(torch.zeros((2,20)),~t.done,DT)
    assert t.palm_reposition.state.tolist()==[[0,1]]*2
    assert t.palm_reposition.attempts.tolist()==[[0,1]]*2
    torch.testing.assert_close(t.palm_targets[:,0],anchor)


def test_invalid_action_hard_failure_clears_close_hold_before_substeps():
    t=held();action=torch.zeros((2,20));action[0,0]=float('nan')
    before=copy.deepcopy(t.close_grasp_hold)
    t.begin_step(action)
    assert t.done[0] and not t.close_grasp_hold.active[0].any()
    for k,v in vars(before).items():torch.testing.assert_close(getattr(t.close_grasp_hold,k)[1],v[1])


def test_force_direction_held_api_preserves_inactive_rows_and_held_diagnostics():
    from go2_pvcnn.control.dh116.tabletop_force_direction_reposition import ForceDirectionReposition
    c=ForceDirectionReposition(torch.ones(2,dtype=torch.bool))
    c.state[1]=2;c.clock[1]=.5;c.dwell[1]=.01;c.goal[1]=.0009
    c.goal[0,0]=.0007;c.goal_clipped[0,0]=False
    before=copy.deepcopy(c)
    c.apply(torch.full((2,2,3),-.004,dtype=torch.float64),torch.zeros((2,2,3),dtype=torch.float64),torch.full((2,2),2.,dtype=torch.float64),
            torch.ones(2,dtype=torch.long),torch.tensor([True,False]),DT,
            torch.full((2,2),-.3),clearance=torch.full((2,2),.0021),
            held_hands=torch.tensor([[True,False],[False,False]]))
    for k,v in vars(before).items():
        if isinstance(v,torch.Tensor):torch.testing.assert_close(getattr(c,k)[1],v[1])
    torch.testing.assert_close(c.goal[0,0],before.goal[0,0])
    assert not c.goal_clipped[0,0]


def test_zero_normal_lift_then_positive_normal_preserves_delayed_ready_bonus_once():
    t=enabled();t.begin_step(torch.zeros((2,20)))
    consume(t,measured(t,2.,2.,(0.,0.)))
    assert (t.phase==3).all()
    assert not t.grasp_ready_awarded.any()
    assert not t.terms['grasp_ready'].any()
    # Lift readiness must retain its established reward behavior after close.
    consume(t,measured(t,2.,2.,(.1,.1)))
    assert not t.close_grasp_hold.ready.any()
    assert t.grasp_ready_awarded.all()
    torch.testing.assert_close(t.terms['grasp_ready'],torch.full((2,),3.,dtype=torch.float64))
    consume(t,measured(t,2.,2.,(.1,.1)))
    torch.testing.assert_close(t.terms['grasp_ready'],torch.full((2,),3.,dtype=torch.float64))
