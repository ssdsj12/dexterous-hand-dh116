import torch


def setup(mask=(True,False)):
    from go2_pvcnn.control.dh116.tabletop_loaded_palm_motion import LoadedPalmReposition
    c=LoadedPalmReposition(torch.tensor(mask));n=len(mask)
    current=torch.zeros((n,2,3),dtype=torch.float64);target=current.clone();target[:,:,2]=-.001
    load=torch.ones((n,2));phase=torch.ones(n,dtype=torch.long);live=torch.ones(n,dtype=torch.bool)
    return c,target,current,load,phase,live


def test_downward_reposition_releases_before_motion_and_recloses_after_arrival():
    c,target,current,load,phase,live=setup()
    out,release=c.apply(target,current,load,phase,live,.01)
    torch.testing.assert_close(out[0],current[0]);torch.testing.assert_close(out[1],target[1])
    assert release.tolist()==[[True,True],[False,False]]
    load.zero_()
    for _ in range(5):out,release=c.apply(target,current,load,phase,live,.01)
    torch.testing.assert_close(out[0],target[0]);assert release[0].all()
    current.copy_(target)
    for _ in range(5):out,release=c.apply(target,current,load,phase,live,.01)
    assert not release[0].any() and c.state[0].tolist()==[3,3]
    load.fill_(1.)
    for _ in range(5):c.apply(target,current,load,phase,live,.01)
    assert c.state[0].tolist()==[0,0]


def test_recontact_stops_move_until_released_and_timeout_does_not_force_motion():
    c,target,current,load,phase,live=setup()
    c.apply(target,current,load,phase,live,.01);load.zero_()
    for _ in range(5):c.apply(target,current,load,phase,live,.01)
    load.fill_(1.);out,release=c.apply(target,current,load,phase,live,.01)
    torch.testing.assert_close(out[0],current[0]);assert release[0].all()
    for _ in range(210):out,release=c.apply(target,current,load,phase,live,.01)
    assert c.aborted[0].all();torch.testing.assert_close(out[0],current[0])


def test_unloaded_downward_positioning_and_upward_motion_are_preserved():
    c,target,current,load,phase,live=setup();load.zero_()
    out,release=c.apply(target,current,load,phase,live,.01)
    torch.testing.assert_close(out,target);assert not release.any()
    load.fill_(1.);target[:,:,2]=.001
    out,release=c.apply(target,current,load,phase,live,.01)
    torch.testing.assert_close(out,target);assert not release.any()
    target[:,:,2]=-.001;phase.fill_(3)
    c.apply(target,current,load,phase,live,.01)
    assert not c.active.any()


def test_shared_selection_inactive_rows_and_task_reset():
    c,target,current,load,phase,live=setup((True,True));load[0,1]=0;live[1]=False
    _,release=c.apply(target,current,load,phase,live,.01)
    assert release.tolist()==[[True,True],[False,False]]
    from test_dh116_coordinated_grasp import enabled
    t=enabled();t.palm_reposition=c;t.reset([1],t.m)
    assert c.state[0].tolist()==[1,1]
    t.reset([0],t.m);assert not c.active.any()


def test_controller_releases_index_and_blocks_force_integration_while_waiting():
    from test_dh116_coordinated_grasp import enabled
    from test_dh116_tensor_contact_window import loads
    t=enabled();c,*_=setup();t.palm_reposition=c;t.m=loads(t,4.)
    t.command.copy_(t.m['q']);before=t.command.clone();pressure=t.f_offset.clone()
    action=torch.ones((2,20),dtype=torch.float64);action[:,14]=action[:,17]=-.5
    t.control(action,~t.done,1/480)
    assert c.state[0].tolist()==[1,1]
    assert (t.command[0,[2,8]]<before[0,[2,8]]).all()
    torch.testing.assert_close(t.palm_targets[0],t.m['adaptive_palm_offset'][0])
    assert (t.f_offset[0]<=pressure[0]).all()
    assert ((t.command-before).abs()<=.20/480+1e-12).all()


def test_close_cannot_enter_lift_during_reposition():
    from test_dh116_coordinated_grasp import enabled
    from test_dh116_tensor_contact_window import loads
    t=enabled();c,*_=setup();t.palm_reposition=c;c.state[0,0]=1
    t.begin_step(torch.zeros((2,20)));m=loads(t,4.)
    for _ in range(100):
        t.consume(m,~t.done,1/480,vertical_force_n=torch.ones(2),object_weight_n=torch.ones(2))
    assert t.phase.tolist()==[1,3]
    assert t.contact_time[0]==0.


def test_contract_migration_is_exact_one_way_and_requires_coordinated_control():
    import pytest
    from go2_pvcnn.control.dh116.tabletop_tensor_training import checkpoint_contract_changes
    from go2_pvcnn.control.dh116.tabletop_loaded_palm_motion import CONTRACT
    from go2_pvcnn.control.dh116.tabletop_coordinated_grasp import CONTRACT as CONTROL
    old=dict(num_envs=4096,physics_hz=480,grasp_controller=dict(CONTROL))
    new=dict(old,palm_reposition=CONTRACT)
    assert checkpoint_contract_changes(old,new)==[CONTRACT['version']]
    for a,b in [(new,old),(old,dict(new,physics_hz=240)),
                (old,dict(new,palm_reposition=dict(CONTRACT,released_load_n=1.))),
                ({'num_envs':4096},{'num_envs':4096,'palm_reposition':CONTRACT})]:
        with pytest.raises(ValueError):checkpoint_contract_changes(a,b)


def test_released_hand_holds_fingers_instead_of_opening_toward_table():
    from test_dh116_coordinated_grasp import enabled
    from test_dh116_tensor_contact_window import loads
    t=enabled();c,*_=setup();t.palm_reposition=c;t.m=loads(t,0.)
    c.state[0]=2;c.goal[0,:,2]=-.001
    t.command.copy_(t.m['q']);before=t.command.clone()
    action=torch.ones((2,20),dtype=torch.float64)
    t.control(action,~t.done,1/480)
    torch.testing.assert_close(t.command[0],before[0])


def test_release_opens_thumb_and_index_without_crossing_mechanical_bounds():
    from test_dh116_coordinated_grasp import enabled
    from test_dh116_tensor_contact_window import loads
    t=enabled();c,*_=setup();t.palm_reposition=c;t.m=loads(t,4.)
    t.command.copy_(t.m['q']);before=t.command.clone()
    action=torch.ones((2,20),dtype=torch.float64);action[:,14]=action[:,17]=-.5
    t.control(action,~t.done,1/480)
    assert (t.command[0,[1,2,7,8]]<before[0,[1,2,7,8]]).all()
    assert (t.command[0]>=t.low).all()


def test_released_move_projects_downward_goal_to_existing_table_clearance():
    c,target,current,load,phase,live=setup();clearance=torch.full((2,2),.0022)
    c.apply(target,current,load,phase,live,.01,clearance=clearance);load.zero_()
    for _ in range(5):out,_=c.apply(target,current,load,phase,live,.01,clearance=clearance)
    torch.testing.assert_close(out[0,:,2],torch.full((2,),-.0001,dtype=torch.float64),atol=1e-9,rtol=0.)
    assert c.goal_clipped[0].all()


def test_released_move_uses_the_effective_world_clearance_floor():
    c,target,current,load,phase,live=setup()
    target[:,:,2]=-.003
    clearance=torch.full((2,2),.0022,dtype=torch.float64)
    c.apply(target,current,load,phase,live,.01,clearance=clearance,
            clearance_floor_m=.0003)
    load.zero_()
    for _ in range(5):
        out,_=c.apply(target,current,load,phase,live,.01,clearance=clearance,
                      clearance_floor_m=.0003)
    torch.testing.assert_close(out[0,:,2],torch.full((2,),-.0019,dtype=torch.float64))
    assert c.goal_clipped[0].all()


def test_coordinated_reposition_receives_the_tasks_effective_clearance_floor():
    from test_dh116_coordinated_grasp import enabled
    from test_dh116_tensor_contact_window import loads
    t=enabled();c,*_=setup();t.palm_reposition=c
    t.palm_descent_clearance_m=.0003
    t.m=loads(t,0.);t.command.copy_(t.m['q'])
    t.m['hand_clearance'].fill_(.0022)
    c.state[0]=1;c.goal[0,:,2]=-.003
    for _ in range(5):t.control(torch.zeros((2,20),dtype=torch.float64),~t.done,.01)
    torch.testing.assert_close(t.palm_targets[0,:,2],torch.full((2,),-.0019,dtype=torch.float64))


def test_release_request_cannot_override_coupling_recovery_or_pressure_relief():
    from test_dh116_coordinated_grasp import enabled
    from test_dh116_tensor_contact_window import loads
    # Each case compares the candidate against the existing governor sign.
    for kind in ('nonpinch_mimic','thumb_mimic','overpressure'):
        t=enabled();c,*_=setup();t.palm_reposition=c
        total_load=0. if kind=='nonpinch_mimic' else 8. if kind=='overpressure' else 4.
        t.m=loads(t,total_load)
        t.command.copy_(t.m['q']);before=t.command.clone()
        c.state[0]=2;c.goal[0,:,2]=-.001
        action=torch.ones((2,20),dtype=torch.float64)
        if kind=='nonpinch_mimic':
            t.m['mimic_errors'][:,2]=-.016;index=3;sign=-1
        elif kind=='thumb_mimic':
            t.m['mimic_errors'][:,0]=.016;index=1;sign=1
        else:
            t.direction[:]=1.;action[:,18:]=-1.;index=3;sign=-1
        t.control(action,~t.done,1/480)
        assert sign*(t.command[0,index]-before[0,index])>0
        if kind=='overpressure':assert t.pressure_limited[0].all()


def test_disabled_candidate_matches_existing_controller_commands_exactly():
    from test_dh116_coordinated_grasp import enabled
    from test_dh116_tensor_contact_window import loads
    from go2_pvcnn.control.dh116.tabletop_loaded_palm_motion import LoadedPalmReposition
    baseline=enabled();candidate=enabled()
    candidate.palm_reposition=LoadedPalmReposition(torch.zeros(2,dtype=torch.bool))
    generator=torch.Generator().manual_seed(481)
    for t in (baseline,candidate):t.m=loads(t,4.);t.command.copy_(t.m['q'])
    for _ in range(20):
        action=2*torch.rand((2,20),generator=generator,dtype=torch.float64)-1
        for t in (baseline,candidate):t.control(action,~t.done,1/480)
        for name in ('command','palm_actions','f_offset','f_target','offset'):
            torch.testing.assert_close(getattr(candidate,name),getattr(baseline,name),rtol=0,atol=0)


def test_moving_hand_timeout_does_not_charge_shared_release_wait():
    c,target,current,load,phase,live=setup()
    c.state[0]=torch.tensor([2,1]);c.goal.copy_(target)
    c.clock[0,0]=1.95;load[0,0]=0.
    c.apply(target,current,load,phase,live,.1)
    assert c.state[0].tolist()==[2,1]
    assert not c.aborted[0].any()
    torch.testing.assert_close(c.clock[0],torch.tensor([1.95,.1],dtype=torch.float64))


def test_reclose_waits_for_every_participating_hand_to_release_and_arrive():
    c,target,current,load,phase,live=setup()
    c.state[0]=torch.tensor([2,1]);c.goal.copy_(target);current.copy_(target)
    load[0,0]=0.
    for _ in range(5):c.apply(target,current,load,phase,live,.01)
    assert c.state[0].tolist()==[2,1]
    load.zero_()
    for _ in range(5):c.apply(target,current,load,phase,live,.01)
    assert c.state[0].tolist()==[2,2]
    for _ in range(5):c.apply(target,current,load,phase,live,.01)
    assert c.state[0].tolist()==[3,3]


def test_next_attempt_waits_for_both_hands_to_finish_reclosing():
    c,target,current,load,phase,live=setup()
    c.state[0]=3;c.goal.copy_(target);load[0]=torch.tensor([1.,0.])
    for _ in range(5):c.apply(target,current,load,phase,live,.01)
    assert c.state[0].tolist()==[3,3]
    load[0,1]=1.
    for _ in range(5):c.apply(target,current,load,phase,live,.01)
    assert c.state[0].tolist()==[0,0]


def test_release_requires_unload_margin_before_recontact_hysteresis():
    c,target,current,load,phase,live=setup()
    c.state[0]=1;c.goal.copy_(target)
    # Saved traces repeatedly stopped opening near .03 N, released after 50 ms,
    # and exceeded .05 N again after only a few descent ticks.
    load.fill_(.03)
    for _ in range(5):c.apply(target,current,load,phase,live,.01)
    assert c.state[0].tolist()==[1,1]
    load.fill_(.005)
    for _ in range(5):c.apply(target,current,load,phase,live,.01)
    assert c.state[0].tolist()==[2,2]
    load.fill_(.03);c.apply(target,current,load,phase,live,.01)
    assert c.state[0].tolist()==[2,2]
    load.fill_(.06);c.apply(target,current,load,phase,live,.01)
    assert c.state[0].tolist()==[1,1]


def test_controller_keeps_opening_until_the_lower_release_threshold():
    from test_dh116_coordinated_grasp import enabled
    from test_dh116_tensor_contact_window import loads
    t=enabled();c,*_=setup();t.palm_reposition=c
    t.m=loads(t,.03);t.command.copy_(t.m['q']);before=t.command.clone()
    c.state[0]=1;c.goal[0,:,2]=-.001
    t.control(torch.zeros((2,20),dtype=torch.float64),~t.done,.01)
    assert (t.command[0,[1,2,7,8]]<before[0,[1,2,7,8]]).all()


def test_exhausted_loaded_downward_request_is_an_explicit_failure():
    c,target,current,load,phase,live=setup()
    c.attempts[:]=2
    out,_=c.apply(target,current,load,phase,live,.01)
    assert c.aborted.tolist()==[[True,True],[False,False]]
    assert c.exhausted_reposition.tolist()==[[True,True],[False,False]]
    assert c.attempts.tolist()==[[2,2],[2,2]]
    torch.testing.assert_close(out[0,:,2],current[0,:,2])
    c.reset([0]);assert not c.exhausted_reposition.any()


def test_exhausted_attempts_preserve_unloaded_small_upward_and_lift_requests():
    for kind in ('unloaded','small','upward','lift'):
        c,target,current,load,phase,live=setup();c.attempts[:]=2
        if kind=='unloaded':load.zero_()
        elif kind=='small':target[:,:,2]=-.0001
        elif kind=='upward':target[:,:,2]=.001
        else:phase.fill_(3)
        c.apply(target,current,load,phase,live,.01)
        assert not c.aborted.any(),kind


def test_exhaustion_on_reclose_completion_cannot_enter_lift_as_ready():
    c,target,current,load,phase,live=setup()
    c.state[0]=3;c.goal.copy_(current);c.attempts[0]=2
    c.dwell[0]=.04
    c.apply(target,current,load,phase,live,.01)
    assert c.state[0].tolist()==[0,0]
    assert c.aborted[0].all()


def test_initially_unloaded_hand_joins_shared_release_without_closing():
    from test_dh116_coordinated_grasp import enabled
    from test_dh116_tensor_contact_window import loads
    t=enabled();c,*_=setup();t.palm_reposition=c
    t.m=loads(t,4.);t.m['hand_object_loads'][0,1]=0.
    t.command.copy_(t.m['q']);before=t.command.clone()
    action=torch.ones((2,20),dtype=torch.float64);action[:,14]=action[:,17]=-.5
    t.control(action,~t.done,.01)
    assert c.state[0].tolist()==[1,1]
    assert c.attempts[0].tolist()==[1,1]
    torch.testing.assert_close(t.command[0,6:],before[0,6:])
    assert (t.command[0,[1,2]]<before[0,[1,2]]).all()


def test_force_direction_reposition_retains_its_own_release_opening_threshold():
    from test_dh116_coordinated_grasp import enabled
    from test_dh116_tensor_contact_window import loads
    from go2_pvcnn.control.dh116.tabletop_force_direction_reposition import ForceDirectionReposition
    t=enabled();c=ForceDirectionReposition(torch.tensor([True,False]))
    t.palm_reposition=c;t.force_direction_reposition=True
    t.m=loads(t,.03);t.command.copy_(t.m['q']);before=t.command.clone()
    t.m['vertical_hand_force_n']=torch.zeros((2,2),dtype=torch.float64)
    c.state[0]=2;c.goal[0,:,2]=-.001
    t.control(torch.zeros((2,20),dtype=torch.float64),~t.done,.01)
    torch.testing.assert_close(t.command[0],before[0])
