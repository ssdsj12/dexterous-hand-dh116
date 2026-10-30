"""Only raw sustained physical lift may publish a DH116 teacher case."""
import importlib
import importlib.util
import json
import pytest
import torch


def module():
    name='go2_pvcnn.control.dh116.tabletop_surface_teacher'
    assert importlib.util.find_spec(name) is not None, 'teacher qualification missing'
    return importlib.import_module(name)


def frame(n=2):
    pose=torch.zeros(n,7,dtype=torch.float64);pose[:,2]=.318;pose[:,3]=1.
    return dict(object_pose_w=pose,bottom=torch.full((n,),.310,dtype=torch.float64),
        digit_normal_load_n=torch.full((n,2,2),2.,dtype=torch.float64),
        hand_normal_load_n=torch.full((n,2),4.,dtype=torch.float64),
        object_velocity=torch.zeros(n,6,dtype=torch.float64),
        requested_hand_force_n=torch.full((n,2),4.,dtype=torch.float64),
        object_weight_n=torch.full((n,),1.1772,dtype=torch.float64),
        vertical_components_n=torch.tensor([[[0.,.5886],[0.,.5886]]],dtype=torch.float64).repeat(n,1,1),
        hand_clearance_m=torch.full((n,2),.005,dtype=torch.float64),
        object_penetration_m=torch.zeros(n,dtype=torch.float64),
        self_penetration_m=torch.zeros(n,dtype=torch.float64),
        fault=torch.zeros(n,dtype=torch.long),reset=torch.zeros(n,dtype=torch.bool),
        observations=torch.zeros(n,261,dtype=torch.float64),
        actions=torch.cat((torch.zeros(n,24,dtype=torch.float64),torch.full((n,2),-.6,dtype=torch.float64)),1))


def test_sustained_actual_lift_and_four_digit_bearing_is_required():
    q=module().Qualification(2,'cpu');m=frame()
    for _ in range(17):assert not q.update(m,1/60).any()
    assert q.update(m,1/60).tolist()==[True,True]
    assert q.longest_hold_s.tolist()==pytest.approx([.3,.3])


@pytest.mark.parametrize('key,index,value',[
    ('digit_normal_load_n',(0,0,0),0.),
    ('digit_normal_load_n',(0,1,1),.01),
    ('digit_normal_load_n',(0,0,0),4.),
    ('hand_normal_load_n',(0,0),13.),
    ('hand_normal_load_n',(0,1),1.),
    ('object_velocity',(0,0),.031),
    ('object_penetration_m',(0,),.0011),
    ('self_penetration_m',(0,),.0011),
    ('hand_clearance_m',(0,0),-.0001),
    ('object_pose_w',(0,2),.317),
])
def test_unqualified_row_never_published_but_other_row_isolated(key,index,value):
    q=module().Qualification(2,'cpu');m=frame();m[key][index]=value
    if key=='object_pose_w':m['bottom'][0]=.309
    for _ in range(30):accepted=q.update(m,1/60)
    assert accepted.tolist()==[False,True]


@pytest.mark.parametrize('key',['fault','reset'])
def test_any_fault_or_reset_disqualifies_entire_candidate(key):
    q=module().Qualification(2,'cpu');m=frame();m[key][0]=1
    q.update(m,1/60);m[key][0]=0
    for _ in range(30):accepted=q.update(m,1/60)
    assert accepted.tolist()==[False,True]


def test_reported_height_cannot_override_actual_object_geometry():
    q=module().Qualification(2,'cpu');m=frame();m['object_pose_w'][:,2]=.308
    # Deliberately retain the false .310 bottom report.
    for _ in range(30):assert not q.update(m,1/60).any()


def test_zero_quaternion_and_nonfinite_load_are_rejected_as_candidate_evidence():
    q=module().Qualification(2,'cpu');m=frame();m['object_pose_w'][0,3:]=0.;m['digit_normal_load_n'][1,0,0]=float('nan')
    for _ in range(30):assert not q.update(m,1/60).any()


def test_hold_break_resets_dwell():
    q=module().Qualification(2,'cpu');m=frame()
    for _ in range(17):q.update(m,1/60)
    m['digit_normal_load_n'][0,0,0]=0.;q.update(m,1/60)
    m['digit_normal_load_n'][0,0,0]=2.
    for _ in range(17):assert not q.update(m,1/60)[0]
    assert q.update(m,1/60)[0]


def test_bad_physics_substep_cannot_be_hidden_by_control_tick_average():
    q=module().Qualification(2,'cpu');m=frame()
    m['physics_interval_qualified']=torch.tensor([False,True])
    for _ in range(30):accepted=q.update(m,1/60)
    assert accepted.tolist()==[False,True]


def trace(frames):
    result={key:torch.stack([m[key] for m in frames]) for key in frames[0]}
    for output,key in [('physics_digit_normal_load_n','digit_normal_load_n'),
        ('physics_hand_normal_load_n','hand_normal_load_n'),('physics_bottom','bottom'),
        ('physics_object_pose_w','object_pose_w'),('physics_object_velocity','object_velocity'),('physics_fault','fault'),
        ('physics_requested_hand_force_n','requested_hand_force_n'),('physics_object_weight_n','object_weight_n'),
        ('physics_vertical_components_n','vertical_components_n')]:
        value=result[key]
        result[output]=value.unsqueeze(2).expand(*value.shape[:2],8,*value.shape[2:]).clone()
    return result


def metadata():
    return dict(active_dof_count=12,passive_dof_count=10,finger_dof_count=22,
        physics=dict(gravity=[0.,0.,-9.81],object_mass_kg=.12,object_material='extreme_grip'),
        source_hashes={'left':'a'*64,'right':'b'*64},observation_size=261,action_size=26,physics_substeps_per_action=8,
        implementation_hashes={name:'c'*64 for name in module().REQUIRED_IMPLEMENTATIONS})


def initial_cases():
    return [dict(active_angles=[.2]*12,wrist_pose_w=[[0.,0.,.4,1.,0.,0.,0.]]*2) for _ in range(2)]


def test_export_rechecks_raw_evidence_and_writes_nothing_for_empty_success(tmp_path):
    m=frame();m['object_pose_w'][:,2]=.308;m['bottom'][:]=.300
    paths=module().export_qualified(tmp_path,trace([m]*30),initial_cases(),metadata(),dt=1/60)
    assert paths==[] and not list(tmp_path.glob('teacher*'))


def test_export_contains_only_qualified_case_and_exact_command_dimensions(tmp_path):
    m=frame();m['digit_normal_load_n'][1,0,0]=0.
    paths=module().export_qualified(tmp_path,trace([m]*30),initial_cases(),metadata(),dt=1/60)
    assert len(paths)==1
    saved=json.loads(paths[0].read_text())
    assert saved['qualified'] and saved['candidate_index']==0 and saved['metadata']['active_dof_count']==12
    data=torch.load(paths[0].with_suffix('.pt'),weights_only=True)
    assert data['actions'].shape==(30,26) and data['observations'].shape==(30,261)


def test_export_cannot_accept_wrong_hand_count_or_nonfinite_actions(tmp_path):
    m=frame();bad=metadata();bad['active_dof_count']=2
    with pytest.raises(ValueError):module().export_qualified(tmp_path,trace([m]*30),initial_cases(),bad,dt=1/60)
    m['actions'][0,0]=float('nan')
    paths=module().export_qualified(tmp_path,trace([m]*30),initial_cases(),metadata(),dt=1/60)
    assert len(paths)==1 and json.loads(paths[0].read_text())['candidate_index']==1


def test_initial_geometry_preflight_uses_current_depths_and_clearance():
    m=module();inside=torch.tensor([0.,.00031,0.,0.],dtype=torch.float64)
    self_depth=torch.tensor([0.,0.,.00031,0.],dtype=torch.float64)
    clearance=torch.tensor([[.001,.001],[.001,.001],[.001,.001],[.001,.0001]],dtype=torch.float64)
    result=m.initial_geometry_mask(inside,torch.zeros_like(inside),self_depth,clearance)
    assert result.tolist()==[True,False,False,False]


def test_export_replays_raw_physics_instead_of_trusting_interval_flag(tmp_path):
    m=frame();m['physics_interval_qualified']=torch.ones(2,dtype=torch.bool)
    data=trace([m]*30)
    data['physics_digit_normal_load_n'][:,0,3,0,0]=0.
    paths=module().export_qualified(tmp_path,data,initial_cases(),metadata(),dt=1/60)
    assert len(paths)==1 and json.loads(paths[0].read_text())['candidate_index']==1


def test_export_rejects_finite_out_of_range_teacher_commands(tmp_path):
    m=frame();m['actions'][0,0]=1.01
    paths=module().export_qualified(tmp_path,trace([m]*30),initial_cases(),metadata(),dt=1/60)
    assert len(paths)==1 and json.loads(paths[0].read_text())['candidate_index']==1


def test_export_isolates_nonfinite_raw_failed_candidate(tmp_path):
    data=trace([frame()]*30);data['physics_object_pose_w'][5,0,2,0]=float('nan')
    data['physics_fault'][5,0,2]=1
    paths=module().export_qualified(tmp_path,data,initial_cases(),metadata(),dt=1/60)
    assert len(paths)==1 and json.loads(paths[0].read_text())['candidate_index']==1


def test_export_requires_complete_raw_substep_evidence(tmp_path):
    data=trace([frame()]*30);data.pop('physics_digit_normal_load_n')
    with pytest.raises(ValueError,match='physics'):
        module().export_qualified(tmp_path,data,initial_cases(),metadata(),dt=1/60)


def test_teacher_export_requires_executed_source_identity(tmp_path):
    data=trace([frame()]*30);missing=metadata();missing.pop('implementation_hashes')
    with pytest.raises(ValueError,match='implementation'):
        module().export_qualified(tmp_path,data,initial_cases(),missing,dt=1/60)


def test_stable_but_wrong_commanded_force_is_not_a_teacher():
    q=module().Qualification(2,'cpu');m=frame();m['requested_hand_force_n'][0,0]=8.
    for _ in range(30):accepted=q.update(m,1/60)
    assert accepted.tolist()==[False,True]


def test_upward_force_support_must_persist_along_with_height():
    q=module().Qualification(2,'cpu');m=frame();m['vertical_components_n'][0].zero_()
    for _ in range(30):accepted=q.update(m,1/60)
    assert accepted.tolist()==[False,True]


def test_raw_substep_command_force_mismatch_is_rejected_at_publication(tmp_path):
    data=trace([frame()]*30);data['physics_requested_hand_force_n'][:,0,4,1]=8.
    paths=module().export_qualified(tmp_path,data,initial_cases(),metadata(),dt=1/60)
    assert len(paths)==1 and json.loads(paths[0].read_text())['candidate_index']==1


def test_logged_force_target_must_match_the_executed_action():
    q=module().Qualification(2,'cpu');m=frame();m['actions'][0,-2:]=0.
    for _ in range(30):accepted=q.update(m,1/60)
    assert accepted.tolist()==[False,True]


def test_support_export_requires_complete_actual_motor_commands(tmp_path):
    data=trace([frame()]*30);meta=metadata();meta['surface_support']={'enabled':True}
    meta['physics']['effort_limit_nm']=.2
    with pytest.raises(ValueError,match='support.*torque'):
        module().export_qualified(tmp_path,data,initial_cases(),meta,dt=1/60)


def test_support_export_rejects_over_budget_motor_commands_per_candidate(tmp_path):
    data=trace([frame()]*30);meta=metadata();meta['surface_support']={'enabled':True}
    meta['physics']['effort_limit_nm']=.2
    data['physics_support_torque_command_nm']=torch.zeros(30,2,8,2,6,dtype=torch.float64)
    data['physics_support_torque_command_nm'][3,0,4,0,0]=1.
    paths=module().export_qualified(tmp_path,data,initial_cases(),meta,dt=1/60)
    assert len(paths)==1 and json.loads(paths[0].read_text())['candidate_index']==1
