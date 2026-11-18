"""Synthetic CPU pressure proposals, never native contact/force validation."""
from copy import deepcopy
import importlib
import importlib.util
import numpy as np
import pytest

from test_tripod_candidate_planner import fixture as candidate_fixture
from go2_pvcnn.control.dh116.linear_coupling import JOINTS

NAME='go2_pvcnn.control.dh116.tabletop_tripod_pressure_hold'


def api():
    assert importlib.util.find_spec(NAME) is not None,'missing fresh native-pressure geometric hold'
    return importlib.import_module(NAME)


def inputs():
    planner,a=candidate_fixture();m=planner.geometry.models[1];s=a['snapshots'][1]
    patch=m.patches['index'];pose=s['poses'][patch['body']]
    c=pose[:3,:3]@np.asarray(patch['center_local_m'])+pose[:3,3]
    n=pose[:3,:3]@np.asarray(patch['reference_outward_normal_local'])
    tangent=pose[:3,:3]@np.array([0.,1.,0.]);angle=s['full_q'][JOINTS.index('finger22')]
    planner.geometry.scene['rod']=dict(pose_w=[*(c+n*.01005),np.cos(angle/2),0.,0.,np.sin(angle/2)],radius_m=.01,length_m=.01)
    loads=np.zeros((2,5));loads[1,1]=.5
    valid=np.zeros((2,5),bool);valid[1,1]=True
    vectors=np.zeros((2,5,3));vectors[1,1]=-(.95*n+np.sqrt(1-.95**2)*tangent)*.5
    points=np.zeros((2,5,3));points[1,1]=c
    a.update(effective_pad_loads_n=loads,effective_pad_vectors_w=vectors,effective_pad_points_w_m=points,
        effective_pad_valid=valid,effective_pad_depth_m=np.zeros((2,5)),rejected_contact_loads_n=np.zeros(2),
        upstream_pressure_action24=np.zeros(24,np.float32))
    a['bearing_mask']=valid.copy()
    a['bearing_provenance']=dict(source='synthetic-not-native',physics_step=a['snapshot_identity']['physics_step'],
        snapshot_identity=a['snapshot_identity'],native_accepted_contacts=True)
    return api().TripodPressureHold(planner),a


def test_reference_uses_only_actual_bearing_and_original_gain_depth_and_rejected_limits():
    _,a=inputs()
    a['effective_pad_loads_n'][1,:3]=[1.,.5,2.]
    a['effective_pad_valid'][1,:3]=True
    a['effective_pad_vectors_w'][1,:3]=[[1.,0.,0.],[0.,1.,0.],[0.,0.,2.]]
    a['effective_pad_depth_m'][1,0]=.00060001
    refs=api().pressure_references(**{k:v for k,v in a.items() if k.startswith('effective_') or k=='rejected_contact_loads_n'})
    assert [r['goal_n'] for r in refs]==[1.5,.75,.75]
    assert [r['speed_m_s'] for r in refs]==pytest.approx([-.001,.0001,-.0005])
    np.testing.assert_array_equal(refs[0]['inward_w'],[-1.,0.,0.])
    assert refs[0]['mandatory_relief']
    a['rejected_contact_loads_n'][1]=.050001
    refs=api().pressure_references(**{k:v for k,v in a.items() if k.startswith('effective_') or k=='rejected_contact_loads_n'})
    assert all(r['speed_m_s']==-.001 and r['mandatory_relief'] for r in refs)


@pytest.mark.parametrize('bad',['no_contact','invalid','zero_vector','nan_point','negative_depth','unknown_rejected'])
def test_unknown_or_missing_pressure_never_acquires_or_imagines_normal(bad):
    hold,a=inputs()
    if bad=='no_contact':a['effective_pad_loads_n'][1,1]=0.
    if bad=='invalid':a['effective_pad_valid'][1,1]=False
    if bad=='zero_vector':a['effective_pad_vectors_w'][1,1]=0.
    if bad=='nan_point':a['effective_pad_points_w_m'][1,1,0]=np.nan
    if bad=='negative_depth':a['effective_pad_depth_m'][1,1]=-.001
    if bad=='unknown_rejected':a['rejected_contact_loads_n'][1]=np.nan
    out=hold.plan(**a)
    np.testing.assert_array_equal(out['action24'],np.zeros(24,np.float32))
    assert out['per_pad'][1]['status'] in ('no_actual_bearing','unknown_measurement')
    assert not out['native_contact_verified'] and out['budget_consumed_m']==0.


def test_fresh_pressure_moves_only_actual_right_index_and_audits_left_tracking():
    hold,a=inputs();a['current_command'][0,0]+=.002
    out=hold.plan(**a)
    assert out['accepted'],out
    assert out['per_pad'][1]['status']=='certified_geometric_pressure_proposal'
    assert np.flatnonzero(out['action24']).tolist()==[8]
    assert out['per_pad'][1]['predicted_normal_displacement_m']>0
    assert out['audit']['predicted_deltas'][-1,0,0]>0
    assert out['audit']['coverage']['source_hulls']==[17,17]
    assert len(out['audit']['swept']['collision']['interhand'])==289
    assert out['reacquisition_budget_untouched']
    assert not out['force_prediction'] and not out['native_contact_verified']


def test_opposite_fresh_actual_vector_changes_pressure_target_without_cached_normal():
    hold,a=inputs();first=hold.plan(**a)
    a['effective_pad_vectors_w'][1,1]*=-1
    second=hold.plan(**a)
    np.testing.assert_allclose(first['per_pad'][1]['inward_w'],-second['per_pad'][1]['inward_w'])
    assert first['action24'][8]*second['action24'][8]<0


def test_solver_rejection_does_not_latch_future_pressure_attempts_or_clear_old_counters():
    hold,a=inputs();p=hold.planner.geometry;m=p.models[1]
    from go2_pvcnn.control.dh116.tabletop_tripod_nonlinear_geometry import world_hull
    hull=m.predict(a['snapshots'][1],np.zeros(6))['hulls'][0]
    p.obstacles.append(world_hull(hull['vertices'],np.eye(4),'synthetic-blocker'))
    for _ in range(4):
        out=hold.plan(**a)
        assert not out['accepted'] and np.all(out['action24']==0)
    p.obstacles.pop()
    out=hold.plan(**a)
    assert out['accepted'] and out['action24'][8]!=0
    assert not hasattr(hold,'bad_predictions')


def test_original_outward_pressure_is_never_replaced_by_inward_after_rebuild():
    hold,a=inputs();ordinary=hold.plan(**a)
    a['upstream_pressure_action24'][8]=-ordinary['action24'][8]
    out=hold.plan(**a)
    assert out['per_pad'][1]['must_preserve_upstream_relief']
    assert out['per_pad'][1]['speed_m_s']<=0
    assert out['per_pad'][1]['predicted_normal_displacement_m']<=0
    assert ordinary['action24'][8]*out['action24'][8]<=0


def test_same_clock_measurements_have_owned_diagnostic_copies_and_do_not_modify_input():
    hold,a=inputs();saved=deepcopy(a)
    out=hold.plan(**a)
    out['per_pad'][1]['actual_point_w_m'][0]+=2.
    np.testing.assert_array_equal(a['effective_pad_points_w_m'],saved['effective_pad_points_w_m'])
    np.testing.assert_array_equal(a['current_command'],saved['current_command'])
    bad=deepcopy(a);bad['snapshots'][1]['measurement_identity']['physics_step']-=1
    with pytest.raises(ValueError,match='identity'):hold.plan(**bad)


def test_pressure_command_arc_length_respects_each_480hz_speed_limit_without_reacq_charge():
    hold,a=inputs();a['current_command']=a['current_command'].astype(np.float64)+1.5e-8
    out=hold.plan(**a)
    assert out['accepted']
    assert np.max(out['per_substep_command_path_m'])<=.001/480+1e-15
    assert out['budget_consumed_m']==0 and out['reacquisition_budget_untouched']


def test_mandatory_depth_relief_never_hides_geometry_block_as_partial_acceptance():
    hold,a=inputs();a['effective_pad_depth_m'][1,1]=.0007
    p=hold.planner.geometry;m=p.models[1]
    from go2_pvcnn.control.dh116.tabletop_tripod_nonlinear_geometry import world_hull
    hull=m.predict(a['snapshots'][1],np.zeros(6))['hulls'][0]
    p.obstacles.append(world_hull(hull['vertices'],np.eye(4),'synthetic-blocker'))
    out=hold.plan(**a)
    assert not out['accepted']
    assert out['per_pad'][1]['mandatory_relief'] and out['per_pad'][1]['relief_blocked']
    np.testing.assert_array_equal(out['action24'],np.zeros(24,np.float32))


def test_pressure_law_matches_original_reference_without_new_force_prediction():
    import torch
    from go2_pvcnn.control.dh116.tabletop_teacher_pressure_reference import pressure_reference_targets
    for loads,depth,rejected in [([1.,.5,2.],[0.,0.,0.],0.),([8.,20.,.051],[.0006,.00060001,0.],0.),([.05,.051,1.],[0.,0.,0.],.05)]:
        _,a=inputs();a['effective_pad_loads_n'][1,:3]=loads;a['effective_pad_depth_m'][1,:3]=depth
        a['effective_pad_valid'][1,:3]=True;a['effective_pad_vectors_w'][1,:3]=np.eye(3)
        a['rejected_contact_loads_n'][1]=rejected
        refs=api().pressure_references(**{k:v for k,v in a.items() if k.startswith('effective_') or k=='rejected_contact_loads_n'})
        goals=np.tile([1.5,.75,.75,0.,0.],(2,1))
        old=pressure_reference_targets(torch.tensor(a['effective_pad_points_w_m'][None]),
            torch.tensor(a['effective_pad_vectors_w'][None]),torch.tensor(a['effective_pad_loads_n'][None]),
            torch.tensor(a['effective_pad_valid'][None]),torch.tensor(goals[None]),horizon=1,
            effective_depth_m=torch.tensor(a['effective_pad_depth_m'][None]))
        np.testing.assert_allclose([r['speed_m_s'] for r in refs],old['speed_m_s'][0,1,:3].numpy(),atol=0,rtol=0)


def test_false_bearing_geometry_mask_cannot_authorize_real_measured_pressure_proposal():
    hold,a=inputs();a['bearing_mask'][1,1]=False
    out=hold.plan(**a)
    assert not out['accepted']
    assert out['per_pad'][1]['status']=='bearing_provenance_mismatch'
    assert not np.any(out['action24'])
