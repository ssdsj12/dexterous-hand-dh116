"""Numerical CPU candidates are not native contacts or drive validation."""
import copy
import importlib
import importlib.util

import numpy as np
import pytest

from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, coupling_matrix
from test_tripod_nonlinear_geometry import safe_audit_fixture

NAME = 'go2_pvcnn.control.dh116.tabletop_tripod_candidate_planner'
IDS = [JOINTS.index(v) for v in ACTIVE]


def api():
    assert importlib.util.find_spec(NAME) is not None, 'missing bimanual candidate planner'
    return importlib.import_module(NAME)


def fixture():
    geometry, snapshots = safe_audit_fixture()
    identity = dict(env_id=4, episode_id=1, physics_step=437, source_snapshot_id='fixture-not-native')
    for s in snapshots:
        s['measurement_identity'] = copy.deepcopy(identity)
        s['full_qd'] = np.zeros(11)
    command = np.stack([s['full_q'][IDS] for s in snapshots]).astype(np.float32)
    planner = api().TripodCandidatePlanner(geometry,
        active_lower=np.zeros((2, 6)), active_upper=np.ones((2, 6))*2,
        full_lower=np.zeros((2, 11)), full_upper=np.ones((2, 11))*2,
        limit_sources={k:'explicit synthetic test' for k in ('active', 'full', 'lead', 'response', 'command')},
        max_candidates=1)
    args = dict(snapshots=snapshots, full_qd=np.zeros((2, 11)), current_command=command,
        snapshot_identity=identity, bearing_mask=np.zeros((2, 5), dtype=bool), bearing_provenance=None)
    return planner, args


def test_decoder_integrates_float32_velocity_each_physics_step_not_held_endpoint():
    m = api(); q = np.ones((2, 6), dtype=np.float32)*.5
    a = np.zeros(24, dtype=np.float32); a[6] = .5
    path = m.decode_velocity_action24(q, q, a, np.zeros_like(q), np.ones_like(q), substeps=8)
    c = q.copy(); expected=[]
    for _ in range(8):
        c = c+a[:12].reshape(2, 6)*np.float32(.3)*np.float32(1/480)
        expected.append(c.copy())
    np.testing.assert_array_equal(path['command_targets'], np.asarray(expected))
    alpha = 1-(1-.5)**(1/8)
    assert path['predicted_deltas'][0, 1, 0] == pytest.approx(alpha*(expected[0][1, 0]-.5))
    assert path['predicted_deltas'][0, 1, 0] < alpha*(expected[-1][1, 0]-.5)
    assert path['command_targets'].dtype == np.float32


def test_double_logical_command_matches_actual_surface_control_sdk_cast_each_substep():
    import torch
    from test_surface_exploration import surface_task
    from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import control
    t,w=surface_task();middle=(t.low+t.high)/2
    t.command[:]=middle+1.234567e-8;t.m['q'][:]=middle
    t.surface_policy_owned_targets=True;t.surface_redistribution=False
    original=t.command[0].reshape(2,6).numpy().copy()
    initial=t.m['q'][0].reshape(2,6).numpy().copy()
    action=np.zeros(24,np.float32);action[6:10]=[.111111,.222223,-.333333,.444441]
    path=api().decode_velocity_action24(original,initial,action,t.low.reshape(2,6).numpy(),
        t.high.reshape(2,6).numpy(),substeps=8)
    commands=[];logical=[];alpha=1-(1-.5)**(1/8);predicted=initial.copy()
    for _ in range(8):
        t.m['q'][:]=torch.from_numpy(predicted.reshape(12)).double()
        control(t,torch.from_numpy(action).double()[None].repeat(2,1),torch.ones(2,dtype=torch.bool),1/480)
        logical.append(t.command[0].reshape(2,6).numpy().copy())
        commands.append(t.command[0].float().reshape(2,6).numpy().copy())
        predicted=predicted+alpha*(commands[-1].astype(float)-predicted)
    np.testing.assert_array_equal(path['logical_command_targets'],np.asarray(logical))
    np.testing.assert_array_equal(path['command_targets'],np.asarray(commands))
    assert path['logical_command_targets'].dtype==np.float64


def test_postguard_double_logical_and_float_sdk_targets_must_bind_exactly():
    p,a=fixture();a['current_command']=a['current_command'].astype(np.float64)+1.234567e-8
    logical=a['current_command'].copy();logical[1,2]+=.0001
    sdk=logical.astype(np.float32)
    with pytest.raises(ValueError,match='logical'):
        p.audit_commands(final_next_command=sdk,**a)
    good=p.audit_commands(final_next_command=sdk,final_logical_command=logical,**a)
    assert good['accepted']
    bad=logical.copy();bad[1,2]+=.000001
    with pytest.raises(ValueError,match='encoded'):
        p.audit_commands(final_next_command=sdk,final_logical_command=bad,**a)


def test_bimanual_trajectory_predicts_left_tracking_and_all_source_hulls_sweep():
    p, a = fixture(); a['current_command'][0, 0] += .002
    out = p.audit_action(action24=np.zeros(24, np.float32), substeps=8, **a)
    assert out['accepted'], out['rejection_reasons']
    assert out['coverage']['physics_substeps'] == 8
    assert out['predicted_deltas'][-1, 0, 0] > 0
    assert out['coverage']['source_hulls'] == [17, 17]
    assert out['coverage']['swept_method'] == 'whole_trajectory_joint_box_certificate'
    assert len(out['swept']['collision']['interhand']) == 289
    assert out['swept']['left_prediction_scope'] == 'explicit_left_joint_increment'
    assert not out['native_contact_verified']


def test_current_pose_full_follower_residual_is_not_projected_to_ideal_mimic():
    p, a = fixture(); a['snapshots'][1]['full_q'][2] += .013
    out = p.audit_action(action24=np.zeros(24, np.float32), substeps=1, **a)
    np.testing.assert_allclose(out['predicted_full_q'][0, 1],
        a['snapshots'][1]['full_q']+coupling_matrix()@out['predicted_deltas'][0, 1])
    assert out['zero_rebase_calibration']['maximum_pose_error'] < 1e-12
    assert out['zero_rebase_calibration']['native_kinematics_validated'] is False


def test_same_clock_identity_and_full_qd_are_required_not_only_six_joints():
    p, a = fixture(); a['snapshots'][1]['measurement_identity']['physics_step'] -= 1
    with pytest.raises(ValueError, match='identity'):
        p.audit_action(action24=np.zeros(24, np.float32), substeps=1, **a)
    p, a = fixture(); a['full_qd'][1, 2] = .01
    with pytest.raises(ValueError, match='velocity'):
        p.audit_action(action24=np.zeros(24, np.float32), substeps=1, **a)


def test_postguard_audit_uses_encoded_float32_command_and_conservative_pad_command_path():
    p, a = fixture(); final = a['current_command'].copy(); final[1, 2] += np.float32(.00001)
    out = p.audit_commands(final_next_command=final, **a)
    assert out['accepted'], out['rejection_reasons']
    np.testing.assert_array_equal(out['first_command'], final)
    bounds = out['approach_command_m']
    assert bounds[1] > 0 and bounds[0] == 0 and bounds[2] == 0
    assert out['distance_kind'] == 'source_pad_command_path_arclength_upper_bound_not_actual_motion'
    # All source pad vertices, dense interpolation, are bounded by the path certificate.
    model = p.geometry.models[1]; patch = model.patches['index']; s=a['snapshots'][1]
    shape = next(v for v in model.shapes if v['body'] == patch['body'])
    previous = None; length = 0.
    for fraction in np.linspace(0, 1, 101):
        delta = a['current_command'][1]+fraction*(final[1]-a['current_command'][1])-s['full_q'][IDS]
        pose = model.predict(s, delta)['poses'][patch['body']]
        point = pose[:3, :3]@shape['vertices'][0]+pose[:3, 3]
        if previous is not None: length += np.linalg.norm(point-previous)
        previous=point
    assert length <= bounds[1]+1e-12


def test_postguard_beyond_original_velocity_and_actual_active_limits_rejected():
    p, a=fixture(); final=a['current_command'].copy(); final[1, 2] += .001
    out=p.audit_commands(final_next_command=final, **a)
    assert not out['accepted'] and 'COMMAND_SLEW' in out['rejection_reasons']
    p, a=fixture(); a['snapshots'][1]['full_q'][0] = 2.01
    out=p.audit_commands(final_next_command=a['current_command'], **a)
    assert not out['accepted'] and 'ACTUAL_ACTIVE_LIMITS' in out['rejection_reasons']


def test_one_float32_ulp_beyond_native_slew_is_not_a_numeric_relaxation():
    p,a=fixture(); final=a['current_command'].copy()
    bound=np.float32(final[1,2]+np.float32(.3/480))
    final[1,2]=np.nextafter(bound,np.float32(np.inf))
    out=p.audit_commands(final_next_command=final,**a)
    assert not out['accepted'] and 'COMMAND_SLEW' in out['rejection_reasons']


def test_near_current_rod_candidate_uses_only_missing_index_and_spends_no_budget():
    p,a=fixture(); model=p.geometry.models[1]; snapshot=a['snapshots'][1]
    patch=model.patches['index'];pose=snapshot['poses'][patch['body']]
    center=pose[:3,:3]@np.array(patch['center_local_m'])+pose[:3,3]
    normal=pose[:3,:3]@np.array(patch['reference_outward_normal_local'])
    angle=snapshot['full_q'][JOINTS.index('finger22')]
    p.geometry.scene['rod']=dict(pose_w=[*(center+normal*.01005),np.cos(angle/2),0.,0.,np.sin(angle/2)],radius_m=.01,length_m=.01)
    out=p.plan(missing_right=np.array([False,True,False]),remaining_approach_m=np.array([.001,.001,.001]),
               substeps=8,**a)
    assert out['accepted'], (out.get('reason'),out.get('rejection_reasons'))
    changed=np.flatnonzero(out['action24'])
    np.testing.assert_array_equal(changed,[8])
    assert 0 < out['approach_command_m'][1] <= .001/60
    assert out['budget_consumed_m']==0
    assert not out['native_contact_verified']
    target=out['targets_current_rod'][1]
    np.testing.assert_array_equal(target['object_pose_w'],p.geometry.scene['rod']['pose_w'])


def test_double_task_budget_cannot_borrow_next_substeps_distance_after_sdk_rounding():
    p,a=fixture();a['current_command']=a['current_command'].astype(np.float64)+1.5e-8
    m=p.geometry.models[1];s=a['snapshots'][1];patch=m.patches['index'];pose=s['poses'][patch['body']]
    c=pose[:3,:3]@np.array(patch['center_local_m'])+pose[:3,3]
    normal=pose[:3,:3]@np.array(patch['reference_outward_normal_local']);angle=s['full_q'][JOINTS.index('finger22')]
    p.geometry.scene['rod']=dict(pose_w=[*(c+normal*.01005),np.cos(angle/2),0.,0.,np.sin(angle/2)],radius_m=.01,length_m=.01)
    out=p.plan(missing_right=np.array([False,True,False]),remaining_approach_m=np.ones(3)*.001,substeps=8,**a)
    assert out['accepted']
    from go2_pvcnn.control.dh116.tabletop_tripod_candidate_planner import _command_path_bounds
    previous=a['current_command'].astype(np.float32)
    for cmd in out['command_targets']:
        assert _command_path_bounds(m,s,previous[1],cmd[1])[1] <= .001/480+1e-15
        previous=cmd


def test_full_sweep_rejects_fixed_original_obstacle_and_missing_region_not_sampled_acceptance():
    p, a=fixture(); model=p.geometry.models[1]; initial=model.predict(a['snapshots'][1],np.zeros(6))
    from go2_pvcnn.control.dh116.tabletop_tripod_nonlinear_geometry import world_hull
    hull=initial['hulls'][0]
    p.geometry.obstacles.append(world_hull(hull['vertices'], np.eye(4), 'original-obstacle-fixture'))
    out=p.audit_action(action24=np.zeros(24,np.float32),substeps=1,**a)
    assert not out['accepted']
    assert any('ORIGINAL_OBSTACLE_UNCERTIFIED' in v for v in out['rejection_reasons'])


def test_plan_is_budget_owned_missing_digit_only_and_no_measured_contact_claim():
    p, a=fixture()
    out=p.plan(missing_right=np.array([False, True, False]), remaining_approach_m=np.array([0.,0.,.001]),
               substeps=1, **a)
    assert not out['accepted'] and out['reason'] == 'NO_ELIGIBLE_RECOVERY_BUDGET'
    np.testing.assert_array_equal(out['action24'],np.zeros(24,np.float32))
    assert out['budget_consumed_m'] == 0 and not out['native_contact_verified']
    # All targets remain tied to actual scene's rod, no commanded wrist pose input exists.
    np.testing.assert_array_equal(out['rod_snapshot']['pose_w'],p.geometry.scene['rod']['pose_w'])


def test_invalid_nonfinger_motion_and_mutating_actual_snapshot_are_not_allowed():
    p,a=fixture(); action=np.zeros(24,np.float32); action[14]=.1
    with pytest.raises(ValueError,match='fixed wrist'):
        p.audit_action(action24=action,substeps=1,**a)
    p,a=fixture(); saved=copy.deepcopy(a)
    p.audit_action(action24=np.zeros(24,np.float32),substeps=1,**a)
    for h in range(2):
        np.testing.assert_array_equal(a['snapshots'][h]['full_q'],saved['snapshots'][h]['full_q'])
        for body in a['snapshots'][h]['poses']:
            np.testing.assert_array_equal(a['snapshots'][h]['poses'][body],saved['snapshots'][h]['poses'][body])


def test_real_source_usd_bimanual_full_coverage_and_command_path_bounds():
    from pathlib import Path
    from go2_pvcnn.control.dh116.tabletop_tripod_nonlinear_geometry import SourceHandGeometry,TripodNonlinearGeometry,GeometryThresholds
    from go2_pvcnn.control.dh116.hand_point_kinematics import HandPointFK
    from go2_pvcnn.control.dh116.linear_coupling import active_limits,expand
    root=Path(__file__).resolve().parents[1]/'assets/fixed_dual_panda_dh116_v1/dependencies'
    models=[];snapshots=[];identity=dict(env_id=0,episode_id=0,physics_step=0,source_snapshot_id='ideal-usd-not-native')
    for side,filename,offset in [('left','d134680ab619_DH116_left.usd',-2.),('right','3046ad2ed855_DH116_right.usd',2.)]:
        source=root/filename;model=SourceHandGeometry.from_usd(source,side)
        q=np.mean(active_limits(),axis=0)
        poses=HandPointFK(source,side,[offset,0.,.5,1.,0.,0.,0.]).poses(q)
        snapshot=model.bind(expand(q),poses);snapshot['measurement_identity']=identity.copy();snapshot['full_qd']=np.zeros(11)
        models.append(model);snapshots.append(snapshot)
    fields=dict(contact_distance_m=.001,normal_cosine_min=.8,object_overlap_m=.0003,
        noncontact_object_clearance_m=0.,table_clearance_m=.0003,self_clearance_m=0.,
        interhand_clearance_m=0.,obstacle_clearance_m=0.,max_command_step_rad=.005,command_lead_rad=.1)
    geometry=TripodNonlinearGeometry(*models,dict(rod=dict(pose_w=[0.,0.,.5,1.,0.,0.,0.],radius_m=.008,length_m=.4),
        table_top_z_m=0.,obstacles=[],obstacles_complete=True,provenance={'synthetic':True}),
        GeometryThresholds(**fields,sources={k:'synthetic test explicit limit' for k in fields}))
    planner=api().TripodCandidatePlanner(geometry,active_lower=np.zeros((2,6)),active_upper=np.ones((2,6))*2,
        full_lower=np.zeros((2,11)),full_upper=np.ones((2,11))*2,
        limit_sources={k:'synthetic bounds, not physical deployment' for k in ('active','full','lead','response','command')})
    command=np.stack([s['full_q'][IDS] for s in snapshots])
    action=np.zeros(24,np.float32);action[6:10]=[.02,-.03,.04,-.01]
    out=planner.audit_action(snapshots=snapshots,full_qd=np.zeros((2,11)),current_command=command,
        snapshot_identity=identity,bearing_mask=np.zeros((2,5),bool),bearing_provenance=None,action24=action,substeps=8)
    assert out['zero_rebase_calibration']['passed']
    assert out['swept']['source_hull_counts']==[17,17]
    assert len(out['swept']['collision']['interhand'])==289
    assert len(out['swept']['collision']['object'])==34
    model=models[1];snapshot=snapshots[1]
    bounds=out['approach_command_m']
    previous=None;distance={digit:0. for digit in ('thumb','index','middle')}
    for fraction in np.linspace(0,1,31):
        encoded=command[1].astype(np.float32).astype(float)+fraction*(out['command_targets'][-1,1].astype(float)-command[1].astype(np.float32))
        moved=model.predict(snapshot,encoded-snapshot['full_q'][IDS])
        points={digit:pose[:3,:3]@np.array(model.patches[digit]['center_local_m'])+pose[:3,3]
            for digit in distance for pose in [moved['poses'][model.patches[digit]['body']]]}
        if previous is not None:
            for digit in distance:distance[digit]+=np.linalg.norm(points[digit]-previous[digit])
        previous=points
    assert all(distance[digit]<=bounds[i]+1e-12 for i,digit in enumerate(distance))
    assert out['native_candidate_eligible'] is False
