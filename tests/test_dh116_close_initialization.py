import hashlib
import json

import numpy as np
import pytest

from go2_pvcnn.control.dh116.close_initialization import (
    from_contact_row, load_close_initialization, robot_initial_arrays, rebase_hand_stiffness,
)
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS, expand


def test_restored_initial_screen_uses_actual_wire_shape_and_volume_corrected_radius():
    from go2_pvcnn.control.dh116.close_initialization import initial_wire_geometry
    nominal=np.array([[0.,0.,.3],[0.,.01,.3],[0.,.02,.3]])
    actual=nominal.copy();actual[:,0]=.005;actual[-1,1]=.019
    payload=dict(rod=dict(dynamic=dict(rod_position=actual.T.tolist()),
                          reference=dict(rest_lengths=[.01,.01])))
    nodes,radius=initial_wire_geometry(payload,nominal,base_radius_m=.008)
    np.testing.assert_array_equal(nodes,actual)
    assert radius==pytest.approx(.008*np.sqrt(.01/.009))
    np.testing.assert_array_equal(initial_wire_geometry(None,nominal)[0],nominal)
    assert initial_wire_geometry(None,nominal)[1]==.008


def contact_row():
    sides={}
    for side in ('left','right'):
        names=[f'{side}_panda_joint{i}' for i in range(1,8)]+list(JOINTS)
        sides[side]=dict(joint_names=names,actual_joint_positions=np.r_[np.zeros(7),expand(np.full(6,.2))].tolist(),
                         actual_joint_velocities=np.zeros(18).tolist())
    return dict(step=8,passed=True,grasp_sequence={'stage':'close'},sides=sides,
                targets={s:dict(arm=[0.]*7,hand=[.2]*6) for s in sides},
                contact={'normal_samples':{s+'_'+d:{'normal_load_n':.5} for s in sides for d in ('thumb','index','middle')}},
                rod_state={'rod_position':[[1.]],'rod_velocity':[[2.]],
                           'rod_director':[[3.]],'rod_omega':[[4.]]})


def test_measured_close_initialization_preserves_state_without_importing_success(tmp_path):
    row=contact_row()
    p=from_contact_row(row,source_run='run',asset_sha256='a'*64,
        contact_manifest_sha256='b'*64,rod_physical={'model_sha256':'c'*64,'reference':{'rest':1}})
    assert p['grasp_qualified'] is False
    assert p['rod']['dynamic']==row['rod_state']
    assert p['robot']['left']['q']==row['sides']['left']['actual_joint_positions']
    assert p['contact_loads_n']==[.5]*6
    path=tmp_path/'initial.json';path.write_text(json.dumps(p))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    loaded=load_close_initialization(path,digest,asset_sha256='a'*64,contact_manifest_sha256='b'*64)
    assert loaded['source']==dict(run='run',step=8)
    with pytest.raises(ValueError,match='identity'):
        load_close_initialization(path,digest,asset_sha256='d'*64,contact_manifest_sha256='b'*64)
    with pytest.raises(ValueError):
        from_contact_row(dict(row,passed=False),source_run='run',asset_sha256='a'*64,
            contact_manifest_sha256='b'*64,rod_physical={})


def test_robot_initialization_maps_actual_q_velocity_and_target_by_name():
    row=contact_row()
    p=from_contact_row(row,source_run='run',asset_sha256='a'*64,
        contact_manifest_sha256='b'*64,rod_physical={})
    saved=p['robot']['left'];names=list(reversed(saved['joint_names']))
    limits=np.tile([-3.,3.],(18,1))
    arrays=robot_initial_arrays(saved,names,limits)
    np.testing.assert_array_equal(arrays['q'],saved['q'][::-1])
    arrays['q'][0]=99.;assert saved['q'][-1]!=99.
    saved['qd'][0]=.2
    with pytest.raises(ValueError,match='velocity'):
        robot_initial_arrays(saved,names,limits)


def test_gain_change_preserves_initial_pd_torque_and_measured_state():
    row=contact_row();row['targets']['left']['hand']=[.21]*6
    p=from_contact_row(row,source_run='run',asset_sha256='a'*64,
        contact_manifest_sha256='b'*64,rod_physical={})
    changed=rebase_hand_stiffness(p,old_stiffness=1.,new_stiffness=6.)
    before=p['robot']['left'];after=changed['robot']['left']
    np.testing.assert_array_equal(after['q'],before['q'])
    np.testing.assert_array_equal(after['qd'],before['qd'])
    for name in ACTIVE:
        i=before['joint_names'].index(name)
        assert 6.*(after['target'][i]-after['q'][i])==pytest.approx(before['target'][i]-before['q'][i])
    assert changed['hand_stiffness_nm_rad']==6.


def test_close_initialization_preserves_the_loaded_controller_reference(tmp_path):
    row=contact_row()
    reference=dict(source_step=0,commanded_penetration_m=[[.0006,.0005,0.],[.0004,.0004,0.]],
                   tracking_bias_rad=[.002]*12)
    p=from_contact_row(row,source_run='run',asset_sha256='a'*64,
        contact_manifest_sha256='b'*64,rod_physical={},controller_reference=reference)
    assert p['controller_reference']==reference
    reference['tracking_bias_rad'][0]=999.
    assert p['controller_reference']['tracking_bias_rad'][0]==.002
    path=tmp_path/'initial.json';path.write_text(json.dumps(p))
    loaded=load_close_initialization(path,hashlib.sha256(path.read_bytes()).hexdigest(),
        asset_sha256='a'*64,contact_manifest_sha256='b'*64)
    assert loaded['controller_reference']==p['controller_reference']
    from go2_pvcnn.control.dh116.grip_load_reference import GripLoadReference
    grip=GripLoadReference(np.ones((2,3)),initial_depth_m=p['controller_reference']['commanded_penetration_m'])
    np.testing.assert_array_equal(grip.depth,p['controller_reference']['commanded_penetration_m'])
    from go2_pvcnn.control.dh116.contact_mpc import ParallelContactMPC
    planner=ParallelContactMPC();planner.restore_tracking_bias(p['controller_reference']['tracking_bias_rad'])
    np.testing.assert_allclose(planner._tracking_bias.numpy(),.002)
    with pytest.raises(ValueError):planner.restore_tracking_bias([.1]*12)
    with pytest.raises(ValueError):planner.restore_tracking_bias([0.]*26)


def test_tracking_calibration_uses_only_completed_prior_physical_intervals():
    from go2_pvcnn.control.dh116.mpc_feedback import estimate_tracking_bias
    q=np.full((2,6),.2);command=np.full(12,.21);nominal=np.full(12,.004)
    rows=[]
    for i in range(20):
        after=q+.5*(command.reshape(2,6)-.01-q)
        rows.append(dict(step=i*100,phase='close',owner='hand_mpc',action=command.tolist(),control_dt_s=.01,
            observation=dict(hand_q=q.tolist(),pad_jacobians=np.ones((2,3,3,6)).tolist(),
                             clearance_jacobians=[]),
            prediction=dict(model='linearized_pad_kinematics',actuator_load_offset_rad=nominal.tolist()),
            outcome=dict(step=(i+1)*100,elapsed_s=.01,passed=True,state=dict(hand_q=after.tolist()))))
        q=after
    expected=estimate_tracking_bias(rows,through_step=2000)
    np.testing.assert_allclose(expected,.006,atol=.00003)
    bad=dict(rows[-1],outcome=dict(step=2100,elapsed_s=.01,passed=True,state=dict(hand_q=np.zeros((2,6)).tolist())))
    np.testing.assert_array_equal(estimate_tracking_bias(rows+[bad],through_step=2000),expected)
    bad['outcome']['step']=1900;bad['outcome']['passed']=False
    np.testing.assert_array_equal(estimate_tracking_bias(rows+[bad],through_step=2000),expected)


def test_gain_rebase_invalidates_old_predictor_bias_but_preserves_radial_reference():
    reference=dict(source_step=0,commanded_penetration_m=[[.0004]*3]*2,tracking_bias_rad=[.002]*12)
    p=from_contact_row(contact_row(),source_run='run',asset_sha256='a'*64,
        contact_manifest_sha256='b'*64,rod_physical={},controller_reference=reference)
    changed=rebase_hand_stiffness(p,old_stiffness=1.,new_stiffness=6.)
    assert 'tracking_bias_rad' not in changed['controller_reference']
    assert changed['controller_reference']['commanded_penetration_m']==reference['commanded_penetration_m']
    same=rebase_hand_stiffness(p,old_stiffness=6.,new_stiffness=6.)
    assert same['controller_reference']==reference


def test_offline_contact_alignment_preserves_wire_velocity_and_initial_drive_deflection():
    from go2_pvcnn.control.dh116.close_initialization import reframe_close_initialization
    row=contact_row();row['targets']['left']['hand']=[.21]*6
    payload=from_contact_row(row,source_run='run',asset_sha256='a'*64,
        contact_manifest_sha256='b'*64,rod_physical={})
    states={s:np.r_[np.full(7,.01),np.full(6,.205)] for s in ('left','right')}
    reframed=reframe_close_initialization(payload,states,alignment={'physics_qualified':False})
    assert reframed['rod']==payload['rod']
    assert reframed['robot']['left']['qd']==payload['robot']['left']['qd']
    assert reframed['grasp_qualified'] is False
    assert reframed['pregrasp_alignment']['physics_qualified'] is False
    for name in ACTIVE:
        old=payload['robot']['left'];new=reframed['robot']['left'];i=old['joint_names'].index(name)
        assert new['q'][i]==pytest.approx(.205)
        assert new['target'][i]-new['q'][i]==pytest.approx(old['target'][i]-old['q'][i])
    assert payload['robot']['left']['q'][0]==0.
