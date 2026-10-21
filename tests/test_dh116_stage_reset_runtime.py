import importlib
import numpy as np
import pytest
from pathlib import Path
from test_dh116_stage_state import state


def api():return importlib.import_module('go2_pvcnn.control.dh116.stage_runtime')


def measurement():
    return dict(settled=True,approach_position_errors_m=[0.]*6,
        approach_normal_errors_rad=[0.]*6,station_heights_m=[.308]*2,
        loads_n=[1.]*6,valid_contacts=[True]*6,
        normals=[[1.,0,0],[-1.,0,0],[0.,0,1],[-1.,0,0],[1.,0,0],[0.,0,1]],
        anchors_local=np.zeros((6,3)),min_surface_clearance_m=0.,
        safety_ok=True,active_constraints=())


@pytest.mark.parametrize('entry,until',[('approach',.1),('close',.1),('hold',1.),('lift',1.)])
def test_restore_revalidates_before_authorizing_stage(entry,until):
    s=state();s['entry_stage']=entry
    s['source']['passed_stages']=list(('settle','approach','close','hold')[:('approach','close','hold','lift').index(entry)+1])
    seq=api().StageResetSequence(s,s['pins'])
    for time in np.arange(0,until,.01):
        r=seq.update(float(time),**measurement())
        assert r['stage']=='restore_validate'
        assert not r['permit_lift']
        assert not seq.transport_velocity(float(time)).any()
    r=seq.update(until,**measurement())
    assert r['stage']==entry
    assert seq.stage_summary(final_passed=False)['settle']=='IMPORTED'
    assert not r['continuous_grasp_qualified']


def test_restore_abort_is_latched_and_no_motion_clock_advances():
    s=state();seq=api().StageResetSequence(s,s['pins']);m=measurement()
    m['safety_ok']=False
    assert seq.update(0.,**m)['stage']=='abort'
    m['safety_ok']=True
    assert seq.update(.01,**m)['stage']=='abort'
    assert not seq.transport_velocity(.01).any()


def test_symmetric_lift_restore_rebuilds_transport_after_contact_validation():
    s=state();s['entry_stage']='lift'
    s['source']['passed_stages']=['settle','approach','close','hold']
    seq=api().StageResetSequence(s,s['pins'],symmetric_lift=True)
    for i in range(101):
        seq.update(i*.01,**measurement())
        np.testing.assert_array_equal(seq.transport_velocity(i*.01),np.zeros((2,3)))
    seq.update(1.01,**measurement())
    np.testing.assert_array_equal(seq.transport_velocity(1.01),np.zeros((2,3)))
    seq.update(1.02,**measurement())
    assert np.all(seq.transport_velocity(1.02)[:,2]>0.)
    assert not seq._result()['continuous_grasp_qualified']


def test_approach_uses_new_local_clock_not_original_run_time():
    s=state();s['entry_stage']='approach';s['source']['passed_stages']=['settle']
    seq=api().StageResetSequence(s,s['pins'])
    for t in np.arange(0,.11,.01):seq.update(float(t),**measurement())
    assert seq.motion_time(.1)==pytest.approx(.5)
    assert seq.motion_time(.2)==pytest.approx(.6)
    assert seq.approach_deadline==pytest.approx(3.6)


def test_robot_mapping_checks_live_limits_and_fixed_root():
    s=state()['robot']['left'];names=s['joint_names'][::-1]
    data=api().robot_restore_arrays(s,names,np.array([s['lower'],s['upper']]).T[::-1],s['root_state'])
    np.testing.assert_array_equal(data['q'],np.array(s['q'])[::-1])
    with pytest.raises(ValueError):api().robot_restore_arrays(s,names,np.tile([0.,.05],(18,1)),s['root_state'])
    root=s['root_state'].copy();root[0]=1.
    with pytest.raises(ValueError):api().robot_restore_arrays(s,names,np.tile([0.,1.],(18,1)),root)


def test_runner_wires_restore_and_checkpoint_without_continuous_claim():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    for token in ('--restore-state','--restore-state-sha256','--save-stage-states',
                  'load_state(', 'engine.restore_physical_state(', 'robot_restore_arrays(',
                  'StageResetSequence(', 'save_state(', "result['stage_reset']",
                  'and restore_payload is None'):
        assert token in source
    assert source.index('robot_restore_arrays(')<source.index('sim.physics_sim_view.update_articulations_kinematic()')


def test_restore_rejects_invalid_station_measurement():
    s=state();seq=api().StageResetSequence(s,s['pins']);m=measurement()
    m['station_heights_m']=[float('nan'),.308]
    with pytest.raises(ValueError):seq.update(0.,**m)


@pytest.mark.parametrize('entry',['approach','close'])
def test_touch_first_restore_rebuilds_measured_readiness_and_touch(entry):
    s=state();s['entry_stage']=entry
    s['source']['passed_stages']=['settle'] if entry=='approach' else ['settle','approach']
    seq=api().StageResetSequence(s,s['pins'],state_triggered=True)
    m=measurement();m['joint_speeds_rad_s']=np.zeros((2,18))
    m['loads_n']=[0.]*6;m['valid_contacts']=[False]*6
    for i in range(11):r=seq.update(i*.01,**m)
    assert r['stage']==entry
    assert seq.touch_output is None
    for i in range(11,17):r=seq.update(i*.01,**m)
    assert r['stage']=='close'
    assert r['touch_first']['phases']==['touch','touch']
    assert not r['continuous_grasp_qualified']


def test_touch_restore_requires_actual_speed_even_before_release():
    s=state();seq=api().StageResetSequence(s,s['pins'],state_triggered=True)
    with pytest.raises(ValueError):seq.update(0.,**measurement())


def test_runner_selects_state_triggered_measured_handoff_and_pad_speeds():
    source=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    for token in ('state_triggered=True','joint_speeds_rad_s=', 'bounded_handoff(',
                  'normal_speeds_m_s=', 'handoff_seed', "'touch-first-v1' if args.grasp_sequence else 'diagnostic'"):
        assert token in source
