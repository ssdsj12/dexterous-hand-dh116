import numpy as np
import pytest
from test_dh116_grasp_sequence import observation
from test_dh116_coordinated_closure import inputs
from go2_pvcnn.control.dh116.speed_profile import GraspSpeedProfile
from go2_pvcnn.control.dh116.grasp_sequence import TabletopGraspSequence
from go2_pvcnn.control.dh116.symmetric_transport import SynchronizedTransport
from go2_pvcnn.control.dh116.coordinated_closure import coordinated_bimanual_increment


def test_sequence_passes_measured_gaps_to_staged_touch_without_authorizing_lift():
    seq = TabletopGraspSequence(material_ids=(36,36,4,4), target_loads_n=[1.]*4,
        state_triggered=True, speed_profile=GraspSpeedProfile(), symmetric_lift=True)
    for i in range(1, 150):
        result = seq.update(i*.01, **observation(loads_n=np.zeros(4), valid_contacts=[False]*4),
            joint_speeds_rad_s=np.zeros((2,18)), surface_gaps_m=[.02,.02,.001,.001])
    assert result['stage'] == 'close' and not result['permit_lift']
    np.testing.assert_allclose(result['touch_first']['normal_speeds_m_s'], [[.02,0],[.001,0]])


def test_staged_sequence_requires_gaps_before_advancing_time():
    seq = TabletopGraspSequence(material_ids=(36,36,4,4), target_loads_n=[1.]*4,
        state_triggered=True, speed_profile=GraspSpeedProfile())
    with pytest.raises(ValueError, match='gap'):
        seq.update(.01, **observation(), joint_speeds_rad_s=np.zeros((2,18)))
    assert seq.last_time is None


def test_configured_lift_speed_preserves_bilateral_height_abort():
    control = SynchronizedTransport(max_speed_m_s=.01)
    heights = np.zeros(2)
    for i in range(40):
        result = control.update(i*.01,rises_m=heights,contacts_valid=np.ones(2,dtype=bool),direction=1)
        heights += result['velocity_w'][:,2]*.01
    np.testing.assert_allclose(result['velocity_w'][:,2], .01)
    with pytest.raises(ValueError, match='2 mm'):
        control.update(.4,rises_m=[.003,0],contacts_valid=np.ones(2,dtype=bool),direction=1)


def test_allocator_explicit_speed_limit_reports_scale_and_preserves_joint_caps():
    J,n,W=inputs(); J=np.stack([J,J]); n=np.stack([n,n]); W=np.stack([W,W])
    delta,report=coordinated_bimanual_increment(J,n,W,np.zeros((2,2)),.0001,
        target_loads_n=np.ones((2,2)), transport_velocity_w=np.zeros(3),
        normal_speeds_m_s=np.array([[.02,0],[.02,0]]),normal_speed_limit_m_s=.02,
        return_diagnostics=True)
    assert report['joint_rate_scale']>1
    assert np.max(np.abs(delta[:,:7]/.0001))<=.05+1e-12
    assert np.max(np.abs(delta[:,7:]/.0001))<=.02+1e-12
    np.testing.assert_allclose(report['allocated_pad_velocity_w_m_s'],np.einsum('shij,sj->shi',J,delta/.0001))
    assert np.max(np.abs(report['allocated_pad_velocity_w_m_s'])) < .02
    with pytest.raises(ValueError):
        coordinated_bimanual_increment(J,n,W,np.zeros((2,2)),.0001,
            target_loads_n=np.ones((2,2)), transport_velocity_w=np.zeros(3),
            normal_speeds_m_s=np.full((2,2),.02))


def test_loaded_force_speed_cap_is_independent_of_free_speed_limit():
    J,n,W=inputs(); J=np.stack([J,J]); n=np.stack([n,n]); W=np.stack([W,W])
    _,report=coordinated_bimanual_increment(J,n,W,np.zeros((2,2)),.0001,
        target_loads_n=np.ones((2,2)),transport_velocity_w=np.zeros(3),
        normal_speed_limit_m_s=.02,force_speed_limit_m_s=.001,return_diagnostics=True)
    assert np.max(np.abs(report['requested_pad_velocity_w_m_s'])) <= .001+1e-12


def test_restore_validates_gaps_before_releasing_staged_control():
    from test_dh116_stage_state import state
    from test_dh116_stage_reset_runtime import measurement
    from go2_pvcnn.control.dh116.stage_runtime import StageResetSequence
    saved=state()
    seq=StageResetSequence(saved,saved['pins'],state_triggered=True,speed_profile=GraspSpeedProfile())
    with pytest.raises(ValueError,match='gap'):
        seq.update(0.,**measurement(),joint_speeds_rad_s=np.zeros((2,18)))
    assert seq.last_time is None
    result=seq.update(0.,**measurement(),joint_speeds_rad_s=np.zeros((2,18)),surface_gaps_m=[.002]*4)
    assert result['stage']=='restore_validate' and not result['permit_lift']


def test_runner_staged_configuration_is_explicit_pinned_and_uses_measured_gaps():
    from pathlib import Path
    import ast
    code=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert "parser.add_argument('--grasp-speed-profile'" in code
    tree=ast.parse(code)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    for name in ('TabletopGraspSequence','StageResetSequence'):
        call=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id==name)
        assert ast.unparse(next(k.value for k in call.keywords if k.arg=='speed_profile'))=='speed_profile'
    updates=[n for n in calls if isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=='grasp_sequence.update']
    gap=next(k.value for k in updates[0].keywords if k.arg=='surface_gaps_m')
    assert ast.unparse(gap)=="[samples[c]['gap_m'] for c in BODY_NAMES]"
    config=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
                and any(ast.unparse(t)=="result['run_config']" for t in n.targets))
    assert any(k.arg=='speed_profile' for k in config.value.keywords)
    assert "record['control_allocation']" in code
