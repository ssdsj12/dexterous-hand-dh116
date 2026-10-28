import numpy as np
import pytest


def test_explicit_load_parameters_expand_force_control_and_checkpoint_contract():
    from go2_pvcnn.control.dh116.tabletop_load_parameters import LoadParameters
    from go2_pvcnn.control.dh116.tabletop_force_feedback import ReactionForceController, force_feedback_contract
    parameters = LoadParameters(max_force_n=4., force_rate_n_s=2., stiffness_max=50., effort_limit_nm=.5)
    control = ReactionForceController(parameters)
    for _ in range(240):
        control.update(action=[1., 1.], forces=[0.,0.,0.,0.], blocked=[False, False], slip_m=0., phase='close', dt=1/60)
    np.testing.assert_allclose(control.requested, [4.,4.])
    np.testing.assert_allclose(control.target, [4.,4.])
    assert np.all(control.offset <= .08)
    contract = force_feedback_contract(parameters)
    assert contract['active_effort_limit_nm'] == .5
    assert contract['active_stiffness_range'] == [10.,50.]
    assert contract['force_feedback']['target_hand_load_range_n'] == [.1,4.]
    assert contract['load_parameters']['calibration_status'] == 'simulation_estimate'
    assert force_feedback_contract() != contract
    assert np.all(np.isfinite(control.observation()))


@pytest.mark.parametrize('kwargs', [dict(max_force_n=np.nan), dict(max_force_n=0.),
    dict(max_force_n=33.),dict(effort_limit_nm=np.inf),dict(effort_limit_nm=-1.),
    dict(stiffness_max=9.),dict(force_rate_n_s=0.),dict(calibration_status='hardware_calibrated')])
def test_unidentified_or_invalid_parameters_are_rejected(kwargs):
    from go2_pvcnn.control.dh116.tabletop_load_parameters import LoadParameters
    with pytest.raises(ValueError):
        LoadParameters(**kwargs)


def test_rigid_coupling_requires_explicit_profile_at_480hz():
    from go2_pvcnn.control.dh116.tabletop_vec_config import validate_mimic_options
    with pytest.raises(ValueError):
        validate_mimic_options(physics_hz=480,mimic_frequency=0)
    validate_mimic_options(physics_hz=480,mimic_frequency=0,allow_rigid=True)
    with pytest.raises(ValueError):
        validate_mimic_options(physics_hz=240,mimic_frequency=0,allow_rigid=True)


def test_physics_readback_rejects_unapplied_effort_and_gain_profile():
    from go2_pvcnn.control.dh116.tabletop_load_parameters import LoadParameters, verify_drive_readback
    parameters = LoadParameters(stiffness_max=50., effort_limit_nm=.5)
    verify_drive_readback(np.full((3,6),50.),np.full((3,6),.2),np.full((3,6),.5),parameters)
    with pytest.raises(ValueError, match='effort'):
        verify_drive_readback(np.full((3,6),50.),np.full((3,6),.2),np.full((3,6),.2),parameters)
    with pytest.raises(ValueError, match='stiffness'):
        verify_drive_readback(np.full((3,6),51.),np.full((3,6),.2),np.full((3,6),.5),parameters)
    # The older joint-target scene intentionally has a distinct 0.15 Nm limit.
    verify_drive_readback(np.full((1,6),10.),np.full((1,6),.2),np.full((1,6),.15),LoadParameters(effort_limit_nm=.15))


def test_twelve_newton_profile_reaches_target_within_acquisition_timeout():
    from pathlib import Path
    from go2_pvcnn.control.dh116.tabletop_load_parameters import read_load_parameters
    from go2_pvcnn.control.dh116.tabletop_force_feedback import ReactionForceController
    parameters=read_load_parameters(Path(__file__).resolve().parents[1]/'configs/dh116_load_profiles/v2_pressure_12n_candidate.json')
    control=ReactionForceController(parameters)
    for step in range(360):
        control.update(action=[1.,1.],forces=[0.,0.,0.,0.],blocked=[False,False],slip_m=0.,phase='close',dt=1/60)
        assert np.all(control.target<=12.)
        if step==59:np.testing.assert_allclose(control.target,[2.1,2.1],atol=1e-10)
    np.testing.assert_allclose(control.target,[12.,12.],atol=1e-10)
    assert parameters.effort_limit_nm==.2
