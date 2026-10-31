import importlib.util
import os
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def solve():
    root=Path(os.environ.get('DH116_AGENT_B_DIR',Path(__file__).parents[1]/'src/go2_pvcnn/control/dh116'))
    path=root/'tabletop_gravity_feasibility.py'
    assert path.exists(), 'task B static solver is not implemented'
    spec=importlib.util.spec_from_file_location('gravity_task_b',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.gravity_feasibility


def grip():
    return np.array([[-.01,0,0],[.01,0,0]]),np.array([[1.,0,0],[-1.,0,0]]),np.zeros((2,3,2))


def test_missing_lift_direction_is_infeasible(solve):
    p,n,j=grip()
    r=solve(p,n,j,[0,0,0],.1,0.,[.2,.2])
    assert not r['static_feasible']
    assert r['dynamic_success'] is False


def test_friction_support_solves_all_six_axes_without_fixed_total(solve):
    p,n,j=grip()
    totals=[]
    for mass in (.1,2.):
        r=solve(p,n,j,[0,0,0],mass,.5,[.2,.2])
        assert r['static_feasible']
        np.testing.assert_allclose(r['forces_on_rod_n'].sum(0),[0,0,mass*9.81],atol=1e-8)
        assert np.linalg.norm(r['wrench_residual'])<1e-8
        totals.append(sum(r['normal_loads_n']))
    assert totals[1]>12. and totals[1]/totals[0]==pytest.approx(20.)


def test_torque_imbalance_cannot_be_hidden_by_force_balance(solve):
    p=np.array([[.1,0,0],[.1,.01,0]])
    n=np.array([[0,0,1],[0,0,1]])
    r=solve(p,n,np.zeros((2,3,1)),[0,0,0],.1,0.,[.2])
    assert not r['static_feasible']


def test_original_motor_cap_and_gravity_torque_are_enforced(solve):
    p,n,j=grip();j[:,2,0]=1.
    assert not solve(p,n,j,[0,0,0],.1,.5,[.2,.2])['static_feasible']
    r=solve(p,n,j,[0,0,0],.1,.5,[2.,.2],gravity_torques=[.1,0.])
    assert r['static_feasible']
    # Jacobians act on HAND, contact force is the reaction to the rod force.
    np.testing.assert_allclose(r['motor_torques_nm'],[1.081,0.],atol=1e-8)
    assert not solve(p,n,j,[0,0,0],.1,.5,[1.,.2],gravity_torques=[.1,0.])['static_feasible']


def test_same_face_digit_equalization_sums_multiple_contacts(solve):
    p=np.zeros((4,3));n=np.array([[1,0,0],[-1,0,0],[-1,0,0],[-1,0,0]])
    r=solve(p,n,np.zeros((4,3,1)),[0,0,0],.2,.5,[.2],
            hand_ids=[0]*4,digit_ids=[4,0,0,2],face_ids=[0,1,1,1])
    assert r['static_feasible']
    f=r['normal_loads_n']
    assert f[0]==pytest.approx(sum(f[1:]))
    assert f[1]+f[2]==pytest.approx(f[3])
    assert f[0]==pytest.approx(2*f[3])


def test_single_face_group_fails_and_separate_hands_do_not_share_budget(solve):
    p,n,j=grip()
    assert not solve(p,n,j,[0,0,0],.1,.5,[.2,.2],hand_ids=[0,1],digit_ids=[0,1],face_ids=[0,1])['static_feasible']


def test_empty_and_invalid_inputs_fail_conservatively(solve):
    r=solve(np.empty((0,3)),np.empty((0,3)),np.empty((0,3,1)),[0,0,0],.1,.5,[.2])
    assert not r['static_feasible']
    p,n,j=grip();n[0]=np.nan
    r=solve(p,n,j,[0,0,0],.1,.5,[.2,.2])
    assert not r['static_feasible'] and r['status']=='invalid_input'
    assert np.isfinite(r['forces_on_rod_n']).all()


def test_zero_mass_requires_no_artificial_normal_budget(solve):
    p,n,j=grip()
    r=solve(p,n,j,[0,0,0],0.,.5,[.2,.2])
    assert r['static_feasible'] and not np.any(r['normal_loads_n'])


def test_nonnumeric_input_is_reported_instead_of_throwing(solve):
    r=solve('bad',[],[],[0,0,0],.1,.5,[.2])
    assert not r['static_feasible'] and r['status']=='invalid_input'


def test_failure_residual_does_not_report_a_zero_balanced_wrench(solve):
    p,n,j=grip()
    r=solve(p,n,j,[0,0,0],.1,0.,[.2,.2])
    assert not r['static_feasible']
    np.testing.assert_allclose(r['wrench_residual'],[0,0,-.981,0,0,0])


def test_friction_cone_and_cap_residuals_match_returned_contact_forces(solve):
    p,n,j=grip();j[:,2,0]=.05
    r=solve(p,n,j,[0,0,0],.1,.37,[.2,.2])
    assert r['static_feasible']
    f=r['forces_on_rod_n'];load=(f*n).sum(-1);tangent=f-load[:,None]*n
    assert (np.linalg.norm(tangent,axis=-1)<=.37*load+1e-9).all()
    np.testing.assert_allclose(r['motor_cap_residual_nm'],np.array([.2,.2])-abs(r['motor_torques_nm']))
