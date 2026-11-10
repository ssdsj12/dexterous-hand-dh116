"""Synthetic CPU batch FK; never Native geometry or control qualification."""
import importlib
from pathlib import Path

import numpy as np
import pytest

from test_bimanual_mpc_model_descriptor import fixture
from test_bimanual_mpc_mounted_kinematics import api as scalar_api, build, hypothesis


def api():
    path=Path(__file__).parents[1]/'src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_batch_kinematics.py'
    assert path.is_file(),'batch mounted FK kernel is missing'
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_mpc_batch_kinematics')


@pytest.mark.parametrize('side',['left','right'])
def test_64_independent_full17_rows_match_scalar_chain_and_keep_real_side_axes(side):
    m=api();_,_,_,tree=build(side)
    q=np.random.default_rng(83).uniform(-.03,.03,(64,17))
    q[:,12]=np.linspace(-.04,.06,64)  # Followers remain independent coordinates.
    result=m.MountedFKBatchKernel(tree).evaluate(q)
    expected=np.asarray([tree.fk(row).link_poses_world for row in q])
    np.testing.assert_allclose(result.link_poses_world,expected,rtol=0,atol=2e-14)
    assert result.link_poses_world.shape==(64,23,4,4)
    assert result.side==side and result.source_clock==tree.clock
    assert result.dof_names==tree.dof_names and result.link_paths==tree.link_paths
    assert result.kind=='geometry_forecast_hypothesis'
    assert not any(result.qualification.values())


def test_independent_analytic_rotation_translation_batch_golden():
    m=api();s=scalar_api();f=fixture()
    sample=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]
    tree=s.MountedTreeKinematics.from_descriptor(sample,side='left',coordinate_hypothesis=hypothesis(s,sample,'left'))
    q=np.zeros((3,17));q[:,0]=[np.pi/2,np.pi,-np.pi/2];q[:,3]=[.03,.06,-.01]
    result=m.MountedFKBatchKernel(tree).evaluate(q)
    point=np.einsum('bij,j->bi',result.link_poses_world[:,22],[0,1,0,1])[:,:3]
    np.testing.assert_allclose(point,[[.03,0,1],[.06,-1,0],[-.01,0,-1]],rtol=0,atol=1e-14)


def test_explicit_sign_zero_and_nontrivial_root_joint_frames_are_preserved():
    m=api();s,sample,h,_=build('right')
    h['joint_coordinates'][2].update(sign=-1,zero_offset=.023)
    tree=s.MountedTreeKinematics.from_descriptor(sample,side='right',coordinate_hypothesis=h)
    q=np.arange(17)[None,:]*.001+np.array([0.,.01])[:,None]
    result=m.MountedFKBatchKernel(tree).evaluate(q)
    np.testing.assert_allclose(result.link_poses_world,[tree.fk(row).link_poses_world for row in q],rtol=0,atol=2e-14)
    assert result.assumption_receipt['joint_coordinates'][2]['sign']==-1
    assert result.assumption_receipt['joint_coordinates'][2]['zero_offset']==.023


def test_kernel_owns_compiled_tree_and_result_is_owned_read_only_numeric_data():
    m=api();_,_,_,tree=build();kernel=m.MountedFKBatchKernel(tree)
    q=np.zeros((2,17));before=q.copy();result=kernel.evaluate(q);poses=result.link_poses_world.copy()
    np.testing.assert_array_equal(q,before)
    q[:]=.1;tree._root_transform[:3,3]+=10;tree._sign[:]=-1
    np.testing.assert_array_equal(result.input_q_sdk,before)
    np.testing.assert_array_equal(result.link_poses_world,poses)
    np.testing.assert_array_equal(kernel.evaluate(before).link_poses_world,poses)
    assert not result.input_q_sdk.flags.writeable and not result.link_poses_world.flags.writeable
    assert not np.shares_memory(result.input_q_sdk,q)
    with pytest.raises(ValueError):result.link_poses_world[0,0,0,0]=0
    with pytest.raises(TypeError):result.qualification['planning_eligible']=True


@pytest.mark.parametrize('q',[np.zeros(17),np.zeros((0,17)),np.zeros((2,16)),
    np.zeros((2,17),dtype=bool),np.full((2,17),np.nan),np.full((2,17),np.inf),
    np.ones((2,17),dtype=complex),np.full((2,17),'0'),np.full((2,17),None,dtype=object),
    [[True]+[0.]*16],[np.ones(17,dtype=bool),np.zeros(17)]])
def test_invalid_batch_rejected_without_poisoning_kernel(q):
    m=api();_,_,_,tree=build();kernel=m.MountedFKBatchKernel(tree)
    with pytest.raises(ValueError):kernel.evaluate(q)
    assert kernel.evaluate(np.zeros((1,17))).link_poses_world.shape==(1,23,4,4)


def test_exact_mounted_model_required_without_backend_or_fake_tree_fallback():
    with pytest.raises(ValueError):api().MountedFKBatchKernel(object())
