import copy
import importlib
import importlib.util

import pytest
import torch


def module():
    name = 'go2_pvcnn.control.dh116.tabletop_surface_contract'
    assert importlib.util.find_spec(name), 'canonical surface contract is missing'
    return importlib.import_module(name)


def legacy():
    return dict(observation_size=351, observation_version='dh116-surface-sharing-lift-objective-v2',
                policy_command_size=24, action='24D_free_fingers_worldXYZ_intrinsicXYZ_surface_sharing_v1',
                physics_hz=480, control_hz=60)


def test_contract_binds_all_twenty_features_without_mutating_legacy():
    m=module(); old=legacy(); saved=copy.deepcopy(old); new=m.bind_contract(old)
    assert old==saved and new['observation_size']==371
    fields=new['surface_contract']['feedback_fields']
    assert fields[0]['slice'][0]==351 and fields[-1]['slice'][1]==371
    assert sum(f['slice'][1]-f['slice'][0] for f in fields)==20
    m.validate_contract(new)


@pytest.mark.parametrize('mutation', ['scale','sha','order','version','dimension'])
def test_tampered_contract_is_rejected(mutation):
    m=module(); c=m.bind_contract(legacy())
    if mutation=='scale':c['surface_contract']['action']['translation_rate_m_s']=.02
    if mutation=='sha':c['surface_contract_sha256']='0'*64
    if mutation=='order':c['surface_contract']['action']['joint_order'].reverse()
    if mutation=='version':c['observation_version']='unknown'
    if mutation=='dimension':c['policy_command_size']=26
    with pytest.raises(ValueError):m.validate_contract(c)


def test_sha_covers_semantics_and_is_independent_of_batch_size_only():
    m=module(); c=m.bind_contract(legacy()); other=copy.deepcopy(c); other['num_envs']=2
    assert m.semantic_sha256(c)==m.semantic_sha256(other)
    other['physics_hz']=240
    assert m.semantic_sha256(c)!=m.semantic_sha256(other)


@pytest.mark.parametrize('size',[20,26,401])
def test_legacy_26d_or_unknown_source_never_silently_migrates(size):
    m=module(); c=legacy(); c['policy_command_size']=size
    with pytest.raises(ValueError):m.bind_contract(c)


def test_action_axis_units_use_existing_physical_limits():
    m=module(); from test_surface_exploration import surface_task
    from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import control
    t,w=surface_task(); q=(t.low+t.high)/2; t.command[:]=q; t.m['q'][:]=q
    x=torch.zeros(2,24,dtype=torch.float64); x[:,0]=.5;x[:,14]=.4;x[:,23]=-.2
    control(t,x,torch.ones(2,dtype=torch.bool),1/480)
    torch.testing.assert_close(t.command[:,0]-q[0],torch.full((2,),.5*.3/480,dtype=torch.float64))
    torch.testing.assert_close(t.palm_targets[:,0,2],torch.full((2,),.4*.01/480,dtype=torch.float64))
    torch.testing.assert_close(t.rotation_targets[:,1,2],torch.full((2,),-.2*.5/480,dtype=torch.float64))
    assert m.ACTION_CONTRACT['joint_rate_rad_s']==.3
