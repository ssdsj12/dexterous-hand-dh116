import importlib

import pytest
import torch


@pytest.fixture
def module():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_vision_camera')


def inputs(n, *, floating=False):
    h,w=7,9
    rgb=torch.tensor([160,90,30],dtype=torch.uint8).expand(n,h,w,3).clone()
    if floating:
        rgb=rgb.float()/255.
        # The final block controls the original whole-batch float normalization.
        rgb[-1]*=255.
    depth=(.8+torch.arange(n)*.0001)[:,None,None,None].expand(n,h,w,1).clone()
    K=torch.eye(3).repeat(n,1,1);K[:,0,0]=100.;K[:,1,1]=100.
    K[:,0,2]=w/2.;K[:,1,2]=h/2.
    T=torch.eye(4).repeat(n,1,1);T[:,0,3]=torch.arange(n)*.01
    hands=torch.zeros(n,2,7);hands[...,3]=1.;hands[:,1,1]=.2
    return rgb,depth,K,T,hands


@pytest.mark.parametrize('n,floating',[(257,False),(137,True)])
def test_frames_chunked_at_64_matches_dense_reference_bit_for_bit(module,monkeypatch,n,floating):
    args=inputs(n,floating=floating)
    kw=dict(num_points=8,min_depth_m=.1,max_depth_m=2.)
    dense=module._frames_to_points_dense
    expected=dense(*args,**kw)
    calls=[]
    def spy(rgb,*rest,**kwargs):
        calls.append(len(rgb));return dense(rgb,*rest,**kwargs)
    monkeypatch.setattr(module,'_frames_to_points_dense',spy)
    actual=module.frames_to_points(*args,**kw)
    assert calls==([64]*(n//64)+([n%64] if n%64 else []))
    assert actual.keys()==expected.keys()
    assert 'rgb' not in actual and 'depth' not in actual
    for key in actual:
        assert torch.equal(actual[key],expected[key]),key
        assert actual[key].shape[0]==n
    if floating:
        assert not actual['object_valid'][:-1].any()
        assert actual['object_valid'][-1]


def test_full_calibration_shape_is_rejected_before_any_dense_work(module,monkeypatch):
    args=list(inputs(137));args[2]=args[2][:-1]
    def forbidden(*args,**kwargs):raise AssertionError('dense processing before full validation')
    monkeypatch.setattr(module,'_frames_to_points_dense',forbidden)
    with pytest.raises(ValueError,match='per-environment'):
        module.frames_to_points(*args,num_points=8,min_depth_m=.1,max_depth_m=2.)
