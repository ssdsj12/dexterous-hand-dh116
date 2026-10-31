"""Clearance batching retains each original CAD vertex and reduction."""
import importlib
import pytest
import torch
from go2_pvcnn.control.dh116.tabletop_tensor_world import rotated_heights

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_ragged_source_hulls_match_original_each_shape_and_each_hand(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    module=importlib.import_module('go2_pvcnn.control.dh116.tabletop_native_height_batch')
    g=torch.Generator().manual_seed(128)
    poses=torch.randn(4,2,6,7,generator=g,dtype=torch.float64).to(device)
    poses[...,3:]/=poses[...,3:].norm(dim=-1,keepdim=True)
    shapes=[(h,b,torch.randn(k,3,generator=g,dtype=torch.float64).to(device))
        for h,b,k in ((0,1,1),(0,3,8),(0,2,31),(1,1,4),(1,5,20))]
    reader=module.HeightBatch(shapes)
    actual=reader(poses)
    expected=torch.stack([rotated_heights(poses[:,h,b],points).amin(-1) for h,b,points in shapes],1)
    torch.testing.assert_close(actual,expected,rtol=0,atol=2e-15)
    for ids in ([0,1,2],[3,4]):
        torch.testing.assert_close(actual[:,ids].amin(-1),expected[:,ids].amin(-1),rtol=0,atol=2e-15)

def test_padding_never_becomes_a_lower_vertex():
    module=importlib.import_module('go2_pvcnn.control.dh116.tabletop_native_height_batch')
    poses=torch.zeros(1,2,2,7,dtype=torch.float64);poses[...,3]=1
    shapes=[(0,0,torch.tensor([[0.,0.,1.]],dtype=torch.float64)),
        (1,1,torch.tensor([[0.,0.,2.],[0.,0.,3.]],dtype=torch.float64))]
    assert module.HeightBatch(shapes)(poses).tolist()==[[1.,2.]]
