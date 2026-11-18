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


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_both_chunk_boundaries_retain_exact_ragged_cad_minimum(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from go2_pvcnn.control.dh116.tabletop_native_height_batch import HeightBatch
    g=torch.Generator().manual_seed(472)
    poses=torch.randn(7,2,4,7,generator=g,dtype=torch.float64).to(device)
    poses[...,3:]/=poses[...,3:].norm(dim=-1,keepdim=True)
    shapes=[(h,b,torch.randn(k,3,generator=g,dtype=torch.float64).to(device))
            for h,b,k in ((0,0,1),(0,2,11),(1,1,37),(1,3,8))]
    expected=torch.stack([rotated_heights(poses[:,h,b],points).amin(-1)
                          for h,b,points in shapes],1)
    reader=HeightBatch(shapes,env_chunk_size=3,vertex_chunk_size=7)
    actual=reader(poses)
    assert actual.dtype==torch.float64
    torch.testing.assert_close(actual,expected,rtol=0,atol=1e-12)


def test_4096_envs_never_allocate_full_env_by_shape_by_vertex_tensor():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves
    from go2_pvcnn.control.dh116.tabletop_native_height_batch import HeightBatch
    allocations=[]
    class AllocationShapes(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            result=func(*args,**(kwargs or {}))
            # These operations allocate data; ignore expand/view aliases.
            if str(func).startswith(('aten.mm.','aten.bmm.','aten.add.',
                                     'aten.masked_fill.','aten.clone.')):
                allocations.extend((str(func),tuple(t.shape),t.numel())
                                   for t in tree_leaves(result) if isinstance(t,torch.Tensor))
            return result
    with FakeTensorMode(),AllocationShapes():
        shapes=[(i//17,i%17,torch.empty(2771-(i%3),3,dtype=torch.float64))
                for i in range(34)]
        reader=HeightBatch(shapes)
        actual=reader(torch.empty(4096,2,17,7,dtype=torch.float64))
    assert actual.shape==(4096,34)
    assert allocations
    # Defaults bound envs to 128 and vertices to 512. Matmul may clone the
    # broadcast three-coordinate point inputs, hence three times that tile.
    limit=128*34*512*3
    excessive=[item for item in allocations if item[2]>limit]
    assert not excessive,excessive[:3]


@pytest.mark.parametrize('name',('env_chunk_size','vertex_chunk_size'))
@pytest.mark.parametrize('value',(0,-1,1.5,True))
def test_chunk_sizes_must_be_explicit_positive_integers(name,value):
    from go2_pvcnn.control.dh116.tabletop_native_height_batch import HeightBatch
    with pytest.raises(ValueError,match='chunk'):
        HeightBatch([(0,0,torch.zeros(1,3))],**{name:value})
