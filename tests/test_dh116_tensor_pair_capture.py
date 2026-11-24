import numpy as np
import pytest
import torch
from go2_pvcnn.control.dh116.tabletop_tensor_pair_capture import tensor_pair_capture
from go2_pvcnn.control.dh116.tabletop_batched_pair_capture import compact_pair_capture

def fixture():
    n=2;v=41;m=3;s=12;c=4
    packed=torch.zeros(n,c,24)
    counts=torch.tensor([2,1],dtype=torch.int32)
    ids=[torch.tensor([[0,2,1],[4,5,0]],dtype=torch.int32),torch.tensor([[7,6,2]],dtype=torch.int32)]
    vector=[torch.zeros(2,12),torch.zeros(1,12)]
    vector[0][:,3]=1;vector[0][:,6]=torch.tensor([2.,3.])
    vector[1][:,4]=1;vector[1][:,7]=4.
    nodes=torch.zeros(n,v,3);wrench=torch.zeros(n,m,6);lanes=[]
    for i in range(n):
        k=len(ids[i]);packed[i,:k,:3]=ids[i].view(torch.float32)
        packed[i,:k,3:15]=vector[i]
        for j in range(k):
            nodes[i,ids[i][j,1]]+=vector[i][j,6:9]/s
            wrench[i,ids[i][j,2],:3]-=vector[i][j,6:9]/s
        lanes.append(dict(schema="dh116.sdf_pair_samples_lane.v1",env_id=i,substeps=s,
            mesh_count=m,capacity=c,attempted=k,overflow=False,last_error=0,
            owner_time_s_before=.1,owner_time_s_after=.1,indices=ids[i].numpy(),vectors=vector[i].numpy()))
    raw=dict(schema="dh116.sdf_pair_device.v1",num_envs=n,mesh_count=m,substeps=s,
        last_error=0,owner_time_s_before=.1,owner_time_s_after=.1,capacity=c,
        counts=counts,samples=packed,node_forces=nodes,mesh_wrenches=wrench)
    old=dict(schema="dh116.sdf_pair_samples_batch.v1",lanes=lanes,
             node_forces_on_rod_world_n=nodes.numpy(),mesh_wrenches_on_mesh_world=wrench.numpy())
    return raw,old

def test_tensor_capture_matches_original_all_pair_reductions():
    raw,old=fixture();a=tensor_pair_capture(raw,n=2,vertices=41,meshes=3,substeps=12,device="cpu")
    b=compact_pair_capture(old,dt_s=1/480,device="cpu")
    for key in b:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    raw["samples"].zero_()
    assert a["vectors"].abs().sum()>0

@pytest.mark.parametrize("fault",["overflow","duplicate","nonfinite","index","total"])
def test_tensor_pair_refuses_same_invalid_evidence(fault):
    raw,_=fixture()
    if fault=="overflow":raw["counts"][1]=5
    elif fault=="duplicate":raw["samples"][0,1,:3]=raw["samples"][0,0,:3]
    elif fault=="nonfinite":raw["samples"][0,0,4]=float("nan")
    elif fault=="index":raw["samples"][0,0,0]=torch.tensor(12,dtype=torch.int32).view(torch.float32)
    else:raw["node_forces"][0,0,0]=.01
    with pytest.raises(ValueError):tensor_pair_capture(raw,n=2,vertices=41,meshes=3,substeps=12,device="cpu")

def test_flat_owned_tensor_abi_equals_original_including_empty_lane():
    from go2_pvcnn.control.dh116.tabletop_tensor_pair_capture import flat_cuda_pair_capture
    raw,old=fixture();rows=raw["samples"];mask=torch.arange(4)[None,:]<raw["counts"][:,None]
    data=rows[mask];owners=torch.arange(2,dtype=torch.int32)[:,None].expand_as(mask)[mask][:,None]
    flat=dict(schema="dh116.sdf_pair_flat_cuda.v1",substeps=12,num_envs=2,mesh_count=3,
        owner_time_s_before=.1,owner_time_s_after=.1,
        indices_gpu=torch.cat((owners,data[:,:3].contiguous().view(torch.int32)),1),
        vectors_gpu=data[:,3:15].clone(),
        lane_meta_gpu=torch.stack((torch.zeros(2,dtype=torch.int32),torch.zeros(2,dtype=torch.int32),
                                  torch.full((2,),4,dtype=torch.int32),raw["counts"]),1),
        node_forces_gpu=raw["node_forces"],mesh_wrenches_gpu=raw["mesh_wrenches"])
    actual=flat_cuda_pair_capture(flat,n=2,vertices=41,meshes=3,substeps=12,device="cpu",expected_owner_time_s=.1)
    expected=compact_pair_capture(old,dt_s=1/480,device="cpu")
    for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
