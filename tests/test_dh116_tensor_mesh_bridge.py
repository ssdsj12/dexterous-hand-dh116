"""Device-only mesh target layout and explicit ownership tests."""
import pytest
import torch
from go2_pvcnn.control.dh116.tabletop_tensor_mesh_bridge import TensorMeshBridge

def fixture():
    n=2;names=("left.pad","table","right.pad")
    b={"left.pad":{"side":"left","body_index":4},"right.pad":{"side":"right","body_index":8}}
    pos=torch.randn(n,3,3);quat=torch.zeros(n,3,4);quat[...,3]=1
    return TensorMeshBridge(names,b,pos,quat),pos,quat

def test_all_pose_updates_keep_static_slots_and_actual_per_lane_link_mapping():
    bridge,pos,quat=fixture()
    links=torch.randn(2,2,23,7);links[...,3:]=torch.tensor([0.,0.,0.,1.])
    actual_p,actual_q=bridge.targets(links)
    torch.testing.assert_close(actual_p[:,0],links[:,0,4,:3])
    torch.testing.assert_close(actual_p[:,2],links[:,1,8,:3])
    torch.testing.assert_close(actual_p[:,1],pos[:,1])
    assert actual_p.is_contiguous() and actual_q.is_contiguous()
    assert torch.equal(links[...,3:],torch.tensor([0.,0.,0.,1.]).expand_as(links[...,3:]))

@pytest.mark.parametrize("fault",["nan","bad_quat","wrong_count"])
def test_mesh_source_rejects_invalid_complete_gpu_input_without_advancing(fault):
    bridge,_,_=fixture();links=torch.zeros(2,2,23,7);links[...,6]=1
    if fault=="nan":links[1,0,2,0]=float("nan")
    if fault=="bad_quat":links[1,0,2,6]=.8
    if fault=="wrong_count":links=links[:1]
    with pytest.raises(ValueError):bridge.targets(links)

def test_material_is_pinned_until_explicit_selected_reset_reverification():
    bridge,_,_=fixture();param=dict(radius=.005,mass=1.,EA=800.,EI=.018)
    bridge.bind_material(param,reset_ids=(1,1))
    assert bridge.material(reset_ids=(1,1))==param
    with pytest.raises(ValueError):bridge.material(reset_ids=(2,1))
    bridge.bind_material(param,reset_ids=(2,1))
    assert bridge.material(reset_ids=(2,1))==param
    with pytest.raises(ValueError):bridge.bind_material(dict(param,radius=.006),reset_ids=(3,1))
