"""GPU/CPU mathematical parity, not native physical qualification."""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation


@pytest.mark.parametrize("device",["cpu","cuda:0"])
def test_batch_metrics_match_original_complete_vertices_and_reject_stale_pose(device):
    if device.startswith("cuda") and not torch.cuda.is_available():pytest.skip("CUDA required")
    from go2_pvcnn.control.dh116.tabletop_batched_geometry_metrics import BatchedGeometryMetrics
    from go2_pvcnn.control.dh116.tabletop_measured_closure_geometry import rod_support_lower
    from scripts.audit_dh116_pinch_initial_pose import pose_matrix,transform
    rng=np.random.default_rng(271)
    n=128;shapes=[(0,3,"source:finger13_link",rng.normal(size=(21,3))*.01),
                  (1,6,"cooked:finger23_link",rng.normal(size=(37,3))*.01)]
    poses=np.zeros((n,2,23,7));poses[...,6]=1
    poses[:,:,:,:3]=rng.normal(size=(n,2,23,3))*.1+[.5,0.,.5]
    poses[...,3:]=Rotation.random(n*2*23,random_state=rng).as_quat().reshape(n,2,23,4)
    poses[...,3:]*=1.+1e-7
    nodes=rng.normal(size=(n,41,3))*.02+[.5,0.,.4]
    metrics=BatchedGeometryMetrics(shapes,device=device)
    metrics.capture(poses,nodes,frame=("fixture",4))
    for lane in range(n):
        metrics.validate(lane,poses[lane],nodes[lane],frame=("fixture",4))
        for column,(side,link,key,vertices) in enumerate(shapes):
            p=poses[lane,side,link]
            world=transform(vertices,pose_matrix(p[[0,1,2,6,3,4,5]]))
            np.testing.assert_allclose(metrics.lower[lane,column],rod_support_lower(world,nodes[lane]),rtol=0,atol=1e-10)
            np.testing.assert_allclose(metrics.minimum[lane,column],world.min(0),rtol=0,atol=1e-10)
            np.testing.assert_allclose(metrics.maximum[lane,column],world.max(0),rtol=0,atol=1e-10)
    damaged=poses[0].copy();damaged[0,0,0]+=1e-6
    with pytest.raises(ValueError,match="pose"):
        metrics.validate(0,damaged,nodes[0],frame=("fixture",4))
    with pytest.raises(ValueError,match="frame"):
        metrics.validate(0,poses[0],nodes[0],frame=("fixture",5))
    bad=nodes.copy();bad[0,0,0]=np.nan
    with pytest.raises(ValueError):metrics.capture(poses,bad,frame=("fixture",5))
