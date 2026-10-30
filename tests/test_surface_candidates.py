"""Bounded DH116 native search coordinates and diverse candidate selection."""
import importlib
import importlib.util
import pytest
import torch
from go2_pvcnn.control.dh116.linear_coupling import active_limits


def module():
    name='go2_pvcnn.control.dh116.tabletop_surface_candidates'
    assert importlib.util.find_spec(name) is not None, 'surface candidate search missing'
    return importlib.import_module(name)


def roots():return torch.tensor([[.5,.2,.42,1.,0.,0.,0.],[.5,-.2,.42,1.,0.,0.,0.]],dtype=torch.float64)


def test_decode_uses_twelve_active_and_external_wrist_bounds():
    x=torch.randn(4,2,12,dtype=torch.float64)*20
    q,pose,offset,angles=module().decode_variables(x,roots())
    lo,hi=map(torch.tensor,active_limits())
    assert q.shape==(4,2,6) and pose.shape==(4,2,7)
    assert (q>=lo).all() and (q<=hi).all()
    assert (offset>=offset.new_tensor([-.03,-.03,-.004])).all()
    assert (offset<=offset.new_tensor([.03,.03,.05])).all() and (angles.abs()<=torch.pi/2).all()
    torch.testing.assert_close(pose[:,:,3:].norm(dim=-1),torch.ones(4,2,dtype=torch.float64))


def test_encode_decode_preserves_reset_seed_without_fixed_pose():
    lo,hi=map(torch.tensor,active_limits());q=((lo+hi)/2).expand(2,-1)
    x=module().encode_variables(q,torch.zeros(2,3),torch.zeros(2,3))
    actual,pose,offset,angles=module().decode_variables(x[None],roots())
    torch.testing.assert_close(actual[0],q)
    torch.testing.assert_close(pose[0],roots())


def test_decode_has_finite_rotation_gradient_at_zero():
    x=torch.zeros(1,2,12,dtype=torch.float64,requires_grad=True)
    q,pose,offset,angles=module().decode_variables(x,roots())
    (q.sum()+pose.sum()).backward()
    assert torch.isfinite(x.grad).all() and (x.grad[:,:,9:].abs()>0).all()


def test_diverse_ranking_rejects_duplicates_and_retains_distinct_grasps():
    rows=[dict(score=.1,active_angles=[.2]*12,wrist_pose_w=roots().tolist()),
          dict(score=.2,active_angles=[.20001]*12,wrist_pose_w=roots().tolist()),
          dict(score=.3,active_angles=[.4]*12,wrist_pose_w=roots().tolist())]
    selected=module().rank_diverse(rows,maximum=4)
    assert [r['score'] for r in selected]==[.1,.3]
    assert all(r.get('qualified',False) is False for r in selected)


def test_nonfinite_variables_and_bad_counts_rejected():
    m=module();x=torch.zeros(1,2,12,dtype=torch.float64);x[0,0,0]=float('nan')
    with pytest.raises(ValueError):m.decode_variables(x,roots())
    with pytest.raises(ValueError):m.decode_variables(torch.zeros(1,2,11),roots())
