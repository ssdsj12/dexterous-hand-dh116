import torch
from test_manual_digit_force_balance import setup


def test_measured_compensation_uses_actual_pad_geometry_and_preserves_lift():
    from go2_pvcnn.control.dh116.tabletop_manual_pinch_geometry import measured_target
    from types import SimpleNamespace
    home=torch.tensor([[[.438,.2,.306],[.422,.2,.306]],
                       [[.438,-.2,.306],[.422,-.2,.306]]],dtype=torch.float64)
    origins=torch.tensor([[0.,0.,0.],[2.,3.,0.]],dtype=torch.float64)
    actual_wrist=torch.tensor([[[.001,0.,.004],[-.001,0.,.004]]],dtype=torch.float64).repeat(2,1,1)
    internal_shift=torch.tensor([[[.0007,0.,.0008],[-.0003,0.,.0011]]],dtype=torch.float64).repeat(2,1,1)
    pads=home[None]+origins[:,None,None]+actual_wrist[:,:,None]+internal_shift[:,:,None]
    wire=torch.tensor([[.43,0.,.308],[2.43,3.,.317]],dtype=torch.float64)
    w=SimpleNamespace(origins=origins,wp=wire,pad_relative=pads.reshape(2,4,3)-wire[:,None])
    t=SimpleNamespace(n=2,manual_pinch_home=home.mean(1),
                      m={'adaptive_palm_offset':actual_wrist})
    nominal=torch.zeros((2,2,3),dtype=torch.float64);nominal[:,:,2]=.01
    target=measured_target(t,w,nominal)
    torch.testing.assert_close(target,nominal-internal_shift,atol=1e-12,rtol=0)
    # An object displacement must not create an extra autonomous wrist lift.
    w.wp[:,2]+=.02;w.pad_relative[:,:,2]-=.02
    torch.testing.assert_close(measured_target(t,w,nominal),target,atol=1e-12,rtol=0)


def test_force_closure_keeps_pinch_midpoint_at_commanded_wrist_placement():
    from go2_pvcnn.control.dh116.tabletop_manual_pinch_geometry import enable
    t=setup([[0.,0.],[0.,0.]])
    geometry={}
    for h,name in enumerate(('left','right')):
        j=torch.zeros((2,3,6),dtype=torch.float64)
        j[0,0,1]=-.08;j[1,0,2]=.09
        j[0,2,1]=.05;j[1,2,2]=.04
        geometry[name]=dict(point_jacobian_m_per_rad=j.tolist(),reference_active_rad=t.reference[0,6*h:6*h+6].tolist())
    enable(t,geometry)
    a=torch.zeros((t.n,20),dtype=torch.float64);a[:,18:]=-.4
    t.control(a,~t.done,1/480)
    dq=(t.command-t.joint_requested_target).reshape(t.n,2,6)
    movement=torch.einsum('hdck,nhk->nhdc',t.manual_pinch_jacobian,dq).mean(2)
    assert movement[:,:,2].abs().max()>0
    torch.testing.assert_close(t.palm_targets+movement,torch.zeros_like(t.palm_targets),atol=1e-12,rtol=0)
