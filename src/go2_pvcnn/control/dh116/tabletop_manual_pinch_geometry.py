"""Manual force closure compensation, from the seed's audited pad Jacobians."""
import torch


def enable(t,derivation,home_pad_positions=None):
    jac=t.command.new_tensor([derivation[h]['point_jacobian_m_per_rad'] for h in ('left','right')])
    reference=t.command.new_tensor([derivation[h]['reference_active_rad'] for h in ('left','right')]).flatten()
    if jac.shape!=(2,2,3,6) or reference.shape!=(12,) or not torch.isfinite(jac).all():
        raise ValueError('finite two-hand pad point Jacobians required')
    if not torch.allclose(t.reference,reference.expand_as(t.reference),atol=1e-6,rtol=0):
        raise ValueError('manual pinch geometry does not match the grasp reference')
    t.manual_pinch_jacobian=jac
    t.manual_pinch_compensation=torch.zeros((t.n,2,3),device=t.device,dtype=t.command.dtype)
    t._seed['manual_pinch_compensation']=t.manual_pinch_compensation[0].clone()
    if home_pad_positions is not None:
        home=t.command.new_tensor(home_pad_positions)
        if home.shape!=(2,2,3) or not torch.isfinite(home).all():
            raise ValueError('finite source FK pad positions required')
        t.manual_pinch_home=home.mean(1)


def compensation(t,applied,requested):
    dq=(applied-requested).reshape(t.n,2,6)
    return -torch.einsum('hdck,nhk->nhdc',t.manual_pinch_jacobian,dq).mean(2)


def measured_target(t,w,nominal):
    """Keep the measured pad midpoint at the user's commanded placement.

    Use measured wrist displacement to remove carriage tracking error from the
    internal finger displacement. Adding wp cancels pad_relative's object frame;
    object motion therefore never creates an autonomous lift command.
    """
    actual_mid=(w.pad_relative+w.wp[:,None,:]).reshape(t.n,2,2,3).mean(2)
    internal=actual_mid-(t.manual_pinch_home[None]+w.origins[:,None,:]
                         +t.m['adaptive_palm_offset'])
    return nominal-internal
