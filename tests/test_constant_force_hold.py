import pytest
import torch
from test_force_region_integration import constrained
from go2_pvcnn.control.dh116.tabletop_policy_lift_hold import PolicyLiftHold


def held():
    t=constrained();t.phase[:]=3;t.f_requested[:]=5.
    t.m['hand_object_loads'][:]=4.8
    h=PolicyLiftHold(2,'cpu');t.policy_lift_hold=h
    live=torch.ones(2,dtype=torch.bool)
    h.capture(t,live,t.m['adaptive_palm_offset'])
    return t,h,live


def test_close_latches_requested_reference_not_noisy_actual_sample():
    t,h,live=held()
    assert (h.force_target==5.).all()
    t.f_requested[:]=12.;t.m['hand_object_loads'][:]=3.
    h.capture(t,live,t.m['adaptive_palm_offset'])
    assert (h.force_target==5.).all()


def test_dropout_and_slip_never_raise_latched_reference():
    t,h,live=held();t.m['hand_object_loads'][:]=0.
    t.m['contact_valid'][:]=False
    for _ in range(50):
        _,_,_,paused=h.targets(t,live,.01)
        assert paused.all()
    assert (h.force_target==5.).all()
    assert not h.force_integral.any()


def test_feedback_compensates_load_error_and_preserves_reference():
    t,h,live=held();t.m['hand_object_loads'][:]=3.
    before=h.command.clone()
    target,_,_,_=h.targets(t,live,.01)
    assert (h.force_filtered<4.8).all()
    assert (h.force_integral>0).all()
    assert (h.closure>0).all() and (h.force_target==5.).all()
    assert not torch.equal(target,before)
    assert h.observe(t).shape==(2,34)


def test_coupling_guard_freezes_integral_and_selected_reset_clears_history():
    t,h,live=held();t.m['hand_object_loads'][:]=3.
    h.targets(t,live,.01);saved=h.force_integral.clone()
    t.m['mimic_errors'][:]=.02
    for _ in range(20):h.targets(t,live,.01)
    torch.testing.assert_close(h.force_integral,saved)
    h.reset(torch.tensor([0]))
    assert not h.force_integral[0].any() and not h.force_filtered[0].any()
    assert h.force_integral[1].any()


def test_lift_waits_for_latched_force_tolerance_while_pi_compensates():
    t,h,live=held();t.m['hand_object_loads'][:]=3.
    t.palm_targets[:,:,2]=.05
    _,_,_,paused=h.targets(t,live,.01)
    assert paused.all() and not h.common_height.any()
    assert h.closure.any()


def test_cli_refuses_force_region_without_latched_hold():
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    source=Path(__file__).resolve().parents[1]/'scripts/train_dh116_tabletop_gpu_ppo.py'
    parsed=ast.parse(source.read_text())
    node=next(n for n in parsed.body if isinstance(n,ast.If)
              and ast.unparse(n.test).startswith('args.force_region_control'))
    args=SimpleNamespace(force_region_control=True,grasp_controller='policy',tensor_pipeline=True,
        vertical_force_reward=True,grasp_readiness='measured_load',policy_lift_hold=False)
    class Parser:
        def error(self,message):raise ValueError(message)
    with pytest.raises(ValueError):exec(compile(ast.Module(body=[node],type_ignores=[]),'<cli-check>','exec'),dict(args=args,parser=Parser()))
