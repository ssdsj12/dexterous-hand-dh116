import numpy as np
import pytest
import torch

from scripts.audit_dh116_pinch_staged_closure import compile_close
from go2_pvcnn.control.dh116.tabletop_command_kernels import (
    finger_pre_guard, surface_final_guard, wrist_requested_targets,
)
from go2_pvcnn.control.dh116.linear_coupling import active_limits


def _plan():
    def hand(q, dz, turn):
        return dict(root_pose=[.5, .2, .4, 1., 0., 0., 0.], q_open=q,
                    q_close=[q[0], q[1]+.025, q[2]+.035, *q[3:]],
                    wrist_delta=[0., 0., dz], euler_command=[0., turn, 0.], steps=64)

    return dict(left=hand([.3,.15,.65,.3,.3,.3],-.0006,-.012),
                right=hand([.7,.04,.71,.3,.3,.3],.0003,.03))


def test_compiled_fixed_close_reaches_both_targets_through_original_kernels():
    plan = _plan()
    action = torch.tensor(compile_close(plan))[None, :]
    start = torch.tensor([plan["left"]["q_open"] + plan["right"]["q_open"]], dtype=torch.float64)
    q = start.clone()
    palms = torch.zeros((1,2,3), dtype=q.dtype)
    angles = palms.clone()
    lower, upper = map(lambda x: torch.from_numpy(np.tile(x,2)), active_limits())
    rates = dict(joint_rate_rad_s=.3, wrist_rate_m_s=.01, rotation_rate_rad_s=.5)
    dt = 1/480
    for _ in range(64):
        requested = finger_pre_guard(q, action, q, torch.zeros((1,10)), lower, upper, dt, rates,
                                     policy_owned=True)
        _, q = surface_final_guard(q, requested, lower, upper, dt)
        palms, angles = wrist_requested_targets(palms, angles, action, rates, dt)
    goal = plan["left"]["q_close"] + plan["right"]["q_close"]
    np.testing.assert_allclose(q.numpy()[0], goal, rtol=0., atol=1e-10)
    np.testing.assert_allclose(palms[0].numpy(), [plan[s]["wrist_delta"] for s in plan], atol=1e-10)
    np.testing.assert_allclose(angles[0].numpy(), [plan[s]["euler_command"] for s in plan], atol=1e-10)


def test_compiler_rejects_translation_norm_outside_original_rate():
    plan = _plan()
    plan["right"]["wrist_delta"] = [.001,.001,0.]
    with pytest.raises(ValueError, match="rate"):
        compile_close(plan)


def test_compiler_does_not_move_non_target_digits():
    plan = _plan()
    plan["left"]["q_close"][3] += .005
    with pytest.raises(ValueError, match="thumb and index"):
        compile_close(plan)
