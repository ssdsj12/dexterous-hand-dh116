import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_deformable_rod_contact import (
    command_at_step, schedule_summary,
)


def test_bounded_target_and_one_exchange_lag():
    q0 = np.zeros(6)
    endpoint = np.array([0., .09, 0., 0., 0., 0.])
    q, a = command_at_step(q0, endpoint, step=49, side="right")
    assert q[1] == pytest.approx(.09 / 160)
    assert a[7] == pytest.approx(.9)
    assert np.count_nonzero(a) == 1
    _, hold = command_at_step(q0, endpoint, step=209, side="right")
    assert np.count_nonzero(hold) == 0
    with pytest.raises(ValueError, match="velocity"):
        command_at_step(q0, endpoint * 2, step=49, side="left")


def test_wrench_evidence_refuses_wrong_exchange_clock():
    base = {"phase_step": 1, "sdk_step": 8, "rod_substep": 12, "sdk_time_s": 8 / 480,
            "applied_wrench_com_w": [0.] * 6, "rod_on_hand_com_w": [0.] * 6,
            "pair_force_residual_n": 0., "failure": None}
    with pytest.raises(ValueError, match="clock"):
        schedule_summary([base, {**base, "phase_step": 2, "sdk_step": 10, "rod_substep": 24}])
