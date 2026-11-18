import numpy as np
import pytest

from go2_pvcnn.control.dh116.native_stage_controller import (
    NativeObservation,
    NativeStageController,
)


def _observation(**updates):
    values = dict(
        contact_established=False,
        contact_duration_s=0.0,
        hold_duration_s=0.0,
        slip_ok=True,
        lift_displacement_m=0.0,
        lift_duration_s=0.0,
        whole_wire_clear=False,
        hand_active_targets={"left": np.zeros(6), "right": np.ones(6)},
    )
    values.update(updates)
    return NativeObservation(**values)


def test_controller_starts_at_close_and_never_emits_approach():
    controller = NativeStageController("close")
    command = controller.step(_observation())
    assert command.stage == "close"
    assert command.stage != "approach"
    assert command.panda_delta_z == {"left": 0.0, "right": 0.0}


def test_lift_moves_both_pandas_by_equal_positive_z_delta():
    controller = NativeStageController("lift")
    command = controller.step(
        _observation(
            contact_established=True,
            hold_duration_s=1.0,
            lift_displacement_m=0.01,
        )
    )
    assert command.panda_delta_z["left"] > 0.0
    assert command.panda_delta_z["left"] == command.panda_delta_z["right"]


def test_controller_rejects_approach_entry():
    with pytest.raises(ValueError, match="close|hold|lift"):
        NativeStageController("approach")

