"""Phase-scoped finger holds preserve the already authorized archive prefix."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import verify_dh116_power_shared_rod as shared


def test_hold_validates_only_future_phase_and_preserves_action_channels():
    actions = np.zeros((3, 24), dtype=np.float64)
    actions[0, 3] = 1.0e-10
    actions[1:, [0, 2, 12, 18]] = .1
    original = actions.copy()
    hold = shared.prepare_finger_hold(actions, task="pinch", first_sdk=240, hold_from_sdk=248)
    assert tuple(hold.held_action_indices) == (3, 4, 5, 9, 10, 11)
    np.testing.assert_array_equal(actions, original)
    with pytest.raises(ValueError, match="held|fixed"):
        shared.prepare_finger_hold(actions, task="pinch", first_sdk=240, hold_from_sdk=240)
    with pytest.raises(ValueError, match="window"):
        shared.prepare_finger_hold(actions, task="pinch", first_sdk=240, hold_from_sdk=247)
    with pytest.raises(ValueError, match="Pinch|Tripod|pinch|tripod"):
        shared.prepare_finger_hold(actions, task="power", first_sdk=240, hold_from_sdk=248)


def test_tripod_holds_four_targets_in_new_actual_window_only():
    value = os.environ.get("DH116_TRIPOD_902_NATIVE_576")
    if not value:
        pytest.skip("sealed native 576 receipt is external")
    root = Path(value).parent / "native"
    actions = json.loads((root / "frozen_plan.json").read_bytes())["tripod_approach"]
    hold = shared.prepare_finger_hold(np.asarray(actions), task="tripod", first_sdk=240, hold_from_sdk=568)
    assert tuple(hold.held_action_indices) == (4, 5, 10, 11)
    with pytest.raises(ValueError, match="held|fixed"):
        shared.prepare_finger_hold(np.asarray(actions), task="tripod", first_sdk=240, hold_from_sdk=560)
    for sdk in range(569, 577):
        hold.observe(json.loads((root / f"tripod_control_{sdk}.json").read_bytes()))
