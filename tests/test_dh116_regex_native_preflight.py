"""CPU admission tests for a prepared, never-launched regex native probe."""
import importlib
import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

MODULE = "scripts.smoke_dh116_regex_articulation_native"


def implementation():
    assert importlib.util.find_spec(MODULE) is not None, "bounded regex native probe is missing"
    return importlib.import_module(MODULE)


def test_preflight_bounds_two_distinct_24d_lanes_and_exact_sdk_budget():
    impl = implementation()
    for style in ("pinch", "tripod", "power"):
        plan = impl.build_plan(style=style, receipt=Path(f"/{style}/result.json"),
                               receipt_sha256="a" * 64, gpu=0)
        assert plan["environments"] == 2 and plan["articulations"] == 2
        assert plan["sdk_initialization_steps"] == 2
        assert plan["setup_steps"] == 8
        assert plan["measurement_steps"] == 16
        assert plan["measurement_start_sdk_step"] == 10
        assert plan["max_sdk_step"] == 26
        assert plan["reset_steps"] == 0
        assert plan["rod_steps"] == 0 and plan["ppo_updates"] == 0
        assert plan["legacy_equivalence_proven"] is False
        assert plan["style"] == style
        first = impl.action_for_tick(plan, 11)
        assert first.shape == (2, 24) and not np.array_equal(first[0], first[1])
        assert np.count_nonzero(first) == 48
        np.testing.assert_array_equal(first, impl.action_for_tick(plan, 18))
        second = impl.action_for_tick(plan, 19)
        assert not np.array_equal(first, second)
        np.testing.assert_array_equal(second, impl.action_for_tick(plan, 26))
        with pytest.raises(ValueError, match="measurement"):
            impl.action_for_tick(plan, 27)
        with pytest.raises(ValueError, match="measurement"):
            impl.action_for_tick(plan, 10)
    with pytest.raises(ValueError, match="style"):
        impl.build_plan(style="unknown", receipt=Path("/x"), receipt_sha256="a" * 64, gpu=0)


def test_peer_reset_proof_detects_target_reaction_and_bit_pattern_mutation():
    impl = implementation()
    before = {side: {field: np.zeros((2, 3), dtype=np.float32)
                     for field in ("q", "qd", "backend_targets", "root_pose_wxyz",
                                   "link_pose_wxyz", "permanent_force", "instantaneous_force")}
              for side in ("left", "right")}
    after = deepcopy(before)
    for values in after.values():
        for value in values.values():
            value[1] = 9
    assert impl.verify_peer_preserved(before, after, selected_env_id=1)["peer_bitwise_preserved"]
    for side, field, value in (("left", "backend_targets", 1),
                               ("right", "instantaneous_force", 2), ("left", "q", -0.)):
        corrupted = deepcopy(after)
        corrupted[side][field][0, 0] = value
        with pytest.raises(ValueError, match="peer"):
            impl.verify_peer_preserved(before, corrupted, selected_env_id=1)


def valid_acquisition():
    return dict(scope="regex_articulation_only", native_execution_performed=True,
                environments=2, articulations=2, sdk_step=26, physics_rows=16,
                setup_steps=8, measurement_windows=2, isolated_reset_peer_bitwise=True,
                distinct_24d_commands=True, dynamic_state_changed=True,
                all_source_clearance_passed=True, backend_targets_verified=True,
                legacy_equivalence_proven=False, full_chain_qualified=False, rod_steps=0,
                ppo_updates=0)


def test_natural_exit_admission_requires_normal_close_cleanup_and_finished_budget():
    impl = implementation()
    close = dict(app_close_returned=True, close_mode="normal")
    lifecycle = dict(close_mode="normal", failure=None, lifetime_released_before_close=True,
                     physics_steps=26, acquisition=valid_acquisition())
    proof = impl.admit_natural_exit(0, close, lifecycle)
    assert proof["native_articulation_api_passed"] and proof["process_exit_verified"]
    assert not proof["full_chain_qualified"] and not proof["legacy_equivalence_proven"]
    for code, changed_close, changed_lifecycle in (
            (1, close, lifecycle),
            (0, dict(close, close_mode="fault"), lifecycle),
            (0, close, dict(lifecycle, lifetime_released_before_close=False)),
            (0, close, dict(lifecycle, physics_steps=25)),
            (0, close, dict(lifecycle, acquisition=dict(valid_acquisition(), measurement_windows=1))),
            (0, close, dict(lifecycle, acquisition=dict(valid_acquisition(), full_chain_qualified=True)))):
        with pytest.raises(ValueError):
            impl.admit_natural_exit(code, changed_close, changed_lifecycle)
