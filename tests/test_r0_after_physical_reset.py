"""New R0 initialization is pure; these are CPU parity checks, not native runs."""
from copy import deepcopy
import numpy as np
import pytest
from test_dh116_deformable_control_step import inputs
from test_dh116_deformable_grasp_task import snapshot, measurement
from go2_pvcnn.control.dh116.tabletop_deformable_grasp_task import RESET_HOOKS
from go2_pvcnn.control.dh116.tabletop_deformable_control_step import FlexibleR0ControlStep
from go2_pvcnn.control.dh116.tabletop_r0_after_physical_reset import (
    PostPhysicalResetR0ControlStep, reset_clock_matches,
)

def clock():
    return dict(run_uuid="fixture",env_id=0,episode_uid="ep-0",reset_id=0,rod_uid="rod0",
                physics_step=2,rod_tick=24,timestamp_s=2/480)

def test_direct_initialization_matches_legacy_state_without_mock_reset_callbacks():
    t,b,w=inputs()
    old=FlexibleR0ControlStep(t,"pinch",env_id=0,budget_config=b,diagnostic_coefficients=w)
    new=PostPhysicalResetR0ControlStep(t,"pinch",env_id=0,budget_config=b,diagnostic_coefficients=w)
    s=snapshot(t,2); radius=measurement(.005)
    old.reset({k:lambda:None for k in RESET_HOOKS},lambda:(s,radius))
    new.initialize_completed_boundary(s,radius,reset_approach=None)
    for name in ("identity","clock","rod_source","initial_median_z","initial_min_z",
                 "radius","radius_source","material_anchor","slip_references","hold_start",
                 "hold_steps","hold_s","best_lift","previous_bilateral","active",
                 "phase"):
        assert getattr(new._episode,name)==getattr(old._episode,name)
    np.testing.assert_array_equal(new._episode.rest_lengths,old._episode.rest_lengths)
    assert new._initialized
    assert new.reset_initialization["actual_physical_reset_executed_by_initializer"] is False
    assert new.reset_initialization["native_qualification"] is False
    assert new.reset_initialization["reset_callbacks_executed"]==0

def test_callback_reset_entry_is_not_silently_reused():
    t,b,w=inputs();r=PostPhysicalResetR0ControlStep(t,"pinch",env_id=0,budget_config=b,diagnostic_coefficients=w)
    with pytest.raises(RuntimeError,match="completed physical"):
        r.reset({k:lambda:None for k in RESET_HOOKS},lambda:(snapshot(t,2),measurement(.005)))

def test_one_initialization_per_budget_and_failures_close_state():
    t,b,w=inputs();r=PostPhysicalResetR0ControlStep(t,"pinch",env_id=0,budget_config=b,diagnostic_coefficients=w)
    r.initialize_completed_boundary(snapshot(t,2),measurement(.005))
    with pytest.raises(RuntimeError):r.initialize_completed_boundary(snapshot(t,2),measurement(.005))

def test_bad_setup_clock_fails_closed():
    t,b,w=inputs();r=PostPhysicalResetR0ControlStep(t,"pinch",env_id=0,budget_config=b,diagnostic_coefficients=w)
    with pytest.raises(ValueError):r.initialize_completed_boundary(snapshot(t,3),measurement(.005))
    assert not r._initialized and not r._episode.active
    with pytest.raises(RuntimeError):r.initialize_completed_boundary(snapshot(t,2),measurement(.005))

def test_same_reset_measured_approach_baseline():
    t,b,w=inputs();r=PostPhysicalResetR0ControlStep(t,"pinch",env_id=0,budget_config=b,diagnostic_coefficients=w)
    reset=dict(sample_identity=dict(env_id=0,episode_uid="ep-0",reset_id=0,rod_uid="rod0",
        physics_step=2,rod_tick=24,timestamp_s=2/480),
        approach_distance_m=measurement([.01,.03]))
    r.initialize_completed_boundary(snapshot(t,2),measurement(.005),reset_approach=reset)
    assert r._episode.approach[0]==pytest.approx(.02)

@pytest.mark.parametrize("key",["env_id","episode_uid","reset_id","rod_uid","physics_step","rod_tick","timestamp_s","run_uuid"])
def test_reset_boundary_rejects_any_identity_or_time_change(key):
    before=clock();after=deepcopy(before)
    after[key]="foreign" if isinstance(after[key],str) else after[key]+1
    with pytest.raises(ValueError):reset_clock_matches(before,after)

def test_matching_live_clock_does_not_grant_native_execution():
    assert reset_clock_matches(clock(),clock()) is None
