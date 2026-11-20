import numpy as np
import pytest

from scripts import audit_dh116_pinch_wrist_sweep as original
from scripts import audit_dh116_pinch_early_approach as approach


def _pads(gaps, *, opposition=-0.8):
    return {"gaps_m": list(gaps), "opposition_cos": opposition}


def test_pair_cannot_be_assembled_from_different_frames_or_hands():
    rows = [
        dict(sdk_step=3, left=_pads((-.0002, .0005)), right=_pads((-.0002, -.0002))),
        dict(sdk_step=4, left=_pads((.0005, -.0002)), right=_pads((-.0002, -.0002))),
        dict(sdk_step=5, left=_pads((-.0002, -.0002)), right=_pads((-.0002, .0005))),
    ]
    result = approach.bilateral_summary(rows)
    assert result["best_bilateral_gap_m"] == .0005
    assert result["bilateral_opposed_target_ticks"] == []


def test_pair_requires_all_four_observed_gaps_and_two_opposed_normals():
    rows = [
        dict(sdk_step=3, left=_pads((None, -.0002)), right=_pads((-.0002, -.0002))),
        dict(sdk_step=4, left=_pads((-.0002, -.0002), opposition=.9), right=_pads((-.0002, -.0002))),
        dict(sdk_step=5, left=_pads((-.0002, -.0002)), right=_pads((-.0002, -.00015))),
    ]
    result = approach.bilateral_summary(rows)
    assert result["best_bilateral_gap_m"] == -.0002
    assert result["bilateral_opposed_target_ticks"] == [5]
    assert result["accepted_for_native"] is False


def test_force_candidate_requires_geometry_proof_along_path():
    rows = [dict(sdk_step=68, left=_pads((-.0002, -.0002)), right=_pads((-.0002, -.0002)))]
    for side in original.SIDES:
        rows[0][side]["table"] = {"proved": True}
        rows[0][side]["non_target_rod"] = {"proved": True}
    rows[0]["left"]["non_target_rod"] = {"proved": False}
    result = approach.bilateral_summary(rows)
    assert result["bilateral_opposed_target_ticks"] == [68]
    assert result["bilateral_geometry_ticks"] == []


def test_horizon_respects_original_translation_rotation_and_digit_slew():
    delta = np.array([.0003, .0004, 0., .025, .015, .04, 0., .03])
    assert approach.minimum_sdk_steps(delta) == 48
    with pytest.raises(ValueError, match="finite"):
        approach.minimum_sdk_steps([float("nan")] * 8)

def test_swapped_raw_link_boundary_is_rejected_even_with_matching_rod_and_sdk():
    keys = ("run_uuid", "env_id", "episode_uid", "reset_id", "rod_uid",
            "physics_step", "rod_tick", "timestamp_s", "solver_time_s")
    identity = dict.fromkeys(keys, 0)
    identity.update(physics_step=20, clock_valid=True)
    raw = dict(identity=dict(identity, physics_step=19), rod_nodes_m=np.zeros((41, 3)))
    post = dict(identity=[identity])
    rod = dict(identity, node_positions=np.zeros((1, 41, 3)))
    row = dict(action_readback={"post": post}, closure_geometry={"raw": raw}, rod=rod, physics_step=20)
    with pytest.raises(ValueError, match="same native boundary"):
        approach._same_clock(row)


def test_loaded_reset_cannot_be_reported_as_safe_candidate_path():
    hand = dict(_pads((-.0002, -.0002)), initial_gaps_m=[-.0002, .001],
                table={"proved": True}, non_target_rod={"proved": True},
                swept_y_bounds_m=[.1, .2])
    other = dict(hand, initial_gaps_m=[.001, .001], swept_y_bounds_m=[-.2, -.1])
    result = approach.bilateral_summary([dict(sdk_step=4,left=hand,right=other,
                                              interhand_sweep_gap_m=.2)])
    assert result["bilateral_opposed_target_ticks"] == [4]
    assert result["bilateral_geometry_ticks"] == []
