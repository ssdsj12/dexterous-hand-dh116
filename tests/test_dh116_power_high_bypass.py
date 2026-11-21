from copy import deepcopy

import numpy as np
import pytest

from scripts.verify_dh116_power_high_bypass import compile_route


def inputs():
    projected = np.array([[[.55, .16, .33, 1., 0., 0., 0.],
                           [.55,-.16, .34, 1., 0., 0., 0.]]])
    row = {"wrist_projection": {"projected_root_poses": projected, "palm_offsets_m": np.zeros((1,2,3)),
        "projected_angles": np.zeros((1,2,3)), "safe_z_clipped": np.zeros((1,2),dtype=bool)}}
    cfg = {"version": "dh116-power-high-bypass-v1", "source_boundary_sdk": 880,
           "runtime_interpolation": "world_xyz_and_reset_relative_intrinsic_XYZ",
           "phase_window_counts": [66,272], "hands": {}}
    for i,side in enumerate(("left","right")):
        xyz = projected[0,i].copy()
        middle, end = xyz.copy(), xyz.copy()
        middle[:3] += [.008, 0., .004]
        end[:3] += [.003, -.013 if side == "left" else .010, .035]
        cfg["hands"][side] = {"waypoints":[xyz,middle,end]}
    return row,cfg


def test_power_routes_separate_hands_at_the_original_action_rates():
    row,cfg = inputs()
    commands,targets = compile_route(row,cfg)
    assert commands.shape == (338,24)
    np.testing.assert_array_equal(commands[:,:12],0.)
    assert commands[-1,13] < 0 < commands[-1,16]
    for i, side in enumerate(("left","right")):
        span = commands[:,12+3*i:15+3*i].sum(axis=0)*(.01/60)
        np.testing.assert_allclose(row["wrist_projection"]["projected_root_poses"][0,i,:3]+span,targets[side][-1,:3])


def test_power_route_rejects_wrong_pose_owner_bounds_and_action_rate():
    row,cfg=inputs()
    mismatch=deepcopy(cfg)
    mismatch["hands"]["right"]["waypoints"][0][0] += .001
    with pytest.raises(ValueError, match="actual projected"):
        compile_route(row,mismatch)
    below=deepcopy(cfg)
    below["hands"]["left"]["waypoints"][-1][2] -= .05
    with pytest.raises(ValueError, match="original wrist range"):
        compile_route(row,below)
    fast=deepcopy(cfg)
    fast["hands"]["left"]["waypoints"][1][0] += .007
    with pytest.raises(ValueError, match="original rate"):
        compile_route(row,fast)
