import pytest


def test_precontact_progress_requires_every_shape_in_every_sample():
    from scripts.screen_dh116_tripod_measured_precontact import check_route

    row = dict(shape_count=22, rod_clearance_m=.0006, support_clearance_m=.002,
               table_clearance_m=.009, pad_gaps_m=[.001, .0012, .0009], opposition=[-.7, -.8])
    samples = [dict(row) for _ in range(17)]
    samples[0]['pad_gaps_m'] = [.002, .002, .002]
    assert check_route(samples)
    for broken in (
        {'rod_clearance_m': .00025}, {'support_clearance_m': 0.},
        {'table_clearance_m': .002}, {'shape_count': 21}, {'pad_gaps_m': [.001, None, .0009]},
    ):
        damaged = samples.copy()
        damaged[5] = dict(row, **broken)
        assert not check_route(damaged)
    assert not check_route(samples[:-1])


def test_measured_route_replay_rejects_stale_geometry_and_mismatched_boundary():
    from scripts.verify_dh116_power_shared_rod import verify_waypoint_boundary

    def row(tick):
        rod = {key: tick for key in ('node_positions', 'velocities', 'rest_lengths_m',
              'external_support_wrenches_world', 'mesh_names', 'physics_step', 'rod_tick')}
        hands = {side: {key: [0] for key in ('q', 'qd', 'backend_targets', 'native_anchor_pose_w')}
                 for side in ('left', 'right')}
        return dict(physics_step=tick, rod=rod, action_readback={'post': {'hands': hands}},
                    sdk_hand_geometry=dict(physics_step=tick, poses={'shape': [1]}))

    actual, reference = row(432), row(432)
    verify_waypoint_boundary(actual, reference, sdk_step=432, exact_geometry=True)
    actual['sdk_hand_geometry']['physics_step'] = 431
    with pytest.raises(ValueError):
        verify_waypoint_boundary(actual, reference, sdk_step=432, exact_geometry=True)
    with pytest.raises(ValueError):
        verify_waypoint_boundary(row(432), row(272), sdk_step=432, exact_geometry=True)


def test_precontact_archive_and_validator_import_without_loading_usd():
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", """
from scripts.screen_dh116_tripod_measured_precontact import archive, check_route
from scripts.prepare_dh116_dual_pinch_interval import assert_clean_import_boundary
assert_clean_import_boundary()
"""], check=True)
