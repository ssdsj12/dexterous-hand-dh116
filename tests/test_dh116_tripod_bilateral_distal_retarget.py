"""Regression coverage for a retargeted supported Tripod approach."""
from copy import deepcopy

from scripts import screen_dh116_tripod_bilateral_retarget as retarget


def sample():
    return dict(shape_count=22, noncontact_rod_clearance_m=.00095, support_clearance_m=.012,
                table_clearance_m=.020, pad_gaps_m=[-.00007, -.00005, -.00008],
                node_ids=[35, 35, 33], opposition=[-.80, -.56])


def test_reviewed_route_requires_margin_and_individual_material_stations():
    rows = [sample() for _ in range(retarget.measured.SAMPLES)]
    assert retarget.route_passes(rows)
    for key, value in (("noncontact_rod_clearance_m", .0007),
                       ("support_clearance_m", 0.), ("pad_gaps_m", [.00015, 0., 0.]),
                       ("node_ids", [30, 35, 33]), ("opposition", [-.1, -.56])):
        modified = deepcopy(rows)
        modified[-1][key] = value
        assert not retarget.route_passes(modified), key
    assert not retarget.route_passes(rows[:-1])


def test_previous_guard_failure_is_only_readable_for_its_named_completed_boundary():
    terminal = dict(exit_code=1, debugger_inferior_exit_code=1, debugger_inferior_signal=None,
                    forced_exit=False, exceeded_budget=False, debugger_status_verified=True)
    failure = dict(guarded_stop_sdk_step=1178, grasp_qualified=False, teacher_admission=False, ppo_updates=0,
                   failure="ValueError: live hand source envelope lost its staged rod clearance")
    assert retarget.diagnostic_boundary_valid(terminal, failure, {"app_close_returned":True})
    for key, value in (("forced_exit",True),("debugger_inferior_signal","SIGSEGV"),("exit_code",0)):
        assert not retarget.diagnostic_boundary_valid(dict(terminal, **{key:value}), failure,
                                                      {"app_close_returned":True})
    assert not retarget.diagnostic_boundary_valid(terminal, dict(failure, guarded_stop_sdk_step=1136),
                                                  {"app_close_returned":True})
    assert not retarget.diagnostic_boundary_valid(terminal, failure, {"app_close_returned":False})
