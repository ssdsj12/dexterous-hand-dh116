import numpy as np
import pytest


def test_inverse_query_bound_uses_exact_chord_and_preserves_unreachable_case():
    from tools import audit_dh116_power_right_index as audit

    assert audit.required_tilt(.010, .020, .002) == pytest.approx(2*np.arcsin(.008/.040))
    assert audit.required_tilt(.001, 0., .002) == 0.
    assert audit.required_tilt(.010, .003, .002) is None


def test_projection_can_exclude_all_axial_slides_and_tangent_rolls():
    from tools import audit_dh116_power_right_index as audit

    # The rod points lie on the roll axis. The hull is displaced 15 mm
    # orthogonally. Axial slides and tangent rolls leave this gap invariant.
    nodes = np.zeros((41, 3))
    nodes[:, 1] = np.linspace(-.15, .15, 41)
    vertices = np.array([[.015, -.02, 0.], [.017, -.02, .001], [.015, .02, 0.], [.017, .02, .001]])
    result = audit.placement_bounds([vertices, vertices], nodes, 9, .002, .005)
    assert result["rotation_invariant_transverse_reserve_m"][9] == pytest.approx(.007498)
    assert result["rotation_invariant_transverse_reserve_m"][3] > 0.
    assert result["geometric_witness_shell_m"] == .0005


def test_tangent_roll_budget_accounts_for_wire_drift_from_the_axis():
    from tools import audit_dh116_power_right_index as audit

    nodes = np.zeros((41, 3))
    nodes[:, 1] = np.linspace(-.15, .15, 41)
    nodes[3, 0] = .010
    vertices = np.array([[.025, -.1, 0.], [.027, -.1, .001], [.025, .1, 0.], [.027, .1, .001]])
    result = audit.placement_bounds([vertices, vertices], nodes, 9, 0., .005)
    assert result["rotation_invariant_transverse_reserve_m"][3] == pytest.approx(-.010502)
    assert result["rotation_invariant_transverse_reserve_m"][9] == pytest.approx(.019498)


def test_measured_geometry_requires_current_rod_and_shape_clock():
    from tools import audit_dh116_power_right_index as audit

    row = dict(physics_step=23, rod=dict(physics_step=22), sdk_hand_geometry=dict(physics_step=23))
    with pytest.raises(ValueError, match="same SDK tick"):
        audit.measured_index(row, {})


def test_original_workspace_can_exclude_station_that_tangent_roll_alone_leaves_open():
    from tools import audit_dh116_power_right_index as audit

    nodes = np.zeros((41, 3))
    nodes[:, 1] = np.linspace(-.2, .2, 41)
    vertices = np.array([[0., -.001, 0.], [.001, .001, 0.], [0., .001, .001]])
    bounds = audit.placement_bounds([vertices], nodes, 9, 0., .005)
    axial = audit.project_workspace_axis(np.zeros(3), np.array([0., 1., 0.]),
                                        np.full(3, -.025), np.full(3, .025))
    lo, hi = bounds["required_axial_slide_interval_m"][19]
    assert axial == pytest.approx((-.025, .025))
    assert lo <= axial[1] and hi >= axial[0]
    lo, _ = bounds["required_axial_slide_interval_m"][30]
    assert lo > axial[1]
    assert bounds["rotation_invariant_transverse_reserve_m"][30] < 0.


def test_receipt_bound_measured_geometry_checks_the_real_raw_bytes(tmp_path, monkeypatch):
    import hashlib
    import json
    from itertools import product

    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import digest
    from tools import audit_dh116_power_right_index as audit

    native = tmp_path / "native"
    native.mkdir()
    model = {"cooked_pads": {audit.INDEX: {
        "capture_sha256": "A", "geometry_sha256": "B",
        "vertices": np.asarray(list(product([-.001, .001], repeat=3))),
    }}}
    monkeypatch.setattr(audit, "load_geometry", lambda *args, **kwargs: model)
    plan = {"budget": {"sdk_steps": 26}}
    frozen = digest(plan)
    monkeypatch.setattr(audit, "FROZEN_SHA256", frozen)
    nodes = np.zeros((1, 41, 3))
    nodes[0, :, 1] = np.linspace(-.25, .25, 41)
    label = "right." + audit.INDEX
    clock = dict(run_uuid="run", env_id=0, episode_uid="episode", rod_uid="rod", reset_id=1,
                 physics_step=26, rod_tick=288, timestamp_s=26/480, solver_time_s=24/480, clock_valid=True)
    row = dict(physics_step=26, rod=dict(clock, node_positions=nodes.tolist(),
        mesh_bindings={label: {"capture_sha256": "A", "source_geometry_sha256": "B"}}),
        guard_failure=None, sdk_hand_geometry=dict(physics_step=26,
        poses={label: [[.020, 0., 0.], [0., 0., 0., 1.]]}))
    inventory = {}
    for name, data in (("frozen_plan.json", plan), ("power_control_26.json", row)):
        raw = json.dumps(data).encode()
        (native/name).write_bytes(raw)
        inventory[name] = hashlib.sha256(raw).hexdigest()
    receipt = tmp_path/"outer.json"
    payload = json.dumps(dict(frozen_sha256=frozen, artifact_sha256=inventory)).encode()
    receipt.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    report = audit.inspect_native(receipt, sha, [26])
    assert not report["actual_load_verified"]
    gap = report["rows"][0]["whole_wire_cooked_gap"]
    assert gap["lower_m"] <= .014 <= gap["upper_m"]
    (native/"power_control_26.json").write_bytes(json.dumps(row).encode() + b" ")
    with pytest.raises(ValueError, match="SHA"):
        audit.inspect_native(receipt, sha, [26])
