import itertools

import numpy as np
import pytest


def test_tripod_search_scores_authored_and_locked_hulls_separately():
    from scripts.solve_dh116_wire_pregrasp import _constraint_violations

    hull = np.array(list(itertools.product((-.001, .001), repeat=3)))
    wire = np.array([[.004, -.010, 0.], [.004, .010, 0.]])
    row = {"shapes": {"source:pad": hull, "cooked:pad": hull + [-.006, 0., 0.]}, "dots": {}}
    violations = _constraint_violations(row, ("pad",), wire, .005)
    assert violations["source:pad:nonpenetration"] > 0.
    assert violations["cooked:pad:nonpenetration"] == 0.
    assert violations["cooked:pad:approach"] > 0.


def test_power_palm_alignment_is_a_rotation_and_aligns_span_with_rod():
    from scripts.prepare_dh116_power_palm_seed import align_palm

    palm = np.zeros((4, 3))
    fingers = np.array([[0., .4, 0.], [-.1, .5, 0.], [0., .5, 0.],
                        [.1, .5, 0.], [.2, .5, 0.]])
    frame = np.eye(3)[:, [2, 0, 1]]
    rotation = align_palm(palm, fingers, frame)
    np.testing.assert_allclose(rotation.apply([1., 0., 0.]), frame[:, 0], atol=1e-12)
    assert np.dot(rotation.apply(fingers.mean(0) - palm.mean(0)), frame[:, 1]) > 0.
    assert np.linalg.det(rotation.as_matrix()) > .999


def test_power_samples_both_proper_axis_signs():
    from scripts.prepare_dh116_power_palm_seed import palm_axis_frames

    original = np.eye(3)
    forward, reverse = palm_axis_frames(original)
    for sign, frame in (forward, reverse):
        np.testing.assert_array_equal(frame[:, 0], sign * original[:, 0])
        np.testing.assert_array_equal(frame[:, 1], original[:, 1])
        assert np.linalg.det(frame) == 1.
    with pytest.raises(ValueError, match="right-handed"):
        palm_axis_frames(np.diag([-1., 1., 1.]))


def test_power_support_clearance_measures_separation_and_keeps_partner():
    from scripts.prepare_dh116_power_palm_seed import support_clearance

    hull = np.array(list(itertools.product((-.001, .001), repeat=3)))
    supports = [dict(center_m=[0., 0., 0.], size_m=[.002] * 3)]
    assert support_clearance({"source:base_link": hull}, supports)[0] == [
        dict(source="source:base_link", support_index=0)]
    hits, separation = support_clearance({"source:base_link": hull + [.005, 0., 0.]}, supports)
    assert hits == []
    assert separation == pytest.approx(.003)


def test_tripod_witness_prefers_proven_endpoint_over_proxy_winner():
    from scripts.solve_dh116_wire_pregrasp import _tripod_witness_rank

    def row(status, dot, good=False, upper=.0001):
        gaps = {"source:thumb": dict(status=status, upper_m=upper)}
        return dict(endpoint_geometry_passed=good,
                    endpoint_witness=dict(gaps=gaps, witness_direction_dots={"cooked:middle": dot}))

    hit = row("intersection_proven", -.8, upper=-.001)
    shallow_hit = row("intersection_proven", -.1, upper=-.0001)
    unknown = row("unresolved", None)
    wrong_direction = row("clear", -.1)
    valid = row("clear", -.4, good=True)
    assert sorted([hit, unknown, wrong_direction, shallow_hit, valid], key=_tripod_witness_rank) == [
        valid, wrong_direction, unknown, shallow_hit, hit]


def test_power_approach_rejects_swept_support_between_clear_endpoints(monkeypatch):
    from scripts import prepare_dh116_power_palm_seed as power

    hull = np.array(list(itertools.product((-.0005, .0005), repeat=3)))
    shapes = {"source:base_link": hull}
    supports = [dict(center_m=[.004, 0., 0.], size_m=[.001, .001, .001])]
    assert power.support_collisions(shapes, supports) == []
    assert power.support_collisions({"source:base_link": hull + [.008, 0., 0.]}, supports) == []
    wire = np.array([[0., -.01, .1], [0., .01, .1]])
    monkeypatch.setattr(power, "APPROACH_STEP_M", .008)
    result = power.audit_fixed_preshape_approach(shapes, wire, .005, [1., 0., 0.], supports)
    assert result["last_accepted_translation_m"] == 0.
    assert result["first_rejected"]["support_intersections"] == [
        {"source": "source:base_link", "support_index": 0}]
    assert result["native_contact_verified"] is False


def test_power_closure_search_uses_independent_finger_commands():
    from scripts.prepare_dh116_power_palm_seed import search_closure_joints

    lower = np.zeros(6)
    upper = np.ones(6)
    target = np.asarray([0., .5, .25, .75, .25, .75])

    def score(q):
        if q[1] > .75:
            return None
        return {"rank": (np.linalg.norm(q - target),)}

    result, count = search_closure_joints(np.zeros(6), lower, upper, score)
    assert np.array_equal(result["active_q"], target)
    assert count <= 34


def test_power_closure_collision_certificate_is_conservative():
    from itertools import product
    from scripts.prepare_dh116_power_palm_seed import thumb_clearance

    box = np.asarray(list(product((0., .01), repeat=3)))
    moving = {"source:finger13_link": box, "cooked:finger23_link": box + [.009, 0., 0.]}
    assert thumb_clearance(moving) <= 0.
    moving["cooked:finger23_link"] += [.003, 0., 0.]
    assert thumb_clearance(moving) == pytest.approx(.002)


@pytest.mark.parametrize("rejected_by", ["table", "support", "thumb_clearance", "same_side"])
def test_power_closure_never_promotes_unproven_roles(monkeypatch, rejected_by):
    import scripts.prepare_dh116_power_palm_seed as power

    keys = ("source:base_link", *("cooked:" + pad for pad in power.DISTAL_PAD_LINKS))
    z = .001 if rejected_by == "table" else .003
    hulls = {key: np.array([[0., 0., z], [.01, .01, z]]) for key in keys}
    table = dict(table_center=[0., 0., -.01], table_size=[1., 1., .02])
    geometry = dict(reference=dict(reference_layout=table), old_candidate=dict(rod=dict(radius_m=.005)))
    row = dict(root=[0., 0., 0., 1., 0., 0., 0.], palm_toward_fingers=[1., 0., 0.],
               active_q=[0.]*6, fixed_preshape_approach=dict(last_accepted_translation_m=.001))

    monkeypatch.setattr(power, "active_limits", lambda: (np.zeros(6), np.ones(6)))
    monkeypatch.setattr(power, "expand", lambda q: q)
    monkeypatch.setattr(power, "shapes_world", lambda *args: hulls)
    monkeypatch.setattr(power, "support_clearance", lambda *args: ([], .001 if rejected_by == "support" else .003))
    monkeypatch.setattr(power, "thumb_clearance", lambda *args: -.001 if rejected_by == "thumb_clearance" else .003)
    monkeypatch.setattr(power, "_pad_gap_interval", lambda *args: dict(lower_m=.0001, upper_m=.0002))
    witness = dict(witness=dict(direction=[1., 0., 0.]))
    monkeypatch.setattr(power, "audit_hand", lambda *args, **kw: dict(
        geometric_approach_passed=True, gaps={key: witness for key in keys},
        witness_direction_dots={key: 1. for key in keys[2:]}))
    results = power.audit_closure_positions(geometry, row, np.zeros((2, 3)), ())
    assert [item["wrist_fraction"] for item in results] == [0., .5, 1.]
    assert not any(item["endpoint_geometry_passed"] for item in results)
    if rejected_by == "same_side":
        assert all(item["palm_finger_direction_dots"] == [1.]*4 for item in results)
    else:
        assert all(item["rejections"][rejected_by] == item["evaluations"] for item in results)


def test_power_closure_cli_status_tracks_independent_screen(monkeypatch, tmp_path):
    import json
    import scripts.prepare_dh116_power_palm_seed as power

    seed = tmp_path / "seed.json"
    seed.write_text("{}")
    output = tmp_path / "audit.json"
    result = dict(geometric_wrap_passed=True, closure_screen=dict(pairs=[]))
    monkeypatch.setattr(power, "prepare_palm_seed", lambda *args, **kwargs: result.copy())
    code = power.main(["--seed", str(seed), "--audit-approach", "--audit-closure",
                       "--left-node", "3", "--right-node", "6", "--output", str(output)])
    assert code == 2
    assert json.loads(output.read_text())["closure_screen"]["pairs"] == []


def test_power_reclosure_can_change_thumb_abduction_when_explicitly_enabled():
    from scripts.prepare_dh116_power_palm_seed import search_closure_joints

    q0 = np.zeros(6)
    target = np.array([.75, .5, .5, .5, .5, .5])
    evaluate = lambda q: {"rank": (np.linalg.norm(q-target),)}
    unchanged, _ = search_closure_joints(q0, q0, np.ones(6), evaluate)
    changed, evaluations = search_closure_joints(q0, q0, np.ones(6), evaluate,
                                                 include_thumb_abduction=True)
    assert unchanged["active_q"][0] == 0.
    np.testing.assert_array_equal(changed["active_q"], target)
    assert evaluations <= 40
