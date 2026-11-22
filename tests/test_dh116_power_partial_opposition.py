import numpy as np

from scripts import screen_dh116_power_partial_opposition as stage


def test_selected_local_pair_must_oppose_at_neighbouring_interior_nodes():
    thumb = "finger14_link"
    little = "finger53_link"
    ring = "finger43_link"

    def witness(node, normal):
        return dict(node_id=node, normal_world=normal, gap_m=.0001, joints_rad=[.5, .4])

    row = dict(independent_near_pads=[thumb, little, ring],
               witnesses={thumb:witness(33, [1.,0.,0.]),
                          little:witness(34, [-1.,0.,0.]),
                          ring:witness(36, [-1.,0.,0.])},
               support_adjacent_witnesses={})
    assert [pair["partner"] for pair in stage.local_opposition_pairs(row, set())] == [little]
    assert not stage.local_opposition_pairs(row, {34})
    row["witnesses"][little]["normal_world"] = [1.,0.,0.]
    assert not stage.local_opposition_pairs(row, set())


def test_partial_commands_lock_other_fingers_and_only_target_pads_may_touch():
    active = np.array([.4, .2, .3, .4, .5, .6])
    pair = dict(thumb_joints_rad=[.7, .8], partner="finger53_link", partner_flexion_rad=.9)
    abduct, close = stage.staged_joints(active, pair)
    np.testing.assert_allclose(abduct, [.7, .2, .3, .4, .5, .6])
    np.testing.assert_allclose(close, [.7, .8, .3, .4, .5, .9])
    assert stage.rod_clearance_required("source:finger13_link", set()) == .00025
    assert stage.rod_clearance_required("source:finger13_link", {"finger14_link", "finger53_link"}) == .00025
    assert stage.rod_clearance_required("cooked:finger53_link", {"finger14_link", "finger53_link"}) == -.00015


def test_negative_lower_bound_and_confirmed_intrusion_are_separate():
    assert stage.clearance_status(lower_m=-.0003, upper_m=-.0002, required_m=-.00015) == "certified_below_limit"
    assert stage.clearance_status(lower_m=-.0003, upper_m=.0001, required_m=-.00015) == "unresolved"
    assert stage.clearance_status(lower_m=.0004, upper_m=.0005, required_m=.00025) == "clear"
