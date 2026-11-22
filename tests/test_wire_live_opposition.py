"""Preserve same-step opposing load and following-SDK reaction identities."""
import numpy as np
import pytest

from go2_pvcnn.control.dh116 import tabletop_wire_live_opposition as probe


def sample(thumb, index, *, index_normal=(-1., 0., 0.)):
    return [{"per_pad": {
        "thumb": {"normal_load_n": a, "normal_on_rod_world": [1., 0., 0.] if a else None},
        "index": {"normal_load_n": b, "normal_on_rod_world": index_normal if b else None},
    }} for a, b in zip(thumb, index, strict=True)]


def test_two_left_contacts_must_oppose_in_the_same_solver_substep():
    one_at_a_time = probe.left_loads(sample([.1, 0.] * 6, [0., .1] * 6))
    assert one_at_a_time["opposed_substeps"] == 0
    assert one_at_a_time["index_loaded_substeps"] == 6
    light = probe.left_loads(sample([.01] * 12, [.03] * 12))
    assert light["opposed_substeps"] == 12
    assert light["loaded_opposed_substeps"] == 0
    load = probe.left_loads(sample([.15] * 12, [.16] * 12))
    assert load["loaded_opposed_substeps"] == 12
    aligned = probe.left_loads(sample([.15] * 12, [.16] * 12, index_normal=[1., 0., 0.]))
    assert aligned["opposed_substeps"] == 0


def test_late_index_observation_latches_only_its_actual_next_composer_channel():
    state = probe.IndexReaction(body_id=17)
    evidence = {"index_loaded_substeps": 12}
    state.observe(2186, evidence, None)
    assert state.first_contact_sdk == 2186 and state.first_transfer_sdk is None
    issued = {"body_ids": [8, 17], "forces": np.array([[[2., 0., 0.], [0., .03, 0.]]]),
              "torques": np.zeros((1, 2, 3))}
    unrelated = probe.IndexReaction(body_id=17)
    unrelated.observe(2186, evidence, None)
    wrong = dict(issued, forces=issued["forces"].copy())
    wrong["forces"][0, 1] = 0.
    with pytest.raises(ValueError, match="index reaction"):
        unrelated.observe(2187, evidence, wrong)
    state.observe(2187, evidence, issued)
    assert state.first_transfer_sdk == 2187


def test_close_actions_use_only_original_left_index_and_respect_rate():
    command, windows = probe.index_closure(.05068016052246094)
    assert windows == 11
    assert np.flatnonzero(command).tolist() == [2]
    assert 0 < command[2] < 1.
    assert command[2] * .3 * windows / 60 == pytest.approx(.05068016052246094)


def test_thumb_handoff_requires_the_final_sample_and_completed_transfer():
    assert not probe.ready_after_thumb(2168, dict(physics_step=2168, substep=5, guard_failure=None))
    assert not probe.ready_after_thumb(None, dict(physics_step=2170, substep=7, guard_failure=None))
    assert probe.ready_after_thumb(2168, dict(physics_step=2170, substep=7, guard_failure=None))
    assert not probe.ready_after_thumb(2168, dict(physics_step=2170, substep=7, guard_failure="joint_limit"))


def test_opposition_end_requires_boundary_receipt_without_promoting_lift(tmp_path):
    import json
    from scripts.record_dh116_teacher_numeric_reset import _settled_left_opposition_end

    first, origin, end = 2170, 1426, 2178
    phase = dict(start_sdk=first, last_sdk=end, physical_samples=8,
                 first_thumb_contact_sdk=2167, first_thumb_transfer_sdk=2168,
                 first_index_contact_sdk=None, first_index_transfer_sdk=None,
                 teacher_admission=False, grasp_qualified=False, lift_promoted=False)
    path = tmp_path / "settled_left_opposition.json"
    path.write_text(json.dumps(phase))
    plan = dict(version=probe.VERSION, quiet_settlement_sdk=origin,
                budget=dict(sdk_steps=3338, solver_steps=40032, control_steps=239))
    result = dict(scope="measured_settled_wire_left_opposition_probe", sdk_steps=end,
                  control_steps=(end-origin)//8, solver_substeps=(end-2)*12, measured_exchanges=end-2,
                  quiet_settlement=dict(qualified=True, last_sdk=origin),
                  approach_feedback=dict(first_contact_sdk=2167, first_transfer_sdk=2168,
                     contact_reached=True, left_opposition=phase, start_sdk=origin, end_sdk=end,
                     readback=dict(clock=dict(clock_valid=True, physics_step=end, rod_tick=(end-2)*12))))
    assert _settled_left_opposition_end(plan, result, tmp_path) == end
    path.write_text(json.dumps(dict(phase, lift_promoted=True)))
    with pytest.raises(ValueError, match="receipt"):
        _settled_left_opposition_end(plan, result, tmp_path)


def test_requested_increment_reserve_rejects_a_currently_safe_selected_hull():
    from itertools import product

    hull = np.array([[x, y, z] for x, y, z in product((.004862, .00497), (-.001, .001), (-.0001, .0001))])
    nodes = np.stack([np.zeros(41), np.linspace(-.25, .25, 41), np.full(41, .5)], axis=-1)[None]
    frames = np.array([[[0., 0., .5, 1., 0., 0., 0.]]])
    kwargs = dict(frames=frames, before_frames=frames, nodes=nodes, before_nodes=nodes,
        shapes=[(0, 0, hull)], selected={(0, 0)}, left_palm=np.array([0., 0., .5]), finger_travel={},
        boxes=[(np.ones(3), np.full(3, 2.)), (-np.full(3, 2.), -np.ones(3))],
        table_top=.3, radius=.005)
    hold = np.zeros(24)
    assert probe.guard_prediction(action=hold, **kwargs)["passed"]
    approaching = hold.copy()
    approaching[12] = -1.
    rejected = probe.guard_prediction(action=approaching, **kwargs)
    assert not rejected["passed"] and rejected["shape_index"] == 0
    assert rejected["requested_hand_motion_m"] == pytest.approx(.01 / 480)


def test_predicted_stop_still_delivers_late_index_wrench_in_hold():
    state = probe.IndexReaction(body_id=17)
    assert not state.deliver_after_stop("hold")
    state.observe(2186, {"index_loaded_substeps": 1}, None)
    assert not state.deliver_after_stop("close")
    assert state.deliver_after_stop("hold")
    issued = {"body_ids": [17], "forces": np.array([[[0., 0., .01]]])}
    state.observe(2187, {"index_loaded_substeps": 0}, issued)
    assert not state.deliver_after_stop("hold")


def test_rigid_wrist_composition_preserves_a_valid_tangential_thumb_increment():
    from itertools import product

    hull = np.array([[x, y, z] for x, y, z in product((.004862, .00497), (-.001, .001), (-.0001, .0001))])
    nodes = np.stack([np.zeros(41), np.linspace(-.25, .25, 41), np.full(41, .5)], axis=-1)[None]
    frames = np.array([[[0., 0., .5, 1., 0., 0., 0.]]])
    palm = np.array([0., -.1, .5])
    seed = np.array([[np.r_[palm, [1., 0., 0., 0.]], [.2, 0., .5, 1., 0., 0., 0.]]])
    kwargs = dict(frames=frames, before_frames=frames, nodes=nodes, before_nodes=nodes,
        shapes=[(0, 0, hull)], selected={(0, 0)}, left_palm=palm, finger_travel={},
        boxes=[(np.ones(3), np.full(3, 2.)), (-np.full(3, 2.), -np.ones(3))], table_top=.3, radius=.005)
    command = np.zeros(24)
    command[12] = .8
    command[20] = .16
    assert not probe.guard_prediction(action=command, **kwargs)["passed"]

    def composed(move):
        return probe.compose_projected_pivot(move, root_seed_w=seed, measured_base_w=seed,
            clearance_m=np.full((1, 2), .2), palm_offsets_m=np.zeros((1, 2, 3)),
            angles_rad=np.zeros((1, 2, 3)))

    verified = probe.guard_prediction(action=command, composed_pivot=lambda: composed(command), **kwargs)
    assert verified["passed"] and verified["composed_hull_count"] == 1
    bad = command.copy()
    bad[12] *= -1
    assert not probe.guard_prediction(action=bad, composed_pivot=lambda: composed(bad), **kwargs)["passed"]
    # The next hull is not selected and must still be rejected after the first exact check.
    kwargs["shapes"].append((0, 1, hull))
    kwargs["frames"] = kwargs["before_frames"] = np.repeat(frames, 2, axis=1)
    later = probe.guard_prediction(action=command, composed_pivot=lambda: composed(command), **kwargs)
    assert not later["passed"] and later["shape_index"] == 1


def test_composed_preview_declines_a_workspace_clamped_wrist_target():
    from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import CONTRACT

    seed = np.array([[[0., 0., .5, 1., 0., 0., 0.], [.2, 0., .5, 1., 0., 0., 0.]]])
    offsets = np.zeros((1, 2, 3))
    offsets[0, 0, 0] = CONTRACT["palm_upper_m"][0]
    observed = seed.copy()
    observed[:, :, :3] += offsets
    action = np.zeros(24)
    action[12] = .8
    action[20] = .16
    assert probe.compose_projected_pivot(action, root_seed_w=seed, measured_base_w=observed,
        clearance_m=np.full((1, 2), .2), palm_offsets_m=offsets, angles_rad=np.zeros((1, 2, 3))) is None
