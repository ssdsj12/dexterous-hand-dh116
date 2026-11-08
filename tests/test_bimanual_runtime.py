from types import SimpleNamespace

import torch

from test_bimanual_tracking_guard import state, actions
from go2_pvcnn.control.dh116.tabletop_bimanual_tracking_guard import BimanualTrackingGuard
from go2_pvcnn.control.dh116.tabletop_bimanual_experiment_runtime import apply_tracking_guard


def test_guard_cancels_only_unsafe_queued_targets_before_native_control(monkeypatch):
    sample = state()
    sample.active_tracking_error_rad[0, 0] = .04
    sample.measured_palm_offset_m[0, 0] = torch.tensor([.001, -.002, .003])
    sample.current_palm_target_m.fill_(.006)
    sample.current_wrist_angles_rad.fill_(.04)
    world = SimpleNamespace(palms=sample.current_palm_target_m.clone(),
                            wrist_angles=sample.current_wrist_angles_rad.clone())
    task = SimpleNamespace(palm_targets=world.palms.clone(),
                           rotation_targets=world.wrist_angles.clone())
    before_right = world.palms[:, 1].clone()
    before_right_angles = world.wrist_angles[:, 1].clone()
    monkeypatch.setattr('go2_pvcnn.control.dh116.tabletop_bimanual_tracking_guard.tracking_state_from_native',
                        lambda w, t: sample)
    raw = actions()
    issued, info = apply_tracking_guard(BimanualTrackingGuard(1, enabled=True), raw, world, task)
    assert info['pause_hand'].tolist() == [[True, False]]
    assert torch.equal(issued[:, :12], raw[:, :12])
    assert issued[:, 12:15].eq(0).all() and issued[:, 18:21].eq(0).all()
    assert torch.equal(world.palms[:, 0], sample.measured_palm_offset_m[:, 0])
    assert torch.equal(task.palm_targets, world.palms)
    assert world.wrist_angles[:, 0].eq(0).all()
    assert torch.equal(task.rotation_targets, world.wrist_angles)
    assert torch.equal(world.palms[:, 1], before_right)
    assert torch.equal(world.wrist_angles[:, 1], before_right_angles)


def coordination_runtime(tmp_path, monkeypatch, *, mask=None):
    from test_bimanual_coordination import state as coordination_state
    from go2_pvcnn.control.dh116.tabletop_bimanual_experiment_runtime import install_coordination
    co, tracking = coordination_state(2), state(2)
    world = SimpleNamespace(n=2, device=torch.device('cpu'), dt=1/480,
        palms=torch.full((2,2,3), .006, dtype=torch.float64),
        wrist_angles=torch.full((2,2,3), .04, dtype=torch.float64))
    task = SimpleNamespace(palm_targets=world.palms.clone(), rotation_targets=world.wrist_angles.clone())
    env = SimpleNamespace(task=task)
    monkeypatch.setattr('go2_pvcnn.control.dh116.tabletop_bimanual_coordination.coordination_state_from_native',
                        lambda *args: co)
    monkeypatch.setattr('go2_pvcnn.control.dh116.tabletop_bimanual_tracking_guard.tracking_state_from_native',
                        lambda *args: tracking)
    tmp_path.mkdir(exist_ok=True)
    install_coordination(env, world, 'z3', tmp_path, active_first_episode_mask=mask)
    return env, world, task, co, tracking


def test_explicit_all_active_mask_matches_default_live_actions_and_targets_bitwise(tmp_path, monkeypatch):
    first = coordination_runtime(tmp_path/'first', monkeypatch)
    raw = actions(2)
    first[4].active_tracking_error_rad[0,0] = .04
    expected = first[0].bimanual_action_adapter(raw).clone()
    expected_targets = [value.clone() for value in (first[1].palms, first[1].wrist_angles,
                                                    first[2].palm_targets, first[2].rotation_targets)]
    second = coordination_runtime(tmp_path/'second', monkeypatch,
        mask=lambda: torch.ones(2, dtype=torch.bool))
    second[4].active_tracking_error_rad[0,0] = .04
    actual = second[0].bimanual_action_adapter(raw)
    assert actual.numpy().tobytes() == expected.numpy().tobytes()
    for wanted, value in zip(expected_targets, (second[1].palms, second[1].wrist_angles,
                                                second[2].palm_targets, second[2].rotation_targets)):
        assert value.numpy().tobytes() == wanted.numpy().tobytes()


def test_closed_new_episode_zero_tail_cannot_create_ascent_or_cancel_targets(tmp_path, monkeypatch):
    first = torch.tensor([True, False])
    env, world, task, co, tracking = coordination_runtime(tmp_path, monkeypatch, mask=lambda: first)
    co.episode_id[1] = 1
    tracking.active_tracking_error_rad[1] = .04
    targets = [world.palms, world.wrist_angles, task.palm_targets, task.rotation_targets]
    before = [value[1].numpy().tobytes() for value in targets]
    source = torch.zeros(2,24, dtype=torch.float64)
    for _ in range(120):
        result = env.bimanual_action_adapter(source)
        assert result[1].eq(0).all()
        assert [value[1].numpy().tobytes() for value in targets] == before


def test_midterminal_remaining_substep_keeps_source_and_targets_untouched(tmp_path, monkeypatch):
    env, world, task, co, tracking = coordination_runtime(tmp_path, monkeypatch,
        mask=lambda: torch.ones(2, dtype=torch.bool))
    co.live[1] = False
    tracking.live[1] = False
    raw = actions(2)
    targets = [world.palms, world.wrist_angles, task.palm_targets, task.rotation_targets]
    before = [value[1].numpy().tobytes() for value in targets]
    result = env.bimanual_action_adapter(raw)
    assert result[1].numpy().tobytes() == raw[1].numpy().tobytes()
    assert [value[1].numpy().tobytes() for value in targets] == before
