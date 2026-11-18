"""CPU regressions for the opt-in absolute Teacher-reference experiment."""
import importlib

import pytest
import torch

from test_vision_teacher_mpc import native_env


def optimizer(env, sequence, **kwargs):
    module = importlib.import_module("go2_pvcnn.control.dh116.tabletop_vision_teacher_absolute_mpc")
    return module.AbsoluteTeacherReferenceMPC(env, "a" * 64, sequence, **kwargs)


def test_lost_opposition_returns_to_absolute_teacher_target_during_zero_velocity_hold():
    env = native_env()
    sequence = torch.zeros(1, 8, 24)
    controller = optimizer(env, sequence)
    original = env.task.command.clone()
    env.task.command[:, 0] += .013
    env.task.m["effective_pad_vectors_w"][..., 0] = 1
    for tick in range(3):
        action, diag = controller.refine(sequence[:, tick], context={"tick": tick})
        env.task.command += action[:, :12].double() * .005
        assert not diag["per_env"][0]["active"]
    torch.testing.assert_close(env.task.command, original, atol=1e-9, rtol=0)
    assert diag["profile"] == "absolute_reference"
    assert diag["trust_radius_rad"] == .001
    assert diag["training_expert_privileged_only"] is True


def test_horizon_uses_time_indexed_absolute_commands_including_future_direction_change():
    env = native_env()
    sequence = torch.zeros(1, 5, 24)
    sequence[0, :, 0] = torch.tensor([1., -.5, -.25, .75, 0.])
    controller = optimizer(env, sequence)
    initial = env.task.command.clone()
    calls = []
    original = controller.native.planner.plan
    def record(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    controller.native.planner.plan = record
    controller.refine(sequence[:, 0], context={"tick": 0})
    expected = initial[0, 0] + torch.tensor([.005, .0025, .00125, .005], dtype=torch.float64)
    torch.testing.assert_close(calls[0]["nominal"][0, :, 0, 0], expected)


def test_total_mpc_offset_is_bounded_across_repeated_replans_and_prediction_matches_issued_action():
    env = native_env()
    sequence = torch.zeros(1, 30, 24)
    controller = optimizer(env, sequence)
    original = env.task.command.clone()
    for tick in range(30):
        # Persistent object displacement asks geometry to keep accumulating motion.
        env.task.m["object_relative"][:, 0] += .0002
        before = env.task.command.clone()
        action, diag = controller.refine(sequence[:, tick], context={"tick": tick})
        issued_target = before + action[:, :12].double() * .005
        assert float((issued_target - original).abs().max()) <= .001 + 1e-9
        if controller.native.pending.any():
            expected = env.task.m["q"].reshape(1, 2, 6) + .5 * (issued_target - before).reshape(1, 2, 6)
            torch.testing.assert_close(controller.native.last_prediction, expected, atol=1e-12, rtol=0)
        env.task.command.copy_(issued_target)
    assert float((env.task.command - original).abs().max()) > 1e-6


def test_invalid_native_row_is_isolated_and_teacher_wrists_are_exact():
    env = native_env(3)
    sequence = torch.zeros(3, 4, 24)
    sequence[:, :, 12:] = .3
    controller = optimizer(env, sequence)
    env.task.m["q"][1, 0] = float("nan")
    action, diag = controller.refine(sequence[:, 0], context={"tick": 0})
    assert [r["active"] for r in diag["per_env"]] == [True, False, True]
    assert torch.isfinite(action).all()
    torch.testing.assert_close(action[:, 12:], sequence[:, 0, 12:], atol=0, rtol=0)


def test_unreachable_trust_region_recovers_at_original_rate_without_disabling_siblings():
    env = native_env(3); sequence = torch.zeros(3, 4, 24)
    controller = optimizer(env, sequence)
    env.task.command[1, 0] += .02
    before = env.task.command.clone()
    action, diag = controller.refine(sequence[:, 0], context={"tick": 0})
    assert [r["active"] for r in diag["per_env"]] == [True, False, True]
    assert action[1, 0] == -1
    assert abs(float(before[1, 0] + action[1, 0] * .005 - .215)) < 1e-8
    assert controller.native.stats["model_fallback_env_steps"] == 1


def test_nonfinite_applied_command_produces_finite_isolated_fallback():
    env = native_env(3); sequence = torch.zeros(3, 4, 24)
    controller = optimizer(env, sequence)
    env.task.command[1, 0] = float("nan")
    action, diag = controller.refine(sequence[:, 0], context={"tick": 0})
    assert torch.isfinite(action).all()
    assert [r["active"] for r in diag["per_env"]] == [True, False, True]
    assert diag["per_env"][1]["reason"] == "invalid_absolute_command"


def test_dead_rows_never_plan_or_restart_on_terminal_reset():
    env = native_env(2)
    sequence = torch.full((2, 4, 24), .2)
    controller = optimizer(env, sequence)
    controller.refine(sequence[:, 0], context={"tick": 0})
    controller.observe_result(torch.tensor([True, False]))
    reference = sequence[:, 1].clone(); reference[0] = 0
    action, diag = controller.refine(reference, context={"tick": 1}, alive=torch.tensor([False, True]))
    assert not action[0].any() and not controller.native.pending[0]
    assert diag["per_env"][0]["reason"] == "inactive_episode"
    with pytest.raises(ValueError, match="ended"):
        controller.refine(sequence[:, 2], context={"tick": 2}, alive=torch.tensor([True, True]))


@pytest.mark.parametrize("context", [None, {}, {"tick": True}, {"tick": -1}, {"tick": 1}, {"tick": 0.0}])
def test_tick_identity_must_be_monotonic_and_explicit(context):
    env = native_env(); sequence = torch.zeros(1, 4, 24)
    controller = optimizer(env, sequence)
    with pytest.raises(ValueError, match="tick"):
        controller.refine(sequence[:, 0], context=context)


def test_source_mismatch_and_repeated_tick_are_rejected_before_planning():
    env = native_env(); sequence = torch.zeros(1, 4, 24)
    controller = optimizer(env, sequence)
    with pytest.raises(ValueError, match="reference"):
        controller.refine(torch.ones(1, 24), context={"tick": 0})
    controller.refine(sequence[:, 0], context={"tick": 0})
    with pytest.raises(ValueError, match="tick"):
        controller.refine(sequence[:, 0], context={"tick": 0})


def test_reference_sequence_is_owned_and_horizon_holds_after_sequence_end():
    env = native_env(); sequence = torch.zeros(1, 1, 24)
    sequence[:, :, 0] = 1
    controller = optimizer(env, sequence)
    sequence.fill_(0)
    action = torch.zeros(1, 24); action[:, 0] = 1
    _, diag = controller.refine(action, context={"tick": 0})
    reference = torch.tensor(diag["per_env"][0]["absolute_reference_trajectory_rad"])
    torch.testing.assert_close(reference[:, 0, 0], torch.full((4,), .205))


@pytest.mark.parametrize("radius", [0, -.001, float("nan"), True])
def test_trust_radius_must_be_positive_finite(radius):
    with pytest.raises(ValueError, match="trust"):
        optimizer(native_env(), torch.zeros(1, 4, 24), trust_radius_rad=radius)


@pytest.mark.parametrize('initial',[1.,1.1])
def test_canonical_substep_roundoff_does_not_falsely_reject_feasible_reference(initial):
    env=native_env();env.task.high[:]=2.
    env.task.command[:,0]=initial;env.task.m['q'][:,0]=initial
    sequence=torch.zeros(1,4,24);sequence[:,:,0]=1.
    controller=optimizer(env,sequence)
    before=env.task.command.clone()
    action,diag=controller.refine(sequence[:,0],context={'tick':0})
    assert diag['per_env'][0]['active'] is True
    assert controller.native.stats['model_fallback_env_steps']==0
    issued=before+action[:,:12].to(before)*.005
    assert float((issued-before).abs().max())<=.005+2e-15
    assert float((issued.reshape(1,2,6)-controller.absolute_commands[:,0]).abs().max())<=.001+2e-15


def test_native_extension_hooks_are_opt_in_and_share_the_original_planner_path():
    from go2_pvcnn.control.dh116.tabletop_vision_teacher_absolute_mpc import _AbsoluteNativeSurfaceMPC
    class Probe(_AbsoluteNativeSurfaceMPC):
        def _activation_mask(self, paired, bottom, reference, ids, valid):
            return valid.any((1, 2))
        def _model_points(self, ids, hand, material, measured):
            return measured + .001
        def _target_trajectory(self, ids, points, targets, wrist_motion, target_points, valid):
            return target_points + .002, torch.zeros_like(valid, dtype=points.dtype)
    env = native_env(); env.task.m["bottom"][:] = .36
    controller = Probe(env, .001)
    controller.reference_trajectory = env.task.command.reshape(1, 1, 2, 6).repeat(1, 4, 1, 1)
    calls = []; original = controller.planner.plan
    def record(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    controller.planner.plan = record
    controller.refine(torch.zeros(1, 24))
    assert controller.pending.tolist() == [True]
    assert len(calls) == 1
    torch.testing.assert_close(calls[0]["points"], torch.full((1, 2, 5, 3), .001, dtype=torch.float64))
    torch.testing.assert_close(calls[0]["target_points"], torch.full((1, 4, 2, 5, 3), .002, dtype=torch.float64))
    assert not calls[0]["weights"].any()
