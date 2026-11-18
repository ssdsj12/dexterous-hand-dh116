"""CPU tests for the privileged Teacher-reference training-label optimizer."""
import importlib
import json
from types import SimpleNamespace as NS

import pytest
import torch


def optimizer(env=None, source_sha="a" * 64):
    module = importlib.import_module("go2_pvcnn.control.dh116.tabletop_vision_teacher_mpc")
    return module.TeacherReferenceMPC(native_env() if env is None else env, source_sha)


def native_env(n=1):
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, MIMIC
    dtype = torch.float64
    tensor = lambda value: torch.as_tensor(value, dtype=dtype)
    names = list(ACTIVE) + [row[0] for row in MIMIC]
    bodies = ["root", "finger14_link", "finger23_link", "finger33_link", "finger43_link", "finger53_link"]
    poses = torch.zeros(n, 6, 7, dtype=dtype)
    poses[..., 3] = 1
    jac = torch.zeros(n, 5, 6, len(names), dtype=dtype)
    jac[:, :, 0, 0] = .01
    hand = NS(joint_names=names, body_names=bodies,
              data=NS(body_link_pose_w=poses, body_com_pos_w=poses[..., :3]),
              root_physx_view=NS(get_jacobians=lambda: jac))
    roots = torch.zeros(n, 2, 7, dtype=dtype)
    roots[..., 3] = 1
    loads = tensor([[[1., .5, .5, 0., 0.], [1., .5, .5, 0., 0.]]]).repeat(n, 1, 1)
    vectors = torch.zeros(n, 2, 5, 3, dtype=dtype)
    vectors[:, :, 0, 0] = 1
    vectors[:, :, 1:3, 0] = -.5
    q = torch.full((n, 12), .2, dtype=dtype)
    measurements = dict(q=q, bottom=torch.full((n,), .39, dtype=dtype),
                        object_relative=torch.zeros(n, 3, dtype=dtype),
                        object_quat=tensor([[1., 0., 0., 0.]]).repeat(n, 1),
                        wrist_pose=roots, effective_pad_loads_n=loads,
                        effective_pad_vectors_w=vectors, effective_pad_valid=loads > 0,
                        effective_pad_points_w=torch.zeros(n, 2, 5, 3, dtype=dtype))
    task = NS(m=measurements, command=q.clone(), low=torch.zeros(12, dtype=dtype),
              high=torch.ones(12, dtype=dtype), surface_reference_bottom_m=tensor(.36))
    world = NS(n=n, device="cpu", t=tensor, hands=[hand, hand], roots=roots,
               root_seed=roots.clone(), palms=torch.zeros(n, 2, 3, dtype=dtype),
               wrist_angles=torch.zeros(n, 2, 3, dtype=dtype))
    return NS(world=world, task=task)


def test_training_labels_use_real_mpc_and_keep_teacher_wrists_and_state_unchanged():
    env = native_env()
    controller = optimizer(env)
    reference = torch.linspace(-.8, .8, 24).reshape(1, 24)
    original = reference.clone()
    before = {key: value.clone() for key, value in env.task.m.items()}
    command = env.task.command.clone()
    issued, diag = controller.refine(reference, context={"tick": 12, "physics_step": 96, "episode": [3]})
    assert issued.shape == (1, 24) and torch.isfinite(issued).all() and issued.abs().max() <= 1
    torch.testing.assert_close(issued[:, 12:], reference[:, 12:], atol=0, rtol=0)
    torch.testing.assert_close(reference, original, atol=0, rtol=0)
    torch.testing.assert_close(env.task.command, command, atol=0, rtol=0)
    for key in before:
        torch.testing.assert_close(env.task.m[key], before[key], atol=0, rtol=0)
    assert diag["optimized_action_dim"] == 12 and diag["returned_action_dim"] == 24
    assert diag["training_expert_privileged_only"] is True
    assert diag["model"] == "local_pad_kinematics_not_object_dynamics"
    assert diag["native_correction_qualified"] is False
    assert diag["student_state_dagger_label"] is False
    row = diag["per_env"][0]
    assert row["active"] is True and row["reason"] == "optimized_local_pad_model"
    assert row["context"] == {"tick": 12, "physics_step": 96, "episode": 3}
    assert row["reference_actions"] == reference[0].tolist()
    assert row["issued_actions"] == issued[0].tolist()
    torch.testing.assert_close(torch.tensor(row["action_delta"]), issued[0] - reference[0])
    json.dumps(diag, allow_nan=False)


def test_real_optimizer_can_change_fingers_without_replacing_teacher_reference():
    controller = optimizer()
    reference = torch.zeros(1, 24, dtype=torch.float64)
    reference[:, :12] = 1
    issued, diag = controller.refine(reference)
    assert (issued[:, :12] - reference[:, :12]).abs().max() > .01
    assert diag["per_env"][0]["changed"] is True
    assert diag["per_env"][0]["reference_actions"] == reference[0].tolist()


@pytest.mark.parametrize("case,reason", [
    ("below_air", "below_activation_lift"),
    ("no_opposition", "no_effective_opposition"),
    ("disabled", "prediction_limit_reached"),
])
def test_original_activation_fallbacks_keep_full_teacher_action(case, reason):
    env = native_env()
    controller = optimizer(env)
    if case == "below_air":
        env.task.m["bottom"][:] = .36299
    elif case == "no_opposition":
        env.task.m["effective_pad_vectors_w"][..., 0] = 1
    else:
        controller.native.bad_predictions[:] = 3
    reference = torch.full((1, 24), .4)
    issued, diag = controller.refine(reference)
    torch.testing.assert_close(issued, reference, atol=0, rtol=0)
    row = diag["per_env"][0]
    assert row["active"] is False and row["reason"] == reason and row["changed"] is False


def test_bad_native_environment_is_isolated_and_snapshot_json_keeps_unknown_as_null():
    env = native_env(3)
    env.task.m["q"][1, 0] = float("nan")
    reference = torch.full((3, 24), .2)
    issued, diag = optimizer(env).refine(reference)
    assert [row["active"] for row in diag["per_env"]] == [True, False, True]
    assert diag["per_env"][1]["reason"] == "native_model_rejected"
    torch.testing.assert_close(issued[1], reference[1], atol=0, rtol=0)
    assert diag["per_env"][1]["snapshot"]["q_rad"][0] is None
    json.dumps(diag, allow_nan=False)


def test_invalid_activation_measurement_is_local_and_retains_reference():
    env = native_env(3)
    env.task.m["effective_pad_loads_n"][1, 0, 0] = -1
    reference = torch.full((3, 24), .3)
    issued, diag = optimizer(env).refine(reference)
    assert [row["active"] for row in diag["per_env"]] == [True, False, True]
    assert diag["per_env"][1]["reason"] == "invalid_activation_measurements"
    torch.testing.assert_close(issued[1], reference[1], atol=0, rtol=0)


def test_snapshot_identity_binds_actual_state_before_action_and_has_owned_values():
    env = native_env()
    controller = optimizer(env)
    reference = torch.zeros(1, 24)
    _, first = controller.refine(reference)
    env.task.command[0, 0] += .001
    _, second = controller.refine(reference)
    a, b = first["per_env"][0], second["per_env"][0]
    assert len(a["snapshot_sha256"]) == 64 and a["snapshot_sha256"] != b["snapshot_sha256"]
    assert a["snapshot"]["command_rad"][0] == pytest.approx(.2)
    assert b["snapshot"]["command_rad"][0] == pytest.approx(.201)
    assert first["source_sha256"] == "a" * 64


def test_snapshot_identity_covers_actual_link_and_jacobian_readback_not_only_active_q():
    env = native_env()
    controller = optimizer(env)
    reference = torch.zeros(1, 24)
    _, first = controller.refine(reference)
    env.world.hands[0].data.body_link_pose_w[0, 1, 0] += .001
    _, pose_changed = controller.refine(reference)
    env.world.hands[0].root_physx_view.get_jacobians()[0, 0, 0, 0] += .001
    _, jacobian_changed = controller.refine(reference)
    fingerprints = {diag["per_env"][0]["snapshot_sha256"]
                    for diag in (first, pose_changed, jacobian_changed)}
    assert len(fingerprints) == 3
    json.dumps(jacobian_changed, allow_nan=False)


@pytest.mark.parametrize("bad", ["shape26", "nan", "inf", "outside", "integer"])
def test_noncanonical_teacher_reference_is_rejected_before_planning(bad):
    controller = optimizer()
    action = torch.zeros(1, 24)
    if bad == "shape26":
        action = torch.zeros(1, 26)
    elif bad == "integer":
        action = action.long()
    else:
        action[0, 0] = {"nan": float("nan"), "inf": float("inf"), "outside": 1.01}[bad]
    with pytest.raises(ValueError):
        controller.refine(action)
    assert not controller.native.pending.any()


@pytest.mark.parametrize("source_sha", [None, "", "a" * 63, "g" * 64])
def test_reference_source_requires_actual_sha256_identity(source_sha):
    with pytest.raises(ValueError, match="source"):
        optimizer(source_sha=source_sha)


def test_observe_result_uses_native_prediction_checks_and_subset_reset():
    env = native_env(2)
    controller = optimizer(env)
    controller.refine(torch.zeros(2, 24))
    env.task.m["q"][0] += .006
    result = controller.observe_result(torch.tensor([False, True]))
    assert controller.native.bad_predictions.tolist() == [1, 0]
    assert controller.native.pending.tolist() == [True, False]
    assert result["prediction_checks"] == 1
    assert result["prediction_peak_rad"] == pytest.approx(.006)
    json.dumps(result, allow_nan=False)


def test_context_validation_rejects_misaligned_identity_before_planning():
    controller = optimizer(native_env(2))
    with pytest.raises(ValueError, match="context"):
        controller.refine(torch.zeros(2, 24), context={"episode": [1, 2, 3]})
    assert not controller.native.pending.any()
