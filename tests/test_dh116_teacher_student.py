import json
import numpy as np
import pytest

from go2_pvcnn.control.dh116.teacher_student import (
    LinearImitationPolicy, make_transition, qualify_proxy_lift,
    score_physics_outcome, write_replay,
)
from go2_pvcnn.control.dh116.mpc_feedback import (
    advance_progress, initial_active_action, peak_body_force, to_numpy,
    top_down_hand_pose, trajectory_action_index,
)


def test_replay_schema_and_hash(tmp_path):
    item = make_transition(obs_student=np.zeros(13), obs_teacher_privileged=np.zeros(14),
                           previous_action=np.zeros(12), mpc_action=np.ones(12),
                           action=np.ones(12), reward=1.0, done=False, phase="grasp")
    manifest = write_replay(tmp_path / "teacher.jsonl", [item], role="teacher")
    assert manifest["action_dim"] == 12
    row = json.loads((tmp_path / "teacher.jsonl").read_text())
    assert row["phase"] == "grasp"


def test_student_checkpoint_loads_and_predicts_bounded_active_targets(tmp_path):
    import hashlib

    payload = {
        "role": "student", "kind": "linear_imitation_checkpoint",
        "input_dim": 25, "output_dim": 12,
        "weights": np.zeros((26, 12), dtype=np.float32).tolist(),
        "arm_state_included": False,
    }
    payload["weights"][-1] = [-10.0] * 12
    canonical = json.dumps(payload, sort_keys=True).encode()
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    checkpoint = tmp_path / "student_checkpoint.json"
    checkpoint.write_text(json.dumps(payload))

    policy = LinearImitationPolicy.load(checkpoint, expected_role="student")
    prediction = policy.predict(np.zeros((2, 25), dtype=np.float32))

    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    lower, _ = active_limits()
    assert prediction.shape == (2, 12)
    np.testing.assert_allclose(prediction, np.tile(lower, 2)[None, :].repeat(2, axis=0))


def test_student_trajectory_checkpoint_interpolates_plan_and_uses_previous_action(tmp_path):
    import hashlib

    from go2_pvcnn.control.dh116.linear_coupling import active_limits
    lower, upper = active_limits()
    start = np.tile((lower + upper) * 0.25, 2)
    finish = np.tile((lower + upper) * 0.75, 2)
    payload = {
        "role": "student", "kind": "trajectory_imitation_checkpoint",
        "input_dim": 25, "output_dim": 12,
        "progress_knots": [0.0, 1.0],
        "actions": [start.tolist(), finish.tolist()],
        "previous_smoothing": 0.25, "arm_state_included": False,
    }
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    checkpoint = tmp_path / "student_trajectory_checkpoint.json"
    checkpoint.write_text(json.dumps(payload))
    policy = LinearImitationPolicy.load(checkpoint, expected_role="student")

    previous = np.tile((lower + upper) * 0.5, 2).astype(np.float32)
    observation = np.concatenate((np.zeros(12), previous, [0.5])).astype(np.float32)
    prediction = policy.predict(observation)

    expected = 0.75 * ((start + finish) * 0.5) + 0.25 * previous
    np.testing.assert_allclose(prediction, expected, atol=1e-6)

    lift_observation = np.concatenate((np.zeros(12), previous, [1.0])).astype(np.float32)
    lift_target = policy.predict(lift_observation)
    np.testing.assert_allclose(lift_target, 0.75 * finish + 0.25 * previous, atol=1e-6)


def test_integrated_hand_adapter_can_run_student_checkpoint(tmp_path):
    import hashlib
    from go2_pvcnn.control.dh116.integrated_hand_mpc import IntegratedHandMPC

    payload = {
        "role": "student", "kind": "linear_imitation_checkpoint",
        "input_dim": 25, "output_dim": 12,
        "weights": np.zeros((26, 12), dtype=np.float32).tolist(),
        "arm_state_included": False,
    }
    payload["weights"][-1] = [0.2] * 12
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    checkpoint = tmp_path / "student_checkpoint.json"
    checkpoint.write_text(json.dumps(payload))
    policy = LinearImitationPolicy.load(checkpoint, expected_role="student")
    controller = IntegratedHandMPC(policy)

    action = controller.plan(np.zeros((2, 7)), np.full((2, 6), 0.1), 0.5,
                             np.full(12, 0.15))

    assert action.shape == (2, 6)
    np.testing.assert_allclose(action, np.full((2, 6), 0.2), atol=1e-6)


def test_replay_rejects_arm_state_and_bad_phase():
    with pytest.raises(ValueError, match="Panda"):
        make_transition(obs_student={"arm": [0]}, obs_teacher_privileged=np.zeros(1),
                        previous_action=np.zeros(12), mpc_action=np.zeros(12),
                        action=np.zeros(12), reward=0, done=False, phase="lift")
    with pytest.raises(ValueError, match="phase"):
        make_transition(obs_student=np.zeros(1), obs_teacher_privileged=np.zeros(1),
                        previous_action=np.zeros(12), mpc_action=np.zeros(12),
                        action=np.zeros(12), reward=0, done=False, phase="approach")


def test_physics_feedback_is_saved_and_lift_requires_stable_contact():
    item = make_transition(obs_student=np.zeros(25), obs_teacher_privileged=np.zeros(26),
                           previous_action=np.zeros(12), mpc_action=np.ones(12),
                           action=np.ones(12), reward=0.5, done=False, phase="grasp",
                           contact_force_n=4.0, wire_lift_m=0.025, physics_valid=True)
    assert item.contact_force_n == pytest.approx(4.0)
    assert item.wire_lift_m == pytest.approx(0.025)
    assert item.physics_valid is True
    assert score_physics_outcome(contact_force_n=4.0, wire_lift_m=0.025) == (True, 1.0)
    assert score_physics_outcome(contact_force_n=360.0, wire_lift_m=0.03) == (False, -1.0)
    assert score_physics_outcome(contact_force_n=0.0, wire_lift_m=0.03) == (False, 0.0)
    assert not qualify_proxy_lift(contact_peak_n=4.0, wire_lift_m=0.03, abort_reason="wire_escape")
    assert qualify_proxy_lift(contact_peak_n=4.0, wire_lift_m=0.03, abort_reason=None)


def test_online_mpc_progress_reacts_to_measured_contact():
    assert advance_progress(0.2, 0.0, contact_force_n=0.0) == pytest.approx((0.28, 0.0))
    assert advance_progress(0.4, 0.0, contact_force_n=0.2) == pytest.approx((0.48, 0.0))
    assert advance_progress(0.4, 0.0, contact_force_n=5.0) == pytest.approx((0.48, 0.0))
    assert advance_progress(0.4, 0.2, contact_force_n=40.0) == pytest.approx((0.32, 0.2))
    assert advance_progress(0.4, 0.2, contact_force_n=0.0,
                            contact_latched=True) == pytest.approx((0.48, 0.2))


def test_mpc_feedback_converts_cuda_plan_to_numpy():
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    result = to_numpy(torch.arange(12, device=device))
    np.testing.assert_array_equal(result, np.arange(12))


def test_hand_only_scene_uses_palm_down_pose_above_target():
    pose = top_down_hand_pose(-0.22, 0.55)
    np.testing.assert_allclose(pose[:3], [-0.22, 0.0, 0.55])
    np.testing.assert_allclose(pose[3:], [0.0, 1.0, 0.0, 0.0])


def test_initial_mpc_reference_matches_all_active_hand_joints():
    np.testing.assert_allclose(initial_active_action(0.1), np.full(12, 0.1))


def test_progress_selects_preplanned_grasp_then_lift_reference_index():
    assert trajectory_action_index(0.0, 0.0, grasp_steps=4, lift_steps=4) == 0
    assert trajectory_action_index(0.5, 0.0, grasp_steps=4, lift_steps=4) == 1
    assert trajectory_action_index(1.0, 0.0, grasp_steps=4, lift_steps=4) == 3
    assert trajectory_action_index(1.0, 0.5, grasp_steps=4, lift_steps=4) == 5
    assert trajectory_action_index(1.0, 1.0, grasp_steps=4, lift_steps=4) == 7


def test_contact_force_diagnostic_identifies_responsible_hand_link():
    name, force = peak_body_force(["base_link", "finger13_link"],
                                  np.asarray([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]))
    assert name == "finger13_link"
    assert force == pytest.approx(5.0)


def test_non_pad_link_collision_filter_preserves_tip_collision():
    pytest.importorskip("pxr.Usd")
    from pxr import Usd, UsdGeom, UsdPhysics
    from go2_pvcnn.control.dh116.hand_collisions import disable_nonpad_collisions
    stage = Usd.Stage.CreateInMemory()
    prototype32 = UsdGeom.Xform.Define(stage, "/Prototype32")
    UsdGeom.Xform.Define(stage, "/Prototype32/finger32_link")
    prototype33 = UsdGeom.Xform.Define(stage, "/Prototype33")
    UsdGeom.Xform.Define(stage, "/Prototype33/finger33_link")
    collision = UsdGeom.Cube.Define(stage, "/Prototype32/finger32_link/collisionShape")
    tip_collision = UsdGeom.Cube.Define(stage, "/Prototype33/finger33_link/collisionShape")
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    UsdPhysics.CollisionAPI.Apply(tip_collision.GetPrim())
    for link, prototype in (("finger32_link", prototype32), ("finger33_link", prototype33)):
        branch = UsdGeom.Xform.Define(stage, f"/Hand/{link}/collisions").GetPrim()
        branch.GetReferences().AddInternalReference(prototype.GetPath())
        branch.SetInstanceable(True)
    disabled = disable_nonpad_collisions(stage, "/Hand")
    assert disabled == ("/Hand/finger32_link/collisions/finger32_link/collisionShape",)
    assert not UsdPhysics.CollisionAPI(stage.GetPrimAtPath(disabled[0])).GetCollisionEnabledAttr().Get()
    tip = stage.GetPrimAtPath("/Hand/finger33_link/collisions/finger33_link/collisionShape")
    assert UsdPhysics.CollisionAPI(tip).GetCollisionEnabledAttr().Get()
