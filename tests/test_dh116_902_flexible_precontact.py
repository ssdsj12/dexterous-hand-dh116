import numpy as np
import pytest

from scripts import screen_dh116_902_flexible_precontact as route


def test_target_pad_material_sides_are_checked_on_every_approach_sample():
    samples = [{"node_ids":[35, 36]}, {"node_ids":[35, 19]}]
    assert not route.material_side_matches(samples, "left")
    assert route.material_side_matches([{"node_ids":[2, 3, 4]}], "right")
    assert not route.material_side_matches([{"node_ids":[20, 21]}], "left")


def test_settled_seed_restores_native_joint_targets_and_zero_wrist_carriage():
    from go2_pvcnn.control.dh116.linear_coupling import ACTIVE
    names = ["wrist_x", *reversed(ACTIVE)]
    q = np.array([[0., 1., 2., 3., 4., 5., 6.]])
    pose = (np.zeros(3), np.array([0., 0., 0., 1.]))
    row = dict(rod={"physics_step":240},native_joint_state={s:dict(names=names,position=q,target=q+.1) for s in ("left","right")},
               hand_link_poses={s+".base_link":pose for s in ("left","right")})
    config=dict(hands={s:dict(pose_wxyz=[0.,0.,0.,1.,0.,0.,0.]) for s in ("left","right")})
    state=route.settled_control_state(row,config)
    np.testing.assert_array_equal(state["wrist_projection"]["projected_angles"], np.zeros((1,2,3)))
    hand=state["action_readback"]["post"]["hands"]["right"]
    assert [hand["joint_names"][i] for i in hand["active_indices"]]==list(ACTIVE)
    assert hand["backend_targets"][0, hand["active_indices"][0]]==pytest.approx(6.1)


def test_frozen_proof_cannot_be_reused_for_a_different_pose(monkeypatch, tmp_path):
    import json

    path = tmp_path / "goals.json"
    proof = tmp_path / "proof.json"
    receipt = tmp_path / "outer.json"
    receipt.write_text("{}")
    target = dict(left={}, right={})
    proof.write_text(json.dumps(dict(all_passed=True, grasp="pinch", goal_sha256="wrong",
        outer_sha256=route.reset.sha(receipt), before_sdk=240)))
    path.write_text(json.dumps(dict(goals={"pinch":target}, screen_sha256={"pinch":route.reset.sha(proof)})))
    monkeypatch.setattr(route, "GOALS", path)
    monkeypatch.setattr(route, "load_settled", lambda *args: ([], ()))
    with pytest.raises(ValueError, match="passing sealed 22-envelope waypoint"):
        route.load_approach(receipt, proof, "pinch", {})
