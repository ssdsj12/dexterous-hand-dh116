"""Teacher-side RGB/privileged registration against the Student calibration."""
from pathlib import Path

import numpy as np
import pytest

from go2_pvcnn.control.dh116.tabletop_stereo_calibration import StereoCalibration
from go2_pvcnn.control.dh116 import tabletop_teacher_stereo_alignment as alignment


ROOT = Path(__file__).resolve().parents[1]


def scene(shift=(0, 0, 0)):
    calibration = StereoCalibration.from_json(ROOT / alignment.STUDENT_CAMERA)
    shift = np.array(shift)
    left = np.array(calibration.T_world_left)
    point = left[:3, 3] + left[:3, 2] + shift
    palms = np.array([[*point, 1, 0, 0, 0], [*(point + (0, .1, 0)), 1, 0, 0, 0]])
    state = dict(teacher_obs67=np.zeros(67).tolist(), rod_nodes_world_m=np.tile(point, (41, 1)).tolist(),
                 palm_poses_world_xyz_wxyz=palms.tolist())
    identity = dict(state_sha256=alignment.state_digest(state), calibration_sha256=calibration.sha256,
                    physics_tick_end=2, sdk_step=2, sdk_timestamp_s=2/480, simulation_timestamp=2/480)
    packet = dict(identity=identity,
                  render_timing=dict(physical_step=2, physical_timestamp_s=2/480,
                                     raw_left_clock={"frame_id": 7}, raw_right_clock={"frame_id": 7}),
                  calibration_sha256=calibration.sha256)
    for side in ("left", "right"):
        t = np.array(getattr(calibration, "T_world_" + side))
        t[:3, 3] += shift
        packet["T_world_" + side] = t[None]
        packet["K_" + side] = np.array(getattr(calibration, "K_" + side))[None]
    return calibration, state, packet, shift


def test_projection_uses_student_optical_axes_and_both_world_palm_poses():
    cal, state, frame, shift = scene()
    result = alignment.aligned_payload(state, frame["identity"], frame, cal, shift[None])
    l, r = [result["cameras"][side] for side in ("left", "right")]
    assert l["rod_pixels_uv"][0] == pytest.approx([320, 240])
    assert l["rod_optical_z_m"][0] == pytest.approx(1)
    assert r["rod_pixels_uv"][0][0] == pytest.approx(320 - .08 * cal.K_right[0, 0].item())
    assert result["rod_nodes_palm_m"]["left"][0] == pytest.approx([0, 0, 0])
    assert result["rod_nodes_palm_m"]["right"][0] == pytest.approx([0, -.1, 0])
    assert result["teacher_obs67"] == state["teacher_obs67"]


def test_whole_environment_shift_preserves_image_and_local_labels():
    cal, state, frame, shift = scene()
    a = alignment.aligned_payload(state, frame["identity"], frame, cal, shift[None])
    cal, state, frame, shift = scene((.8, -2, .7))
    b = alignment.aligned_payload(state, frame["identity"], frame, cal, shift[None])
    np.testing.assert_allclose(a["cameras"]["right"]["rod_pixels_uv"],
                               b["cameras"]["right"]["rod_pixels_uv"], atol=1e-9)
    np.testing.assert_allclose(a["rod_nodes_palm_m"]["right"], b["rod_nodes_palm_m"]["right"])


@pytest.mark.parametrize("mismatch", ["state", "clock", "lens", "camera_frame"])
def test_mismatched_privilege_or_camera_packet_is_rejected(mismatch):
    cal, state, frame, shift = scene()
    if mismatch == "state":
        state["teacher_obs67"][0] += 1
    elif mismatch == "clock":
        frame["render_timing"]["physical_step"] += 1
    elif mismatch == "lens":
        frame["K_left"][0, 0, 0] += 1
    else:
        frame["render_timing"]["raw_right_clock"] = {"frame_id": 8}
    with pytest.raises(ValueError):
        alignment.aligned_payload(state, frame["identity"], frame, cal, shift[None])


def test_raw_geometry_must_decode_from_actual_teacher_observation():
    from types import SimpleNamespace
    import torch

    nodes = torch.zeros(1, 41, 3)
    origin = torch.tensor([[0., 0., 0.]])
    reference = torch.tensor([.43, 0., .368])
    indices = torch.tensor([0, 10, 20, 30, 40])
    palm = torch.tensor([[[.43, 0., .368, 1., 0., 0., 0.]]])
    hand = SimpleNamespace(data=SimpleNamespace(body_link_pose_w=palm))
    obs = torch.zeros(1, 67)
    obs[:, 30:38] = palm[0, 0, 3:].repeat(2)
    obs[:, 48:63] = ((nodes[:, indices] - origin[:, None] - reference) * 5).flatten(1)
    runtime = SimpleNamespace(rope_pos_view=nodes, env_origins=origin, wire_ref_center=reference,
        wire_nodes_idx=indices, get_obs=lambda: obs, left_hands=hand, right_hands=hand,
        l_base_id=0, r_base_id=0)
    assert alignment.runtime_state(runtime)["teacher_wire_node_indices"] == indices.tolist()
    obs[0, 48] += .01
    with pytest.raises(ValueError, match="differs from frozen Teacher"):
        alignment.runtime_state(runtime)


def test_independent_camera_plan_pins_student_lens_and_rejects_old_digest(tmp_path):
    from scripts import diagnose_dh116_teacher_stereo_alignment as worker

    plan = worker.freeze_plan(ROOT, tmp_path/"out", worker.legacy.BINDINGS)
    assert plan["teacher_stereo"]["calibration_sha256"] == StereoCalibration.from_json(
        ROOT/alignment.STUDENT_CAMERA).sha256
    assert plan["teacher_stereo"]["capture_pairs"] == 1
    assert plan["budget"]["ppo_updates"] == plan["budget"]["solver_steps"] == 0
    worker.verify_plan(plan, worker.legacy.digest(plan))
    with pytest.raises(ValueError, match="fingerprint"):
        worker.verify_plan(plan, "0"*64)
