import math

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import project_policy_wrist_targets
from go2_pvcnn.control.dh116.tabletop_free_wrist_grasp import compose_rotation


def inputs():
    seed = torch.tensor([[[.2, .1, .36, 1., 0., 0., 0.],
                          [.2, -.1, .40, 1., 0., 0., 0.]]], dtype=torch.float64)
    roots = seed.clone()
    roots[:, :, 2] += .012
    return dict(root_seed_w=seed, measured_base_w=roots, common_lift_offset_m=torch.tensor([.01]).double(),
        measured_clearance_m=torch.tensor([[.001, .012]]).double(),
        palm_offsets_m=torch.tensor([[[0., 0., -.001], [0., 0., .002]]]).double(),
        requested_palm_offsets_m=torch.tensor([[[0., 0., -.00102], [0., 0., .00198]]]).double(),
        angles_rad=torch.zeros((1, 2, 3), dtype=torch.float64),
        requested_angles_rad=torch.tensor([[[.0005, 0., 0.], [0., 0., -.0005]]]).double(),
        allowed=torch.tensor([True]), live=torch.tensor([True]))


def test_preserve_per_hand_safe_z_and_cancel_queued_descent():
    x = inputs()
    packet = project_policy_wrist_targets(**x)
    expected = x["measured_base_w"][0, 0, 2] + .0021 - x["measured_clearance_m"][0, 0]
    assert packet["projected_root_poses"][0, 0, 2] == pytest.approx(float(expected))
    assert packet["projected_root_poses"][0, 1, 2] == pytest.approx(.41198)
    assert packet["safe_z_clipped"].tolist() == [[True, False]]
    torch.testing.assert_close(packet["projected_root_poses"][..., 3:],
                               compose_rotation(x["root_seed_w"][..., 3:], x["requested_angles_rad"]))


def test_live_but_inhibited_still_cancels_queued_descent():
    x = inputs()
    x["allowed"][:] = False
    result = project_policy_wrist_targets(**x)
    assert result["safe_z_clipped"].tolist() == [[True, False]]
    torch.testing.assert_close(result["projected_angles"], x["angles_rad"])
    assert result["projected_root_poses"][0, 1, 2] == pytest.approx(.412)


def test_inactive_rows_do_not_raise_or_move():
    x = inputs()
    x["live"][:] = False
    result = project_policy_wrist_targets(**x)
    assert not result["safe_z_clipped"].any()
    torch.testing.assert_close(result["palm_offsets_m"], x["palm_offsets_m"])
    torch.testing.assert_close(result["projected_angles"], x["angles_rad"])


def test_original_slew_limit_is_vector_norm():
    x = inputs()
    x["measured_clearance_m"][:] = .02
    x["requested_palm_offsets_m"] = x["palm_offsets_m"] + .02
    result = project_policy_wrist_targets(**x)
    delta = result["palm_offsets_m"] - x["palm_offsets_m"]
    assert delta.norm(dim=-1).max() == pytest.approx(.01 / 480)


@pytest.mark.parametrize("field,update", [
    ("measured_clearance_m", lambda x: x.fill_(float("nan"))),
    ("root_seed_w", lambda x: x[..., 3:].zero_()),
    ("measured_base_w", lambda x: x[..., 3:].zero_()),
    ("requested_angles_rad", lambda x: x.fill_(math.pi)),
    ("live", lambda x: x.to(torch.float64)),
])
def test_invalid_input_rejected(field, update):
    args = inputs()
    args[field] = update(args[field])
    with pytest.raises(ValueError):
        project_policy_wrist_targets(**args)


def test_floor_matches_original_task_and_safety_precedes_slew():
    import json
    from pathlib import Path
    from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M

    root = Path(__file__).resolve().parents[1]
    args = json.loads((root / "configs/dh116_training/stereo_native_task_v1.json").read_text())["arguments"]
    assert args[args.index("--palm-exploration-axis") + 1] == "xyz"
    assert CLEARANCE_M == .0021
    x = inputs()
    result = project_policy_wrist_targets(**x)
    assert (result["palm_offsets_m"] - x["palm_offsets_m"])[0, 0, 2] > .01 / 480
    assert result["projection_reasons"][0, 0].item() & 2
    assert not result["native_clearance_qualified"]
