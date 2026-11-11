"""Replay the opt-in RGB depth auxiliary head on one archived frame using CPU."""
import argparse
import json
from pathlib import Path

import torch

from go2_pvcnn.control.dh116.tabletop_stereo_contract import file_sha256
from go2_pvcnn.control.dh116.tabletop_stereo_depth_auxiliary import (
    StereoDepthAuxiliary,
    depth_auxiliary_loss,
    native_depth_targets,
)
from go2_pvcnn.control.dh116.tabletop_stereo_encoder import StereoEncoder
from go2_pvcnn.control.dh116.tabletop_stereo_error_cli import load_capture


def _metrics(predicted, reference, valid):
    difference = (predicted[valid] - reference[valid]).detach().abs()
    if not difference.numel():
        return {"supervised_samples": 0, "mae_m": None, "rmse_m": None, "p95_m": None}
    return {
        "supervised_samples": difference.numel(),
        "mae_m": float(difference.mean()),
        "rmse_m": float(difference.square().mean().sqrt()),
        "p95_m": float(torch.quantile(difference, 0.95)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--proof-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if file_sha256(args.proof) != args.proof_sha256:
        raise ValueError("capture proof SHA mismatch")
    capture, references, calibration, config, proof = load_capture(args.proof)
    frame = capture["frames"][0]
    if frame.get("native_evidence") is not True or frame.get("render_passes") != 1:
        raise ValueError("first frame requires synchronized native RGB evidence")
    depth_range = {
        "min_depth_m": config["min_depth_m"],
        "max_depth_m": config["max_depth_m"],
    }
    torch.manual_seed(0)
    encoder = StereoEncoder(image_processing_batch_size=1)
    model = StereoDepthAuxiliary(encoder, **depth_range).eval()
    left = frame["left_rgb"].detach().requires_grad_()
    right = frame["right_rgb"].detach().requires_grad_()
    prediction = model(left, right, torch.ones((len(left), 2), dtype=torch.bool))
    target = native_depth_targets(frame, references[0], **depth_range)
    loss = depth_auxiliary_loss(prediction, target)
    loss.backward()
    mask = prediction.valid & target.valid
    report = {
        "mode": "archived_native_reference_cpu_gradient_diagnostic",
        "capture_proof_sha256": args.proof_sha256,
        "capture_artifact_sha256": proof["capture_artifact"]["sha256"],
        "reference_artifact_sha256": proof["reference_artifacts"][0]["sha256"],
        "cohort_sha256": proof["cohort_sha256"],
        "calibration_sha256": calibration.sha256,
        "frame_id": frame["frame_id"],
        "simulation_time_s": frame["simulation_time"],
        "rgb_hw": list(left.shape[-2:]),
        "grid_hw": list(prediction.depth_m.shape[-2:]),
        "checkpoint_saved": False,
        "optimizer_updates": 0,
        "native_physics_steps": 0,
        "deployment_qualified": False,
        "reference_used_as_input": False,
        "loss_m": float(loss.detach()),
        "gradient_l2": {
            "left_rgb": float(left.grad.norm()),
            "right_rgb": float(right.grad.norm()),
            "shared_backbone_first_parameter": float(next(encoder.backbone.parameters()).grad.norm()),
            "left_head_weight": float(model.project_depth[0].weight.grad.norm()),
            "right_head_weight": float(model.project_depth[1].weight.grad.norm()),
        },
        "environments": [
            {
                "identity": identity,
                **{
                    side: _metrics(prediction.depth_m[index, view], target.depth_m[index, view], mask[index, view])
                    for view, side in enumerate(("left", "right"))
                },
            }
            for index, identity in enumerate(frame["identity"])
        ],
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
