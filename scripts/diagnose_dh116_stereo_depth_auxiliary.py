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


def _optimize_archived_frame(model, left, right, image_valid, target, mask, steps):
    """Fit the same CPU frame for a fixed diagnostic budget without persisting weights."""
    sides = ("left", "right")
    supervised = mask.any(dim=(0, 2, 3)).tolist()
    parameters = list(model.encoder.parameters())
    heads = [tuple(head.parameters()) for head in model.project_depth]
    before = [[parameter.detach().clone() for parameter in head] for head in heads]
    backbone = next(model.encoder.backbone.parameters())
    before_backbone = backbone.detach().clone()
    for has_labels, head in zip(supervised, heads, strict=True):
        if has_labels:
            parameters.extend(head)
    learning_rate = 0.01
    optimizer = torch.optim.SGD(parameters, lr=learning_rate)
    losses = []
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        prediction = model(left, right, image_valid)
        loss = depth_auxiliary_loss(prediction, target)
        if not torch.isfinite(loss):
            raise ValueError("non-finite depth auxiliary diagnostic loss")
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = model(left, right, image_valid)
        losses.append(float(depth_auxiliary_loss(final, target)))
        head_changes = {
            side: max(float((p - old).abs().max()) for p, old in zip(head, initial, strict=True))
            for side, head, initial in zip(sides, heads, before, strict=True)
        }
        final_mask = final.valid & target.valid
        return {
            "scope": "same_archived_frame_optimizer_diagnostic",
            "optimizer": "SGD",
            "learning_rate": learning_rate,
            "same_frame_loss_m": losses,
            "head_parameter_max_abs_change": head_changes,
            "shared_backbone_first_parameter_max_abs_change": float((backbone - before_backbone).abs().max()),
            "post_update_environments": [
                {
                    side: _metrics(final.depth_m[index, view], target.depth_m[index, view], final_mask[index, view])
                    for view, side in enumerate(sides)
                }
                for index in range(len(left))
            ],
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--proof-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimizer-steps", type=int, default=0,
                        help="optional same-frame CPU diagnostic updates, 0–8, without saving a model")
    args = parser.parse_args(argv)
    if not 0 <= args.optimizer_steps <= 8:
        parser.error("optimizer steps must be between 0 and 8")
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
    image_valid = torch.ones((len(left), 2), dtype=torch.bool)
    prediction = model(left, right, image_valid)
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
    if args.optimizer_steps:
        report["mode"] = "archived_native_reference_cpu_optimizer_diagnostic"
        report["optimization"] = _optimize_archived_frame(
            model, left.detach(), right.detach(), image_valid, target, mask, args.optimizer_steps)
        report["optimizer_updates"] = args.optimizer_steps
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
