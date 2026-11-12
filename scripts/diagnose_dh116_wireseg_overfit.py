#!/usr/bin/env python3
"""Fixed-data learning diagnostic; never grants Stereo or policy qualification."""

import argparse
import json
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.pretrain_dh116_wireseg import (
    verified_sources, prepare_depth, read_subset, training_sample, save_training,
)
from scripts.relay_wireseg_blobs import digest_file
from go2_pvcnn.control.dh116.tabletop_wireseg_pretraining import (
    load_wireseg_sample, WireSegPretrainer, pretraining_losses,
)


def make_view(sample, mode, size, device):
    if mode == "center":
        return training_sample(sample, size, device)
    if mode != "full_field_resize":
        raise ValueError("unknown diagnostic view")
    rgb = torch.from_numpy(sample.rgb.copy()).permute(2, 0, 1)[None].to(device).float() / 255
    rgb = F.interpolate(rgb, (size, size), mode="bilinear", align_corners=False, antialias=True)
    masks = torch.from_numpy(sample.instance_masks.copy()).to(device)
    masks = (F.interpolate(masks.float()[None], (size, size), mode="nearest")[0].bool()
             if len(masks) else torch.empty((0, size, size), dtype=torch.bool, device=device))
    depth = valid = None
    if sample.depth_m is not None:
        depth = torch.from_numpy(sample.depth_m.copy())[None, None].to(device)
        valid = torch.from_numpy(sample.depth_valid.copy())[None, None].to(device)
        depth = F.interpolate(depth, (size, size), mode="nearest")[:, 0]
        valid = F.interpolate(valid.float(), (size, size), mode="nearest")[:, 0].bool()
    return rgb, masks, depth, valid


def error_metrics(predicted, target, valid):
    valid = valid & np.isfinite(target) & (target > 0)
    if not np.isfinite(predicted[valid]).all():
        raise ValueError("nonfinite prediction at valid depth label")
    error = (predicted[valid] - target[valid]).astype(np.float64)
    return dict(valid_pixels=int(error.size), mae_m=float(np.abs(error).mean()) if error.size else None,
                rmse_m=float(np.sqrt(np.square(error).mean())) if error.size else None,
                p95_m=float(np.percentile(np.abs(error), 95)) if error.size else None)


def depth_metrics(predicted, target, valid, roi):
    return {name: error_metrics(predicted, target, valid & mask)
            for name, mask in (("all_valid", np.ones_like(valid)), ("wire_roi", roi))}


def diagnostic_loss(prediction, masks, depth, valid, mode):
    if mode == "depth-only":
        if depth is None or valid is None or not valid.any():
            raise ValueError("depth-only requires verified paired valid depth")
        valid = valid & torch.isfinite(depth) & (depth > 0)
        return F.smooth_l1_loss(prediction["metric_depth"][valid], depth[valid], beta=.01)
    if mode != "seg-only":
        raise ValueError("unknown head diagnostic")
    terms = pretraining_losses(prediction, [masks], None, None)
    return terms["total"]


def has_supervision(view, mode):
    if mode == "seg-only":
        return True  # Empty target is a real negative segmentation label.
    depth, valid = view[2:]
    return depth is not None and valid is not None and bool(
        (valid & torch.isfinite(depth) & (depth > 0)).any())


def validate_fixed_cohort(identities):
    if not 8 <= len(identities) <= 16:
        raise ValueError("fixed diagnostic requires 8 to 16 images")
    if len({entry["file_name"] for entry in identities}) != len(identities):
        raise ValueError("diagnostic identities must be unique")
    if not all(entry["pair_sha256"] for entry in identities):
        raise ValueError("diagnostic requires exact verified RGB-depth pairs")
    return identities


def constant_depth(arrays):
    values = np.concatenate([x[np.isfinite(x) & (x > 0)].ravel() for x in arrays])
    if not len(values):
        raise ValueError("no valid training depth for constant baseline")
    return float(np.median(values))


def quantiles(values):
    valid = np.asarray(values)
    valid = valid[np.isfinite(valid)]
    return {str(p): float(np.percentile(valid, p)) if valid.size else None for p in (0, 5, 50, 95, 100)}


@torch.inference_mode()
def evaluate(model, samples, view, size, device, baseline):
    model.eval()
    records, predictions, targets, validity, rois = [], [], [], [], []
    intersections = unions = target_count = predicted_count = pixels = 0
    for sample in samples:
        rgb, masks, depth, valid = make_view(sample, view, size, device)
        result = model(rgb)
        truth = masks.any(0)
        predicted = result["visible"][0] > 0
        intersection = int((predicted & truth).sum())
        union = int((predicted | truth).sum())
        intersections += intersection
        unions += union
        target_count += int(truth.sum())
        predicted_count += int(predicted.sum())
        pixels += truth.numel()
        row = dict(file_name=sample.file_name, family=sample.scene_family, pair_sha256=sample.pair_sha256,
                   view=view, gt_pixels=int(truth.sum()), predicted_pixels=int(predicted.sum()),
                   empty_crop=not bool(truth.any()), iou=intersection/union if union else None)
        if depth is not None:
            p, t, v, r = (x.detach().cpu().numpy().ravel()
                          for x in (result["metric_depth"], depth, valid, truth))
            row.update(depth=depth_metrics(p, t, v, r),
                       target_depth_quantiles_m=quantiles(t[v]),
                       predicted_depth_quantiles_m=quantiles(p[v]),
                       valid_depth_pixels=int(v.sum()))
            predictions.append(p); targets.append(t); validity.append(v); rois.append(r)
        records.append(row)
    depth_result = constant_result = None
    if targets:
        p, t, v, r = [np.concatenate(x) for x in (predictions, targets, validity, rois)]
        depth_result = depth_metrics(p, t, v, r)
        constant_result = depth_metrics(np.full_like(t, baseline), t, v, r)
    return dict(samples=records, rgb_count=len(records),
                empty_mask_count=sum(row["empty_crop"] for row in records),
                visible_iou=intersections/unions if unions else None,
                gt_coverage=target_count/pixels, predicted_coverage=predicted_count/pixels,
                depth=depth_result, training_constant_depth_m=baseline,
                constant_depth_metrics=constant_result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--wall-seconds", type=int, default=300)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=116)
    parser.add_argument("--heads", nargs="+", choices=("seg-only", "depth-only"), default=["seg-only", "depth-only"])
    args = parser.parse_args()
    if not (8 <= args.samples <= 16 and 1 <= args.steps <= 512 and 32 <= args.size <= 256
            and 1 <= args.wall_seconds <= 600):
        parser.error("bounded diagnostic: 8-16 images, <=512 steps/head, size32-256, <=600s/head")
    args.output.mkdir(parents=True, exist_ok=False)
    catalogue_sha, names = verified_sources(args.dataset_root)
    members = prepare_depth(args.dataset_root, names, True)
    train_rows = list(islice(read_subset(args.dataset_root, args.subset_manifest, names, catalogue_sha, "train"),
                             args.samples))
    train = [load_wireseg_sample(row, args.dataset_root / "metric_depth_npy", depth_members=members)
             for row in train_rows]
    identities = validate_fixed_cohort([dict(file_name=s.file_name, family=s.scene_family,
                                            pair_sha256=s.pair_sha256) for s in train])
    validation = [load_wireseg_sample(row, args.dataset_root / "metric_depth_npy", depth_members=members)
                  for row in read_subset(args.dataset_root, args.validation_manifest, names, catalogue_sha, "validation")]
    if {s.file_name for s in train} & {s.file_name for s in validation}:
        raise ValueError("train/validation identity leakage")
    baseline = constant_depth([s.depth_m[s.depth_valid] for s in train])
    contract = dict(source_sha256=digest_file(Path(__file__)), kind="wireseg-fixed-cohort-overfit-v1", fresh_model=True, strict_resume=False,
                    source_catalogue_sha256=catalogue_sha, subset_sha256=digest_file(args.subset_manifest),
                    validation_sha256=digest_file(args.validation_manifest), identities=identities,
                    size=args.size, optimization_view="center", learning_rate=args.learning_rate,
                    seed=args.seed, max_updates_per_head=args.steps, wall_seconds_per_head=args.wall_seconds,
                    labels_unit="metre_from_pinned_dataset", depth_semantics="source_metric_depth_not_assumed_optical_z",
                    real_test_used=False, stereo_qualified=False, policy_qualified=False,
                    generalization_qualified=False)
    (args.output / "frozen_cohort.json").write_text(json.dumps(contract, indent=2) + "\n")
    result = dict(contract=contract, heads={})
    device = torch.device(args.device)
    for mode in args.heads:
        head_dir = args.output / mode
        head_dir.mkdir()
        torch.manual_seed(args.seed)
        model = WireSegPretrainer().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        before = {view: evaluate(model, train, view, args.size, device, baseline)
                  for view in ("center", "full_field_resize")}
        steps, start, skipped = [], time.monotonic(), 0
        if not any(has_supervision(make_view(s, "center", args.size, device), mode) for s in train):
            raise ValueError("fixed cohort has no valid labels for requested head")
        with (head_dir / "updates.jsonl").open("x") as stream:
            for index in range(args.steps * len(train)):
                if len(steps) >= args.steps or time.monotonic() - start >= args.wall_seconds:
                    break
                sample = train[index % len(train)]
                model.train()
                rgb, masks, depth, valid = make_view(sample, "center", args.size, device)
                if not has_supervision((rgb, masks, depth, valid), mode):
                    skipped += 1
                    stream.write(json.dumps(dict(visit=index+1, file_name=sample.file_name,
                        updated=False, reason="no_valid_depth_label", updates_completed=len(steps))) + "\n")
                    stream.flush()
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss = diagnostic_loss(model(rgb), masks, depth, valid, mode)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite diagnostic loss")
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                grad_norm = torch.sqrt(sum(g.square().sum() for g in gradients))
                if not torch.isfinite(grad_norm) or grad_norm <= 0:
                    raise ValueError("zero or nonfinite diagnostic gradient")
                optimizer.step()
                record = dict(update=len(steps)+1, visit=index+1, updated=True, file_name=sample.file_name,
                              pair_sha256=sample.pair_sha256, loss=float(loss.detach()),
                              grad_norm=float(grad_norm), empty_mask=not bool(masks.any()),
                              elapsed_s=time.monotonic()-start)
                steps.append(record)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
        optimization_wall_s = time.monotonic()-start
        save_training(head_dir / "diagnostic.pt", model, optimizer, device,
                      {**contract, "mode": mode}, steps, {"rgb_updates": len(steps)})
        after = {view: evaluate(model, train, view, args.size, device, baseline)
                 for view in ("center", "full_field_resize")}
        heldout = {view: evaluate(model, validation, view, args.size, device, baseline)
                   for view in ("center", "full_field_resize")}
        result["heads"][mode] = dict(before=before, after=after, validation=heldout,
            completed_updates=len(steps), skipped_unlabelled_visits=skipped, optimization_wall_s=optimization_wall_s,
            max_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            checkpoint_sha256=digest_file(head_dir / "diagnostic.pt"),
            checkpoint=str(head_dir / "diagnostic.pt"), loss_first=steps[0]["loss"] if steps else None,
            loss_last=steps[-1]["loss"] if steps else None, stop_reason="budget" if len(steps)==args.steps else "wall_clock")
        (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        del optimizer, model
    print(json.dumps({m: {"completed_updates": r["completed_updates"],
                         "train_iou": r["after"]["center"]["visible_iou"],
                         "train_depth": r["after"]["center"]["depth"]}
                      for m,r in result["heads"].items()}))


if __name__ == "__main__":
    main()
