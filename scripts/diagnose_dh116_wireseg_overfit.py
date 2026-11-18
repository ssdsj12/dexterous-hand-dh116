#!/usr/bin/env python3
"""Fixed-data learning diagnostic; never grants Stereo or policy qualification."""

import argparse
import json
import math
import random
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.pretrain_dh116_wireseg import (
    verified_sources, prepare_depth, read_subset, training_sample,
)
from scripts.relay_wireseg_blobs import digest_file
from scripts.wireseg_checkpoint_state import (
    audit_checkpoint, canonical_sha256, capture_rng, read_checkpoint, require_v2_state, restore_diagnostic_checkpoint,
    restore_rng, save_diagnostic_checkpoint,
)
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


def sample_identity(sample):
    return dict(file_name=sample.file_name, family=sample.scene_family, pair_sha256=sample.pair_sha256)


def diagnostic_training_config(args, mode, device):
    scripts = Path(__file__).resolve().parent
    source = scripts.parent / "src/go2_pvcnn/control/dh116"
    implementation = (
        scripts / "diagnose_dh116_wireseg_overfit.py", scripts / "pretrain_dh116_wireseg.py",
        scripts / "wireseg_checkpoint_state.py", source / "tabletop_wireseg_pretraining.py",
        source / "tabletop_stereo_encoder.py",
    )
    return dict(mode=mode, device=str(device), model="WireSegPretrainer", optimizer="AdamW",
                learning_rate=args.learning_rate, size=args.size, seed=args.seed, optimization_view="center",
                sampler="fixed_order_cyclic_v1", torch_version=str(torch.__version__), numpy_version=np.__version__,
                torch_threads=torch.get_num_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
                cuda_version=torch.version.cuda if device.type == "cuda" else None,
                implementation_sha256={path.name: digest_file(path) for path in implementation})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--subset-manifest", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=128, help="maximum new optimizer updates per head")
    parser.add_argument("--wall-seconds", type=int, default=300)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=.0003)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=116)
    parser.add_argument("--heads", nargs="+", choices=("seg-only", "depth-only"), default=["seg-only", "depth-only"])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--audit-checkpoint", type=Path, help="read metadata and list strict-resume gaps")
    action.add_argument("--resume", type=Path, help="resume a complete v2 checkpoint into a new output")
    parser.add_argument("--resume-sha256", help="SHA256 of the source checkpoint bytes")
    args = parser.parse_args()
    if args.audit_checkpoint is not None:
        if args.resume_sha256 is not None:
            parser.error("--resume-sha256 requires --resume")
        print(json.dumps(audit_checkpoint(args.audit_checkpoint), indent=2))
        return
    if not all((args.dataset_root, args.subset_manifest, args.validation_manifest, args.output)):
        parser.error("training requires dataset root, both manifests and a new output directory")
    if not (8 <= args.samples <= 16 and 1 <= args.steps <= 512 and 32 <= args.size <= 256
            and 1 <= args.wall_seconds <= 600 and 0 <= args.seed <= 2**32 - 1
            and math.isfinite(args.learning_rate) and args.learning_rate > 0):
        parser.error("bounded diagnostic: 8-16 images, <=512 steps/head, size32-256, <=600s/head, valid seed and LR")
    if len(set(args.heads)) != len(args.heads):
        parser.error("diagnostic heads must be unique")
    if (args.resume is None) != (args.resume_sha256 is None) or args.resume is not None and len(args.heads) != 1:
        parser.error("strict resume requires one explicit --heads value, --resume and --resume-sha256")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    source_state = source_sha = None
    if args.resume is not None:
        source_state, source_sha = read_checkpoint(args.resume, args.resume_sha256)
        require_v2_state(source_state)
    catalogue_sha, names = verified_sources(args.dataset_root)
    members = prepare_depth(args.dataset_root, names, True)
    train_rows = list(islice(read_subset(args.dataset_root, args.subset_manifest, names, catalogue_sha, "train"),
                             args.samples))
    train = [load_wireseg_sample(row, args.dataset_root / "metric_depth_npy", depth_members=members)
             for row in train_rows]
    identities = validate_fixed_cohort([sample_identity(sample) for sample in train])
    validation = [load_wireseg_sample(row, args.dataset_root / "metric_depth_npy", depth_members=members)
                  for row in read_subset(args.dataset_root, args.validation_manifest, names, catalogue_sha, "validation")]
    if {s.file_name for s in train} & {s.file_name for s in validation}:
        raise ValueError("train/validation identity leakage")
    baseline = constant_depth([s.depth_m[s.depth_valid] for s in train])
    dataset_manifest = dict(source_catalogue_sha256=catalogue_sha, subset_sha256=digest_file(args.subset_manifest),
                            validation_sha256=digest_file(args.validation_manifest), training_cohort=identities,
                            validation_cohort=[sample_identity(sample) for sample in validation])
    contract = dict(source_sha256=digest_file(Path(__file__)), kind="wireseg-fixed-cohort-overfit-v2",
                    fresh_model=source_state is None, strict_resume=source_state is not None,
                    resume_checkpoint_sha256=source_sha, dataset_manifest_sha256=canonical_sha256(dataset_manifest),
                    source_catalogue_sha256=catalogue_sha, subset_sha256=dataset_manifest["subset_sha256"],
                    validation_sha256=dataset_manifest["validation_sha256"], identities=identities,
                    size=args.size, optimization_view="center", learning_rate=args.learning_rate,
                    seed=args.seed, max_updates_per_head=args.steps, wall_seconds_per_head=args.wall_seconds,
                    labels_unit="metre_from_pinned_dataset", depth_semantics="source_metric_depth_not_assumed_optical_z",
                    real_test_used=False, stereo_qualified=False, policy_qualified=False,
                    generalization_qualified=False)
    result = dict(contract=contract, heads={})
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    for mode in args.heads:
        training_config = diagnostic_training_config(args, mode, device)
        head_contract = {**contract, "mode": mode, "training_config_sha256": canonical_sha256(training_config)}
        supervised = [has_supervision(make_view(sample, "center", args.size, device), mode) for sample in train]
        if not any(supervised):
            raise ValueError("fixed cohort has no valid labels for requested head")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = WireSegPretrainer().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        steps, visits = [], []
        if source_state is not None:
            steps, visits = restore_diagnostic_checkpoint(
                source_state, model, optimizer, device, dataset_manifest, training_config, supervised)
            source_state = None
        if mode == args.heads[0]:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "frozen_cohort.json").write_text(json.dumps(contract, indent=2) + "\n")
        head_dir = args.output / mode
        head_dir.mkdir()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rng_before_evaluation = capture_rng(device)
        before = {view: evaluate(model, train, view, args.size, device, baseline)
                  for view in ("center", "full_field_resize")}
        restore_rng(rng_before_evaluation, device)
        initial_step, initial_visit = len(steps), len(visits)
        start, skipped = time.monotonic(), 0
        with (head_dir / "updates.jsonl").open("x") as stream:
            for index in range(initial_visit, initial_visit + args.steps * len(train)):
                if len(steps) - initial_step >= args.steps or time.monotonic() - start >= args.wall_seconds:
                    break
                sample_index = index % len(train)
                sample = train[sample_index]
                visit = dict(visit=index + 1, file_name=sample.file_name,
                             pair_sha256=sample.pair_sha256, updated=supervised[sample_index])
                model.train()
                rgb, masks, depth, valid = make_view(sample, "center", args.size, device)
                if has_supervision((rgb, masks, depth, valid), mode) != visit["updated"]:
                    raise ValueError("fixed cohort supervision changed during a visit")
                if not visit["updated"]:
                    skipped += 1
                    visits.append(visit)
                    stream.write(json.dumps(dict(visit, reason="no_valid_depth_label",
                                                 updates_completed=len(steps))) + "\n")
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
                record = dict(visit, update=len(steps) + 1, loss=float(loss.detach()),
                              grad_norm=float(grad_norm), empty_mask=not bool(masks.any()),
                              elapsed_s=time.monotonic() - start)
                visits.append(visit)
                steps.append(record)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
        optimization_wall_s = time.monotonic()-start
        save_diagnostic_checkpoint(head_dir / "diagnostic.pt", model, optimizer, device,
                                   head_contract, dataset_manifest, training_config, steps, visits)
        after = {view: evaluate(model, train, view, args.size, device, baseline)
                 for view in ("center", "full_field_resize")}
        heldout = {view: evaluate(model, validation, view, args.size, device, baseline)
                   for view in ("center", "full_field_resize")}
        result["heads"][mode] = dict(before=before, after=after, validation=heldout,
            completed_updates=len(steps) - initial_step, global_step=len(steps),
            visits_completed=len(visits), skipped_unlabelled_visits=skipped, optimization_wall_s=optimization_wall_s,
            training_config_sha256=head_contract["training_config_sha256"],
            max_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            checkpoint_sha256=digest_file(head_dir / "diagnostic.pt"),
            checkpoint=str(head_dir / "diagnostic.pt"),
            loss_first=steps[initial_step]["loss"] if len(steps) > initial_step else None,
            loss_last=steps[-1]["loss"] if len(steps) > initial_step else None,
            stop_reason="budget" if len(steps) - initial_step == args.steps else "wall_clock")
        (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        del optimizer, model
    print(json.dumps({m: {"completed_updates": r["completed_updates"],
                         "global_step": r["global_step"],
                         "train_iou": r["after"]["center"]["visible_iou"],
                         "train_depth": r["after"]["center"]["depth"]}
                      for m,r in result["heads"].items()}))


if __name__ == "__main__":
    main()
