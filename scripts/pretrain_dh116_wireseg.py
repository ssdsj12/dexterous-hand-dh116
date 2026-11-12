#!/usr/bin/env python3
"""Audit a pinned local WireSeg cache or train its monocular visual heads."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from scripts.relay_wireseg_blobs import cache_path, digest_file, read_catalogue, verify_blob
from go2_pvcnn.control.dh116.tabletop_wireseg_pretraining import (
    WIRESEG_REVISION, WireSegPretrainer, extract_depth_shard,
    load_wireseg_sample, pretraining_losses, sample_split,
)

def verified_sources(root):
    manifest_sha, blobs = read_catalogue(root / "hf_revision_manifest.json")
    names = {}
    for name, blob in blobs.items():
        path = cache_path(root, name)
        if not path.is_file():
            continue
        if not verify_blob(path, blob):
            raise ValueError(f"HF blob checksum or size mismatch: {name}")
        names[name] = blob.sha256
    return manifest_sha, names


def wireseg_rows(root, names, split=None, *, include_source=False, metadata_only=False):
    import pyarrow.parquet as parquet

    seen = set()
    for name in sorted(names):
        if not name.endswith(".parquet"):
            continue
        real = name.startswith("real/")
        if split is not None and real != (split == "real_test"):
            continue
        columns = ["file_name", "scene_family"] if metadata_only else [
            "image", "file_name", "scene_family", "masks_rle", "num_instances"
        ]
        if not real and not metadata_only:
            columns.append("n_cable")
        for batch in parquet.ParquetFile(root / name).iter_batches(batch_size=4, columns=columns):
            for row in batch.to_pylist():
                identity = row["file_name"]
                if identity in seen:
                    raise ValueError(f"duplicate WireSeg row identity: {identity}")
                seen.add(identity)
                category = sample_split(row["scene_family"])
                if real != (category == "real_test"):
                    raise ValueError("WireSeg source family does not match official configuration")
                if split is None or split == category:
                    yield (name, row) if include_source else row


def prepare_depth(root, names, enabled):
    seen = set()
    if enabled:
        for name in sorted(names):
            if not name.endswith(".tar.gz"):
                continue
            members = set(extract_depth_shard(root / name, root / "metric_depth_npy"))
            if seen & members:
                raise ValueError("duplicate depth identities across shards")
            seen.update(members)
    return seen


def audit(root, names, output, limit, depth_members):
    depth_root = root / "metric_depth_npy"
    count = Counter()
    with (output / "samples.jsonl").open("x", encoding="utf-8") as out:
        for row in wireseg_rows(root, names):
            if limit is not None and count["rows"] >= limit:
                break
            sample = load_wireseg_sample(row, depth_root, depth_members=depth_members)
            if sample.split != "real_test" and sample.rgb.shape != (1024, 1024, 3):
                raise ValueError("synthetic WireSeg RGB dimensions differ from published contract")
            entry = dict(file_name=sample.file_name, family=sample.scene_family, split=sample.split,
                         n_cable=row.get("n_cable"), instances=len(sample.instance_masks),
                         shape=sample.rgb.shape[:2], pair_sha256=sample.pair_sha256,
                         metric_depth_status="paired" if sample.depth_m is not None else (
                             "unavailable_in_official_real" if sample.split == "real_test" else "missing"
                         ))
            if sample.depth_m is not None:
                values = sample.depth_m[sample.depth_valid]
                entry.update(valid_depth_pixels=int(values.size), depth_min_m=float(values.min()) if values.size else None,
                             depth_max_m=float(values.max()) if values.size else None)
            count["rows"] += 1
            count[sample.split] += 1
            count[entry["metric_depth_status"]] += 1
            print(json.dumps(entry, allow_nan=False), file=out)
    return dict(count)


def make_subset(root, names, manifest_sha, split, limit):
    from itertools import islice

    rows = wireseg_rows(root, names, split, include_source=True, metadata_only=True)
    samples = [{"blob": blob, "file_name": row["file_name"]} for blob, row in islice(rows, limit)]
    if not samples:
        raise ValueError(f"no verified WireSeg rows for split {split}")
    return {"version": 1, "revision": WIRESEG_REVISION, "catalogue_sha256": manifest_sha,
            "split": split, "blobs": {entry["blob"]: names[entry["blob"]] for entry in samples},
            "samples": samples}


def read_subset(root, path, names, manifest_sha, split):
    subset = json.loads(path.read_bytes())
    if (subset["version"] != 1 or subset["revision"] != WIRESEG_REVISION
            or subset["catalogue_sha256"] != manifest_sha or subset["split"] != split):
        raise ValueError("WireSeg subset revision, catalogue or split mismatch")
    blobs = subset["blobs"]
    if not blobs or any(name not in names or names[name] != sha for name, sha in blobs.items()):
        raise ValueError("WireSeg subset requires fully verified source blobs")
    samples = [(entry["blob"], entry["file_name"]) for entry in subset["samples"]]
    if not samples or len({name for _, name in samples}) != len(samples) or {
            blob for blob, _ in samples} != set(blobs):
        raise ValueError("WireSeg subset contains duplicate or unbound sample identities")
    requested = set(samples)
    actual = [(blob, row["file_name"]) for blob, row in wireseg_rows(
        root, blobs, split, include_source=True, metadata_only=True
    ) if (blob, row["file_name"]) in requested]
    if actual != samples:
        raise ValueError("WireSeg subset contains missing, reordered or wrong-split rows")
    for blob, row in wireseg_rows(root, blobs, split, include_source=True):
        identity = (blob, row["file_name"])
        if identity not in requested:
            continue
        yield row
        requested.remove(identity)
        if not requested:
            return
    raise ValueError("verified WireSeg source changed while decoding the subset")


def restore_training(path, contract, model, optimizer, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("contract") != contract:
        raise ValueError("WireSeg resume contract differs from the current data or configuration")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"], device)
    return state["global_step"], state["steps"], Counter(state["counts"])


def training_sample(sample, crop, device):
    height, width = sample.rgb.shape[:2]
    crop = min(crop, height, width)
    y, x = (height - crop) // 2, (width - crop) // 2
    view = np.s_[y:y + crop, x:x + crop]
    masks = torch.from_numpy(sample.instance_masks[:, view[0], view[1]].copy()).to(device)
    masks = masks[masks.flatten(1).any(1)]
    rgb = torch.from_numpy(sample.rgb[view].copy()).permute(2, 0, 1)[None].to(device).float() / 255
    depth = valid = None
    if sample.depth_m is not None:
        depth = torch.from_numpy(sample.depth_m[view].copy())[None].to(device)
        valid = torch.from_numpy(sample.depth_valid[view].copy())[None].to(device)
    return rgb, masks, depth, valid


def save_training(path, model, optimizer, device, contract, steps, counts):
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                 contract=contract, global_step=len(steps), steps=steps, counts=dict(counts),
                 backbone_transfer=model.transfer_state(), stereo_correspondence_supervised=False,
                 dh116_policy_supervised=False)
    with NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", delete=False) as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


@torch.inference_mode()
def validate(model, rows, depth_root, depth_members, crop, device):
    model.eval()
    count, metrics, intersection, union = Counter(), Counter(), 0, 0
    for row in rows:
        sample = load_wireseg_sample(row, depth_root, depth_members=depth_members)
        rgb, masks, depth, valid = training_sample(sample, crop, device)
        result = model(rgb)
        terms = pretraining_losses(result, [masks], depth, valid)
        metrics.update({name: float(value) for name, value in terms.items()})
        predicted = result["visible"].reshape(masks.shape[1:]) > 0
        target = masks.any(dim=0)
        intersection += int((predicted & target).sum())
        union += int((predicted | target).sum())
        count["rgb"] += 1
        count["nonempty_crop_masks"] += bool(len(masks))
        count["depth_supervised"] += int(valid is not None and bool(valid.any()))
    return {"counts": dict(count), "visible_iou": intersection / union if union else None,
            "mean_loss": {name: total / count["rgb"] for name, total in metrics.items()}}


def train(root, names, output, args, manifest_sha, depth_members):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = WireSegPretrainer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_manifest = args.subset_manifest
    if train_manifest is None:
        train_manifest = output / "verified_subset_manifest.json"
        subset = make_subset(root, names, manifest_sha, "train", args.max_samples or args.steps)
        train_manifest.write_text(json.dumps(subset, indent=2) + "\n", encoding="utf-8")
    depth_sources = {name: sha for name, sha in names.items() if name.endswith(".tar.gz")}
    contract = dict(mode=args.mode, device=str(device), crop=args.crop, seed=args.seed,
                    learning_rate=args.learning_rate, manifest_sha256=manifest_sha,
                    depth_sources=depth_sources if args.mode == "seg-depth" else {},
                    train_subset_sha256=digest_file(train_manifest),
                    validation_subset_sha256=(digest_file(args.validation_manifest)
                                              if args.validation_manifest else None))
    global_step, steps, counts = (0, [], Counter())
    if args.resume:
        global_step, steps, counts = restore_training(args.resume, contract, model, optimizer, device)
    if not 0 <= global_step == len(steps) < args.steps:
        raise ValueError("WireSeg --steps must be an absolute target above the resume step")
    completed = iter(entry["file_name"] for entry in steps)
    pending = next(completed, None)
    checkpoint = output / "wireseg_monocular.pt"
    for row in read_subset(root, train_manifest, names, manifest_sha, "train"):
        sample = load_wireseg_sample(row, root / "metric_depth_npy", depth_members=depth_members)
        rgb, masks, depth, valid = training_sample(sample, args.crop, device)
        if pending is not None:
            if len(masks):
                if sample.file_name != pending:
                    raise ValueError("WireSeg completed rows do not match the resume prefix")
                pending = next(completed, None)
            continue
        counts["rgb"] += 1
        if not len(masks):
            counts["empty_crop_masks"] += 1
            continue
        counts["nonempty_crop_masks"] += 1
        optimizer.zero_grad(set_to_none=True)
        losses = pretraining_losses(model(rgb), [masks], depth, valid)
        if not torch.isfinite(losses["total"]):
            raise ValueError("nonfinite WireSeg loss")
        losses["total"].backward()
        optimizer.step()
        pixels = int(valid.sum()) if valid is not None else 0
        counts["depth_supervised"] += bool(pixels)
        steps.append({"file_name": sample.file_name, "pair_sha256": sample.pair_sha256,
                      "depth_valid_pixels": pixels,
                      **{name: float(value.detach()) for name, value in losses.items()}})
        if len(steps) % args.checkpoint_every == 0 or len(steps) == args.steps:
            save_training(checkpoint, model, optimizer, device, contract, steps, counts)
        if len(steps) == args.steps:
            break
    if len(steps) != args.steps:
        raise ValueError(f"only {len(steps)} of {args.steps} bounded training steps available")
    if args.mode == "seg-depth" and not counts["depth_supervised"]:
        raise ValueError("seg-depth mode had no exact paired depth supervision")
    evaluation = None
    if args.validation_manifest:
        rows = read_subset(root, args.validation_manifest, names, manifest_sha, "validation")
        evaluation = validate(model, rows, root / "metric_depth_npy", depth_members, args.crop, device)
    return {"steps": steps, "counts": dict(counts), "contract": contract,
            "unique_training_image_ids": len({entry["file_name"] for entry in steps}),
            "training_scene_family": "flying_wire", "validation": evaluation,
            "generalization_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "subset", "pretrain", "evaluate"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract-depth", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--seed", type=int, default=116)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--subset-manifest", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--mode", choices=("seg-only", "seg-depth"), default="seg-only")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=32)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.command == "pretrain" and (args.steps is None or args.steps < 1 or args.crop < 32):
        parser.error("pretraining requires a positive bounded --steps and --crop >= 32")
    if (args.max_samples is not None and args.max_samples < 1) or args.checkpoint_every < 1:
        parser.error("sample and checkpoint bounds must be positive")
    if args.command == "subset" and args.max_samples is None:
        parser.error("subset requires --max-samples")
    if args.command == "evaluate" and not (args.checkpoint and args.validation_manifest):
        parser.error("evaluate requires --checkpoint and --validation-manifest")
    if args.resume and not args.subset_manifest:
        parser.error("--resume requires the original --subset-manifest")
    if args.command == "pretrain" and args.mode == "seg-depth" and not args.extract_depth:
        parser.error("seg-depth requires --extract-depth for exact verified pairs")
    root = args.dataset_root.resolve(strict=True)
    manifest_sha, names = verified_sources(root)
    if not any(name.startswith(("data/", "real/")) for name in names):
        raise ValueError("no verified WireSeg RGB parquet shard is available")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.command == "subset":
        subset = make_subset(root, names, manifest_sha, args.split, args.max_samples)
        (args.output / "verified_subset_manifest.json").write_text(
            json.dumps(subset, indent=2) + "\n", encoding="utf-8")
        return
    depth_members = prepare_depth(root, names, args.extract_depth and
                                  (args.command == "audit" or args.mode == "seg-depth"))
    if args.command == "audit":
        result = audit(root, names, args.output, args.limit, depth_members)
    elif args.command == "pretrain":
        result = train(root, names, args.output, args, manifest_sha, depth_members)
    else:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        contract = state["contract"]
        if contract["manifest_sha256"] != manifest_sha:
            raise ValueError("WireSeg checkpoint catalogue differs from the validation catalogue")
        model = WireSegPretrainer().to(args.device)
        model.load_state_dict(state["model"])
        rows = read_subset(root, args.validation_manifest, names, manifest_sha, "validation")
        result = validate(model, rows, root / "metric_depth_npy", depth_members,
                          contract["crop"], args.device)
        result.update(checkpoint_sha256=digest_file(args.checkpoint),
                      validation_subset_sha256=digest_file(args.validation_manifest),
                      global_step=state["global_step"], generalization_qualified=False)
    (args.output / "result.json").write_text(json.dumps({
        "command": args.command, "dataset_revision": WIRESEG_REVISION, "manifest_sha256": manifest_sha,
        "verified_source_blobs": names, "source_blob_complete": len(names) == 148,
        "verified_depth_members": len(depth_members),
        "split_policy": "exclusive_scene_family_with_official_real_test",
        "stereo_qualified": False, "policy_qualified": False, "result": result,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
