#!/usr/bin/env python3
"""Audit a pinned local WireSeg cache or train its monocular visual heads."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from scripts.relay_wireseg_blobs import cache_path, read_catalogue, verify_blob
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


def wireseg_rows(root, names, split=None):
    import pyarrow.parquet as parquet

    seen = set()
    for name in sorted(names):
        if not name.endswith(".parquet"):
            continue
        real = name.startswith("real/")
        if split is not None and real != (split == "real_test"):
            continue
        for batch in parquet.ParquetFile(root / name).iter_batches(
            batch_size=4,
            columns=["image", "file_name", "scene_family", "masks_rle", "num_instances", "n_cable"]
            if not real else ["image", "file_name", "scene_family", "masks_rle", "num_instances"],
        ):
            for row in batch.to_pylist():
                identity = row["file_name"]
                if identity in seen:
                    raise ValueError(f"duplicate WireSeg row identity: {identity}")
                seen.add(identity)
                category = sample_split(row["scene_family"])
                if real != (category == "real_test"):
                    raise ValueError("WireSeg source family does not match official configuration")
                if split is None or split == category:
                    yield row


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


def train(root, names, output, args, manifest_sha, depth_members):
    torch.manual_seed(args.seed)
    model = WireSegPretrainer().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rows = wireseg_rows(root, names, split="train")
    losses_by_step = []
    for row in rows:
        sample = load_wireseg_sample(row, root / "metric_depth_npy", depth_members=depth_members)
        # WireSeg source images are square. A fixed central crop preserves
        # the pixel alignment and native metric values of all supervision.
        height, width = sample.rgb.shape[:2]
        crop = min(args.crop, height, width)
        y, x = (height - crop) // 2, (width - crop) // 2
        view = np.s_[y:y + crop, x:x + crop]
        masks = torch.from_numpy(sample.instance_masks[:, view[0], view[1]].copy()).to(args.device)
        masks = masks[masks.flatten(1).any(1)]
        if not len(masks):
            continue
        rgb = torch.from_numpy(sample.rgb[view].copy()).permute(2, 0, 1)[None].to(args.device).float() / 255
        depth = valid = None
        if sample.depth_m is not None:
            depth = torch.from_numpy(sample.depth_m[view].copy())[None].to(args.device)
            valid = torch.from_numpy(sample.depth_valid[view].copy())[None].to(args.device)
        optimizer.zero_grad(set_to_none=True)
        losses = pretraining_losses(model(rgb), [masks], depth, valid)
        if not torch.isfinite(losses["total"]):
            raise ValueError("nonfinite WireSeg loss")
        losses["total"].backward()
        optimizer.step()
        losses_by_step.append({
            "file_name": sample.file_name, "pair_sha256": sample.pair_sha256,
            "depth_valid_pixels": int(valid.sum()) if valid is not None else 0,
            **{name: float(value.detach()) for name, value in losses.items()},
        })
        if len(losses_by_step) == args.steps:
            break
    if len(losses_by_step) != args.steps:
        raise ValueError(f"only {len(losses_by_step)} of {args.steps} bounded training steps available")
    # The checkpoint is an isolated monocular pretraining artifact. Native
    # stereo qualification and DH116 policy claims require other datasets.
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    backbone_transfer=model.transfer_state(), seed=args.seed,
                    manifest_sha256=manifest_sha, source_blobs=names,
                    stereo_correspondence_supervised=False, dh116_policy_supervised=False),
               output / "wireseg_monocular.pt")
    return {"steps": losses_by_step, "seed": args.seed, "crop": args.crop}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "pretrain"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract-depth", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--seed", type=int, default=116)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.command == "pretrain" and (args.steps is None or args.steps < 1 or args.crop < 32):
        parser.error("pretraining requires a positive bounded --steps and --crop >= 32")
    root = args.dataset_root.resolve(strict=True)
    manifest_sha, names = verified_sources(root)
    if not any(name.startswith(("data/", "real/")) for name in names):
        raise ValueError("no verified WireSeg RGB parquet shard is available")
    args.output.mkdir(parents=True, exist_ok=False)
    depth_members = prepare_depth(root, names, args.extract_depth)
    if args.command == "audit":
        result = audit(root, names, args.output, args.limit, depth_members)
    else:
        result = train(root, names, args.output, args, manifest_sha, depth_members)
    (args.output / "result.json").write_text(json.dumps({
        "command": args.command, "dataset_revision": WIRESEG_REVISION, "manifest_sha256": manifest_sha,
        "verified_source_blobs": names, "source_blob_complete": len(names) == 148,
        "verified_depth_members": len(depth_members),
        "split_policy": "exclusive_scene_family_with_official_real_test",
        "stereo_qualified": False, "policy_qualified": False, "result": result,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
