"""Read-only CPU depth-error attribution for immutable WireSeg diagnostics."""
import argparse
import json
import signal
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from scripts.continue_dh116_wireseg_legacy import validate_source
from scripts.diagnose_dh116_wireseg_overfit import (
    constant_depth, depth_metrics, diagnostic_loss, make_view, quantiles, sample_identity,
)
from scripts.pretrain_dh116_wireseg import read_subset
from scripts.relay_wireseg_blobs import digest_file, read_catalogue, verify_blob, write_atomic
from scripts.wireseg_checkpoint_state import read_checkpoint
from go2_pvcnn.control.dh116.tabletop_wireseg_pretraining import WireSegPretrainer, load_wireseg_sample


@torch.inference_mode()
def inspect_sample(model, sample, view, size, baseline):
    rgb, masks, depth, valid = make_view(sample, view, size, torch.device("cpu"))
    if depth is None or valid is None:
        raise ValueError("read-only attribution requires an exact depth pair")
    model.eval()
    result = model(rgb)
    arrays = tuple(value.detach().cpu().numpy().ravel() for value in (
        result["metric_depth"], depth, valid, masks.any(0)))
    predicted, target, validity, roi = arrays
    validity = validity & np.isfinite(target) & (target > 0)
    arrays = predicted, target, validity, roi
    residual = predicted[validity] - target[validity]
    if not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError("nonfinite or negative metric-depth output")
    row = dict(file_name=sample.file_name, family=sample.scene_family, pair_sha256=sample.pair_sha256,
               view=view, depth=depth_metrics(*arrays),
               constant=depth_metrics(np.full_like(target, baseline), target, validity, roi),
               target_quantiles_m=quantiles(target[validity]),
               prediction_quantiles_m=quantiles(predicted[validity]),
               full_output_quantiles_m=quantiles(predicted),
               signed_bias_m=float(residual.mean(dtype=np.float64)) if residual.size else None,
               linear_smooth_l1_fraction=float((np.abs(residual) >= .01).mean()) if residual.size else None,
               positive_output_fraction=float((predicted > 0).mean()),
               near_zero_output_fraction=float((predicted < .01).mean()),
               valid_depth_fraction=float(validity.mean()), wire_fraction=float(roi.mean()),
               valid_depth_outside_wire=int((validity & ~roi).sum()),
               rgb_minmax=[float(rgb.min()), float(rgb.max())],
               loss=float(diagnostic_loss(result, masks, depth, valid, "depth-only")) if validity.any() else None)
    return row, arrays


def fit_train_calibration(arrays):
    if not arrays:
        return None
    predicted, target, valid, _ = [np.concatenate(parts) for parts in zip(*arrays)]
    valid = valid & np.isfinite(predicted) & (predicted > 0)
    if not valid.any():
        return None
    return dict(fit_split="train", fit_view="center", fit_pixels=int(valid.sum()),
                offset_m=float(np.median(target[valid] - predicted[valid])),
                scale=float(np.median(target[valid] / predicted[valid])),
                applied_to_checkpoint=False)


def summarize(arrays, baseline, calibration):
    if not arrays:
        return None
    predicted, target, valid, roi = [np.concatenate(parts) for parts in zip(*arrays)]
    result = dict(depth=depth_metrics(predicted, target, valid, roi),
                  constant=depth_metrics(np.full_like(target, baseline), target, valid, roi),
                  valid_depth_fraction=float(valid.mean()), wire_fraction=float(roi.mean()),
                  valid_depth_outside_wire=int((valid & ~roi).sum()),
                  prediction_quantiles_m=quantiles(predicted[valid]), target_quantiles_m=quantiles(target[valid]),
                  signed_bias_m=float((predicted[valid] - target[valid]).mean(dtype=np.float64)) if valid.any() else None,
                  image_count=len(arrays))
    if calibration is not None:
        result["offset_diagnostic"] = depth_metrics(
            predicted + calibration["offset_m"], target, valid, roi)
        result["scale_diagnostic"] = depth_metrics(
            predicted * calibration["scale"], target, valid, roi)
    return result


def inspect_views(model, samples, *, size, baseline, deadline, clock=time.monotonic, publish=lambda row: None):
    arrays = {"center": [], "full_field_resize": []}
    for sample in samples:
        for view in arrays:
            if clock() >= deadline:
                return dict(completed=False, arrays=arrays)
            row, values = inspect_sample(model, sample, view, size, baseline)
            arrays[view].append(values)
            publish(row)
    return dict(completed=True, arrays=arrays)


def verify_unchanged_weights(model, expected):
    observed = model.state_dict()
    if observed.keys() != expected.keys() or any(
            observed[name].dtype != value.dtype or not torch.equal(observed[name], value)
            for name, value in expected.items()):
        raise ValueError("read-only inference changed model weights")
    return True


class BudgetExpired(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-sha256", action="append", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=4)
    parser.add_argument("--validation-samples", type=int, default=4)
    parser.add_argument("--wall-seconds", type=int, default=50)
    args = parser.parse_args()
    if (len(args.checkpoint) != len(args.checkpoint_sha256) or not 1 <= len(args.checkpoint) <= 2
            or not 1 <= args.train_samples <= 16 or not 1 <= args.validation_samples <= 128
            or not 1 <= args.wall_seconds <= 55):
        parser.error("bounded CPU audit: 1-2 checkpoints, train1-16/validation1-128, wall1-55s")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(status="partial", real_training_updates=0, optimizer_constructed=False,
                  device="cpu", torch_threads=torch.get_num_threads(), records=[], summaries={},
                  requested_train_inference=args.train_samples, requested_validation_inference=args.validation_samples,
                  original_fixed_train_count=16, original_validation_count=128,
                  source_checkpoints=[dict(path=str(p), sha256=s) for p, s in zip(args.checkpoint, args.checkpoint_sha256)],
                  labels_unit="metre_from_pinned_dataset", applied_depth_rescaling=False,
                  output_transform="softplus", loss="smooth_l1(beta=0.01), positive finite labels only",
                  source_sha256=digest_file(Path(__file__)), phase="source_validation")
    def snapshot():
        result["wall_seconds"] = time.monotonic() - started
        write_atomic(args.output / "result.json", [json.dumps(result, indent=2, allow_nan=False).encode() + b"\n"])
    def expire(signum, frame):
        raise BudgetExpired()
    snapshot()
    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, args.wall_seconds)
    try:
        sources = []
        for path, sha in zip(args.checkpoint, args.checkpoint_sha256):
            package, observed = read_checkpoint(path, sha)
            sources.append(validate_source(package, observed))
        contract = sources[0]["state"]["contract"]
        evidence = sources[0]["evidence"]
        if any(source["state"]["contract"]["identities"] != contract["identities"]
               or source["evidence"]["train"] != evidence["train"]
               or source["evidence"]["validation"] != evidence["validation"] for source in sources):
            raise ValueError("all compared heads must use the same frozen original data")
        result["phase"] = "data_validation"
        snapshot()
        catalogue_sha, blobs = read_catalogue(Path(evidence["catalogue"]["path"]))
        names = {}
        for key in ("train", "validation"):
            manifest = json.loads(Path(evidence[key]["path"]).read_bytes())
            for name, sha in manifest["blobs"].items():
                if name not in blobs or blobs[name].sha256 != sha or not verify_blob(args.dataset_root / name, blobs[name]):
                    raise ValueError("audit requires verified original RGB sources")
                names[name] = sha
        loaded = {}
        for split, key, limit in (("train", "train", len(contract["identities"])),
                                  ("validation", "validation", args.validation_samples)):
            rows = islice(read_subset(args.dataset_root, Path(evidence[key]["path"]), names, catalogue_sha, split), limit)
            loaded[split] = [load_wireseg_sample(row, args.dataset_root / "metric_depth_npy") for row in rows]
        if [sample_identity(sample) for sample in loaded["train"]] != contract["identities"]:
            raise ValueError("audit original training pairs differ from checkpoint")
        if {s.file_name for s in loaded["train"]} & {s.file_name for s in loaded["validation"]}:
            raise ValueError("audit train/validation samples overlap")
        baseline = constant_depth([s.depth_m[s.depth_valid] for s in loaded["train"]])
        result.update(training_constant_depth_m=baseline, baseline_fit="all16_original_train_full_frame_valid_pixels",
                      phase="readonly_inference",
                      inferred_train_identities=[sample_identity(s) for s in loaded["train"][:args.train_samples]],
                      inferred_validation_identities=[sample_identity(s) for s in loaded["validation"]])
        snapshot()
        for source in sources:
            head = source["state"]["contract"]["mode"]
            model = WireSegPretrainer().cpu()
            model.load_state_dict(source["state"]["model"], strict=True)
            all_arrays = {}
            complete = True
            for split in ("train", "validation"):
                samples = loaded[split][:args.train_samples] if split == "train" else loaded[split]
                def publish(row):
                    result["records"].append(dict(row, checkpoint_head=head, split=split))
                    snapshot()
                batch = inspect_views(model, samples, size=contract["size"], baseline=baseline,
                                      deadline=started + args.wall_seconds, publish=publish)
                all_arrays[split] = batch["arrays"]
                complete &= batch["completed"]
            calibration = fit_train_calibration(all_arrays["train"]["center"])
            result["summaries"][head] = dict(completed=complete, calibration=calibration,
                weights_unchanged=verify_unchanged_weights(model, source["state"]["model"]),
                groups={split: {view: summarize(arrays, baseline, calibration) for view, arrays in views.items()}
                        for split, views in all_arrays.items()})
            snapshot()
            del model
        result["status"] = "complete" if all(value["completed"] for value in result["summaries"].values()) else "partial"
        result["phase"] = "finished"
    except BudgetExpired:
        result.update(status="partial", stop_reason="wall_deadline")
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        snapshot()
    print(json.dumps(dict(status=result["status"], records=len(result["records"]), wall_seconds=result["wall_seconds"])))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
