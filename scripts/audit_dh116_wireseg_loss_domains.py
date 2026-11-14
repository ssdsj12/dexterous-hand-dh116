"""Read-only loss-domain audit with exact-identity reuse of completed views."""
import argparse
import copy
import hashlib
import inspect
import json
import math
import signal
import time
from itertools import islice
from pathlib import Path

import torch

from scripts.audit_dh116_wireseg_depth_error import (
    BudgetExpired, inspect_sample, verify_unchanged_weights,
)
from scripts.continue_dh116_wireseg_legacy import validate_source
from scripts.diagnose_dh116_wireseg_overfit import constant_depth, sample_identity
from scripts.pretrain_dh116_wireseg import read_subset
from scripts.relay_wireseg_blobs import digest_file, read_catalogue, verify_blob, write_atomic
from scripts.wireseg_checkpoint_state import read_checkpoint
from go2_pvcnn.control.dh116.tabletop_wireseg_pretraining import WireSegPretrainer, load_wireseg_sample

FORMAT = "wireseg-readonly-loss-domains-v1"
LEGACY_DOMAIN_SOURCE_SHA = "88d9348da8a191779267f55c6f9cbe4648b3a1afdfcf120e926a6f2ad40dbe18"
LEGACY_AGGREGATION_SHA = "37de45ce092b2ea5a6a20327289e7c52c167b8ff173fd9b768e22e4900cc175c"
AGGREGATION_VERSION = "wireseg-domain-error-moments-v1"
LEGACY_AUDIT_SOURCE_SHA = "ecb263c2c41ea31a061b09574c866da8f8cd8e6c7bbd7c9d1eaf260d2bd41125"
LEGACY_VIEW_IMPLEMENTATION = {
    "scripts/diagnose_dh116_wireseg_overfit.py": "41cbe657c79c3bac8842814c94a6bf3665be3bbb3fbfae5240a02c8b2570e3a2",
    "scripts/pretrain_dh116_wireseg.py": "cc55db717439693ab7a88a67314d31fc3ed43d77099724c903f9200d98738b68",
    "src/go2_pvcnn/control/dh116/tabletop_wireseg_pretraining.py": "7e693a951ef3b69ec84d76bc4809daedc7eebf03795065855ff8b1e838300bff",
    "src/go2_pvcnn/control/dh116/tabletop_stereo_encoder.py": "44e75dfe67c9faaa3ccee1a04f5c8bbce01e02b85a8d6d1a56f64f4ce66c3a44",
}


def view_binding(size):
    root = Path(__file__).resolve().parents[1]
    aggregation = inspect.getsource(_domain_moments) + "\n" + inspect.getsource(summarize_domain)
    return dict(size=size, views=["center", "full_field_resize"],
                implementation_sha256={name: digest_file(root / name) for name in LEGACY_VIEW_IMPLEMENTATION},
                audit_helper_sha256=digest_file(Path(__file__).with_name("audit_dh116_wireseg_depth_error.py")),
                aggregation=dict(version=AGGREGATION_VERSION,
                                 function_source_sha256=hashlib.sha256(aggregation.encode()).hexdigest(),
                                 audit_source_sha256=digest_file(Path(__file__))))


def _domain_moments(row, name, metric_key):
    full = row[metric_key]["all_valid"]
    roi = row[metric_key]["wire_roi"]
    if roi["valid_pixels"] > full["valid_pixels"]:
        raise ValueError("ROI valid pixels exceed full valid support")
    def moments(metric):
        count = metric["valid_pixels"]
        if not count:
            return 0, 0., 0.
        return count, count * metric["mae_m"], count * metric["rmse_m"]**2
    if name != "background":
        return moments(row[metric_key][name])
    all_count, all_abs, all_square = moments(full)
    roi_count, roi_abs, roi_square = moments(roi)
    if all_abs + 1e-8 < roi_abs or all_square + 1e-8 < roi_square:
        raise ValueError("ROI error sums exceed full-domain sums")
    return all_count - roi_count, max(0., all_abs - roi_abs), max(0., all_square - roi_square)


def summarize_domain(rows, expected_names, *, pixels_per_image):
    if (len(set(expected_names)) != len(expected_names)
            or len({row["file_name"] for row in rows}) != len(rows)
            or any(row["file_name"] not in expected_names for row in rows)):
        raise ValueError("domain rows must be a distinct subset of the fixed cohort")
    fixed = len(expected_names) * pixels_per_image
    observed = len(rows) * pixels_per_image
    total_valid = sum(row["depth"]["all_valid"]["valid_pixels"] for row in rows)
    losses = [row for row in rows if row["loss"] is not None]
    if any((row["loss"] is None) != (row["depth"]["all_valid"]["valid_pixels"] == 0) for row in rows):
        raise ValueError("loss supervision differs from valid-label support")
    result = dict(expected_images=len(expected_names), completed_images=len(rows),
                  missing_images=[name for name in expected_names if name not in {row["file_name"] for row in rows}],
                  fixed_total_pixels=fixed, observed_total_pixels=observed,
                  valid_fraction_observed=total_valid / observed if observed else None,
                  valid_fraction_fixed_lower_bound=total_valid / fixed if fixed else None,
                  empty_depth_images=sum(row["depth"]["all_valid"]["valid_pixels"] == 0 for row in rows),
                  empty_wire_depth_images=sum(row["depth"]["wire_roi"]["valid_pixels"] == 0 for row in rows),
                  images_with_background_depth=sum(row["depth"]["all_valid"]["valid_pixels"] >
                                                   row["depth"]["wire_roi"]["valid_pixels"] for row in rows),
                  loss=dict(supervised_images=len(losses),
                            equal_image_smooth_l1=sum(row["loss"] for row in losses) / len(losses) if losses else None,
                            pixel_weighted_smooth_l1=sum(row["loss"] * row["depth"]["all_valid"]["valid_pixels"]
                                                        for row in losses) / total_valid if total_valid else None),
                  p95_status="not_reconstructible_from_cached_per_image_summaries")
    for metric_key in ("depth", "constant"):
        result[metric_key] = {}
        for name in ("all_valid", "wire_roi", "background"):
            moments = [_domain_moments(row, name, metric_key) for row in rows]
            count = sum(item[0] for item in moments)
            supported = [item for item in moments if item[0]]
            result[metric_key][name] = dict(
                valid_pixels=count, supported_images=len(supported),
                pixel_mae_m=sum(item[1] for item in moments) / count if count else None,
                pixel_rmse_m=math.sqrt(sum(item[2] for item in moments) / count) if count else None,
                equal_image_mae_m=sum(item[1] / item[0] for item in supported) / len(supported) if supported else None,
                pixel_p95_m=None)
    for field in ("prediction_quantiles_m", "target_quantiles_m"):
        values = [row[field] for row in rows if field in row and row[field]["0"] is not None]
        result[field.replace("_quantiles_m", "_range_m")] = (
            [min(value["0"] for value in values), max(value["100"] for value in values)] if values else None)
    return result


def inspect_or_reuse(model, sample, split, view, size, baseline, checkpoint_sha, cache):
    identity = (split, sample.file_name, view)
    if identity in cache:
        row = copy.deepcopy(cache[identity])
        if (row["pair_sha256"] != sample.pair_sha256 or row["family"] != sample.scene_family
                or row["checkpoint_sha256"] != checkpoint_sha):
            raise ValueError("cached checkpoint/data/view identity differs")
        row["reused"] = True
        return row
    row, _ = inspect_sample(model, sample, view, size, baseline)
    return dict(row, split=split, checkpoint_head="depth-only", checkpoint_sha256=checkpoint_sha, reused=False)


def load_cache(path, expected_sha, checkpoint_sha, baseline, binding):
    if path is None:
        return {}
    if digest_file(path) != expected_sha:
        raise ValueError("cached report SHA256 differs")
    if binding != view_binding(binding["size"]):
        raise ValueError("runtime view/helper/aggregation binding differs")
    report = json.loads(path.read_bytes())
    if report.get("real_training_updates") != 0 or report.get("training_constant_depth_m") != baseline:
        raise ValueError("cache must use the unchanged training-only baseline and no training")
    if report.get("format") == FORMAT:
        if report["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("cached checkpoint SHA256 differs")
        if report.get("source_sha256") == binding["aggregation"]["audit_source_sha256"]:
            if report.get("view_binding") != binding:
                raise ValueError("cached view/helper/aggregation implementation differs")
        elif report.get("source_sha256") == LEGACY_DOMAIN_SOURCE_SHA:
            legacy_binding = dict(size=192, views=["center", "full_field_resize"],
                                  implementation_sha256=LEGACY_VIEW_IMPLEMENTATION)
            if (report.get("view_binding") != legacy_binding or binding["size"] != 192
                    or binding["implementation_sha256"] != LEGACY_VIEW_IMPLEMENTATION
                    or binding["audit_helper_sha256"] != LEGACY_AUDIT_SOURCE_SHA
                    or binding["aggregation"]["function_source_sha256"] != LEGACY_AGGREGATION_SHA):
                raise ValueError("legacy domain cache helper/view/aggregation is not the reviewed implementation")
        else:
            raise ValueError("cached aggregation source SHA256 is unsupported")
    else:
        # Older 4+4 rows are usable only with the actual reviewed helper still installed.
        if (report.get("source_sha256") != LEGACY_AUDIT_SOURCE_SHA
                or binding["size"] != 192 or binding["implementation_sha256"] != LEGACY_VIEW_IMPLEMENTATION
                or binding["audit_helper_sha256"] != LEGACY_AUDIT_SOURCE_SHA):
            raise ValueError("legacy cache helper/view provenance is not the reviewed implementation")
        head_sources = {}
        for ref in report["source_checkpoints"]:
            package, sha = read_checkpoint(Path(ref["path"]), ref["sha256"])
            source = validate_source(package, sha)
            head = source["state"]["contract"]["mode"]
            if head in head_sources:
                raise ValueError("legacy cache has ambiguous checkpoint heads")
            head_sources[head] = sha
        if head_sources.get("depth-only") != checkpoint_sha:
            raise ValueError("cached depth checkpoint SHA256 differs")
    cache = {}
    for row in report["records"]:
        if row.get("checkpoint_head") != "depth-only":
            continue
        key = row["split"], row["file_name"], row["view"]
        if key in cache:
            raise ValueError("duplicate completed cache identity")
        bound = dict(row, checkpoint_sha256=checkpoint_sha)
        if row.get("checkpoint_sha256", checkpoint_sha) != checkpoint_sha:
            raise ValueError("cached row checkpoint SHA256 differs")
        cache[key] = bound
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-report", type=Path)
    parser.add_argument("--reuse-sha256")
    parser.add_argument("--wall-seconds", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.wall_seconds <= 55 or (args.reuse_report is None) != (args.reuse_sha256 is None):
        parser.error("audit requires wall1-55s and both cache path/SHA or neither")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = dict(format=FORMAT, status="partial", real_training_updates=0, optimizer_constructed=False,
                  device="cpu", torch_threads=2, checkpoint_sha256=args.checkpoint_sha256,
                  cache_report_sha256=args.reuse_sha256, records=[], groups={}, expected={},
                  source_sha256=digest_file(Path(__file__)), phase="preflight")
    size = None
    def snapshot():
        if size is not None:
            for split, expected in result["expected"].items():
                for family in sorted({entry["family"] for entry in expected}):
                    names = [entry["file_name"] for entry in expected if entry["family"] == family]
                    for view in ("center", "full_field_resize"):
                        rows = [row for row in result["records"] if
                                row["split"] == split and row["family"] == family and row["view"] == view]
                        result["groups"][f"{split}/{family}/{view}"] = summarize_domain(
                            rows, names, pixels_per_image=size * size)
        result["wall_seconds"] = time.monotonic() - start
        result["reused_views"] = sum(row["reused"] for row in result["records"])
        result["new_inferred_views"] = sum(not row["reused"] for row in result["records"])
        write_atomic(args.output / "result.json", [json.dumps(result, indent=2, allow_nan=False).encode() + b"\n"])
    def expire(signum, frame):
        raise BudgetExpired()
    snapshot()
    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, args.wall_seconds)
    try:
        package, checkpoint_sha = read_checkpoint(args.checkpoint, args.checkpoint_sha256)
        source = validate_source(package, checkpoint_sha)
        state, evidence = source["state"], source["evidence"]
        contract = state["contract"]
        if contract["mode"] != "depth-only" or len(contract["identities"]) != 16:
            raise ValueError("audit requires the original fixed16 depth diagnostic")
        size = contract["size"]
        result["view_binding"] = view_binding(size)
        manifests = {key: json.loads(Path(evidence[key]["path"]).read_bytes()) for key in ("train", "validation")}
        if len(manifests["validation"]["samples"]) != 128:
            raise ValueError("audit requires the original128 validation manifest")
        result["expected"] = dict(train=contract["identities"], validation=[
            dict(file_name=row["file_name"], family=row["file_name"].split("/")[0])
            for row in manifests["validation"]["samples"]])
        result["data_binding"] = {key: evidence[key]["sha256"] for key in ("train", "validation", "catalogue")}
        result["phase"] = "data_validation"
        snapshot()
        catalogue_sha, blobs = read_catalogue(Path(evidence["catalogue"]["path"]))
        names = {}
        for manifest in manifests.values():
            for name, sha in manifest["blobs"].items():
                if name not in blobs or blobs[name].sha256 != sha or not verify_blob(args.dataset_root / name, blobs[name]):
                    raise ValueError("audit requires verified original source blobs")
                names[name] = sha
        train_rows = islice(read_subset(args.dataset_root, Path(evidence["train"]["path"]), names, catalogue_sha, "train"), 16)
        train = [load_wireseg_sample(row, args.dataset_root / "metric_depth_npy") for row in train_rows]
        if [sample_identity(sample) for sample in train] != contract["identities"]:
            raise ValueError("fixed16 exact pairs changed")
        baseline = constant_depth([sample.depth_m[sample.depth_valid] for sample in train])
        result.update(training_constant_depth_m=baseline, baseline_fit="all16_original_train_full_frame_valid_pixels")
        cache = load_cache(args.reuse_report, args.reuse_sha256, checkpoint_sha, baseline, result["view_binding"])
        model = WireSegPretrainer().cpu()
        model.load_state_dict(state["model"], strict=True)
        validation_rows = read_subset(args.dataset_root, Path(evidence["validation"]["path"]), names, catalogue_sha, "validation")
        validation = (load_wireseg_sample(row, args.dataset_root / "metric_depth_npy") for row in validation_rows)
        result["phase"] = "readonly_inference"
        for split, samples in (("train", train), ("validation", validation)):
            for sample in samples:
                expected = result["expected"][split]
                if not any(item["file_name"] == sample.file_name and item["family"] == sample.scene_family for item in expected):
                    raise ValueError("sample is outside fixed split/family denominator")
                for view in ("center", "full_field_resize"):
                    row = inspect_or_reuse(model, sample, split, view, size, baseline, checkpoint_sha, cache)
                    result["records"].append(row)
                    snapshot()
        result["weights_unchanged"] = verify_unchanged_weights(model, state["model"])
        result.update(status="complete", phase="finished")
    except BudgetExpired:
        result.update(status="partial", stop_reason="wall_deadline")
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        snapshot()
    print(json.dumps(dict(status=result["status"], records=len(result["records"]),
                          reused_views=result["reused_views"], new_inferred_views=result["new_inferred_views"],
                          wall_seconds=result["wall_seconds"])))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
