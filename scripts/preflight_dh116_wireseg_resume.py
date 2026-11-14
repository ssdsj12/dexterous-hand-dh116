"""CPU-only per-head checkpoint and data preflight; executes no training."""
import argparse
import json
from pathlib import Path

import torch

from scripts.continue_dh116_wireseg_legacy import validate_source
from scripts.import_dh116_wireseg_legacy import UNKNOWN_STATE, verified_cohort
from scripts.relay_wireseg_blobs import digest_file
from scripts.wireseg_checkpoint_state import read_checkpoint


def validate_head_support(source, expected_head, cohort, supervised):
    state = source["state"]
    if expected_head not in ("seg-only", "depth-only") or state["contract"]["mode"] != expected_head:
        raise ValueError("checkpoint head differs; segmentation and depth must remain separate")
    if cohort != state["contract"]["identities"]:
        raise ValueError("frozen paired cohort or order changed")
    if supervised != source["supervised"] or any(type(value) is not bool for value in supervised):
        raise ValueError("observed supervision differs from checkpoint data support")
    sampler = source["sampler"]
    if (not cohort or len(supervised) != len(cohort) or type(sampler["visits_completed"]) is not int
            or sampler["visits_completed"] < 0
            or sampler["next_index"] != sampler["visits_completed"] % len(cohort)):
        raise ValueError("sampler cursor does not match the fixed cohort")
    if expected_head == "seg-only" and not all(supervised):
        raise ValueError("empty masks remain supervised segmentation negatives")


def preview_visits(source, updates):
    if type(updates) is not int or not 1 <= updates <= 16:
        raise ValueError("preview requires 1-16 hypothetical updates")
    cohort = source["state"]["contract"]["identities"]
    supervised = source["supervised"]
    if not any(supervised):
        raise ValueError("no valid supervision in fixed cohort")
    validate_head_support(source, source["state"]["contract"]["mode"], cohort, supervised)
    visit = source["sampler"]["visits_completed"]
    update = source["state"]["global_step"]
    initial_update = update
    rows = []
    while update - initial_update < updates:
        index = visit % len(cohort)
        visit += 1
        updated = supervised[index]
        if updated:
            update += 1
        rows.append(dict(visit=visit, file_name=cohort[index]["file_name"],
                         pair_sha256=cohort[index]["pair_sha256"], updated=updated,
                         update_if_executed=update if updated else None))
    return rows


def inspect_head(checkpoint, expected_sha, dataset_root, expected_head, preview_updates):
    package, observed_sha = read_checkpoint(checkpoint, expected_sha)
    source = validate_source(package, observed_sha)
    if source["state"]["contract"]["mode"] != expected_head:
        raise ValueError("checkpoint head differs; refusing cross-head resume")
    cohort, supervised = verified_cohort(dataset_root, source["evidence"], source["state"]["contract"])
    validate_head_support(source, expected_head, cohort, supervised)
    preview = preview_visits(source, preview_updates)
    state = source["state"]
    steps = [float(slot["step"]) for slot in state["optimizer"]["state"].values()]
    cuda_rng = source["rng"]["torch_cuda"] if source["rng"] is not None else state["cuda_rng"]
    if digest_file(checkpoint) != observed_sha:
        raise ValueError("checkpoint bytes changed during read-only preflight")
    return dict(mode=expected_head, status="checkpoint_and_frozen_data_preflight_only",
                checkpoint=str(checkpoint), checkpoint_sha256=observed_sha,
                original_checkpoint_sha256=source["source_checkpoint_sha256"],
                origin_import_sha256=source["origin_import_sha256"],
                global_step=state["global_step"], optimizer_slots=len(steps),
                optimizer_step_range=[min(steps), max(steps)] if steps else None,
                visits_completed=source["sampler"]["visits_completed"],
                next_index=source["sampler"]["next_index"],
                fixed_cohort_size=len(cohort), supervised_images=sum(supervised),
                unlabelled_images=len(cohort) - sum(supervised),
                hypothetical_preview=preview, hypothetical_updates=preview_updates,
                hypothetical_skips=sum(not row["updated"] for row in preview),
                current_rng_policy="restore_complete_current_rng" if source["rng"] is not None else
                                   "preserve_torch_rng_and_require_explicit_new_python_numpy_seed",
                unknown_historical_state=list(UNKNOWN_STATE),
                required_execution_device_family="cuda" if cuda_rng is not None else "cpu",
                execution_device_used="cpu_readonly", real_training_updates=0,
                optimizer_constructed=False, inference_views=0, training_budget_authorized=False,
                strict_resume_eligible=False, historical_bitwise_equivalence=False,
                native_qualified=False, stereo_qualified=False, policy_qualified=False,
                data_and_checkpoint_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg-checkpoint", type=Path, required=True)
    parser.add_argument("--seg-sha256", required=True)
    parser.add_argument("--depth-checkpoint", type=Path, required=True)
    parser.add_argument("--depth-sha256", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-updates", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.preview_updates <= 16:
        parser.error("preview is limited to 1-16 hypothetical updates, never executed")
    if args.seg_sha256 == args.depth_sha256:
        parser.error("segmentation and depth checkpoints must have separate identities")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    report = dict(status="readonly_resume_preflight", source_sha256=digest_file(Path(__file__)),
                  real_training_updates=0, inference_views=0, optimizer_constructed=False,
                  training_budget_authorized=False, torch_threads=2, heads={})
    for head, path, sha in (("seg-only", args.seg_checkpoint, args.seg_sha256),
                            ("depth-only", args.depth_checkpoint, args.depth_sha256)):
        report["heads"][head] = inspect_head(path, sha, args.dataset_root, head, args.preview_updates)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({head: {key: data[key] for key in (
        "global_step", "visits_completed", "next_index", "supervised_images", "hypothetical_skips")}
                      for head, data in report["heads"].items()}))


if __name__ == "__main__":
    main()
