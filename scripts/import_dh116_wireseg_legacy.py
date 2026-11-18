"""Audit-only import of pinned legacy diagnostics; never a strict-v2 checkpoint."""
import argparse
import json
import math
import re
from itertools import islice
from pathlib import Path

import torch

from scripts.relay_wireseg_blobs import digest_file, read_catalogue, verify_blob
from scripts.wireseg_checkpoint_state import canonical_sha256, read_checkpoint, write_checkpoint

FORMAT = "wireseg-legacy-diagnostic-import-v1"
UNKNOWN_STATE = ["python_rng", "numpy_rng", "complete_historical_runtime_config"]
SUPPORTED_SOURCES = {
    "be68eb2fbf4ec5ed96d1a8f9a539881201b8acbff03591d8883832ed8a974c24": "updates_only",
    "e03905dc889f89108eca9c39d1df76c78ce70918d356496a72dfd87d757d2410": "visits_with_skips",
}
EVIDENCE_KEYS = {"checkpoint", "ledger", "source", "cohort", "train", "validation", "catalogue"}


def verify_evidence(evidence):
    if not isinstance(evidence, dict) or set(evidence) != EVIDENCE_KEYS:
        raise ValueError("complete explicit legacy evidence bindings required")
    for name, ref in evidence.items():
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                or not isinstance(ref["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", ref["sha256"])
                or digest_file(Path(ref["path"])) != ref["sha256"]):
            raise ValueError(f"legacy evidence SHA256 mismatch: {name}")


def verified_cohort(root, evidence, contract):
    """Read exact original pairs only; do not extract, change or replace data."""
    from scripts.pretrain_dh116_wireseg import read_subset
    from scripts.diagnose_dh116_wireseg_overfit import has_supervision, make_view, sample_identity
    from go2_pvcnn.control.dh116.tabletop_wireseg_pretraining import load_wireseg_sample

    catalogue_sha, catalogue = read_catalogue(Path(evidence["catalogue"]["path"]))
    names = {}
    for key, split in (("train", "train"), ("validation", "validation")):
        manifest = json.loads(Path(evidence[key]["path"]).read_bytes())
        if manifest["split"] != split:
            raise ValueError("legacy manifest split differs")
        for name, sha in manifest["blobs"].items():
            if name not in catalogue or sha != catalogue[name].sha256 or not verify_blob(root / name, catalogue[name]):
                raise ValueError("legacy manifest requires verified original source blobs")
            names[name] = sha
    train_rows = list(islice(read_subset(root, Path(evidence["train"]["path"]), names, catalogue_sha, "train"),
                             len(contract["identities"])))
    samples = [load_wireseg_sample(row, root / "metric_depth_npy") for row in train_rows]
    cohort = [sample_identity(sample) for sample in samples]
    validation = [row["file_name"] for row in read_subset(
        root, Path(evidence["validation"]["path"]), names, catalogue_sha, "validation")]
    if set(row["file_name"] for row in cohort) & set(validation):
        raise ValueError("legacy train/validation identities overlap")
    supervised = [has_supervision(make_view(sample, "center", contract["size"], torch.device("cpu")),
                                  contract["mode"]) for sample in samples]
    return cohort, supervised


def reconstruct_sampler(state, ledger, cohort, supervised, profile):
    steps = state.get("steps")
    count = state.get("global_step")
    if (profile not in {"updates_only", "visits_with_skips"} or not isinstance(steps, list)
            or type(count) is not int or count <= 0 or count != len(steps)
            or state.get("counts", {}).get("rgb_updates") != count
            or not 8 <= len(cohort) <= 16 or len(supervised) != len(cohort)
            or any(type(x) is not bool for x in supervised)
            or len({x["file_name"] for x in cohort}) != len(cohort)):
        raise ValueError("legacy checkpoint update count or fixed cohort is invalid")
    if profile == "updates_only" and (state["contract"]["mode"] != "seg-only" or not all(supervised)):
        raise ValueError("updates-only source proves no skip path only for segmentation")
    history, successful = [], []
    for visit, row in enumerate(ledger, 1):
        index = (visit - 1) % len(cohort)
        sample = cohort[index]
        updated = supervised[index]
        identity = dict(visit=visit, file_name=sample["file_name"], pair_sha256=sample["pair_sha256"],
                        updated=updated)
        if not isinstance(row, dict) or row.get("file_name") != sample["file_name"]:
            raise ValueError("legacy ledger order differs from the fixed cohort")
        if profile == "visits_with_skips":
            if (type(row.get("visit")) is not int or row["visit"] != visit
                    or type(row.get("updated")) is not bool or row["updated"] != updated):
                raise ValueError("legacy ledger visit or supervision differs")
        elif "visit" in row or "updated" in row:
            raise ValueError("legacy ledger differs from its pinned source profile")
        if updated:
            if (type(row.get("update")) is not int or row["update"] != len(successful) + 1
                    or row.get("pair_sha256") != sample["pair_sha256"]
                    or any(not isinstance(row.get(key), (float, int)) or not math.isfinite(row[key])
                           for key in ("loss", "grad_norm", "elapsed_s"))
                    or row["grad_norm"] <= 0 or row["elapsed_s"] < 0):
                raise ValueError("legacy ledger update or pair SHA differs")
            successful.append(row)
        else:
            expected = dict(visit=visit, file_name=sample["file_name"], updated=False,
                            reason="no_valid_depth_label", updates_completed=len(successful))
            if row != expected or type(row["updates_completed"]) is not int:
                raise ValueError("legacy ledger skip record is invalid")
        history.append(identity)
    # Both pinned loops stop immediately on reaching their update budget.
    if successful != steps or not history or not history[-1]["updated"]:
        raise ValueError("legacy ledger differs from checkpoint steps or has trailing unbound skips")
    return dict(kind="fixed_order_cyclic_v1", cohort_size=len(cohort), visits_completed=len(history),
                next_index=len(history) % len(cohort), skipped_visits=len(history) - count,
                visit_history=history)


def validate_legacy_state(state):
    if state.get("checkpoint_format") is not None:
        raise ValueError("only original unversioned legacy checkpoints may be imported")
    model, adam = state.get("model"), state.get("optimizer")
    if not isinstance(model, dict) or not model or not isinstance(adam, dict):
        raise ValueError("legacy model and Adam state are required")
    if not adam.get("state") or not adam.get("param_groups"):
        raise ValueError("legacy Adam state cannot be reset")
    for tensor in model.values():
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError("legacy model contains invalid tensors")
    ids = [index for group in adam["param_groups"] for index in group["params"]]
    if len(ids) != len(set(ids)) or not set(adam["state"]) <= set(ids):
        raise ValueError("legacy Adam parameter IDs differ")
    for slot in adam["state"].values():
        if set(slot) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("legacy Adam moments are incomplete")
        step, first, second = (slot[key] for key in ("step", "exp_avg", "exp_avg_sq"))
        if (not all(isinstance(t, torch.Tensor) for t in (step, first, second))
                or step.numel() != 1 or float(step) != state["global_step"]
                or first.shape != second.shape or first.dtype != second.dtype
                or not torch.isfinite(first).all() or not torch.isfinite(second).all()):
            raise ValueError("legacy Adam moments or step are invalid")
    try:
        torch.Generator().set_state(state["rng"])
    except (KeyError, TypeError, RuntimeError) as error:
        raise ValueError("legacy Torch CPU RNG is missing or invalid") from error
    if "cuda_rng" not in state:
        raise ValueError("legacy Torch CUDA RNG presence must be explicit")
    cuda = state["cuda_rng"]
    if cuda is not None and (not isinstance(cuda, torch.Tensor) or cuda.dtype != torch.uint8
                             or cuda.ndim != 1 or not cuda.numel()):
        raise ValueError("legacy Torch CUDA RNG is invalid")


def import_legacy(evidence, dataset_root, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    verify_evidence(evidence)
    state, checkpoint_sha = read_checkpoint(Path(evidence["checkpoint"]["path"]), evidence["checkpoint"]["sha256"])
    validate_legacy_state(state)
    contract = state["contract"]
    frozen = json.loads(Path(evidence["cohort"]["path"]).read_bytes())
    if contract != {**frozen, "mode": contract.get("mode")} or contract.get("mode") not in {"seg-only", "depth-only"}:
        raise ValueError("legacy frozen cohort differs from checkpoint contract")
    if contract.get("kind") != "wireseg-fixed-cohort-overfit-v1" or contract.get("optimization_view") != "center":
        raise ValueError("unsupported legacy training contract")
    for key, field in (("train", "subset_sha256"), ("validation", "validation_sha256"),
                       ("catalogue", "source_catalogue_sha256")):
        if evidence[key]["sha256"] != contract[field]:
            raise ValueError("legacy manifest SHA256 differs from checkpoint contract")
    source_sha = evidence["source"]["sha256"]
    profile = SUPPORTED_SOURCES.get(source_sha)
    if profile is None or ("source_sha256" in contract and contract["source_sha256"] != source_sha):
        raise ValueError("legacy diagnostic source is unsupported or differs from checkpoint")
    cohort, supervised = verified_cohort(dataset_root, evidence, contract)
    if cohort != contract["identities"]:
        raise ValueError("legacy exact paired cohort differs from checkpoint")
    ledger = [json.loads(line) for line in Path(evidence["ledger"]["path"]).read_text().splitlines()]
    sampler = reconstruct_sampler(state, ledger, cohort, supervised, profile)
    report = dict(status="legacy_diagnostic_import_only", source_checkpoint_sha256=checkpoint_sha,
                  source_profile=profile, global_step=state["global_step"],
                  visits_completed=sampler["visits_completed"], skipped_visits=sampler["skipped_visits"],
                  next_index=sampler["next_index"], new_training_updates=0, strict_resume_eligible=False,
                  historical_bitwise_equivalence=False, native_qualified=False,
                  stereo_qualified=False, policy_qualified=False, generalization_qualified=False,
                  source_binding=("checkpoint_recorded" if "source_sha256" in contract else "external_snapshot_only"))
    imported = dict(checkpoint_format=FORMAT, legacy_state=state, sampler=sampler,
                    supervised=supervised, evidence=evidence, evidence_sha256=canonical_sha256(evidence),
                    unknown_state=list(UNKNOWN_STATE), audit=report)
    verify_evidence(evidence)  # Reject source changes during dataset validation.
    write_checkpoint(output, imported)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True, help="JSON: seven path/SHA256 bindings")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(import_legacy(json.loads(args.evidence.read_bytes()), args.dataset_root, args.output),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
