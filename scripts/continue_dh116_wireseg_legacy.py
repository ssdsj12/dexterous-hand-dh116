"""Bounded diagnostic continuation of an explicit legacy import, never strict resume."""
import argparse
import copy
import json
import random
import re
import signal
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_dh116_wireseg_overfit import diagnostic_loss, has_supervision, make_view, sample_identity
from scripts.import_dh116_wireseg_legacy import (
    FORMAT as IMPORT_FORMAT, UNKNOWN_STATE, SUPPORTED_SOURCES, reconstruct_sampler,
    validate_legacy_state, verified_cohort, verify_evidence,
)
from scripts.relay_wireseg_blobs import digest_file
from scripts.wireseg_checkpoint_state import (
    _validate_adam_state, _validated_history, _validated_rng, canonical_sha256, capture_rng,
    read_checkpoint, restore_rng, write_checkpoint,
)

FORMAT = "wireseg-legacy-diagnostic-continuation-v1"


def state_equal(left, right):
    if isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(right, dict):
        return isinstance(left, dict) and left.keys() == right.keys() and all(state_equal(left[k], v) for k, v in right.items())
    if isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(state_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def validate_import(package):
    if (package.get("checkpoint_format") != IMPORT_FORMAT
            or package.get("unknown_state") != UNKNOWN_STATE
            or package.get("evidence_sha256") != canonical_sha256(package.get("evidence"))
            or any(package.get("audit", {}).get(name) is not False for name in (
                "strict_resume_eligible", "historical_bitwise_equivalence", "native_qualified",
                "stereo_qualified", "policy_qualified", "generalization_qualified"))):
        raise ValueError("explicit diagnostic-only legacy import with unknown historical state required")
    validate_legacy_state(package["legacy_state"])
    state = package["legacy_state"]
    evidence = package["evidence"]
    verify_evidence(evidence)
    original, sha = read_checkpoint(Path(evidence["checkpoint"]["path"]), evidence["checkpoint"]["sha256"])

    if not state_equal(state, original) or package["audit"]["source_checkpoint_sha256"] != sha:
        raise ValueError("import does not preserve original checkpoint state")
    ledger = [json.loads(line) for line in Path(evidence["ledger"]["path"]).read_text().splitlines()]
    sampler = reconstruct_sampler(state, ledger, state["contract"]["identities"], package["supervised"],
                                  SUPPORTED_SOURCES.get(evidence["source"]["sha256"]))
    if sampler != package["sampler"]:
        raise ValueError("import sampler differs from immutable evidence")
    return state



def canonical_steps(state, sampler):
    updates = iter(state["steps"])
    return [{**next(updates), **visit} for visit in sampler["visit_history"] if visit["updated"]]


def validate_source(package, source_sha, depth=0):
    if depth > 64:
        raise ValueError("diagnostic parent chain exceeds 64 checkpoints")
    if package.get("checkpoint_format") == IMPORT_FORMAT:
        state = validate_import(package)
        return dict(state=state, sampler=package["sampler"], supervised=package["supervised"],
                    evidence=package["evidence"], evidence_sha256=package["evidence_sha256"],
                    rng=None, origin_import_sha256=source_sha,
                    source_checkpoint_sha256=package["audit"]["source_checkpoint_sha256"])
    if (package.get("checkpoint_format") != FORMAT
            or package.get("unknown_historical_state") != UNKNOWN_STATE
            or any(package.get("audit", {}).get(name) is not False for name in (
                "strict_resume_eligible", "historical_bitwise_equivalence", "native_qualified",
                "stereo_qualified", "policy_qualified", "generalization_qualified"))):
        raise ValueError("diagnostic continuation must retain unknown historical state and qualification limits")
    parent_ref = package.get("parent_checkpoint")
    if (not isinstance(parent_ref, dict) or set(parent_ref) != {"path", "sha256"}
            or not isinstance(parent_ref["sha256"], str)
            or not re.fullmatch("[0-9a-f]{64}", parent_ref["sha256"])):
        raise ValueError("diagnostic parent path and SHA256 are required")
    parent_package, parent_sha = read_checkpoint(Path(parent_ref["path"]), parent_ref["sha256"])
    parent = validate_source(parent_package, parent_sha, depth + 1)
    state, sampler, rng = package["continued_state"], package["sampler"], package["current_rng"]
    configuration = package["continuation_config"]
    audit = package["audit"]
    if (package["origin_import_sha256"] != parent["origin_import_sha256"]
            or package["inherited_evidence"] != parent["evidence"]
            or package["inherited_evidence_sha256"] != parent["evidence_sha256"]
            or audit["parent_checkpoint_sha256"] != parent_sha
            or audit["source_import_sha256"] != parent["origin_import_sha256"]
            or audit["source_checkpoint_sha256"] != parent["source_checkpoint_sha256"]
            or state["contract"] != parent["state"]["contract"]
            or package["continuation_config_sha256"] != canonical_sha256(configuration)):
        raise ValueError("diagnostic parent lineage or configuration differs")
    _validated_history({**state, "sampler": sampler}, state["contract"]["identities"], parent["supervised"])
    previous_records = canonical_steps(parent["state"], parent["sampler"])
    previous_history = parent["sampler"]["visit_history"]
    new_updates = state["global_step"] - parent["state"]["global_step"]
    if (not 0 <= new_updates <= 16 or state["steps"][:len(previous_records)] != previous_records
            or sampler["visit_history"][:len(previous_history)] != previous_history
            or audit["initial_global_step"] != parent["state"]["global_step"]
            or audit["global_step"] != state["global_step"] or audit["new_training_updates"] != new_updates):
        raise ValueError("diagnostic sampler or update history differs from parent")
    _validated_rng(rng, torch.device("cpu" if rng["torch_cuda"] is None else "cuda"))
    validate_legacy_state({**state, "rng": rng["torch_cpu"], "cuda_rng": rng["torch_cuda"],
                           "counts": {"rgb_updates": state["global_step"]}})
    return dict(state=state, sampler=sampler, supervised=parent["supervised"],
                evidence=parent["evidence"], evidence_sha256=parent["evidence_sha256"],
                rng=rng, origin_import_sha256=parent["origin_import_sha256"],
                source_checkpoint_sha256=parent["source_checkpoint_sha256"])

def run_continuation(package, source_sha, model, optimizer, samples, output, *,
                     steps, wall_seconds, device, new_rng_seed=None, source_path=None, stop_requested=None):
    if (type(steps) is not int or not 1 <= steps <= 16
            or type(wall_seconds) is not int or not 1 <= wall_seconds <= 120):
        raise ValueError("diagnostic budget requires 1-16 updates and 1-120 seconds")
    if not isinstance(source_sha, str) or not re.fullmatch("[0-9a-f]{64}", source_sha):
        raise ValueError("source import SHA256 required")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if source_path is None or digest_file(source_path) != source_sha:
        raise ValueError("source checkpoint path and expected SHA256 are required")
    disk_package, _ = read_checkpoint(source_path, source_sha)
    if not state_equal(package, disk_package):
        raise ValueError("source checkpoint state differs from SHA256-bound file")
    source = validate_source(package, source_sha)
    original = source["state"]
    if source["rng"] is None:
        if type(new_rng_seed) is not int or not 0 <= new_rng_seed <= 2**32 - 1:
            raise ValueError("explicit new Python/NumPy RNG seed required; historical states remain unknown")
    elif new_rng_seed is not None:
        raise ValueError("cannot seed a continuation with recorded current RNG; omit --new-rng-seed")
    contract = original["contract"]
    cohort = contract["identities"]
    if len(samples) != len(cohort):
        raise ValueError("continuation sample count differs from frozen cohort")
    # Production CLI supplies decoded exact samples; the helper also supports CPU fixture identities.
    identities = [sample if isinstance(sample, dict) else sample_identity(sample) for sample in samples]
    if identities != cohort:
        raise ValueError("continuation sample order or pair SHA differs")
    current = model.state_dict()
    if (current.keys() != original["model"].keys() or any(
            current[name].shape != value.shape or current[name].dtype != value.dtype
            for name, value in original["model"].items())):
        raise ValueError("continuation model differs from legacy state")
    _validate_adam_state(original["optimizer"], sorted(original["optimizer"]["state"]),
                         optimizer, original["global_step"])
    if device.type not in ("cpu", "cuda") or (device.type == "cpu") != (source["rng"]["torch_cuda"] is None if source["rng"] is not None else original["cuda_rng"] is None):
        raise ValueError("continuation device differs from saved Torch RNG")
    output.mkdir(parents=True, exist_ok=False)
    model.load_state_dict(original["model"], strict=True)
    optimizer.load_state_dict(copy.deepcopy(original["optimizer"]))
    # Seed only at the import boundary; later diagnostic checkpoints restore every current RNG.
    if source["rng"] is None:
        random.seed(new_rng_seed)
        np.random.seed(new_rng_seed)
        torch.set_rng_state(original["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(original["cuda_rng"], device)
        rng_scope = "new_python_numpy_only"
    else:
        restore_rng(source["rng"], device)
        rng_scope = "restored_current_rng"
    history = copy.deepcopy(source["sampler"]["visit_history"])
    records = copy.deepcopy(canonical_steps(original, source["sampler"]))
    initial_step = len(records)
    stopped = stop_requested or (lambda: False)
    started = time.monotonic()
    with (output / "updates.jsonl").open("x") as stream:
        while len(records) - initial_step < steps and not stopped() and time.monotonic() - started < wall_seconds:
            index = len(history) % len(cohort)
            sample = samples[index]
            view = make_view(sample, "center", contract["size"], device)
            updated = has_supervision(view, contract["mode"])
            if updated != source["supervised"][index]:
                raise ValueError("diagnostic supervision differs from imported cohort")
            visit = dict(visit=len(history) + 1, file_name=cohort[index]["file_name"],
                         pair_sha256=cohort[index]["pair_sha256"], updated=updated)
            if updated:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = diagnostic_loss(model(view[0]), *view[1:], contract["mode"])
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite diagnostic loss")
                loss.backward()
                gradient = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None))
                if not torch.isfinite(gradient) or gradient <= 0:
                    raise ValueError("zero or nonfinite diagnostic gradient")
                optimizer.step()
                record = dict(visit, update=len(records) + 1, loss=float(loss.detach()),
                              grad_norm=float(gradient), empty_mask=not bool(view[1].any()),
                              elapsed_s=time.monotonic() - started)
                records.append(record)
            else:
                record = dict(visit, reason="no_valid_depth_label", updates_completed=len(records))
            history.append(visit)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
    current_rng = capture_rng(device)
    sampler = dict(kind="fixed_order_cyclic_v1", cohort_size=len(cohort), visits_completed=len(history),
                   next_index=len(history) % len(cohort), skipped_visits=len(history) - len(records),
                   visit_history=history)
    result = dict(status="diagnostic_continuation_only", source_import_sha256=source["origin_import_sha256"],
                  parent_checkpoint_sha256=source_sha,
                  source_checkpoint_sha256=source["source_checkpoint_sha256"],
                  new_training_updates=len(records) - initial_step, global_step=len(records),
                  initial_global_step=initial_step, initial_visits=source["sampler"]["visits_completed"],
                  visits_completed=len(history), new_rng_seed=new_rng_seed,
                  rng_seed_scope=rng_scope, historical_bitwise_equivalence=False,
                  strict_resume_eligible=False, native_qualified=False, stereo_qualified=False,
                  policy_qualified=False, generalization_qualified=False,
                  stop_reason=("interrupted" if stopped() else "budget" if len(records) - initial_step == steps else "wall_clock"),
                  optimization_wall_s=time.monotonic() - started, max_updates=steps,
                  max_wall_seconds=wall_seconds, evaluation_episodes=0)
    continued = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     global_step=len(records), steps=records, contract=contract)
    configuration = dict(mode=contract["mode"], device=str(device), size=contract["size"],
                         optimization_view="center", learning_rate=contract["learning_rate"],
                         new_rng_seed=new_rng_seed, rng_seed_scope=rng_scope,
                         budget_scope="optimization_loop_only", max_updates=steps,
                         max_wall_seconds=wall_seconds, torch_version=str(torch.__version__),
                         numpy_version=np.__version__, torch_threads=torch.get_num_threads(),
                         source_sha256=digest_file(Path(__file__)), scheduler=None, scaler=None)
    checkpoint = dict(checkpoint_format=FORMAT, continued_state=continued, current_rng=current_rng,
                      continuation_config=configuration,
                      continuation_config_sha256=canonical_sha256(configuration),
                      sampler=sampler, unknown_historical_state=list(UNKNOWN_STATE),
                      parent_checkpoint={"path": str(source_path.resolve()), "sha256": source_sha},
                      origin_import_sha256=source["origin_import_sha256"],
                      inherited_evidence=source["evidence"],
                      inherited_evidence_sha256=source["evidence_sha256"], audit=result)
    write_checkpoint(output / "diagnostic.pt", checkpoint)
    with (output / "result.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--import-checkpoint", dest="checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", "--import-sha256", dest="checkpoint_sha256", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--wall-seconds", type=int, default=120)
    parser.add_argument("--device", required=True)
    parser.add_argument("--new-rng-seed", type=int, help="required only for import; forbidden on subsequent continuation")
    args = parser.parse_args()
    if not 1 <= args.steps <= 16 or not 1 <= args.wall_seconds <= 120:
        parser.error("diagnostic budget is at most 16 updates and 120 seconds")
    package, sha = read_checkpoint(args.checkpoint, args.checkpoint_sha256)
    source = validate_source(package, sha)
    state = source["state"]
    cohort, supervised = verified_cohort(args.dataset_root, source["evidence"], state["contract"])
    if cohort != state["contract"]["identities"] or supervised != source["supervised"]:
        raise ValueError("continuation data differs from import")
    from scripts.pretrain_dh116_wireseg import read_subset
    from go2_pvcnn.control.dh116.tabletop_wireseg_pretraining import WireSegPretrainer, load_wireseg_sample
    train_path = Path(source["evidence"]["train"]["path"])
    manifest = json.loads(train_path.read_bytes())
    rows = islice(read_subset(args.dataset_root, train_path, manifest["blobs"],
                             manifest["catalogue_sha256"], "train"), len(cohort))
    samples = [load_wireseg_sample(row, args.dataset_root / "metric_depth_npy") for row in rows]
    device = torch.device(args.device)
    model = WireSegPretrainer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=state["contract"]["learning_rate"])
    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
    previous = {number: signal.signal(number, request_stop) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = run_continuation(package, sha, model, optimizer, samples, args.output, steps=args.steps,
                                  wall_seconds=args.wall_seconds, device=device, new_rng_seed=args.new_rng_seed,
                                  source_path=args.checkpoint, stop_requested=lambda: stop["requested"])
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    print(json.dumps(result, indent=2, allow_nan=False))
    if stop["requested"] or result["stop_reason"] == "interrupted":
        raise SystemExit(130)


if __name__ == "__main__":
    main()
