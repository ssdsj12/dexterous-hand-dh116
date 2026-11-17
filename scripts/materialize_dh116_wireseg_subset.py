#!/usr/bin/env python3
"""Materialize an already audited local WireSeg depth shard without training.

The PNG/RLE/NPY decoders, pinned catalogue and ordered subset reader remain the
existing implementations. Only the independently bound derived directory is new.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from scripts import pretrain_dh116_wireseg as pretrain
from scripts.relay_wireseg_blobs import cache_path, digest_file, read_catalogue, verify_blob
from go2_pvcnn.control.dh116 import tabletop_wireseg_pretraining as wire
from go2_pvcnn.control.dh116.tabletop_wireseg_pretraining import extract_depth_shard

FORMAT = "wireseg-audited-materialization-v1"
INPUTS = ("subset", "ledger", "admission", "heldout")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _pinned_bytes(path, expected):
    path = Path(path).resolve(strict=True)
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("expected SHA256 must be explicit")
    raw = path.read_bytes()
    if _sha(raw) != expected:
        raise ValueError(f"input SHA256 mismatch: {path}")
    return {"path": str(path), "sha256": expected}, raw


class _SubsetSnapshot:
    """Supply the existing subset reader with precisely the verified bytes."""
    def __init__(self, raw):
        self.raw = raw

    def read_bytes(self):
        return self.raw


def _implementation():
    root = Path(__file__).resolve().parents[1]
    paths = ("scripts/materialize_dh116_wireseg_subset.py",
             "scripts/pretrain_dh116_wireseg.py", "scripts/relay_wireseg_blobs.py",
             "src/go2_pvcnn/control/dh116/tabletop_wireseg_pretraining.py")
    return {name: digest_file(root / name) for name in paths}


def _prepare(args):
    root = Path(args["data_root"]).resolve(strict=True)
    checked = {key: _pinned_bytes(args[key], args[key + "_sha256"]) for key in INPUTS}
    binding = {key: value[0] for key, value in checked.items()}
    subset = json.loads(checked["subset"][1])
    admission = json.loads(checked["admission"][1])
    ledger = [json.loads(line) for line in checked["ledger"][1].splitlines()]
    catalogue_sha, catalogue = read_catalogue(root / "hf_revision_manifest.json")
    if (admission["revision"] != wire.WIRESEG_REVISION
            or admission["catalogue_file_sha256"] != catalogue_sha
            or admission["subset_manifest_sha256"] != binding["subset"]["sha256"]
            or admission["candidate_manifest_sha256"] != binding["subset"]["sha256"]
            or admission["audit_ledger_sha256"] != binding["ledger"]["sha256"]):
        raise ValueError("admission identity differs from pinned inputs")
    heldout = json.loads(checked["heldout"][1])
    depth = admission["new_depth"]
    names = dict(subset["blobs"])
    for name, sha in heldout["blobs"].items():
        if name in names and names[name] != sha:
            raise ValueError("heldout blob identity conflict")
        names[name] = sha
    names[depth["relative_path"]] = depth["sha256"]
    for name, sha in names.items():
        if (name not in catalogue or catalogue[name].sha256 != sha
                or not verify_blob(cache_path(root, name), catalogue[name])):
            raise ValueError(f"source blob verification failed: {name}")
    rgb = admission["new_rgb"]
    for item in (rgb, depth):
        blob = catalogue[item["relative_path"]]
        if item["size"] != blob.size or item["sha256"] != blob.sha256:
            raise ValueError("admission blob size or SHA mismatch")
    if subset["blobs"] != {rgb["relative_path"]: rgb["sha256"]}:
        raise ValueError("admission RGB does not bind the exact subset")
    count = len(subset["samples"])
    if (not count or len(ledger) != count or depth["member_count"] != count
            or admission["exact_pairs_new_depth"] != count
            or admission["counts"]["decoded_rows"] != count
            or admission["counts"]["valid_train_rows"] != count
            or admission["counts"]["missing_depth_rows"] != 0):
        raise ValueError("admitted sample count mismatch")
    seen = set()
    for identity, row in zip(subset["samples"], ledger):
        file_name = identity["file_name"]
        member = wire.depth_member(file_name, row["family"])
        if (row["file_name"] != file_name or row["rgb_blob"] != identity["blob"]
                or file_name in seen or row["depth_member"] != member
                or row["split"] != "train" or wire.sample_split(row["family"]) != "train"
                or row["admitted_train_sample"] is not True
                or row["depth_dtype"] != "float32" or row["depth_units"] != "m"):
            raise ValueError("ledger order, identity, admission or depth domain mismatch")
        seen.add(file_name)
        for key in ("depth_npy_sha256", "rgb_sha256", "coco_mask_json_sha256", "pair_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise ValueError("ledger SHA256 malformed")
        pair = _sha(f'{file_name}\n{row["rgb_sha256"]}\n{row["depth_npy_sha256"]}\n'.encode())
        if pair != row["pair_sha256"]:
            raise ValueError("ledger pair SHA256 mismatch")
    # Reuse the full subset reader to enforce its revision, family, duplicate
    # identity and metadata ordering contract. Only holdout metadata is retained.
    held_families, held_names = set(), set()
    for row in pretrain.read_subset(root, _SubsetSnapshot(checked["heldout"][1]), names,
                                    catalogue_sha, "validation"):
        held_families.add(row["scene_family"])
        held_names.add(row["file_name"])
    train_families = {row["family"] for row in ledger}
    if train_families & held_families or seen & held_names:
        raise ValueError("training and heldout source identities overlap")
    binding.update(format=FORMAT, data_root=str(root), catalogue_sha256=catalogue_sha,
                   revision=wire.WIRESEG_REVISION, verified_blobs=names,
                   implementation_sha256=_implementation(),
                   source_family_holdout=dict(training_families=sorted(train_families),
                       heldout_families=sorted(held_families),
                       heldout_samples=len(held_names), training_samples=count))
    return binding, ledger, cache_path(root, depth["relative_path"]), checked["subset"][1]


def _write_exclusive(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".materialize-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # Link publishes a complete file and refuses to replace an existing file.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _scan(output, ledger, *, complete):
    allowed = {row["depth_member"] for row in ledger}
    top = {"binding.json", "materialization.json", ".materialize.lock"}
    depth_root = output / "metric_depth_npy"
    if output.is_symlink():
        raise ValueError("derived output cannot be a symlink")
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"derived symlink rejected: {path}")
        if path.is_dir():
            continue
        rel = path.relative_to(output).as_posix()
        if rel in top:
            continue
        if not rel.startswith("metric_depth_npy/") or rel[len("metric_depth_npy/"):] not in allowed:
            raise ValueError(f"unbound derived file: {rel}")
    entries = []
    for row in ledger:
        path = depth_root / row["depth_member"]
        if not path.is_file():
            if complete:
                raise ValueError(f"missing materialized member: {row['depth_member']}")
            continue
        if digest_file(path) != row["depth_npy_sha256"]:
            raise ValueError(f"materialized member SHA256 mismatch: {row['depth_member']}")
        entries.append(dict(file_name=row["file_name"], depth_member=row["depth_member"],
                            depth_npy_sha256=row["depth_npy_sha256"], bytes=path.stat().st_size,
                            pair_sha256=row["pair_sha256"]))
    return entries


def _manifest(binding, ledger, entries):
    return dict(format=FORMAT, binding=binding, samples=entries,
                depth_root="metric_depth_npy", depth_units="m",
                depth_validity="isfinite and >0; no scaling",
                source_family_holdout=binding["source_family_holdout"],
                counts=dict(samples=len(ledger), instances=sum(row["instances"] for row in ledger),
                            valid_depth_pixels=sum(row["depth_valid_pixels"]["full"] for row in ledger)),
                optimizer_updates=0, model_inference_calls=0, native_started=False,
                quality_qualified=False, stereo_correspondence_supervised=False,
                existing_training_cohort_replaced=False)


def _decoded(binding, ledger, output, subset_raw):
    rows = pretrain.read_subset(Path(binding["data_root"]), _SubsetSnapshot(subset_raw),
                               binding["verified_blobs"], binding["catalogue_sha256"], "train")
    members = {row["depth_member"] for row in ledger}
    count = 0
    for row in rows:
        if count >= len(ledger):
            raise ValueError("decoded subset has excess rows")
        expected = ledger[count]
        sample = wire.load_wireseg_sample(row, output / "metric_depth_npy", depth_members=members)
        if (sample.file_name != expected["file_name"] or sample.pair_sha256 != expected["pair_sha256"]
                or sample.scene_family != expected["family"] or sample.split != expected["split"]
                or list(sample.rgb.shape) != expected["rgb_shape"]
                or _sha(row["image"]["bytes"]) != expected["rgb_sha256"]
                or _sha(row["masks_rle"].encode()) != expected["coco_mask_json_sha256"]
                or len(sample.instance_masks) != expected["instances"]
                or sample.depth_valid is None
                or int(sample.depth_valid.sum()) != expected["depth_valid_pixels"]["full"]):
            raise ValueError(f"decoded sample differs from prior audit: {expected['file_name']}")
        count += 1
        yield sample
    if count != len(ledger):
        raise ValueError("decoded subset incomplete")


@contextmanager
def _locked(output):
    lock = output / ".materialize.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def materialize(**args):
    """Resume only the identical input/source binding; never replace data bytes."""
    output_arg = Path(args["output"])
    output = output_arg.resolve()
    root = Path(args["data_root"]).resolve(strict=True)
    if output_arg.is_symlink() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("derived output must be independent of the original cache")
    binding, ledger, archive, subset_raw = _prepare(args)
    binding["output"] = str(output)
    output.mkdir(parents=True, exist_ok=True)
    with _locked(output):
        _scan(output, ledger, complete=False)
        binding_path = output / "binding.json"
        if binding_path.exists():
            if json.loads(binding_path.read_bytes()) != binding:
                raise ValueError("resume binding mismatch")
        else:
            if any((output / "metric_depth_npy").rglob("*")):
                raise ValueError("unbound preexisting derived data cannot be adopted")
            _write_exclusive(binding_path, binding)
        result = output / "materialization.json"
        if result.exists():
            expected = _manifest(binding, ledger, _scan(output, ledger, complete=True))
            if json.loads(result.read_bytes()) != expected:
                raise ValueError("completed materialization manifest mismatch")
            return expected
        members = extract_depth_shard(archive, output / "metric_depth_npy")
        if members != sorted(row["depth_member"] for row in ledger):
            raise ValueError("archive members differ from the admitted subset")
        entries = _scan(output, ledger, complete=True)
        for _sample in _decoded(binding, ledger, output, subset_raw):
            pass
        manifest = _manifest(binding, ledger, entries)
        _write_exclusive(result, manifest)
        return manifest


def read_materialized(output, *, expected_sha256):
    """Read this independently pinned cohort through the existing sample decoder."""
    output = Path(output)
    result = output / "materialization.json"
    _, raw = _pinned_bytes(result, expected_sha256)
    stored = json.loads(raw)
    old = stored["binding"]
    args = dict(data_root=old["data_root"])
    for key in INPUTS:
        args[key], args[key + "_sha256"] = old[key]["path"], old[key]["sha256"]
    binding, ledger, _, subset_raw = _prepare(args)
    binding["output"] = str(output.resolve())
    if (binding != old or json.loads((output / "binding.json").read_bytes()) != binding
            or stored != _manifest(binding, ledger, _scan(output, ledger, complete=True))):
        raise ValueError("materialized reader identity or manifest mismatch")
    yield from _decoded(binding, ledger, output, subset_raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for key in INPUTS:
        parser.add_argument("--" + key, type=Path, required=True)
        parser.add_argument("--" + key + "-sha256", required=True)
    args = vars(parser.parse_args())
    result = materialize(**args)
    print(json.dumps(dict(output=str(args["output"]),
                          manifest_sha256=digest_file(args["output"] / "materialization.json"),
                          counts=result["counts"], optimizer_updates=0), sort_keys=True))


if __name__ == "__main__":
    main()
