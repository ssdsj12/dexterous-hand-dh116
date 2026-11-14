"""Offline materialization uses real PNG/RLE/NPY/parquet fixtures, no model."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import pretrain_dh116_wireseg as pretrain
from scripts.relay_wireseg_blobs import Blob, digest_file
from go2_pvcnn.control.dh116 import tabletop_wireseg_pretraining as wire


def module():
    from scripts import materialize_dh116_wireseg_subset
    return materialize_dh116_wireseg_subset


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return digest_file(path)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "data").mkdir()
    (root / "depth").mkdir()
    png = io.BytesIO()
    Image.new("RGB", (2, 2), (12, 34, 56)).save(png, format="PNG")
    masks = json.dumps([{"size": [2, 2], "counts": [0, 1, 3]}])
    train = [dict(file_name=f"flying_wire/2_cable/{i:06d}.png", scene_family="flying_wire",
                  image={"bytes": png.getvalue()}, masks_rle=masks, num_instances=1, n_cable=2)
             for i in range(2)]
    held = [dict(train[0], file_name="wire_on_plane/2_cable/000000.png",
                 scene_family="wire_on_plane")]
    train_blob, val_blob, depth_blob = "data/train-a.parquet", "data/train-b.parquet", "depth/depth-a.tar.gz"
    pq.write_table(pa.Table.from_pylist(train), root / train_blob)
    pq.write_table(pa.Table.from_pylist(held), root / val_blob)
    payloads = []
    with tarfile.open(root / depth_blob, "w:gz") as bundle:
        for row in train:
            stream = io.BytesIO()
            np.save(stream, np.array([[1.25, 0], [0, 0]], dtype=np.float32))
            payload = stream.getvalue()
            member = wire.depth_member(row["file_name"], row["scene_family"])
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
            payloads.append((member, payload))
    catalog = {name: Blob(name, (root / name).stat().st_size, digest_file(root / name))
               for name in (train_blob, val_blob, depth_blob)}
    catalog_sha = dump(root / "hf_revision_manifest.json", {"fixture": True})
    monkeypatch.setattr(module(), "read_catalogue", lambda path: (catalog_sha, catalog))
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    subset = pretrain.make_subset(root, {train_blob: catalog[train_blob].sha256},
                                  catalog_sha, "train", 2)
    heldout = pretrain.make_subset(root, {val_blob: catalog[val_blob].sha256},
                                   catalog_sha, "validation", 1)
    subset_path, heldout_path = evidence / "subset.json", evidence / "heldout.json"
    subset_sha, heldout_sha = dump(subset_path, subset), dump(heldout_path, heldout)
    ledger = []
    for row, (member, payload) in zip(train, payloads):
        rgb_sha = hashlib.sha256(row["image"]["bytes"]).hexdigest()
        depth_sha = hashlib.sha256(payload).hexdigest()
        pair = hashlib.sha256(f'{row["file_name"]}\n{rgb_sha}\n{depth_sha}\n'.encode()).hexdigest()
        ledger.append(dict(file_name=row["file_name"], rgb_blob=train_blob,
                           family="flying_wire", split="train", depth_member=member,
                           depth_npy_sha256=depth_sha, rgb_sha256=rgb_sha,
                           coco_mask_json_sha256=hashlib.sha256(masks.encode()).hexdigest(),
                           pair_sha256=pair, admitted_train_sample=True, instances=1,
                           rgb_shape=[2, 2, 3], depth_dtype="float32", depth_units="m",
                           depth_valid_pixels={"full": 1}, empty_instances=0))
    ledger_path = evidence / "samples.jsonl"
    ledger_path.write_text("".join(json.dumps(row) + "\n" for row in ledger))
    admission = dict(revision=wire.WIRESEG_REVISION, catalogue_file_sha256=catalog_sha,
                     subset_manifest_sha256=subset_sha, candidate_manifest_sha256=subset_sha,
                     audit_ledger_sha256=digest_file(ledger_path), exact_pairs_new_depth=2,
                     new_depth=dict(relative_path=depth_blob, sha256=catalog[depth_blob].sha256,
                                    size=catalog[depth_blob].size, member_count=2),
                     new_rgb=dict(relative_path=train_blob, sha256=catalog[train_blob].sha256,
                                  size=catalog[train_blob].size),
                     counts=dict(decoded_rows=2, valid_train_rows=2, missing_depth_rows=0,
                                 instance_masks=2))
    admission_path = evidence / "admission.json"
    admission_sha = dump(admission_path, admission)
    kwargs = dict(data_root=root, subset=subset_path, subset_sha256=subset_sha,
                  ledger=ledger_path, ledger_sha256=digest_file(ledger_path),
                  admission=admission_path, admission_sha256=admission_sha,
                  heldout=heldout_path, heldout_sha256=heldout_sha,
                  output=tmp_path / "derived")
    return kwargs, ledger, payloads


def test_materializes_exact_pairs_readable_by_existing_loader_and_idempotent(fixture):
    args, ledger, payloads = fixture
    manifest = module().materialize(**args)
    assert manifest["counts"] == {"samples": 2, "instances": 2, "valid_depth_pixels": 2}
    assert manifest["source_family_holdout"]["training_families"] == ["flying_wire"]
    assert manifest["source_family_holdout"]["heldout_families"] == ["wire_on_plane"]
    assert manifest["optimizer_updates"] == 0
    assert manifest["quality_qualified"] is False
    paths = [args["output"] / "metric_depth_npy" / member for member, _ in payloads]
    mtimes = [p.stat().st_mtime_ns for p in paths]
    for p, (_, raw) in zip(paths, payloads):
        assert p.read_bytes() == raw
    result = args["output"] / "materialization.json"
    original = result.read_bytes()
    assert module().materialize(**args) == manifest
    assert result.read_bytes() == original
    assert [p.stat().st_mtime_ns for p in paths] == mtimes
    assert not (args["data_root"] / "metric_depth_npy").exists()
    samples = list(module().read_materialized(args["output"], expected_sha256=digest_file(result)))
    assert [s.pair_sha256 for s in samples] == [row["pair_sha256"] for row in ledger]
    assert samples[0].depth_m[0, 0] == 1.25


@pytest.mark.parametrize("key", ["subset_sha256", "ledger_sha256", "admission_sha256", "heldout_sha256"])
def test_input_digest_mismatch_rejected_before_output(fixture, key):
    args, _, _ = fixture
    args[key] = "0" * 64
    with pytest.raises(ValueError, match="SHA"):
        module().materialize(**args)
    assert not args["output"].exists()


def repin(args, key, value):
    if key == "ledger":
        args[key].write_text("".join(json.dumps(row) + "\n" for row in value))
    else:
        dump(args[key], value)
    args[key + "_sha256"] = digest_file(args[key])
    if key in ("subset", "ledger"):
        admission = json.loads(args["admission"].read_text())
        if key == "ledger":
            admission["audit_ledger_sha256"] = args["ledger_sha256"]
        else:
            admission["subset_manifest_sha256"] = args["subset_sha256"]
            admission["candidate_manifest_sha256"] = args["subset_sha256"]
        args["admission_sha256"] = dump(args["admission"], admission)


@pytest.mark.parametrize("change", ["reorder", "pair", "depth_member", "split", "admitted", "duplicate"])
def test_ledger_identity_tampering_rejected(fixture, change):
    args, ledger, _ = fixture
    if change == "reorder":
        ledger.reverse()
    elif change == "duplicate":
        ledger[1] = ledger[0]
    else:
        key, value = {"pair": ("pair_sha256", "0" * 64),
                      "depth_member": ("depth_member", "../escape.npy"),
                      "split": ("split", "validation"),
                      "admitted": ("admitted_train_sample", False)}[change]
        ledger[0][key] = value
    repin(args, "ledger", ledger)
    with pytest.raises(ValueError):
        module().materialize(**args)
    assert not (args["output"] / "materialization.json").exists()


def test_actual_rgb_blob_tamper_is_not_admitted(fixture):
    args, ledger, _ = fixture
    (args["data_root"] / ledger[0]["rgb_blob"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="blob"):
        module().materialize(**args)
    assert not args["output"].exists()


def test_interrupted_extraction_resumes_without_replacing_existing_payload(fixture, monkeypatch):
    args, _, payloads = fixture
    real = module().extract_depth_shard
    def interrupted(archive, root):
        member, raw = payloads[0]
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        raise RuntimeError("simulated interruption")
    monkeypatch.setattr(module(), "extract_depth_shard", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        module().materialize(**args)
    first = args["output"] / "metric_depth_npy" / payloads[0][0]
    mtime = first.stat().st_mtime_ns
    monkeypatch.setattr(module(), "extract_depth_shard", real)
    assert module().materialize(**args)["counts"]["samples"] == 2
    assert first.stat().st_mtime_ns == mtime


@pytest.mark.parametrize("damage", ["payload", "binding", "extra", "symlink", "manifest"])
def test_resume_and_reader_reject_tampered_derived_state(fixture, damage):
    args, _, payloads = fixture
    module().materialize(**args)
    result = args["output"] / "materialization.json"
    expected = digest_file(result)
    target = args["output"] / "metric_depth_npy" / payloads[0][0]
    if damage == "payload":
        target.write_bytes(b"corrupt")
    elif damage == "binding":
        binding = json.loads((args["output"] / "binding.json").read_text())
        binding["subset"]["sha256"] = "0" * 64
        dump(args["output"] / "binding.json", binding)
    elif damage == "extra":
        (target.parent / "unbound.npy").write_bytes(target.read_bytes())
    elif damage == "symlink":
        raw = target.read_bytes()
        target.unlink()
        outside = args["output"].parent / "outside.npy"
        outside.write_bytes(raw)
        target.symlink_to(outside)
    else:
        report = json.loads(result.read_text())
        report["counts"]["samples"] = 999
        dump(result, report)
    with pytest.raises(ValueError):
        module().materialize(**args)
    with pytest.raises(ValueError):
        list(module().read_materialized(args["output"], expected_sha256=expected))


def test_output_cannot_modify_original_cache(fixture):
    args, _, _ = fixture
    args["output"] = args["data_root"] / "derived"
    with pytest.raises(ValueError, match="independent"):
        module().materialize(**args)


def test_heldout_cannot_be_training_family(fixture):
    args, _, _ = fixture
    train = json.loads(args["subset"].read_text())
    repin(args, "heldout", train)
    with pytest.raises(ValueError):
        module().materialize(**args)
    assert not args["output"].exists()


def test_runtime_helper_change_refuses_resume_and_readback(fixture, monkeypatch):
    args, _, _ = fixture
    module().materialize(**args)
    result = args["output"] / "materialization.json"
    expected = digest_file(result)
    old = module()._implementation()
    old["scripts/pretrain_dh116_wireseg.py"] = "0" * 64
    monkeypatch.setattr(module(), "_implementation", lambda: old)
    with pytest.raises(ValueError, match="binding"):
        module().materialize(**args)
    with pytest.raises(ValueError, match="identity"):
        list(module().read_materialized(args["output"], expected_sha256=expected))


def test_concurrent_writer_refuses_without_replacing_manifest(fixture):
    import fcntl
    args, _, _ = fixture
    module().materialize(**args)
    original = (args["output"] / "materialization.json").read_bytes()
    with (args["output"] / ".materialize.lock").open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            module().materialize(**args)
    assert (args["output"] / "materialization.json").read_bytes() == original


@pytest.mark.parametrize("change", ["mask_sha", "shape", "valid_count", "instances"])
def test_actual_decode_must_match_audit_not_just_pair_sha(fixture, change):
    args, ledger, _ = fixture
    if change == "mask_sha":
        ledger[0]["coco_mask_json_sha256"] = "0" * 64
    elif change == "shape":
        ledger[0]["rgb_shape"] = [9, 9, 3]
    elif change == "instances":
        ledger[0]["instances"] = 2
    else:
        ledger[0]["depth_valid_pixels"]["full"] = 4
    repin(args, "ledger", ledger)
    with pytest.raises(ValueError, match="decoded sample"):
        module().materialize(**args)
    assert not (args["output"] / "materialization.json").exists()


def test_reordered_subset_and_ledger_cannot_bypass_actual_parquet_order(fixture):
    args, ledger, _ = fixture
    subset = json.loads(args["subset"].read_text())
    subset["samples"].reverse()
    ledger.reverse()
    repin(args, "subset", subset)
    repin(args, "ledger", ledger)
    with pytest.raises(ValueError, match="reordered"):
        module().materialize(**args)
    assert not (args["output"] / "materialization.json").exists()


@pytest.mark.parametrize("key", ["subset", "admission", "ledger", "heldout"])
def test_changed_metadata_read_cannot_bypass_expected_digest(fixture, monkeypatch, key):
    args, ledger, _ = fixture
    path = args[key]
    good = path.read_bytes()
    if key == "ledger":
        ledger[0]["pair_sha256"] = "0" * 64
        repin(args, key, ledger)
    else:
        bad = json.loads(good)
        bad["revision" if key == "admission" else "version"] = "invalid"
        repin(args, key, bad)
    original_digest = module().digest_file
    def digest_then_swap(candidate):
        value = original_digest(candidate)
        if Path(candidate) == path:
            path.write_bytes(good)
        return value
    monkeypatch.setattr(module(), "digest_file", digest_then_swap)
    with pytest.raises(ValueError):
        module().materialize(**args)
    assert not (args["output"] / "materialization.json").exists()


def test_changed_manifest_read_cannot_bypass_expected_digest(fixture, monkeypatch):
    args, _, _ = fixture
    module().materialize(**args)
    path = args["output"] / "materialization.json"
    good = path.read_bytes()
    bad = json.loads(good)
    bad["counts"]["samples"] = 999
    expected = dump(path, bad)
    original_digest = module().digest_file
    def digest_then_swap(candidate):
        value = original_digest(candidate)
        if Path(candidate) == path:
            path.write_bytes(good)
        return value
    monkeypatch.setattr(module(), "digest_file", digest_then_swap)
    with pytest.raises(ValueError):
        list(module().read_materialized(args["output"], expected_sha256=expected))


def test_decoder_uses_verified_subset_snapshot_even_if_path_changes(fixture, monkeypatch):
    args, _, _ = fixture
    original = pretrain.read_subset
    def swap_path_after_check(root, subset, names, sha, split):
        if split == "train":
            bad = json.loads(args["subset"].read_bytes())
            bad["version"] = "changed after verification"
            dump(args["subset"], bad)
        yield from original(root, subset, names, sha, split)
    monkeypatch.setattr(pretrain, "read_subset", swap_path_after_check)
    manifest = module().materialize(**args)
    assert manifest["counts"]["samples"] == 2
    # The accepted rows came from verified bytes; the changed path is not trusted
    # on a future resume/read.
    with pytest.raises(ValueError, match="SHA"):
        module().materialize(**args)
