import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import diagnose_dh116_wireseg_overfit as diagnostic
from scripts.relay_wireseg_blobs import digest_file


class SmallDiagnostic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1)

    def forward(self, rgb):
        return self.conv(torch.nn.functional.dropout(rgb, p=.25, training=self.training))

    def transfer_state(self):
        return self.state_dict()


def assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            assert_state_equal(actual[name], value)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected):
            assert_state_equal(got, want)
    else:
        assert actual == expected


@pytest.fixture
def launch(tmp_path, monkeypatch):
    train_manifest = tmp_path / "train.json"
    train_manifest.write_text("fixed train cohort\n")
    validation_manifest = tmp_path / "validation.json"
    validation_manifest.write_text("fixed heldout cohort\n")

    def sample(index):
        return SimpleNamespace(
            file_name=f"sample-{index}", scene_family="synthetic-cpu", pair_sha256=f"{index+1:064x}",
            rgb=np.full((32, 32, 3), 15 + index, dtype=np.uint8),
            instance_masks=np.zeros((0, 32, 32), dtype=bool),
            depth_m=np.full((32, 32), 1 + index / 10, dtype=np.float32),
            depth_valid=np.full((32, 32), index not in {0, 4}, dtype=bool),
        )

    monkeypatch.setattr(diagnostic, "verified_sources", lambda *_: ("a" * 64, {}))
    monkeypatch.setattr(diagnostic, "prepare_depth", lambda *_: set())
    monkeypatch.setattr(diagnostic, "read_subset", lambda *args: iter(
        [sample(i) for i in (range(8) if args[-1] == "train" else (8,))]))
    monkeypatch.setattr(diagnostic, "load_wireseg_sample", lambda row, *a, **kw: row)
    monkeypatch.setattr(diagnostic, "WireSegPretrainer", SmallDiagnostic)
    def evaluate(*args):
        random.random()
        np.random.random()
        torch.rand(1)
        return {"visible_iou": None, "depth": None}

    monkeypatch.setattr(diagnostic, "evaluate", evaluate)

    def loss(pred, masks, depth, valid, mode):
        adjustment = (random.random() + np.random.random()) / 10
        return (pred[:, 0][valid] - depth[valid] - adjustment).square().mean()

    monkeypatch.setattr(diagnostic, "diagnostic_loss", loss)

    def run(name, steps, *, resume=None, head="depth-only"):
        output = tmp_path / name
        command = [
            "diagnose", "--dataset-root", str(tmp_path), "--subset-manifest", str(train_manifest),
            "--validation-manifest", str(validation_manifest), "--output", str(output), "--samples", "8",
            "--size", "32", "--steps", str(steps), "--heads", head,
        ]
        if resume is not None:
            command += ["--resume", str(resume), "--resume-sha256", digest_file(resume)]
        monkeypatch.setattr(sys, "argv", command)
        diagnostic.main()
        return output / head / "diagnostic.pt"

    return run, train_manifest


def test_cpu_resume_is_identical_and_preserves_skipped_visits(launch):
    run, _ = launch
    whole = torch.load(run("uninterrupted", 8), weights_only=True)
    first_path = run("first", 3)
    first = torch.load(first_path, weights_only=True)
    initial_sha = digest_file(first_path)
    second_path = run("resumed", 5, resume=first_path)
    resumed = torch.load(second_path, weights_only=True)
    for name in ("model", "optimizer", "rng_states", "sampler", "global_step", "dataset_manifest_sha256",
                 "training_config_sha256", "scheduler", "scaler"):
        assert_state_equal(resumed[name], whole[name])
    assert resumed["global_step"] == 8 and resumed["scheduler"] is resumed["scaler"] is None
    assert first["sampler"]["visits_completed"] == 4 and first["sampler"]["skipped_visits"] == 1
    assert resumed["sampler"]["visits_completed"] == 11 and resumed["sampler"]["skipped_visits"] == 3
    assert [(row["visit"], row["updated"]) for row in resumed["sampler"]["visit_history"]] == [
        (i, i not in (1, 5, 9)) for i in range(1, 12)
    ]
    resumed_log = [json.loads(row) for row in second_path.with_name("updates.jsonl").read_text().splitlines()]
    assert resumed_log[0]["visit"] == 5 and resumed_log[0]["updated"] is False
    assert [row["update"] for row in resumed_log if row["updated"]] == [4, 5, 6, 7, 8]
    assert digest_file(first_path) == initial_sha


def test_resume_rejects_manifest_cursor_and_missing_optimizer(launch, tmp_path):
    run, manifest = launch
    source = run("source", 3)
    original_sha = digest_file(source)
    state = torch.load(source, weights_only=True)
    state["sampler"]["visits_completed"] += 1
    tampered = tmp_path / "tampered.pt"
    torch.save(state, tampered)
    with pytest.raises(ValueError, match="sampler"):
        run("invalid-cursor", 1, resume=tampered)

    state = torch.load(source, weights_only=True)
    del state["optimizer"]
    torch.save(state, tampered)
    with pytest.raises(ValueError, match="optimizer"):
        run("invalid-optimizer", 1, resume=tampered)
    state = torch.load(source, weights_only=True)
    adam = next(iter(state["optimizer"]["state"].values()))
    del adam["exp_avg"]
    torch.save(state, tampered)
    with pytest.raises(ValueError, match="AdamW"):
        run("invalid-moment", 1, resume=tampered)
    state = torch.load(source, weights_only=True)
    state["training_config"]["size"] = 33
    torch.save(state, tampered)
    with pytest.raises(ValueError, match="training config"):
        run("invalid-config", 1, resume=tampered)

    manifest.write_text("changed bytes but same mocked sample identity\n")
    with pytest.raises(ValueError, match="dataset manifest"):
        run("invalid-manifest", 1, resume=source)
    assert digest_file(source) == original_sha


def test_checkpoint_requires_the_expected_source_digest(launch):
    from scripts.wireseg_checkpoint_state import read_checkpoint

    checkpoint = launch[0]("digest-source", 1)
    with pytest.raises(ValueError, match="SHA256 differs"):
        read_checkpoint(checkpoint, "0" * 64)
    loaded, digest = read_checkpoint(checkpoint, digest_file(checkpoint))
    assert digest == digest_file(checkpoint) and loaded["global_step"] == 1


def test_checkpoint_save_and_output_are_no_overwrite(launch, tmp_path):
    from scripts.wireseg_checkpoint_state import write_checkpoint

    run, _ = launch
    checkpoint = run("original", 1)
    original_sha = digest_file(checkpoint)
    with pytest.raises(FileExistsError):
        run("original", 1)
    with pytest.raises(FileExistsError):
        write_checkpoint(checkpoint, {})
    assert digest_file(checkpoint) == original_sha
    assert not list(tmp_path.glob("**/.checkpoint-*"))


def test_legacy_audit_reports_missing_fields_and_resume_refuses_before_data(launch, tmp_path, monkeypatch, capsys):
    legacy = tmp_path / "legacy.pt"
    torch.save({
        "model": {}, "optimizer": {"state": {1: {}}, "param_groups": []},
        "rng": torch.get_rng_state(), "cuda_rng": None, "global_step": 128,
        "steps": [{"update": 128, "visit": 158}], "contract": {"mode": "depth-only"},
    }, legacy)
    expected_sha = digest_file(legacy)
    monkeypatch.setattr(sys, "argv", ["diagnose", "--audit-checkpoint", str(legacy)])
    diagnostic.main()
    audit = json.loads(capsys.readouterr().out)
    assert audit["status"] == "legacy_strict_resume_blocked"
    assert audit["global_step"] == 128 and audit["last_recorded_visit"] == 158
    assert audit["sha256"] == expected_sha
    assert {"rng_states.python", "rng_states.numpy", "sampler", "scheduler", "scaler"} <= set(audit["missing"])
    monkeypatch.setattr(diagnostic, "verified_sources", lambda *_: pytest.fail("legacy resume read dataset"))
    with pytest.raises(ValueError, match="legacy.*strict resume"):
        launch[0]("blocked", 1, resume=legacy)
    assert digest_file(legacy) == expected_sha
