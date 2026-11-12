import copy
import json
from pathlib import Path

import pytest
import torch

from scripts import import_dh116_wireseg_legacy as migration
from scripts.relay_wireseg_blobs import digest_file
from scripts.wireseg_checkpoint_state import require_v2_state
from tests.test_dh116_wireseg_checkpoint_resume import assert_state_equal


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    cohort = [dict(file_name=f"fixture/{i}.png", family="fixture", pair_sha256=f"{i + 1:064x}")
              for i in range(8)]
    supervised = [i not in (0, 4) for i in range(8)]
    files = {}
    def put(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value) if not isinstance(value, str) else value)
        files[name] = dict(path=str(path), sha256=digest_file(path))
    put("source", "fixture source whose fixed cyclic algorithm is selected explicitly")
    monkeypatch.setitem(migration.SUPPORTED_SOURCES, files["source"]["sha256"], "visits_with_skips")
    put("catalogue", {})
    put("train", {})
    put("validation", {})
    contract = dict(kind="wireseg-fixed-cohort-overfit-v1", mode="depth-only",
                    identities=cohort, source_sha256=files["source"]["sha256"],
                    source_catalogue_sha256=files["catalogue"]["sha256"],
                    subset_sha256=files["train"]["sha256"], validation_sha256=files["validation"]["sha256"],
                    size=32, optimization_view="center", learning_rate=.0003, seed=116)
    put("cohort", {k: v for k, v in contract.items() if k != "mode"})
    ledger, steps = [], []
    for i, sample in enumerate(cohort):
        row = dict(visit=i + 1, file_name=sample["file_name"], updated=supervised[i])
        if supervised[i]:
            row.update(update=len(steps) + 1, pair_sha256=sample["pair_sha256"],
                       loss=1., grad_norm=1., empty_mask=False, elapsed_s=float(i))
            steps.append(copy.deepcopy(row))
        else:
            row.update(reason="no_valid_depth_label", updates_completed=len(steps))
        ledger.append(row)
    put("ledger", "".join(json.dumps(row) + "\n" for row in ledger))
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    for _ in steps:
        optimizer.zero_grad()
        model(torch.ones(1, 1)).square().sum().backward()
        optimizer.step()
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), rng=torch.get_rng_state(),
                 cuda_rng=None, contract=contract, global_step=len(steps), steps=steps,
                 counts={"rgb_updates": len(steps)})
    path = tmp_path / "checkpoint"
    torch.save(state, path)
    files["checkpoint"] = dict(path=str(path), sha256=digest_file(path))
    monkeypatch.setattr(migration, "verified_cohort", lambda *a: (cohort, supervised))
    return files, state, ledger, cohort, supervised


def test_import_preserves_state_bytes_and_records_unknowns(fixture, tmp_path):
    files, state, _, _, _ = fixture
    before = {name: digest_file(Path(ref["path"])) for name, ref in files.items()}
    path = tmp_path / "import.pt"
    report = migration.import_legacy(files, tmp_path, path)
    result = torch.load(path, weights_only=True)
    assert result["checkpoint_format"] == migration.FORMAT
    assert_state_equal(result["legacy_state"], state)
    assert result["unknown_state"] == ["python_rng", "numpy_rng", "complete_historical_runtime_config"]
    assert result["sampler"]["visits_completed"] == 8
    assert result["sampler"]["skipped_visits"] == 2
    assert result["sampler"]["next_index"] == 0
    assert report["new_training_updates"] == 0
    assert report["strict_resume_eligible"] is False
    assert report["historical_bitwise_equivalence"] is False
    assert report["native_qualified"] is False
    with pytest.raises(ValueError, match="strict resume"):
        require_v2_state(result)
    assert before == {name: digest_file(Path(ref["path"])) for name, ref in files.items()}
    with pytest.raises(FileExistsError):
        migration.import_legacy(files, tmp_path, path)


@pytest.mark.parametrize("name", ["checkpoint", "ledger", "source", "cohort", "train", "validation", "catalogue"])
def test_evidence_bytes_must_match_expected_sha(fixture, tmp_path, name):
    files, *_ = fixture
    path = Path(files[name]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        migration.import_legacy(files, tmp_path, tmp_path / "rejected.pt")
    assert not (tmp_path / "rejected.pt").exists()


@pytest.mark.parametrize("corruption", ["order", "drop_skip", "skip_count", "pair_sha", "update",
                                      "extra_skip", "checkpoint_step", "supervision"])
def test_reconstruction_rejects_inconsistent_ledger(fixture, corruption):
    _, state, ledger, cohort, supervised = fixture
    if corruption == "order":
        ledger[1], ledger[2] = ledger[2], ledger[1]
    elif corruption == "drop_skip":
        ledger.pop(0)
    elif corruption == "skip_count":
        ledger[4]["updates_completed"] += 1
    elif corruption == "pair_sha":
        ledger[1]["pair_sha256"] = "f" * 64
    elif corruption == "update":
        ledger[1]["update"] += 1
    elif corruption == "extra_skip":
        ledger.append(dict(visit=9, file_name=cohort[0]["file_name"], updated=False,
                           reason="no_valid_depth_label", updates_completed=len(state["steps"])))
    elif corruption == "checkpoint_step":
        state["steps"][0]["loss"] += 1
    else:
        supervised[0] = True
    with pytest.raises(ValueError):
        migration.reconstruct_sampler(state, ledger, cohort, supervised, "visits_with_skips")


def test_segmentation_source_infers_visits_without_fabricating_skips(fixture):
    _, state, _, cohort, _ = fixture
    state["contract"]["mode"] = "seg-only"
    ledger = [dict(update=i + 1, file_name=row["file_name"], pair_sha256=row["pair_sha256"],
                   loss=1., grad_norm=1., empty_mask=True, elapsed_s=float(i))
              for i, row in enumerate(cohort)]
    state.update(steps=copy.deepcopy(ledger), global_step=8, counts={"rgb_updates": 8})
    result = migration.reconstruct_sampler(state, ledger, cohort, [True] * 8, "updates_only")
    assert result["visits_completed"] == 8 and result["skipped_visits"] == 0
    assert all(row["updated"] for row in result["visit_history"])
    with pytest.raises(ValueError):
        migration.reconstruct_sampler(state, ledger, list(reversed(cohort)), [True] * 8, "updates_only")


def test_rehashed_ledger_must_still_match_checkpoint(fixture, tmp_path):
    files, _, ledger, *_ = fixture
    ledger[1]["loss"] += 1
    path = Path(files["ledger"]["path"])
    path.write_text("".join(json.dumps(row) + "\n" for row in ledger))
    files["ledger"]["sha256"] = digest_file(path)
    with pytest.raises(ValueError, match="checkpoint"):
        migration.import_legacy(files, tmp_path, tmp_path / "rejected.pt")


def test_diagnostic_continuation_preserves_adam_rng_and_sampler(fixture, tmp_path, monkeypatch):
    from scripts import continue_dh116_wireseg_legacy as runner
    files, state, _, cohort, supervised = fixture
    path = tmp_path / "import.pt"
    migration.import_legacy(files, tmp_path, path)
    package = torch.load(path, weights_only=True)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    reference = torch.nn.Linear(1, 1)
    reference.load_state_dict(state["model"])
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=.0003)
    reference_optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    torch.set_rng_state(state["rng"])
    reference_optimizer.zero_grad(set_to_none=True)
    expected_loss = (reference(torch.ones(1, 1)) - torch.rand(1, 1)).square().sum()
    expected_loss.backward()
    reference_optimizer.step()
    expected_rng = torch.get_rng_state()
    monkeypatch.setattr(runner, "make_view", lambda sample, *args: (
        torch.ones(1, 1), torch.ones(1, 1, 1, dtype=torch.bool), torch.ones(1, 1, 1),
        torch.full((1, 1, 1), supervised[cohort.index(sample)], dtype=torch.bool)))
    monkeypatch.setattr(runner, "diagnostic_loss",
                        lambda prediction, *args: (prediction - torch.rand(1, 1)).square().sum())
    result = runner.run_continuation(package, digest_file(path), model, optimizer, cohort,
                                     tmp_path / "continued", steps=1, wall_seconds=120,
                                     device=torch.device("cpu"), new_rng_seed=116, source_path=path)
    saved = torch.load(tmp_path / "continued/diagnostic.pt", weights_only=True)
    assert saved["checkpoint_format"] == runner.FORMAT
    assert_state_equal(saved["continued_state"]["model"], reference.state_dict())
    assert_state_equal(saved["continued_state"]["optimizer"], reference_optimizer.state_dict())
    assert torch.equal(saved["current_rng"]["torch_cpu"], expected_rng)
    assert saved["unknown_historical_state"] == migration.UNKNOWN_STATE
    assert saved["sampler"]["visits_completed"] == 10
    assert saved["sampler"]["skipped_visits"] == 3
    assert saved["sampler"]["next_index"] == 2
    assert result["global_step"] == 7 and result["new_training_updates"] == 1
    assert result["new_rng_seed"] == 116 and result["strict_resume_eligible"] is False
    assert result["source_import_sha256"] == digest_file(path)
    with pytest.raises(ValueError, match="strict resume"):
        require_v2_state(saved)
    with pytest.raises(FileExistsError):
        runner.run_continuation(package, digest_file(path), model, optimizer, cohort,
                                tmp_path / "continued", steps=1, wall_seconds=120,
                                device=torch.device("cpu"), new_rng_seed=116, source_path=path)


@pytest.mark.parametrize("steps,seconds", [(0, 120), (17, 120), (1, 0), (1, 121)])
def test_continuation_budget_rejected_before_restore(fixture, tmp_path, steps, seconds):
    from scripts import continue_dh116_wireseg_legacy as runner
    files, *_ = fixture
    path = tmp_path / "import.pt"
    migration.import_legacy(files, tmp_path, path)
    package = torch.load(path, weights_only=True)
    model = torch.nn.Linear(1, 1)
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    with pytest.raises(ValueError, match="budget"):
        runner.run_continuation(package, digest_file(path), model, optimizer, [],
                                tmp_path / "rejected", steps=steps, wall_seconds=seconds,
                                device=torch.device("cpu"), new_rng_seed=116, source_path=path)
    assert_state_equal(before, model.state_dict())
    assert not (tmp_path / "rejected").exists()


@pytest.mark.parametrize("damage", ["unknown", "qualification", "model", "sampler", "optimizer"])
def test_continuation_rejects_tampered_import_before_restore(fixture, tmp_path, damage):
    from scripts import continue_dh116_wireseg_legacy as runner
    files, *_ = fixture
    path = tmp_path / "import.pt"
    migration.import_legacy(files, tmp_path, path)
    package = torch.load(path, weights_only=True)
    if damage == "unknown":
        package["unknown_state"] = []
    elif damage == "qualification":
        package["audit"]["strict_resume_eligible"] = True
    elif damage == "model":
        next(iter(package["legacy_state"]["model"].values())).add_(1)
    elif damage == "sampler":
        package["sampler"]["next_index"] = 1
    else:
        next(iter(package["legacy_state"]["optimizer"]["state"].values()))["exp_avg"].add_(1)
    model = torch.nn.Linear(1, 1)
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    with pytest.raises(ValueError):
        runner.run_continuation(package, digest_file(path), model, optimizer, [],
                                tmp_path / "rejected", steps=1, wall_seconds=120,
                                device=torch.device("cpu"), new_rng_seed=116, source_path=path)
    assert_state_equal(before, model.state_dict())
    assert not (tmp_path / "rejected").exists()


def test_wall_deadline_saves_diagnostic_state_without_new_update(fixture, tmp_path, monkeypatch):
    from scripts import continue_dh116_wireseg_legacy as runner
    files, state, _, cohort, _ = fixture
    path = tmp_path / "import.pt"
    migration.import_legacy(files, tmp_path, path)
    package = torch.load(path, weights_only=True)
    clock = iter([0., 2., 2.])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    result = runner.run_continuation(package, digest_file(path), model, optimizer, cohort,
                                     tmp_path / "deadline", steps=16, wall_seconds=1,
                                     device=torch.device("cpu"), new_rng_seed=116, source_path=path)
    saved = torch.load(tmp_path / "deadline/diagnostic.pt", weights_only=True)
    assert result["stop_reason"] == "wall_clock" and result["new_training_updates"] == 0
    assert_state_equal(saved["continued_state"]["model"], state["model"])
    assert_state_equal(saved["continued_state"]["optimizer"], state["optimizer"])
    assert torch.equal(saved["current_rng"]["torch_cpu"], state["rng"])
    assert saved["sampler"] == package["sampler"]
    assert saved["unknown_historical_state"] == migration.UNKNOWN_STATE
    assert saved["continuation_config"]["new_rng_seed"] == 116
    assert saved["continuation_config"]["budget_scope"] == "optimization_loop_only"


def test_continuation_chain_is_bitwise_equal_after_new_rng_boundary(fixture, tmp_path, monkeypatch):
    import random
    import numpy as np
    from scripts import continue_dh116_wireseg_legacy as runner
    files, _, _, cohort, supervised = fixture
    imported = tmp_path / "import.pt"
    migration.import_legacy(files, tmp_path, imported)
    monkeypatch.setattr(runner, "make_view", lambda sample, *args: (
        torch.ones(1, 1), torch.ones(1, 1, 1, dtype=torch.bool), torch.ones(1, 1, 1),
        torch.full((1, 1, 1), supervised[cohort.index(sample)], dtype=torch.bool)))
    monkeypatch.setattr(runner, "diagnostic_loss", lambda prediction, *args:
                        (prediction - torch.rand(1, 1) - random.random() - np.random.random()).square().sum())

    def run(parent, name, updates, seed):
        package = torch.load(parent, weights_only=True)
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
        runner.run_continuation(package, digest_file(parent), model, optimizer, cohort,
                                tmp_path / name, steps=updates, wall_seconds=120,
                                device=torch.device("cpu"), new_rng_seed=seed, source_path=parent)
        return tmp_path / name / "diagnostic.pt"

    whole_path = run(imported, "whole", 8, 116)
    first = run(imported, "first", 3, 116)
    resumed_path = run(first, "resumed", 5, None)
    whole, resumed = [torch.load(p, weights_only=True) for p in (whole_path, resumed_path)]
    for name in ("model", "optimizer", "global_step"):
        assert_state_equal(whole["continued_state"][name], resumed["continued_state"][name])
    assert_state_equal(whole["current_rng"], resumed["current_rng"])
    assert whole["sampler"] == resumed["sampler"]
    assert resumed["unknown_historical_state"] == migration.UNKNOWN_STATE
    assert resumed["parent_checkpoint"]["sha256"] == digest_file(first)
    assert resumed["origin_import_sha256"] == digest_file(imported)
    assert resumed["audit"]["rng_seed_scope"] == "restored_current_rng"
    with pytest.raises(ValueError, match="seed"):
        run(first, "reseed-rejected", 1, 116)
    broken = torch.load(first, weights_only=True)
    broken["parent_checkpoint"]["sha256"] = "f" * 64
    tampered = tmp_path / "changed-parent.pt"
    torch.save(broken, tampered)
    with pytest.raises(ValueError, match="SHA256"):
        run(tampered, "bad-parent", 1, None)
    broken = torch.load(first, weights_only=True)
    broken["sampler"]["next_index"] += 1
    torch.save(broken, tampered)
    with pytest.raises(ValueError, match="sampler"):
        run(tampered, "bad-chain-sampler", 1, None)


@pytest.mark.parametrize("stop", ["soft_deadline", "interrupt"])
def test_stop_at_safe_update_boundary_retains_checkpoint(fixture, tmp_path, monkeypatch, stop):
    from scripts import continue_dh116_wireseg_legacy as runner
    files, state, _, cohort, supervised = fixture
    path = tmp_path / "import.pt"
    migration.import_legacy(files, tmp_path, path)
    package = torch.load(path, weights_only=True)
    clock = {"value": 0., "interrupt": False}
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock["value"])
    monkeypatch.setattr(runner, "make_view", lambda sample, *args: (
        torch.ones(1, 1), torch.ones(1, 1, 1, dtype=torch.bool), torch.ones(1, 1, 1),
        torch.full((1, 1, 1), supervised[cohort.index(sample)], dtype=torch.bool)))

    def loss(prediction, *args):
        clock["value"] = 2.  # The in-flight update deliberately overshoots the 1s soft deadline.
        clock["interrupt"] = stop == "interrupt"
        return prediction.square().sum()

    monkeypatch.setattr(runner, "diagnostic_loss", loss)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    result = runner.run_continuation(package, digest_file(path), model, optimizer, cohort,
                                     tmp_path / "stopped", steps=16, wall_seconds=1,
                                     device=torch.device("cpu"), new_rng_seed=116, source_path=path,
                                     stop_requested=lambda: clock["interrupt"])
    saved_path = tmp_path / "stopped/diagnostic.pt"
    saved = torch.load(saved_path, weights_only=True)
    assert result["new_training_updates"] == 1 and result["global_step"] == state["global_step"] + 1
    assert result["stop_reason"] == ("interrupted" if stop == "interrupt" else "wall_clock")
    assert result["optimization_wall_s"] == 2.
    assert saved["sampler"]["visits_completed"] == 10
    recovered = runner.validate_source(saved, digest_file(saved_path))
    assert recovered["state"]["global_step"] == 7
