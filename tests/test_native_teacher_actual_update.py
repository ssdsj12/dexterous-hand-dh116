"""C05 checkpoint sealing with the original PPO and explicit CPU physics doubles."""
import hashlib
import pytest
from test_native_teacher_checkpoint import contract, trained
from test_native_teacher_collector import fixture, operations
from test_native_teacher_context import backend_profile


def update():
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_actual_update
    return tabletop_native_teacher_actual_update


def test_existing_checkpoint_is_rejected_before_physical_step(monkeypatch, tmp_path, backend_profile):
    collector, _, rod, *_ = fixture(monkeypatch, tmp_path, backend_profile, rollout_steps=32)
    checkpoint = tmp_path / "teacher_clean.pt"
    checkpoint.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        update().run_fresh_update(collector, checkpoint_contract=contract(collector.trainer))
    assert checkpoint.read_bytes() == b"original"
    assert rod.rope.steps == 0


@pytest.mark.parametrize("invalid", ["policy", "partially_used_trainer", "scope", "initial_clock"])
def test_invalid_training_origin_rejected_before_step(monkeypatch, tmp_path, backend_profile, invalid):
    collector, _, rod, _, sim, _ = fixture(monkeypatch, tmp_path, backend_profile, rollout_steps=32)
    trainer = collector.trainer
    training_contract = contract(trainer)
    if invalid == "policy":
        trainer.policy_version = 1
    elif invalid == "partially_used_trainer":
        trainer.counters["environment_steps"] = 1
    elif invalid == "scope":
        training_contract["scope"] = "owner_source_unqualified"
    elif invalid == "initial_clock":
        sim.current_time_step_index += 1
    with pytest.raises(ValueError, match="fresh|scope|clock"):
        update().run_fresh_update(collector, checkpoint_contract=training_contract)
    assert rod.rope.steps == 0
    assert not (tmp_path / "teacher_clean.pt").exists()


def test_incomplete_episode_cannot_reset_into_fresh_checkpoint(monkeypatch, tmp_path, backend_profile):
    collector, _, rod, *_ = fixture(monkeypatch, tmp_path, backend_profile, rollout_steps=32, cap_controls=1)
    module = update()
    with pytest.raises(module.TrainingStopped, match="fresh_verified_reset_factories_required"):
        module.run_fresh_update(collector, checkpoint_contract=contract(collector.trainer))
    assert rod.rope.steps == 8
    assert collector.trainer.counters["updates"] == 0
    assert not (tmp_path / "teacher_clean.pt").exists()
    assert operations(collector)[-1]["operation"] == "training_stopped"


@pytest.mark.parametrize("swap", ["reject", "replace_staging_file"])
def test_checkpoint_publication_uses_audited_bytes(monkeypatch, tmp_path, backend_profile, swap):
    collector = trained(monkeypatch, tmp_path, backend_profile)
    module = update()
    path = tmp_path / "teacher_clean.pt"
    original_audit = module.audit_checkpoint_bytes
    immutable = []

    def audit(raw, *args, **kwargs):
        immutable.append(raw)
        if swap == "reject":
            raise ValueError("withheld by independent audit")
        staged, = tmp_path.glob(".teacher-audit-*/teacher_clean.pt")
        staged.write_bytes(b"different from audited bytes")
        return original_audit(raw, *args, **kwargs)

    monkeypatch.setattr(module, "audit_checkpoint_bytes", audit)
    if swap == "reject":
        with pytest.raises(ValueError, match="withheld"):
            module._seal_checkpoint(path, collector.trainer, contract(collector.trainer))
        assert not path.exists()
    else:
        receipt = module._seal_checkpoint(path, collector.trainer, contract(collector.trainer))
        assert path.read_bytes() == immutable[0]
        assert receipt["checkpoint_sha256"] == hashlib.sha256(immutable[0]).hexdigest()
    assert len(immutable) == 1 and not list(tmp_path.glob(".teacher-audit-*"))
