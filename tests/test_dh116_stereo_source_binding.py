"""CPU upper replay only: lower native gate is assumed, never qualified."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch

from go2_pvcnn.control.dh116 import tabletop_stereo_promotion as promotion
from go2_pvcnn.control.dh116 import tabletop_stereo_supervision_dataset as datasets
from test_dh116_stereo_collection import FakeEnv, TinyStudent, TinyTeacher, module, record


def verifier(data):
    from go2_pvcnn.control.dh116.tabletop_stereo_source_binding import verify_stereo_dataset_sources
    return verify_stereo_dataset_sources(data)


def formal_fixture_without_binding():
    data = datasets.build_dataset([record()], source_sha256="f"*64, schema_only=True)
    data.update(schema_only=False, purpose="dagger_visited_states", source_bindings=[])
    return data


def test_missing_formal_source_binding_fails_before_native_lookup():
    with pytest.raises(ValueError, match="trusted source bindings"):
        verifier(formal_fixture_without_binding())


def test_missing_native_proof_cannot_be_replaced_with_payload_hash(tmp_path):
    data = formal_fixture_without_binding()
    data["source_bindings"] = [dict(proof_path=str(tmp_path/"absent.json"),
        proof_sha256="a"*64, row_indices=[0])]
    with pytest.raises(ValueError, match="native source proof"):
        verifier(data)


def test_schema_fixture_has_explicit_unqualified_exemption():
    data = datasets.build_dataset([record()], source_sha256="f"*64, schema_only=True)
    before = deepcopy(data)
    assert verifier(data) is None
    assert data["rows"][0]["teacher_mean"].equal(before["rows"][0]["teacher_mean"])
    assert data["native_qualified"] is False
    assert data["successful_demonstrations"] is False


class SealedCPU:
    """Only _native_run is mocked. All path/SHA/completion/row joins stay real."""
    def __init__(self, tmp_path, monkeypatch, *, successful=False):
        original = module("teacher_collector").collect_step(
            FakeEnv(), TinyStudent(), TinyTeacher(), teacher_probability=0., schema_only=True)
        self.data = datasets.build_dataset([original], source_sha256="f"*64, schema_only=True,
            purpose="dagger_visited_states")
        self.records = [deepcopy(original)]
        self.records[0]["schema_only"] = False
        self.records[0]["info"]["evidence_scope"] = "native_runtime_capture_unqualified"
        if successful:
            self.records[0]["done"].fill_(True)
            self.records[0]["info"]["success"].fill_(True)
        self.raw = tmp_path/"raw.jsonl"
        self.raw.write_text("{}\n", encoding="utf-8")
        self.records_path = tmp_path/"records.pt"
        self.packet = dict(version="dh116-stereo-teacher-candidate-records-v1",
            teacher_checkpoint_sha256="f"*64, records=self.records)
        torch.save(self.packet, self.records_path)
        self.manifest_path = tmp_path/"manifest.json"
        self.manifest = dict(version="dh116-stereo-source-manifest-v1",
            teacher_checkpoint_sha256="f"*64, native_raw_artifact=self.binding(self.raw),
            records_artifact=self.binding(self.records_path))
        self.write_json(self.manifest_path, self.manifest)
        self.completion_path = tmp_path/"completion.json"
        self.completion = dict(produced_artifacts=[
            self.binding(self.raw), self.binding(self.records_path), self.binding(self.manifest_path)])
        self.write_json(self.completion_path, self.completion)
        self.proof_path = tmp_path/"proof.json"
        self.proof = dict(raw_artifact=self.binding(self.raw),
            completion_artifact=self.binding(self.completion_path),
            stereo_source_manifest_artifact=self.binding(self.manifest_path))
        self.write_json(self.proof_path, self.proof)
        self.data.update(schema_only=False, rows=datasets.rows_from_records(self.records),
            source_bindings=[dict(proof_path=str(self.proof_path.resolve()),
                proof_sha256=self.binding(self.proof_path)["sha256"],
                row_indices=list(range(len(self.data["rows"]))))])
        self.native_calls = []
        def assumed_native_gate(proof, base):
            self.native_calls.append((proof, base))
            return {"assumed_native_gate_only": True}
        monkeypatch.setattr(promotion, "_native_run", assumed_native_gate)

    @staticmethod
    def binding(path):
        return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def rehash(self, *, manifest=False, records=False, completion=False):
        # Explicit CPU mutation setup; never used on real evidence.
        if records:
            torch.save(self.packet, self.records_path)
            self.manifest["records_artifact"] = self.binding(self.records_path)
        if manifest or records:
            self.write_json(self.manifest_path, self.manifest)
            self.proof["stereo_source_manifest_artifact"] = self.binding(self.manifest_path)
        if completion:
            self.completion["produced_artifacts"] = [
                self.binding(self.raw), self.binding(self.records_path), self.binding(self.manifest_path)]
            self.write_json(self.completion_path, self.completion)
            self.proof["completion_artifact"] = self.binding(self.completion_path)
        self.write_json(self.proof_path, self.proof)
        self.data["source_bindings"][0]["proof_sha256"] = self.binding(self.proof_path)["sha256"]


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    return SealedCPU(tmp_path, monkeypatch)


def test_actual_upper_verifier_replays_exact_records_without_qualification(sealed):
    assert verifier(sealed.data) is None
    assert len(sealed.native_calls) == 1
    assert sealed.data["native_qualified"] is False
    assert sealed.data["successful_demonstrations"] is False


@pytest.mark.parametrize("change", ["left_rgb", "right_rgb", "proprio", "privileged",
    "privileged_valid", "teacher_mean", "issued_actions", "owner_and_action", "dtype",
    "frame_id", "reset_id", "policy_tick", "physical_initial_condition_sha256",
    "source_episode_group", "order", "mode", "qualification", "extra_label", "label_mask"])
def test_actual_upper_verifier_rejects_row_changes_with_same_proof(sealed, change):
    data = sealed.data
    row = data["rows"][0]
    if change in ("left_rgb", "right_rgb", "proprio"):
        row["history"][change].reshape(-1)[0] += .125
    elif change in ("privileged", "teacher_mean", "issued_actions"):
        row[change].reshape(-1)[0] += .125
    elif change == "privileged_valid":
        row[change][0] = ~row[change][0]
    elif change == "owner_and_action":
        for field in ("teacher_mean", "student_nominal", "issued_actions"):
            row[field].fill_(.9)
    elif change == "dtype":
        row["history"]["left_rgb"] = row["history"]["left_rgb"].double()
    elif change == "order":
        data["rows"].reverse()
    elif change == "mode":
        row["mode"] = "teacher_reference"
    elif change == "qualification":
        row["teacher_qualification"] = {"qualified": True}
    elif change == "extra_label":
        row["extra_labels"]["invented"] = torch.tensor([.5])
    elif change == "label_mask":
        row["label_validity"]["invented"] = torch.tensor([True])
    else:
        value = row["identity"][change]
        row["identity"][change] = value+1 if type(value) is int else value+"-changed"
    with pytest.raises(ValueError, match="rows differ"):
        verifier(data)


@pytest.mark.parametrize("indices", [[], [1, 0], [0, 0], [2], [-1], [True], [0.0], (0,)])
def test_raw_row_selection_must_be_nonempty_ordered_unique_in_range(sealed, indices):
    sealed.data["source_bindings"][0]["row_indices"] = indices
    with pytest.raises(ValueError, match="ordered unique"):
        verifier(sealed.data)


def test_subset_and_ordered_multiple_bindings_replay_in_binding_order(sealed):
    first = deepcopy(sealed.data["source_bindings"][0])
    second = deepcopy(first)
    first["row_indices"], second["row_indices"] = [0], [1]
    sealed.data["source_bindings"] = [first, second]
    assert verifier(sealed.data) is None
    sealed.data["source_bindings"].reverse()
    with pytest.raises(ValueError, match="rows differ"):
        verifier(sealed.data)


@pytest.mark.parametrize("change", ["extra", "relative", "directory", "sha", "no_list"])
def test_malformed_binding_is_rejected(sealed, change):
    binding = sealed.data["source_bindings"][0]
    if change == "extra": binding["qualified"] = True
    elif change == "relative": binding["proof_path"] = "proof.json"
    elif change == "directory": binding["proof_path"] = str(sealed.proof_path.parent)
    elif change == "sha": binding["proof_sha256"] = "not-a-sha"
    else: sealed.data["source_bindings"] = {"qualified": True}
    with pytest.raises(ValueError):
        verifier(sealed.data)


def test_proof_bytes_changed_after_binding_fails_before_native_gate(sealed):
    sealed.proof_path.write_bytes(sealed.proof_path.read_bytes()+b"\n")
    with pytest.raises(ValueError, match="native source proof"):
        verifier(sealed.data)
    assert not sealed.native_calls


@pytest.mark.parametrize("artifact", ["raw", "manifest", "records", "completion"])
def test_actual_bound_artifact_byte_change_is_rejected(sealed, artifact):
    path = {"raw": sealed.raw, "manifest": sealed.manifest_path,
        "records": sealed.records_path, "completion": sealed.completion_path}[artifact]
    path.write_bytes(path.read_bytes()+b"\n")
    with pytest.raises(ValueError, match="SHA"):
        verifier(sealed.data)


@pytest.mark.parametrize("missing", ["manifest", "records"])
def test_source_artifact_from_different_run_completion_is_rejected(sealed, missing):
    path = sealed.manifest_path if missing == "manifest" else sealed.records_path
    sealed.completion["produced_artifacts"] = [
        row for row in sealed.completion["produced_artifacts"] if row["path"] != str(path)]
    sealed.write_json(sealed.completion_path, sealed.completion)
    sealed.proof["completion_artifact"] = sealed.binding(sealed.completion_path)
    sealed.rehash()
    with pytest.raises(ValueError, match="absent from actual external process completion"):
        verifier(sealed.data)


def test_rehashed_manifest_and_replacement_records_cannot_self_sign_source(sealed):
    sealed.packet["records"][0]["teacher_mean"].fill_(.9)
    replacement = sealed.records_path.with_name("replacement.pt")
    torch.save(sealed.packet, replacement)
    sealed.manifest["records_artifact"] = sealed.binding(replacement)
    sealed.rehash(manifest=True)
    with pytest.raises(ValueError, match="absent from actual external process completion"):
        verifier(sealed.data)


@pytest.mark.parametrize("change", ["version", "checkpoint", "raw", "extra"])
def test_manifest_contract_is_exact_and_binds_raw_and_teacher(sealed, change):
    if change == "version": sealed.manifest["version"] = "unknown"
    elif change == "checkpoint": sealed.manifest["teacher_checkpoint_sha256"] = "e"*64
    elif change == "raw": sealed.manifest["native_raw_artifact"] = sealed.binding(sealed.records_path)
    else: sealed.manifest["qualified"] = True
    sealed.rehash(manifest=True, completion=True)
    with pytest.raises(ValueError, match="manifest identity"):
        verifier(sealed.data)


@pytest.mark.parametrize("change", ["version", "checkpoint", "records"])
def test_records_packet_binds_checkpoint_and_format(sealed, change):
    if change == "version": sealed.packet["version"] = "unknown"
    elif change == "checkpoint": sealed.packet["teacher_checkpoint_sha256"] = "e"*64
    else: sealed.packet["records"] = {}
    sealed.rehash(records=True, completion=True)
    with pytest.raises(ValueError, match="sealed stereo records"):
        verifier(sealed.data)


def test_missing_manifest_does_not_upgrade_old_native_proof(sealed):
    del sealed.proof["stereo_source_manifest_artifact"]
    sealed.rehash()
    before = sealed.proof_path.read_bytes()
    with pytest.raises(ValueError, match="artifact"):
        verifier(sealed.data)
    assert sealed.proof_path.read_bytes() == before


def test_no_native_gate_assumption_refuses_cpu_proof(tmp_path):
    data = formal_fixture_without_binding()
    path = tmp_path/"proof.json"
    path.write_text("{}")
    data["source_bindings"] = [dict(proof_path=str(path),
        proof_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), row_indices=[0])]
    with pytest.raises(ValueError, match="native source scope"):
        verifier(data)


def test_same_preserves_nan_bits_dtype_shape_and_python_types():
    from go2_pvcnn.control.dh116.tabletop_stereo_source_binding import _same
    first = torch.tensor([2143289345], dtype=torch.int32).view(torch.float32)
    second = torch.tensor([2143289346], dtype=torch.int32).view(torch.float32)
    assert _same(first, first.clone())
    assert not _same(first, second)
    assert not _same(first, first.double())
    assert not _same(first, first.reshape(1, 1))
    assert not _same([1], (1,))
    assert not _same({"v": 1}, {"v": True})
    assert not _same({"v": 1}, {"v": 1.0})
    assert not _same(first, [float("nan")])


def test_build_copies_bindings_without_turning_schema_fixture_into_qualification():
    bindings = [dict(proof_path="/untrusted", proof_sha256="a"*64, row_indices=[0])]
    data = datasets.build_dataset([record()], source_sha256="f"*64, schema_only=True,
        source_bindings=bindings)
    bindings[0]["row_indices"].append(1)
    assert data["source_bindings"][0]["row_indices"] == [0]
    assert data["native_qualified"] is False


def test_sealed_admission_rejects_source_before_teacher_gate():
    from go2_pvcnn.control.dh116.tabletop_stereo_supervision_dataset import validate_sealed_stereo_dataset
    with pytest.raises(ValueError, match="trusted source bindings"):
        validate_sealed_stereo_dataset(formal_fixture_without_binding())


def test_sealed_schema_exemption_requires_explicit_allowance():
    from go2_pvcnn.control.dh116.tabletop_stereo_supervision_dataset import validate_sealed_stereo_dataset
    data = datasets.build_dataset([record()], source_sha256="f"*64, schema_only=True)
    with pytest.raises(ValueError, match="schema-only"):
        validate_sealed_stereo_dataset(data)
    checked = validate_sealed_stereo_dataset(data, allow_schema_only=True)
    assert checked["native_qualified"] is False and checked["successful_demonstrations"] is False


def test_disk_loader_refuses_legacy_formal_file_without_modifying_it(tmp_path):
    data = formal_fixture_without_binding()
    path = tmp_path/"legacy.pt"
    torch.save(data, path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="trusted source bindings"):
        datasets.load_dataset(path)
    assert path.read_bytes() == before


def test_disk_loader_replays_sources_before_teacher_gate(sealed, tmp_path):
    path = tmp_path/"sealed.pt"
    sealed.data["rows"][0]["privileged"][0] += .1
    torch.save(sealed.data, path)
    with pytest.raises(ValueError, match="rows differ"):
        datasets.load_dataset(path)


def test_valid_source_does_not_bypass_teacher_success_pool(sealed):
    from go2_pvcnn.control.dh116.tabletop_stereo_supervision_dataset import validate_sealed_stereo_dataset
    data = sealed.data
    data.update(purpose="successful_bc", qualification=None, successful_demonstrations=True)
    records = deepcopy(sealed.records)
    for value in records:
        value.update(mode="teacher_reference", teacher_qualification=None, candidate_only=False)
    data["rows"] = datasets.rows_from_records(records)
    assert verifier(data) is None
    with pytest.raises(ValueError):
        validate_sealed_stereo_dataset(data)
    assert len(sealed.native_calls) == 2


def test_shared_schema_validation_and_batch_do_not_require_final_close():
    data = datasets.build_dataset([record()], source_sha256="f"*64, schema_only=True)
    data.pop("source_bindings", None)
    assert datasets.validate_dataset(data, allow_schema_only=True) is data
    assert datasets.batch_dataset(data, [0])["schema_only"] is True
