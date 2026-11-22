"""Source-delta contracts for a separately built raw friction observation."""
from pathlib import Path
import hashlib

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEADER = "third_party/yarnball/KittenEngine/YarnBall/YarnBall.h"
CONTACT = "third_party/yarnball/KittenEngine/YarnBall/sim/contact.cuh"


def module():
    from scripts import prepare_dh116_deformx_contact_friction as prep
    return prep


def test_only_diagnostic_record_and_binding_change_and_old_exports_remain_exact():
    prep = module()
    old, new = prep.locked_sources(ROOT), prep.planned_sources(ROOT)
    assert {n for n in old if old[n] != new[n]} == {HEADER, CONTACT, "csrc/bindings.cpp"}
    before = old["csrc/bindings.cpp"]
    after = new["csrc/bindings.cpp"]
    start = after.index('        .def("contact_friction_samples_v1",')
    stop = after.index('        .def("contact_pair_samples_v2",', start)
    assert after[:start] + after[stop:] == before
    assert all(old[name] == new[name] for name in (
        "csrc/batch.cu", "csrc/sdf.cu", "csrc/sdf.h",
        "third_party/yarnball/KittenEngine/YarnBall/sim/iteration.cu",
        "third_party/yarnball/KittenEngine/YarnBall/sim/cosserat.cu"))


@pytest.mark.parametrize("name", [HEADER, CONTACT])
def test_added_record_fields_reverse_to_the_exact_original_force_kernel(name):
    prep = module()
    old, new = prep.locked_sources(ROOT), prep.planned_sources(ROOT)
    text = new[name]
    for filename, before, after in reversed(prep.RECORD_PATCHES):
        if filename == name:
            assert text.count(after) == 1
            text = text.replace(after, before)
    assert text == old[name]


def test_prepare_copies_all_sources_without_overwriting_the_old_binary(tmp_path):
    prep = module()
    destination = tmp_path / "independent"
    old_sha = hashlib.sha256(prep.OLD_BACKEND.read_bytes()).hexdigest()
    result = prep.prepare(ROOT, destination)
    assert result["compiled"] is False and result["solver_executed"] is False
    assert len(result["prepared_source_sha256"]) == 72
    assert result["upstream_commit"] == "96b8514c7c55c724f1abba1fbf6ed6642b7db72b"
    for name, digest in result["prepared_source_sha256"].items():
        assert hashlib.sha256((destination/"source"/name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(prep.OLD_BACKEND.read_bytes()).hexdigest() == old_sha
    assert "-DCMAKE_CUDA_ARCHITECTURES=89" in result["commands"]["configure"]
    assert result["commands"]["environment"]["CUDA_VISIBLE_DEVICES"] == ""
    with pytest.raises(ValueError, match="new empty"):
        prep.prepare(ROOT, destination)


def test_tampered_locked_source_is_rejected_before_output(tmp_path, monkeypatch):
    prep = module()
    original = prep.source._read
    target = prep.OLD_SOURCE / "csrc/batch.cu"
    def altered(path, digest):
        if Path(path) == target:
            raise ValueError("source SHA differs")
        return original(path, digest)
    monkeypatch.setattr(prep.source, "_read", altered)
    destination = tmp_path/"uncreated"
    with pytest.raises(ValueError, match="SHA"):
        prep.prepare(ROOT, destination)
    assert not destination.exists()
