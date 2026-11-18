import pytest

from scripts.audit_dh116_pinch_v4_sources import table_source

FILTER = "/World/PinchTable"


def test_zero_gpu_table_buffer_with_warning_is_unknown():
    log = f"[Warning] [omni.physx.tensors.plugin] GPU contact filter for collider '{FILTER}' is not supported\n"
    result = table_source(log, filter_path=FILTER, counts=[0] * 238)
    assert result["unsupported_filter_warnings"] == 1
    assert result["reported_contacts"] == 0
    assert result["observed_contacts"] is None
    assert result["absence_verified"] is False


def test_empty_without_warning_has_no_positive_capability_evidence():
    result = table_source("", filter_path=FILTER, counts=[0] * 238)
    assert result["unsupported_filter_warnings"] == 0
    assert result["observed_contacts"] is None
    assert result["absence_verified"] is False


def test_warning_requires_exact_path_and_preserves_reported_nonzero_count():
    log = "[omni.physx.tensors.plugin] GPU contact filter for collider '/World/Other' is not supported"
    result = table_source(log, filter_path=FILTER, counts=[1, 2])
    assert result["unsupported_filter_warnings"] == 0
    assert result["reported_contacts"] == 3
    assert result["observed_contacts"] is None


@pytest.mark.parametrize("counts", [[], [True], [-1], [0.5]])
def test_contact_record_counts_cannot_be_invented(counts):
    with pytest.raises(ValueError, match="contact counts"):
        table_source("", filter_path=FILTER, counts=counts)
