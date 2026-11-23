import numpy as np
import pytest

from scripts import audit_dh116_isaac_batch_scene as audit


@pytest.mark.parametrize("count", [32, 64])
def test_power_layout_names_and_spaces_original_tables(count):
    lanes = audit.scene_lanes(count, [0.6, 0.8, 0.3])
    assert len(lanes) == count
    assert len({path for path, _ in lanes}) == count
    assert lanes[0] == ("/World/DH116BatchAudit/lane_000", (0.0, 0.0, 0.0))
    for i, (_, origin) in enumerate(lanes):
        assert origin[2] == 0
        for _, other in lanes[:i]:
            assert abs(origin[0] - other[0]) >= 0.6 or abs(origin[1] - other[1]) >= 0.8


def test_one_lane_write_must_leave_all_other_native_views_unchanged():
    before = np.zeros((32, 2, 17))
    after = before.copy()
    after[31, 0, 0] = 0.001
    audit.verify_isolated_write(before, after, 31)
    with pytest.raises(ValueError, match="another lane"):
        after[0, 0, 0] = 1e-6
        audit.verify_isolated_write(before, after, 31)



def test_batch_ownership_preserves_every_witness_including_live_tail():
    from weakref import ref

    class Owner:
        pass

    dead = ref(Owner())
    tail = Owner()
    references = {str(i): dead for i in range(512)}
    references["last"] = ref(tail)
    assert hasattr(audit, "batch_snapshot"), "lifetime audit must cover the full 64-lane stage"
    summary = audit.batch_snapshot(references)
    assert set(summary["resources"]) == set(references)
    assert summary["resources"]["last"]["alive"] is True
    assert all(summary["resources"][str(i)]["alive"] is False for i in range(512))
    assert summary["native_release_verified"] is False
