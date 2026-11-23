"""CPU checks for measured bilateral 902 Pinch endpoint refitting."""
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import fit_dh116_902_pinch_alignment as alignment


def test_pad_fit_requires_opposition_on_the_original_shared_material_station():
    assert alignment.opposed_at_station("left", [-.00004, -.00008], [36, 36], -.65)
    assert alignment.opposed_at_station("right", [-.00004, -.00008], [4, 4], -.65)
    assert not alignment.opposed_at_station("left", [-.00004, -.00008], [36, 37], -.65)
    assert not alignment.opposed_at_station("left", [-.00004, -.00008], [4, 4], -.65)
    assert not alignment.opposed_at_station("left", [-.00004, -.00008], [None, None], -.65)
    assert not alignment.opposed_at_station("right", [-.00004, -.00008], [4, 4], -.19)
    assert not alignment.opposed_at_station("right", [.00001, -.00008], [4, 4], -.65)


def test_archived_pinch_refit_keeps_full_guards_and_identifies_the_limiting_hull():
    receipt = os.environ.get("DH116_PINCH_ALIGNMENT_RECEIPT")
    if not receipt:
        pytest.skip("sealed 902 Pinch approach receipt is external")
    report = alignment.fit(Path(receipt), max_evaluations=8, starts=1)
    assert report["sdk_range"] == [424, 432]
    assert report["last_row_sha256"] == "ea6985622f9b45a280f005f56c5f48ba2cc2ab9482b62394632d31f7e0206888"
    assert set(report["hands"]) == {"left", "right"}
    for side, expected_gaps in (("left", [1.418459, 1.504077]), ("right", [1.580713, 1.454248])):
        data = report["hands"][side]
        base = data["measured"]
        np.testing.assert_allclose(np.asarray(base["pad_gaps_m"]) * 1000, expected_gaps, atol=1e-4)
        candidate = data["best"]
        assert len(candidate["rod_envelopes"]) == 22
        assert candidate["limiting_nonpad_hull"] in candidate["rod_envelopes"]
        assert candidate["limiting_support"]["hull"] in candidate["rod_envelopes"]
        assert max(candidate["pad_gaps_m"]) < max(base["pad_gaps_m"])
        np.testing.assert_array_equal(candidate["active_q_rad"][3:], base["active_q_rad"][3:])
    assert report["native_contact_verified"] is False
    assert report["teacher_admission"] is False
