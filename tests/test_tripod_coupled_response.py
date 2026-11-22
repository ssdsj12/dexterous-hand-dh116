"""Regression for separating bilateral qualification from individual pad deficits."""
import pytest

from scripts.audit_dh116_tripod_coupled_response import summarize_window


def _hand(weak_pad=None):
    names = ("thumb", "index", "middle")
    loads = {name: 0.04 if name == weak_pad else 0.15 for name in names}
    substeps = [
        dict(substep=step, all_required_pads_on_opposing_faces=True,
             reasons=[f"required_pad_normal_load_not_above_0.05n:{weak_pad}"] if weak_pad else [],
             total_normal_load_n=sum(loads.values()),
             per_pad={name: dict(normal_load_n=load, force_on_rod_world_n=[0., 0., load])
                      for name, load in loads.items()})
        for step in range(12)
    ]
    return dict(loaded=weak_pad is None, per_substep=substeps)


def test_bilateral_response_reports_the_deficit_on_each_hand_separately():
    report = summarize_window([
        dict(per_hand=dict(left=_hand("index"), right=_hand()), bilateral=False),
        dict(per_hand=dict(left=_hand(), right=_hand("middle")), bilateral=False),
    ])
    assert report["bilateral_qualified_ticks"] == 0
    assert report["hands"]["left"]["qualified_ticks"] == report["hands"]["right"]["qualified_ticks"] == 1
    assert report["hands"]["left"]["pad_minimum_quantiles_n"]["index"][0] == .04
    assert report["hands"]["right"]["reason_counts"] == {
        "required_pad_normal_load_not_above_0.05n:middle": 12}
    assert report["hands"]["left"]["opposing_geometry_substeps"] == 24
    with pytest.raises(ValueError, match="12 ordered"):
        summarize_window([dict(per_hand=dict(left=dict(_hand(), per_substep=[]), right=_hand()),
                              bilateral=True)])
