import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_undamped_rod_config_has_zero_damping():
    cfg = json.loads((ROOT / "configs/dh116_tasks/flexible_pinch_undamped_v1.json").read_text())
    assert cfg["rod"]["damping"] == 0.0
    assert cfg["rod"]["drag"] == 0.2


def test_lazy_extension_cache_prepare_does_not_require_existing_attr():
    from test_preview_extension_cache import SOURCE, fixture
    from go2_pvcnn.control.dh116.tabletop_preview_extension_cache import prepare_extension_cache_release

    app, getter, _, events = fixture()
    del getter.extensions  # Simulate lazy cache not populated yet at startup
    receipt = prepare_extension_cache_release(app, getter, source=SOURCE)
    app._framework.unload_all_plugins()
    assert events == ["unload"]
    assert receipt["framework_unload_returned"] is True


def test_curriculum_reward_config_and_smooth_shaping():
    from go2_pvcnn.control.dh116.tabletop_training_reward import (
        CURRICULUM,
        Measurement,
        RewardEndpoint,
        SourceStamp,
        SubmittedRequest,
        TrainingReward,
        load_config,
    )

    cfg = load_config(ROOT / "configs/dh116_training/flexible_train_curriculum_stage1.json")
    assert cfg.profile == CURRICULUM
    assert cfg.success["net_lift_m"] == pytest.approx(0.01)
    assert cfg.success["hold_s"] == pytest.approx(0.5)
    assert cfg.success["max_slip_m"] == pytest.approx(0.002)
    assert cfg.costs["slip"] == pytest.approx(0.25)

    def _stamp(tick):
        return SourceStamp(
            episode_uid="ep1",
            physics_tick=tick,
            solver_time_s=tick / 480.0,
            source="cpu_test",
            scope="cpu_fixture",
        )

    def _m(val, unit, tick):
        return Measurement(raw=val, unit=unit, valid=True, applicable=True, stamp=_stamp(tick))

    def _endpoint(tick, median_z=0.305, min_z=0.305, off_table=False, bilateral=False, cov=0.0, opposed=False):
        fields = {
            "distance": _m(0.05, "m", tick),
            "numerical_health": _m(True, "dimensionless", tick),
            "constraint_failure": _m(False, "dimensionless", tick),
            "median_z": _m(median_z, "m", tick),
            "min_z": _m(min_z, "m", tick),
            "coverage_left": _m(cov, "dimensionless", tick),
            "coverage_right": _m(cov, "dimensionless", tick),
            "opposed_left": _m(opposed, "dimensionless", tick),
            "opposed_right": _m(opposed, "dimensionless", tick),
            "bilateral_support": _m(bilateral, "dimensionless", tick),
            "off_table": _m(off_table, "dimensionless", tick),
            "no_external_support": _m(True, "dimensionless", tick),
            "hold_motion": _m(True, "dimensionless", tick),
            "prohibited_support": _m(False, "dimensionless", tick),
            "slip": Measurement(raw=None, unit="m", valid=True, applicable=False, stamp=_stamp(tick), history_reason="no_bilateral_contact_since_reset"),
        }
        return RewardEndpoint(
            stamp=_stamp(tick),
            fields=fields,
            anchors=(),
            required_pad_ids=(("left", ("pad0",)), ("right", ("pad0",))),
            target_ids=("wire",),
        )

    req0 = SubmittedRequest(token="tok0", policy_uid="pol0", stamp=_stamp(0), values=(0.0,) * 24, representation="normalized_submitted_command24_preprojection")
    tr = TrainingReward(cfg)
    tr.reset_from_verified_state(_endpoint(0), previous_request=req0)
    req1 = SubmittedRequest(token="tok1", policy_uid="pol0", stamp=_stamp(0), values=(0.0,) * 24, representation="normalized_submitted_command24_preprojection")
    tr.begin(req1, remaining_substeps=8)
    # Even with off_table=False (one end still touching), lifting median_z by 5mm gives positive lift potential
    tr.observe(_endpoint(1, median_z=0.310, min_z=0.305, off_table=False, bilateral=True, cov=0.5, opposed=False))
    assert tr._terms["lift"]["contribution"] > 0.0
