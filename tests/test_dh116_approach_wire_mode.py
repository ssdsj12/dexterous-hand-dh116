import pytest

from go2_pvcnn.control.dh116.approach_wire_mode import wire_dynamics_enabled


def test_frozen_mode_only_skips_approach():
    assert wire_dynamics_enabled("settle", "frozen")
    assert not wire_dynamics_enabled("approach", "frozen")
    assert wire_dynamics_enabled("close", "frozen")
    assert wire_dynamics_enabled("hold", "frozen")
    assert wire_dynamics_enabled("lift", "frozen")


def test_dynamic_mode_keeps_all_stages():
    for stage in ("settle", "approach", "close", "hold", "lift"):
        assert wire_dynamics_enabled(stage, "dynamic")


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        wire_dynamics_enabled("approach", "invalid")
