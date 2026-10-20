import pytest

from go2_pvcnn.control.dh116.render_schedule import render_due


def test_render_decimation_does_not_remove_physics_steps():
    physics_steps = list(range(31))
    assert [s for s in physics_steps if render_due(s, 10, False)] == [0, 10, 20, 30]
    assert len(physics_steps) == 31


def test_baseline_and_headless():
    assert all(render_due(s, 1, False) for s in range(31))
    assert not any(render_due(s, 10, True) for s in range(31))


@pytest.mark.parametrize('interval', [0, -1])
def test_invalid_interval(interval):
    with pytest.raises(ValueError):
        render_due(0, interval, False)
