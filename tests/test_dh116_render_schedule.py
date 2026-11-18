import pytest

from go2_pvcnn.control.dh116.render_schedule import render_due, should_publish_wire


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


def test_headless_never_publishes_on_wall_clock():
    assert not should_publish_wire(headless=True, now=10.0, next_publish=0.0, ui_hz=15.0)


def test_wall_clock_schedule_publishes_when_due():
    assert should_publish_wire(headless=False, now=2.0, next_publish=1.9, ui_hz=15.0)
    assert not should_publish_wire(headless=False, now=2.0, next_publish=2.1, ui_hz=15.0)


@pytest.mark.parametrize('ui_hz', [0.0, -1.0, float('nan')])
def test_wall_clock_schedule_rejects_invalid_rate(ui_hz):
    with pytest.raises(ValueError):
        should_publish_wire(headless=False, now=0.0, next_publish=0.0, ui_hz=ui_hz)
