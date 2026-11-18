"""Presentation-only scheduling; never use it to skip physics or contact updates."""

import math


def render_due(step: int, interval: int, headless: bool) -> bool:
    if interval < 1:
        raise ValueError('render interval must be positive')
    return not headless and step % interval == 0


def should_publish_wire(*, headless: bool, now: float, next_publish: float, ui_hz: float) -> bool:
    """Return whether a wall-clock GUI publication is due."""
    if not isinstance(headless, bool):
        raise TypeError('headless must be a bool')
    if not math.isfinite(now) or not math.isfinite(next_publish):
        raise ValueError('wall-clock deadlines must be finite')
    if not math.isfinite(ui_hz) or ui_hz <= 0:
        raise ValueError('ui_hz must be finite and positive')
    return not headless and now >= next_publish
