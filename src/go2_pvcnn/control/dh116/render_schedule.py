"""Presentation-only decimation; never use to skip physics or contact updates."""


def render_due(step: int, interval: int, headless: bool) -> bool:
    if interval < 1:
        raise ValueError('render interval must be positive')
    return not headless and step % interval == 0
