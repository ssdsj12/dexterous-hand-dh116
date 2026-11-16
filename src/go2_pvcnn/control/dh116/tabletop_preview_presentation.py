"""Glue for rendering an owned endpoint before observing its RGB pair."""
from copy import deepcopy


class PreviewPresentation:
    def __init__(self, *, capture_snapshot, display, camera, read_clock, sink):
        self.capture_snapshot, self.display, self.camera = capture_snapshot, display, camera
        self.read_clock, self.sink = read_clock, sink

    def __call__(self, pre, exchange):
        before = deepcopy(self.read_clock())
        snapshot = self.capture_snapshot(exchange)
        if self.read_clock() != before:
            raise RuntimeError("physical clock changed while copying display snapshot")
        self.display.publish_snapshot(snapshot, read_clock=self.read_clock)
        if self.read_clock() != before:
            raise RuntimeError("physical clock changed while publishing display snapshot")
        rgb = self.camera.capture()
        if self.read_clock() != before:
            raise RuntimeError("physical clock changed while capturing RGB")
        self.sink(rgb, snapshot, exchange.get("preview_terminal_reason"))
