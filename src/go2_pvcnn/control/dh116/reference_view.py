"""World layout and camera presets shared by Isaac previews and physics runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REFERENCE_TABLE_CENTER = (0.60, 0.0, 0.25)
REFERENCE_TABLE_SIZE = (0.80, 1.20, 0.10)
REFERENCE_WIRE_START = (0.43, -0.25, 0.308)
REFERENCE_WIRE_END = (0.43, 0.25, 0.308)


@dataclass(frozen=True)
class CameraPreset:
    """An Isaac camera placement matching one panel of the reference figure."""

    eye: tuple[float, float, float]
    target: tuple[float, float, float]


_CAMERA_PRESETS = {
    # Match the reference viewport: view from world -Y, with the arm extending
    # left-to-right along world +X and the tabletop visible beyond the wrist.
    "perspective": CameraPreset((0.43, -1.50, 0.58), (0.43, 0.0, 0.31)),
    # World +Z view, with X horizontal and Y vertical like the reference top panel.
    "top": CameraPreset((0.43, 0.0, 1.50), (0.43, 0.0, 0.31)),
    # World -Y view, with X horizontal and Z vertical like the reference side panel.
    "side": CameraPreset((0.43, -1.50, 0.52), (0.43, 0.0, 0.31)),
}


def camera_preset(name: str) -> CameraPreset:
    """Return a named reference camera, rejecting accidental view drift."""

    try:
        return _CAMERA_PRESETS[name]
    except KeyError as exc:
        names = ", ".join(_CAMERA_PRESETS)
        raise ValueError(f"unknown reference camera {name!r}; choose {names}") from exc


def reference_wire_points(count: int = 41) -> np.ndarray:
    """Return the straight tabletop wire in world coordinates as ``(3, count)``."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("wire point count must be an integer >= 2")
    start = np.asarray(REFERENCE_WIRE_START, dtype=float)
    end = np.asarray(REFERENCE_WIRE_END, dtype=float)
    return np.linspace(start, end, count, dtype=float).T

