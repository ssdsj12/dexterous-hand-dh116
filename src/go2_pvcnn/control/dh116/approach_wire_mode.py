"""Stage-local wire dynamics policy for continuous DH116 runs."""

APPROACH_WIRE_MODES = ("dynamic", "frozen")


def wire_dynamics_enabled(stage: str, mode: str) -> bool:
    """Return whether DeformX should advance for the current sequence stage."""
    if mode not in APPROACH_WIRE_MODES:
        raise ValueError(f"unknown approach wire mode: {mode!r}")
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a non-empty string")
    return not (mode == "frozen" and stage == "approach")


__all__ = ["APPROACH_WIRE_MODES", "wire_dynamics_enabled"]
