"""Verify lane placement without changing the archived hand collision identity."""

from copy import deepcopy
from pathlib import PurePosixPath

from .tabletop_deformable_r0_replay import _digest


def verify_translated_bindings(original, actual, stage_prefix, original_sha):
    """Allow only path rebinding for the recorded world-downstream lane."""
    if _digest(original) != original_sha:
        raise ValueError("canonical archived collision binding SHA differs")
    prefix = PurePosixPath(stage_prefix)
    if not prefix.is_absolute() or ".." in prefix.parts:
        raise ValueError("absolute native lane collision prefix required")
    expected = deepcopy(original)
    for key, row in expected.items():
        side, body = row["side"], row["body_name"]
        if (side not in ("left", "right") or key != f"{side}.{body}"
                or tuple(PurePosixPath(row["prim_path"]).parts[-2:]) != (f"{side}Hand", body)
                or "/" in body or body in ("", ".", "..")):
            raise ValueError("canonical lane collision side or link identity differs")
        row["prim_path"] = str(prefix / f"{side}Hand" / body)
    translated = _digest(expected)
    if translated != _digest(actual):
        raise ValueError("actual lane collision layout differs from the archived source")
    return translated
