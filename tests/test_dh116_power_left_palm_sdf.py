import json

import pytest

from scripts import bake_dh116_power_left_palm_sdf as palm


def test_left_bake_packet_uses_native_convex_fan_and_preserves_source_digest(tmp_path, monkeypatch):
    d = tmp_path / "gpu" / "native"
    d.mkdir(parents=True)
    f = d / "left_base_link.json"
    row = {"side": "left", "link": "base_link", "raw": "original"}
    f.write_text(json.dumps(row))
    sha = palm.surface.parity.file_sha(f)
    state = dict(gpu_row_sha256=sha, matches_convex_boundary=True, exact_convex_facets=True)
    monkeypatch.setattr(palm.surface, "evaluate",
                        lambda *_: {"palms": {"left": state, "right": dict(matches_convex_boundary=False)}})
    source = {"vertices": [[0.,0.,0.]], "faces": [[0,0,0]]}
    monkeypatch.setattr(palm.surface, "_pad_hull", lambda value: (source, 0.))
    packets = palm.input_left_palm(tmp_path)
    assert packets == {"left.base_link": (row, source, 0., sha)}
    state["exact_convex_facets"] = False
    with pytest.raises(ValueError, match="exact original palm triangle surface"):
        palm.input_left_palm(tmp_path)
