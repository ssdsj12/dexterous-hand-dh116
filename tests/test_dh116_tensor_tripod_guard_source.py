"""Real locked assets, synthetic CPU SDK source; no native qualification."""
import numpy as np
import torch
import pytest

from test_native_teacher_context import backend_profile
from test_native_teacher_policy_pinch_distance import sdk_templates
from test_native_teacher_policy_tripod_distance import tripod_case


def test_complete_tripod_adapter_matches_original_source_and_raw_convex_guards(
        monkeypatch, tmp_path, backend_profile, sdk_templates):
    from go2_pvcnn.control.dh116.tabletop_tensor_tripod_guard import TensorTripodGuard
    reader, event, _, _ = tripod_case(monkeypatch, tmp_path, backend_profile, sdk_templates)
    original = reader.geometry
    packet = event["exchange"]["closure_geometry"]
    raw = packet["raw"]
    names = tuple(tuple(raw["hands"][s]["body_names"]) for s in ("left", "right"))
    poses = torch.from_numpy(np.stack([raw["hands"][s]["link_poses_xyzw"][0]
        for s in ("left", "right")])[None].copy())
    nodes = torch.from_numpy(np.array(raw["rod_nodes_m"])[None].copy())
    owner = TensorTripodGuard.from_policy_geometry(original, names,
        env_origins=torch.zeros(1, 3, dtype=torch.float64), device="cpu")
    result = owner.capture(poses, nodes, frame="cpu_current_source", radius=.005)
    assert result.source_complete.tolist() == [packet["source_complete"]]
    expected = packet["supported_geometry_failures"]
    actual = {("left", "right")[owner.support.shapes[col][0]]+"."+owner.support.shapes[col][2]
              for (lane, col) in result.supported.details}
    assert actual == set(expected)
    for (_, col), details in result.supported.details.items():
        key = ("left", "right")[owner.support.shapes[col][0]]+"."+owner.support.shapes[col][2]
        assert details["conditions"] == expected[key]["conditions"]
        np.testing.assert_allclose(details["rod_gap_lower_m"], expected[key]["rod_gap_lower_m"], atol=1e-12, rtol=0)
    assert result.geometry_passed.tolist() == [packet["passed"]]
    # Complete SDF coverage cannot erase any raw source/cooked failure.
    assert result.source_complete.item() and expected
    assert not result.geometry_passed.item()
    # Routes require a separate completed source-qualified native substep.
    assert not result.tripod_supported_guard_covered and not result.native_qualification
    assert owner.require_ready(frame="cpu_current_source") is result

    # Same-body identity does not excuse a changed authenticated pad capture.
    original._mesh_bindings["left.finger14_link"]["capture_sha256"] = "e"*64
    with pytest.raises(ValueError, match="SDF|capture"):
        TensorTripodGuard.from_policy_geometry(original, names,
            env_origins=torch.zeros(1, 3, dtype=torch.float64), device="cpu")
