"""Reserved-GPU production loader parity over source-owned full geometry."""
from dataclasses import fields
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_cuda_geometry import make_tensor_geometry
from go2_pvcnn.control.dh116.tabletop_tensor_geometry import TensorGeometry, _rotation_xyzw
from test_dh116_tensor_geometry_fusion import inventory, inputs


@pytest.fixture(scope="module")
def gpu():
    if os.environ.get("DH116_CUDA_GEOMETRY_TEST") != "1":
        pytest.skip("explicit GPU2 reservation and DH116_CUDA_GEOMETRY_TEST=1 required")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert torch.cuda.get_device_capability() == (8, 9)
    with patch.object(torch, "compile", side_effect=AssertionError("runtime graph compilation is disabled")):
        yield "cuda:0"


def parity(actual, expected):
    for field in fields(actual):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if not isinstance(a, torch.Tensor):
            assert a == b, field.name
        elif a.is_floating_point():
            torch.testing.assert_close(a, b, rtol=0, atol=1e-12, equal_nan=True, msg=field.name)
        else:
            assert torch.equal(a, b), field.name


@pytest.fixture(scope="module")
def corpus(gpu):
    from go2_pvcnn.control.dh116.tabletop_tensor_collector_replay import PinchOracle
    from go2_pvcnn.control.dh116.tabletop_measured_closure_geometry import REQUIRED
    from scripts.prepare_dh116_pinch_closure_margin import sample_contact
    from scripts.audit_dh116_pinch_initial_pose import pose_matrix, transform
    directory = Path("/home/xk/.local/share/dh116-autonomous-3c80166/evidence/teacher-collision-ordered-fk-20261119/tensor-n2-pinch-comfix-gpu3-dn1hpaa2/native")
    meta = torch.load(directory/"tensor_n2_startup.pt", map_location="cpu", weights_only=True)
    row = torch.load(directory/"tensor_n2_sdk_3.pt", map_location="cpu", weights_only=True)
    oracle = PinchOracle(meta)
    source = SimpleNamespace(_geometry={label: {
        "source": dict(oracle.shapes[side][:17]), "cooked": dict(oracle.shapes[side][17:])}
        for side, label in enumerate(("left", "right"))}, REQUIRED_PADS=REQUIRED,
        _grids={label: oracle.grids[side] for side, label in enumerate(("left", "right"))},
        sources={label: {"sdf_binding": {name: meta["bindings"][label+"."+name] for name in REQUIRED}}
            for label in ("left", "right")}, _sdf=sample_contact, _pose_matrix=pose_matrix, _transform=transform)
    names = tuple(tuple(n) for n in meta["body_names"])
    eager = make_tensor_geometry(source, names, device=gpu, backend="eager")
    actual = make_tensor_geometry(source, names, device=gpu, backend="cuda_cabi")
    assert len(actual._shapes) == 44 and sum(actual.vertex_counts) == 28893
    assert actual.cuda_hulls.vertex_counts == eager.vertex_counts
    assert actual.backend == "eager" and actual.measurement_backend == "cuda_cabi"
    assert type(actual).capture is TensorGeometry.capture
    return eager, actual, meta, row


@pytest.mark.parametrize("count", (2, 128))
def test_real_source_every_field_and_repeat_owned(corpus, count, gpu):
    eager, actual, meta, row = corpus
    p = row["state_post"]["link_pose_xyzw"].repeat(count//2, 1, 1, 1).double().to(gpu)
    n = row["nodes_post"].repeat(count//2, 1, 1).double().to(gpu)
    shift = (torch.arange(count, device=gpu, dtype=torch.float64)//2)*.25
    p[..., 0] += shift[:, None, None]
    n[..., 0] += shift[:, None]
    radius = torch.full((count,), meta["radius"], dtype=torch.float64, device=gpu)
    a = actual.capture(p, n, frame=(count, "one"), radius=radius)
    parity(a, eager.capture(p, n, frame=(count, "one"), radius=radius))
    assert bool(a.source_complete.all()) and not bool(a.fallback_rows.any())
    saved = {f.name: getattr(a, f.name).clone() for f in fields(a) if isinstance(getattr(a, f.name), torch.Tensor)}
    b = actual.capture(p, n, frame=(count, "two"), radius=radius)
    for name, tensor in saved.items():
        torch.testing.assert_close(getattr(a, name), tensor, rtol=0, atol=0, equal_nan=True)
        assert getattr(a, name).data_ptr() != getattr(b, name).data_ptr()
    # Arbitrary strided legacy capture inputs remain legal and value-identical.
    ns = n.transpose(1, 2).contiguous().transpose(1, 2)
    ps = p.transpose(2, 3).contiguous().transpose(2, 3)
    rs = radius[:, None].expand(-1, 2).contiguous()[:, 0]
    assert not ns.is_contiguous() and not ps.is_contiguous() and not rs.is_contiguous()
    parity(actual.capture(ps, ns, frame=(count, "stride"), radius=rs),
        eager.capture(ps, ns, frame=(count, "stride"), radius=rs))


@pytest.mark.parametrize("kind,value", (("pose", float("nan")), ("pose", float("inf")),
    ("node", float("nan")), ("node", float("inf")), ("radius", float("nan")),
    ("radius", float("inf")), ("quaternion", 0.), ("quaternion", 2.)))
def test_original_invalid_input_guard_and_epoch(gpu, kind, value):
    from go2_pvcnn.control.dh116.tabletop_cuda_geometry import CudaMeasurementGeometry
    shapes, pads = inventory()
    actual = CudaMeasurementGeometry(shapes, pads, device=gpu)
    eager = TensorGeometry(shapes, pads, device=gpu, backend="eager")
    p, n = inputs(gpu)
    for g in (actual, eager):
        g.capture(p, n, frame="valid")
        bp, bn, r = p.clone(), n.clone(), .005
        if kind == "pose": bp[0, 0, 0, 0] = value
        if kind == "node": bn[0, 0, 0] = value
        if kind == "radius": r = value
        if kind == "quaternion": bp[0, 0, 0, 3:] = value
        with pytest.raises(ValueError, match="finite unit"):
            g.capture(bp, bn, frame="invalid", radius=r)
        assert g.frame is None and g.result is None
        with pytest.raises(ValueError): g.require_ready(frame="valid")


def test_original_shape_metadata_and_sm_guard(corpus, monkeypatch, gpu):
    from go2_pvcnn.control.dh116.tabletop_cuda_geometry import CudaMeasurementGeometry
    _, actual, _, _ = corpus
    p, n = inputs(gpu)
    for poses, nodes in ((p[:, :, :22], n), (p, n[:, :40])):
        with pytest.raises(ValueError, match="complete same-device"):
            actual.capture(poses, nodes, frame="shape")
        assert actual.result is None
    with pytest.raises(ValueError, match="source binding"):
        actual.validate_metadata(bindings={})
    shapes, pads = inventory()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (8, 0))
    with pytest.raises(ValueError, match="SM89"):
        CudaMeasurementGeometry(shapes, pads, device=gpu)


def test_original_1um_table_thresholds_and_fallbacks(gpu):
    from go2_pvcnn.control.dh116.tabletop_cuda_geometry import CudaMeasurementGeometry
    from go2_pvcnn.control.dh116.tabletop_deformable_wrist_projection import CLEARANCE_M
    shapes, pads = inventory()
    eager = TensorGeometry(shapes, pads, device=gpu, backend="eager")
    actual = CudaMeasurementGeometry(shapes, pads, device=gpu)
    for um in (-3, -2, -1, 0, 1, 2, 3):
        p, n = inputs(gpu)
        p[1, 1, 1, 2] = .3-float(shapes[4][3][:, 2].min())+CLEARANCE_M+um*1e-6
        parity(actual.capture(p, n, frame=um), eager.capture(p, n, frame=um))


def test_finite_fp64_float32_overflow_pad_routing_matches(gpu):
    from go2_pvcnn.control.dh116.tabletop_cuda_geometry import CudaMeasurementGeometry
    shapes, pads = inventory()
    eager = TensorGeometry(shapes, pads, device=gpu, backend="eager")
    actual = CudaMeasurementGeometry(shapes, pads, device=gpu)
    p, n = inputs(gpu)
    p[..., 0] = 1e39
    n[..., 0] = 1e39
    r = torch.full((2,), .005, dtype=torch.float64, device=gpu)
    rotation = _rotation_xyzw(p)
    for a, b in zip(actual._pad_metrics(p, n, r, rotation, (0.,)), eager._pad_metrics(p, n, r, rotation, (0.,))):
        if a.dtype == torch.bool:
            assert torch.equal(a, b)
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=1e-12, equal_nan=True)
