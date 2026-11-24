"""Owned training snapshots preserve values without pretending to be durable evidence."""
from copy import deepcopy
import numpy as np
import pytest


def test_memory_equality_checks_units_dtype_shape_and_value_without_json():
    from go2_pvcnn.control.dh116.tabletop_owned_memory_geometry import owned_equal
    original={"q":np.arange(6,dtype=np.float32).reshape(1,6),"unit":"rad","ids":[1,2]}
    assert owned_equal(original,deepcopy(original))
    for field,value in [("q",original["q"].astype(np.float64)),
                        ("q",original["q"].reshape(6)),("unit","deg"),("ids",[2,1])]:
        damaged=deepcopy(original);damaged[field]=value
        assert not owned_equal(original,damaged)
    damaged=deepcopy(original);damaged["q"][0,0]+=1e-8
    assert not owned_equal(original,damaged)


def test_owned_tree_freezes_arrays_and_containers_and_keeps_values():
    from go2_pvcnn.control.dh116.tabletop_owned_memory_geometry import freeze_owned,owned_equal
    raw={"q":np.arange(6,dtype=np.float32),"clock":[{"tick":12}]}
    frozen=freeze_owned(raw)
    assert owned_equal(raw,frozen)
    assert frozen["q"] is raw["q"]
    with pytest.raises(ValueError):frozen["q"][0]=2.
    with pytest.raises(TypeError):frozen["new"]=1
    with pytest.raises(TypeError):frozen["clock"][0]["tick"]=24
    assert deepcopy(frozen) is frozen


def test_real_asset_geometry_math_matches_legacy_and_rejects_changed_sdk_state():
    from pathlib import Path
    from test_dh116_measured_closure_geometry import case
    from go2_pvcnn.control.dh116.tabletop_native_teacher_sources import _canonical_pinch_history
    from go2_pvcnn.control.dh116.tabletop_native_teacher_policy_pinch_distance import PolicyOwnedPinchGeometry
    from go2_pvcnn.control.dh116.tabletop_owned_memory_geometry import OwnedGeometryMode,owned_equal
    from go2_pvcnn.control.dh116.tabletop_training_reward import SourceFailure
    class Fast(OwnedGeometryMode,PolicyOwnedPinchGeometry):
        pass
    with _canonical_pinch_history():
        legacy=PolicyOwnedPinchGeometry(Path(__file__).resolve().parents[1],scope="cpu_fixture")
    row,sim,bindings,read,hands,clock=case.__wrapped__(legacy)
    row["material_director"]={"source_scope":"cpu_fixture"}
    legacy._radius=lambda exchange:{"value":.005}
    fast=Fast.__new__(Fast);fast.__dict__.update(legacy.__dict__)
    fast.fast_memory=True;fast._frozen_sources=None
    expected=legacy.capture(row,sim=sim,bindings=bindings,read_clock=read)
    actual=fast.capture(row,sim=sim,bindings=bindings,read_clock=read)
    for key in expected:
        if key!="raw_sha256":assert owned_equal(expected[key],actual[key]),key
    assert expected["raw_sha256"] and actual["raw_sha256"] is None
    assert actual["durable_geometry_evidence"] is False
    with pytest.raises(SourceFailure,match="not durable"):
        fast.validate(actual,row)
    saved=actual["raw"]["hands"]["left"]["link_poses_xyzw"].copy()
    hands[0].transforms+=1.
    np.testing.assert_array_equal(actual["raw"]["hands"]["left"]["link_poses_xyzw"],saved)
    # A no-tick external write must still fail; a clock-only shortcut is forbidden.
    hands[0].q[0,0]+=.001
    with pytest.raises(SourceFailure,match="changed"):
        fast.capture(row,sim=sim,bindings=bindings,read_clock=read)


def test_batched_full_vertex_metrics_preserve_complete_guard_math_and_stale_rejection():
    from pathlib import Path
    from types import SimpleNamespace
    from test_dh116_measured_closure_geometry import case
    from go2_pvcnn.control.dh116.tabletop_native_teacher_sources import _canonical_pinch_history
    from go2_pvcnn.control.dh116.tabletop_native_teacher_policy_pinch_distance import PolicyOwnedPinchGeometry
    from go2_pvcnn.control.dh116.tabletop_owned_memory_geometry import OwnedGeometryMode,owned_equal
    from go2_pvcnn.control.dh116.tabletop_batched_geometry_metrics import BatchedGeometryMetrics
    from go2_pvcnn.control.dh116.tabletop_training_reward import SourceFailure
    class Fast(OwnedGeometryMode,PolicyOwnedPinchGeometry):pass
    with _canonical_pinch_history():
        legacy=PolicyOwnedPinchGeometry(Path(__file__).resolve().parents[1],scope="cpu_fixture")
    row,sim,bindings,read,hands,clock=case.__wrapped__(legacy)
    row["material_director"]={"source_scope":"cpu_fixture"}
    legacy._radius=lambda exchange:{"value":.005}
    expected=legacy.capture(row,sim=sim,bindings=bindings,read_clock=read)
    shapes=[(s,hands[s].body_names.index(name),kind+":"+name,vertices)
        for s,side in enumerate(("left","right")) for kind,group in legacy._geometry[side].items()
        for name,vertices in group.items()]
    cache=BatchedGeometryMetrics(shapes,device="cpu")
    poses=np.stack([h.transforms[0] for h in hands])[None]
    cache.capture(poses,row["rod"]["node_positions"],frame=clock)
    fast=Fast.__new__(Fast);fast.__dict__.update(legacy.__dict__)
    fast.fast_memory=True;fast._frozen_sources=None;fast.geometry_cache=cache;fast.geometry_lane=0
    fast.geometry_body_names=tuple(tuple(h.body_names) for h in hands)
    fast.radius_source=SimpleNamespace(source=SimpleNamespace(_boundary=lambda:deepcopy(clock)))
    actual=fast.capture(row,sim=sim,bindings=bindings,read_clock=read)
    assert actual.get("batched_geometry_used") is True
    for key in ("passed","failure_reasons","terminal_reason","source_complete"):
        assert actual[key]==expected[key]
    assert actual["inter_hand_separation_lower_m"]==pytest.approx(expected["inter_hand_separation_lower_m"],abs=1e-10)
    for side in ("left","right"):
        for key in ("table_clearance_m","non_target_surface_gap_lower_m"):
            assert actual["sides"][side][key]==pytest.approx(expected["sides"][side][key],abs=1e-10)
        assert owned_equal(actual["sides"][side]["required_pad_sdf"],expected["sides"][side]["required_pad_sdf"])
    baseline=hands[0].transforms.copy()
    finger=hands[0].body_names.index("finger22_link")
    gap=expected["sides"]["left"]["table_clearance_m"]["source:finger22_link"]
    for epsilon in (-1e-9,1e-9):
        hands[0].transforms[:]=baseline
        # Native buffers are float32; use the original verifier's actual rounding.
        hands[0].transforms[0,finger,2]+=.0021-gap+epsilon
        oracle=legacy.capture(row,sim=sim,bindings=bindings,read_clock=read)
        cache.capture(np.stack([h.transforms[0] for h in hands])[None],
                      row["rod"]["node_positions"],frame=clock)
        measured=fast.capture(row,sim=sim,bindings=bindings,read_clock=read)
        assert measured["failure_reasons"]==oracle["failure_reasons"]
        assert measured["sides"]["left"]["table_clearance_m"]["source:finger22_link"]==oracle["sides"]["left"]["table_clearance_m"]["source:finger22_link"]
    hands[0].transforms[0,0,0]+=1e-6
    with pytest.raises(SourceFailure,match="pose"):
        fast.capture(row,sim=sim,bindings=bindings,read_clock=read)
