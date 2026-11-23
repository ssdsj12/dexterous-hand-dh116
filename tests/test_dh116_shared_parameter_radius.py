"""Explicit shared parameter radius cannot pass an old one-env source identity."""
from dataclasses import asdict
import pytest
from test_native_teacher_context import backend_profile
from test_dh116_shared_teacher_source import reader


def row(source,lane):
    from go2_pvcnn.control.dh116.tabletop_shared_teacher_live_source import lane_clock
    return {"rod":lane_clock(source,lane),"material_director":{"source_scope":source.scope}}


def test_original_geometry_radius_default_is_still_original_reader(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_policy_pinch_distance as m
    value={"value":.005}
    monkeypatch.setattr(m,"_radius",lambda exchange:value)
    geometry=object.__new__(m.PolicyOwnedPinchGeometry)
    assert geometry._radius({}) is value


def test_selected_shared_row_binds_complete_parent_and_reset_epoch(backend_profile):
    from go2_pvcnn.control.dh116.tabletop_shared_parameter_radius import SharedParameterRadius
    source,runtime,_,_=reader(backend_profile)
    radius=SharedParameterRadius(source)
    with pytest.raises(ValueError):
        radius.read(row(source,1))
    radius.capture()
    value=radius.read(row(source,1))
    assert value["value"]==pytest.approx(.005)
    assert value["source"]["parent_num_envs"]==2
    assert value["source"]["env_row"]==1
    assert value["source"]["parent_solver_id"]==id(runtime.solver)
    old=row(source,1)
    runtime.resets[1]+=1
    runtime.episodes=(runtime.episodes[0],"new-ep1")
    with pytest.raises(ValueError):
        radius.read(old)
    radius.capture()
    with pytest.raises(ValueError):
        radius.read(old)
    assert radius.read(row(source,1))["source"]["reset_id"]==2
