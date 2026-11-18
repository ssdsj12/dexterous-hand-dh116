"""CPU-only fixture checks; old native rows must not acquire missing raw counters."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from go2_pvcnn.control.dh116.tabletop_live_effective_pad_source import LiveEffectivePadSource,CompletePairLiveR0Observer
from go2_pvcnn.control.dh116.tabletop_complete_pair_capture import read_complete_pair_capture
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
from scripts.audit_dh116_effective_right_tracking import RECEIPT
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def recorded():
    receipt=json.loads(Path(RECEIPT).read_bytes())
    path=Path(receipt["output"])/"physical.jsonl"
    reset=last=None
    for line in path.open():
        r=decode_owned(json.loads(line))
        if r.get("operation")=="reset":reset=r["payload"]["reset"]["after_clock"]
        elif r.get("operation")=="physical":last=r["payload"]["exchange"]
    return reset,last

def synthetic_complete_fixture(x):
    # Explicit test fixture reconstructed from derived pairs. NEVER native evidence.
    x=deepcopy(x);rod=x["rod"];cap=rod["rod_mesh_contact_pairs"]
    from go2_pvcnn.control.dh116.tabletop_dual_hand_rod_runtime import MESH_NAMES
    names=list(MESH_NAMES)
    pairs=cap["pairs"];n=len(pairs)
    raw=dict(substeps=12,mesh_count=len(names),capacity=max(1,n+1),attempted=n,overflow=False,
        indices=np.array([[r["substep"],r["node_index"],names.index(r["mesh_name"])] for r in pairs],dtype=np.int64).reshape(n,3),
        vectors=np.array([r["contact_point_world_m"]+r["outward_normal_world"]+r["force_on_rod_world_n"]+
            r["torque_on_mesh_about_origin_world_nm"] for r in pairs],dtype=float).reshape(n,12),
        node_forces_on_rod_world_n=np.array(rod["distributed_contact"])[0],
        mesh_wrenches_on_mesh_world=np.array([rod["named_wrenches"][k] for k in names]))
    rod["rod_mesh_contact_pairs"]=read_complete_pair_capture(SimpleNamespace(contact_pair_samples=lambda:raw),
        dt_s=1/480,physics_step=cap["physics_step"],rod_tick=cap["rod_tick"],mesh_names=names)
    return x

@pytest.fixture(scope="module")
def source(recorded):
    return LiveEffectivePadSource(ROOT,reset_clock=recorded[0])

def test_actual_historical_native_row_without_counters_is_rejected(recorded,source):
    with pytest.raises(ValueError,match="complete"):source.capture(recorded[1])

def test_complete_fixture_preserves_negative_only_and_real_sides(recorded,source):
    x=synthetic_complete_fixture(recorded[1])
    result=source.capture(x)
    assert result["raw_capacity_overflow_retained"]
    assert result["hands"]["left"]["known_no_effective_contact"]
    assert result["hands"]["right"]["known_no_effective_contact"]
    assert result["hands"]["right"]["raw_pair_count"]>0
    assert result["hands"]["right"]["value"]["material_s_m"] is None
    assert not result["teacher_admission"] and not result["native_qualification"]

@pytest.mark.parametrize("mutation",["raw_missing","pair_removed","side","raw_clock","aggregate","episode","reset"])
def test_live_packet_crosschecks_sources_and_identity(recorded,source,mutation):
    x=synthetic_complete_fixture(recorded[1])
    if mutation=="raw_missing":del x["rod"]["rod_mesh_contact_pairs"]["complete_capture"]
    elif mutation=="pair_removed":x["rod"]["rod_mesh_contact_pairs"]["pairs"].pop()
    elif mutation=="side":x["rod"]["mesh_bindings"]["right.finger14_link"]["side"]="left"
    elif mutation=="raw_clock":x["closure_geometry"]["raw"]["identity"]["timestamp_s"]+=.1
    elif mutation=="aggregate":x["rod"]["named_wrenches"]["table"][0]+=.1
    elif mutation=="episode":x["action_readback"]["post"]["identity"][0]["episode_uid"]="wrong"
    else:x["rod"]["reset_id"]+=1
    with pytest.raises(ValueError):source.capture(x)

def test_observer_only_replaces_proven_contact_not_stability_slip(recorded,source):
    x=synthetic_complete_fixture(recorded[1])
    observer=object.__new__(CompletePairLiveR0Observer);observer._grasp="pinch";observer.effective_pad_source=source
    m,d,na=observer._measurements({"exchange":x},x["action_readback"]["post"]["identity"][0],None)
    assert m["left"]["value"]["pads"]==[] and m["right"]["value"]["pads"]==[]
    assert m["stable"]["value"] is None and "stable" in na
    assert "material_anchor_slip" in na
    assert d["complete_effective_pad_source"]["positive_effective_contact_qualified"] is False
