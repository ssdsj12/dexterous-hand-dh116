"""CPU adversarial source guards. These fabricated rows are never native evidence."""
from copy import deepcopy
import pytest
from test_live_effective_pad_source import ROOT, recorded, source, synthetic_complete_fixture
from go2_pvcnn.control.dh116 import tabletop_live_effective_pad_source as module
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _digest
from go2_pvcnn.control.dh116.tabletop_complete_pair_capture import read_complete_pair_capture
from types import SimpleNamespace

def test_any_accepted_region_keeps_positive_contact_unknown(recorded,source,monkeypatch):
    x=synthetic_complete_fixture(recorded[1])
    original=module.classify_pair
    def candidate(*args,**kwargs):
        out=original(*args,**kwargs)
        out["accepted_by_original_patch"]=True
        return out
    monkeypatch.setattr(module,"classify_pair",candidate)
    result=source.capture(x)
    right=result["hands"]["right"]
    assert right["raw_pair_count"]>0
    assert right["value"] is None
    assert not right["known_no_effective_contact"]
    assert result["positive_effective_contact_qualified"] is False

@pytest.mark.parametrize("key,value",[("clock_valid",False),("rod_tick",12),("physics_step",-1),
    ("physics_step",True),("timestamp_s",float("nan"))])
def test_invalid_reset_boundary_rejected(recorded,key,value):
    reset=deepcopy(recorded[0]);reset[key]=value
    with pytest.raises(ValueError):module.LiveEffectivePadSource(ROOT,reset_clock=reset)

def test_reset_epoch_is_not_hardcoded_to_two_sdk_steps(recorded):
    reset=deepcopy(recorded[0]);x=synthetic_complete_fixture(recorded[1])
    delta_step=997;delta_time=delta_step/480
    reset["physics_step"]+=delta_step;reset["timestamp_s"]+=delta_time
    # Shift all owned identity copies, preserving the same post-reset duration.
    def shift(obj):
        if isinstance(obj,dict):
            if all(k in obj for k in ("episode_uid","physics_step","timestamp_s")):
                obj["physics_step"]+=delta_step;obj["timestamp_s"]+=delta_time
            for value in obj.values():shift(value)
        elif isinstance(obj,list):
            for value in obj:shift(value)
    shift(x)
    cap=x["rod"]["rod_mesh_contact_pairs"];complete=cap["complete_capture"]
    clock=x["action_readback"]["post"]["identity"][0]
    x["rod"]["rod_mesh_contact_pairs"]=read_complete_pair_capture(
        SimpleNamespace(contact_pair_samples=lambda:complete["raw"]),
        dt_s=complete["dt_s"],physics_step=clock["physics_step"],rod_tick=clock["rod_tick"],
        mesh_names=complete["mesh_names"])
    x["closure_geometry"]["raw_sha256"]=_digest(x["closure_geometry"]["raw"])
    result=module.LiveEffectivePadSource(ROOT,reset_clock=reset).capture(x)
    assert result["identity"]["physics_step"]==clock["physics_step"]
    assert not result["teacher_admission"]
