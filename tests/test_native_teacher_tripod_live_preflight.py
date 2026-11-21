"""CPU preflight routing with captured Reset; earlier native I/O validators are isolated.

This tests the later health-dispatch boundary, not full live source qualification.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_native_teacher_tripod_admission import ROOT,config
from test_dh116_fresh_supported_rod import fresh_proximal_case
from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned


def test_tripod_live_preflight_uses_the_supported17_health_reader(monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_native_teacher_preflight as pre
    from go2_pvcnn.control.dh116 import tabletop_mesh_observation_source as source
    from go2_pvcnn.control.dh116 import tabletop_dual_hand_reset as reset_owner
    from go2_pvcnn.control.dh116 import tabletop_mesh_reset_numeric_state as numeric
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
    from go2_pvcnn.control.dh116.tabletop_power_shared_rod import PowerSharedRodRuntime
    from go2_pvcnn.control.dh116.tabletop_tripod_reset_distance import MeasuredTripodResetApproach

    profile=BackendProfile.load(ROOT)
    manifest=pre.load_reset_manifest(config(),project=ROOT)
    path=Path("/mnt/mydisk/xk/.local/share/dh116-autonomous-3c80166/evidence/native-output-tripod-smoke8-f2940452/teacher.tripod-ppo-fresh-smoke8-v3/attempt-riwkhjjt/native/training_events.jsonl")
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='76f17475792ff4b3da1572a74893f7200d30e7e1481c02397cdb5450c31b8039'
    rows=[decode_owned(json.loads(line)) for line in path.read_text().splitlines()]
    receipt=deepcopy(next(x["payload"]["reset"] for x in rows if x["operation"]=="owner_event"))
    audit=deepcopy(next(x["payload"] for x in rows if x["operation"]=="numeric_reset"))
    parameters=deepcopy(next(x["payload"]["parameters"] for x in rows if x["operation"]=="reset_sources"))
    expected=receipt["after_clock"]
    base,kwargs=fresh_proximal_case(monkeypatch,nodes=parameters["node_positions_m"][0])
    kwargs["run_uuid"]=expected["run_uuid"]
    runtime=PowerSharedRodRuntime(base,**kwargs)
    poses={name:(np.zeros(3),np.array([0.,0.,0.,1.])) for name in runtime.bindings}
    runtime.reset(physics_step=2,episode_uid=expected["episode_uid"],reset_id=1,pad_poses=poses)
    runtime.rope.last_error=0
    sim=SimpleNamespace(current_time_step_index=2,current_time=receipt["after_actual_clocks"]["sdk_timestamp_s"],
                        get_physics_dt=lambda:1/480)
    backend=SimpleNamespace(__file__=str(profile.binary_path),RopeBatch=type(runtime.rope))
    monkeypatch.setitem(sys.modules,"ropesim._ropesim",backend)
    parameters["source_instance_id"]=id(runtime.rope) # Explicit CPU transport owner, no mutation of archive.
    bindings=()
    owner=SimpleNamespace(rod=runtime,sim=sim,bindings=bindings,backend_profile=profile)
    mounted=dict(identity=dict(expected,policy_step=0),source_packets=[
        dict(reset_seed=dict(root_seed_w=receipt["seed"]["authored_root_seed_w"]))])
    current=deepcopy(audit["samples"][1]["state"])
    # Hardware reads are already tested by their owning suites. They are outside
    # this routing case; the actual Python type/binary/clock validators still
    # execute over the explicit CPU doubles, without native qualification.
    monkeypatch.setattr(source,"verify_runtime",lambda runtime:None)
    monkeypatch.setattr(pre,"validate_live_numeric",lambda *a,**k:None)
    monkeypatch.setattr(reset_owner,"_hands",lambda *a,**k:deepcopy(receipt["hands"]))
    monkeypatch.setattr(numeric,"_capture",lambda *a,**k:deepcopy(current))
    monkeypatch.setattr(pre,"validate_live_mesh",lambda *a,**k:None)
    monkeypatch.setattr(pre,"validate_live_mounted",lambda *a,**k:dict(valid_channels=66,unknown_channels=305))
    contract=pre.LiveResetContract(owner,sim,runtime,bindings,receipt,audit,mounted,parameters,
                                  MeasuredTripodResetApproach(ROOT),object())
    result=pre.live_preflight(owner,contract,backend_profile=profile,manifest=manifest)
    assert result["reset_identity"]==expected
    assert result["rod_instance_id"]==id(runtime.rope)
    assert result["new_native_steps"]==result["ppo_updates"]==0
    assert result["native_qualification"] is result["teacher_admission"] is False
    assert runtime.rope.steps==0 and runtime.clock()==expected
    runtime.rope.last_error=4
    with pytest.raises(ValueError,match="numerical health"):
        pre.live_preflight(owner,contract,backend_profile=profile,manifest=manifest)
    assert runtime.rope.steps==0 and runtime.clock()==expected
