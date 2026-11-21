"""CPU-scoped source adapter replay over real saved Tripod Reset buffers.

Only native getters are replaced; no new SDK/solver step or qualification.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import importlib

import numpy as np
import pytest
import torch

from test_tripod_reset_manifest import ROOT, RUN


def module():
    value = importlib.import_module("go2_pvcnn.control.dh116.tabletop_native_teacher_sources")
    assert hasattr(value,"NativeTripodSources"), "independent Tripod source adapter missing"
    return value


def arrange(monkeypatch,tmp_path):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import BackendProfile
    from go2_pvcnn.control.dh116.tabletop_native_teacher_collector import NativeJournal
    from go2_pvcnn.control.dh116.tabletop_tripod_reset_manifest import load_tripod_reset_manifest
    from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import decode_owned
    from go2_pvcnn.control.dh116.tabletop_deformx_parameter_snapshot import validate_snapshot
    from go2_pvcnn.control.dh116.tabletop_native_teacher_action_history import NativeAction24HistoryOwner
    from go2_pvcnn.control.dh116.tabletop_r0_health_separation import make_solver_status
    from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as parameters
    from test_native_teacher_model import raw_cpu_packets
    import hashlib,sys
    manifest_path = RUN.parent/"tripod-teacher-resolved-reset-v1.json"
    binding=dict(path=str(manifest_path),sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    manifest=load_tripod_reset_manifest(dict(style="tripod",resolved_reset_manifest=binding),project=ROOT)
    reset=decode_owned(json.loads((RUN/"native/episode_reset.json").read_bytes()))["reset"]
    numeric=decode_owned(json.loads((RUN/"native/mesh_reset_numeric_state.json").read_bytes()))
    profile=BackendProfile.load(ROOT)
    clock=deepcopy(reset["after_clock"])
    rope=object()
    journal=NativeJournal(tmp_path/"source.jsonl",scope="cpu_fixture")
    exchange=SimpleNamespace(_step=lambda action,substep: None)
    cfg=json.loads((ROOT/"configs/dh116_tasks/flexible_pinch_candidate_v1.json").read_bytes())
    owner=SimpleNamespace(
        sim=SimpleNamespace(current_time_step_index=2),
        rod=SimpleNamespace(rope=rope,bindings=deepcopy(reset["rod_reset"]["mesh_bindings"]),
            mesh_names=tuple(reset["rod_reset"]["mesh_names"]),_base=SimpleNamespace(cfg=cfg),
            clock=lambda:deepcopy(clock)),
        episode=SimpleNamespace(_exchange=exchange,_end=66,_control_index=0),
        bindings=(SimpleNamespace(hand=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(1,17)))),),
        episode_uid=clock["episode_uid"],ledger=journal,_restores=[])
    contract=SimpleNamespace(config=SimpleNamespace(
        preflight=dict(style="tripod",profile="approach_wiring_smoke_v1"),project=ROOT,episode_cap_s=1.),
        reset_manifest=manifest,backend_profile=profile,reward_config=SimpleNamespace(profile="approach_wiring_smoke_v1"),
        budget=dict(wall_timeout_s=180.),runtime=SimpleNamespace(remaining_s=lambda:100.,remaining_steps=lambda:64))
    _,_,template,_=raw_cpu_packets(profile)
    packet=deepcopy(template)
    state=numeric["samples"][1]["state"]
    packet.update(identity=deepcopy(clock),source_instance_id=id(rope),source_scope="cpu_fixture",
        raw_native_parameters=deepcopy(state["parameters"]),
        native_parameters=validate_snapshot(state["parameters"],expected_num_envs=1,expected_nodes=41,expected_time_s=0.),
        node_positions_m=np.array(state["positions"],copy=True),
        segment_quaternions_xyzw=np.array(state["quaternions"][:,:40],copy=True))
    packet["reward_radius_sample"]["source"]["identity"]=deepcopy(clock)
    packet["reward_radius_sample"]["value"]=float(state["parameters"]["contact_radius_m"])
    class Getter:
        def __init__(self,nodes,*,device):
            np.testing.assert_array_equal(nodes,state["positions"][0])
        def capture_reset(self,row,**kwargs):
            assert row["physics_step"]==2 and row["rod_tick"]==0
            return deepcopy(packet)
    monkeypatch.setattr(parameters,"NativeParameterEndpoints",Getter)
    monkeypatch.setitem(sys.modules,"ropesim._ropesim",SimpleNamespace())
    source=module().NativeTripodSources(owner,contract,scope="cpu_fixture")
    monkeypatch.setattr(source,"_health",lambda c:make_solver_status(c,0,source_scope="cpu_fixture",
        source_instance_id=id(rope),backend_sha256=profile.binary_sha256))
    history=NativeAction24HistoryOwner(initial_action24=torch.zeros(1,24),seed_source="CPU-owned reset zero",
        physics_budget_end=66,solver_substeps=12)
    history.wrap_controller_reset(lambda c:dict(owner="controller",episode_uid=c["episode_uid"],
        status="cleared",source="CPU actual memory reset"))(
            dict(clock,source_seed_sha256=reset["seed"]["sha256"]))
    owner.history=history
    return source,owner,reset,journal


def test_tripod_source_binds_six_pads_reward_and_context_identity(monkeypatch,tmp_path):
    module()
    source,owner,reset,journal=arrange(monkeypatch,tmp_path)
    try:
        from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import profile_reader_scope
        with profile_reader_scope(source.contract.backend_profile):
            source.after_reset(dict(reset=reset))
        from go2_pvcnn.control.dh116.tabletop_tripod_reset_distance import MeasuredTripodResetApproach
        from go2_pvcnn.control.dh116.tabletop_native_teacher_policy_tripod_distance import PolicyOwnedTripodApproach
        assert type(source.reset_distance) is MeasuredTripodResetApproach
        assert type(source.distance) is PolicyOwnedTripodApproach
        assert source._last.fields["distance"].raw > 0
        assert source._last.fields["distance"].stamp.source == "PolicyOwnedTripodApproach.fixed_material41"
        assert source._last.stamp.scope == "cpu_fixture"
        assert not {"contact","slip","force_proxy"} & set(source._last.fields)
        source.initialize_context(dict(reset=reset))
        assert owner.context.current.metadata.grasp == "tripod"
        assert owner.context.current.native_qualified is False
        row=json.loads(journal.path.read_text().splitlines()[-1])
        assert row["operation"] == "reset_sources"
        assert row["payload"]["distance"]["missing_reason"] is None
        assert all(len(hand["required_pads"])==3 for hand in row["payload"]["distance"]["per_hand"].values())
    finally:
        journal.close()


def test_missing_middle_sdf_coverage_stays_unknown_and_cannot_seed_reward(monkeypatch,tmp_path):
    module()
    source,owner,reset,journal=arrange(monkeypatch,tmp_path)
    try:
        grid=source.reset_distance.geometry._grids["right"]["finger33_link"]
        grid["origin"]=np.asarray(grid["origin"])+10. # Explicit CPU coverage failure.
        from go2_pvcnn.control.dh116.tabletop_training_reward import SourceFailure
        from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import profile_reader_scope
        with profile_reader_scope(source.contract.backend_profile):
            with pytest.raises(SourceFailure,match="Tripod distance unavailable"):
                source.after_reset(dict(reset=reset))
        assert source._last is None
        assert not journal.path.read_text()
    finally:
        journal.close()
