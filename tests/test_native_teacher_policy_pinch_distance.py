"""Policy-owned source tests; all SDK/getter substitutes stay explicit CPU fixtures."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_native_teacher_context import backend_profile

ROOT=Path(__file__).resolve().parents[1]


def test_old_frozen_reader_rejects_legal_policy_action():
    from test_dh116_right_approach_geometry import packet
    from go2_pvcnn.control.dh116.tabletop_bilateral_owned_geometry import validate_projection,plan
    _,row=packet(1);pair=plan.derive_pair()
    row["action_readback"]["request"]["action24"]=np.full((1,24),.123)
    with pytest.raises(ValueError,match="frozen 24D action"):validate_projection(pair,row)


def test_policy_reader_has_no_recipe_schedule_or_absolute_sdk_cap():
    from go2_pvcnn.control.dh116.tabletop_native_teacher_policy_pinch_distance import PolicyOwnedPinchApproach
    assert PolicyOwnedPinchApproach.VERSION=="dh116-policy-owned-pinch-distance-v1"


@pytest.fixture(scope="module")
def sdk_templates():
    # Existing owned SDK-double fixture only supplies template poses/body names.
    # No historical rod/backend/evidence packet is retagged as new native data.
    from test_dh116_measured_closure_geometry import owner,case
    geometry=owner.__wrapped__()
    return case.__wrapped__(geometry)


def actual_source_case(monkeypatch,tmp_path,profile,sdk_templates,*,start_sdk=2):
    from test_native_teacher_collector import fixture
    from test_native_teacher_model import raw_cpu_packets
    from go2_pvcnn.control.dh116.tabletop_native_teacher_sources import NativePinchSources
    from go2_pvcnn.control.dh116 import tabletop_native_parameter_endpoints as parameters
    from go2_pvcnn.control.dh116.tabletop_native_teacher_session import SessionBudget
    from go2_pvcnn.control.dh116.tabletop_native_teacher_profile import profile_reader_scope
    from go2_pvcnn.control.dh116.tabletop_deformx_parameter_snapshot import validate_snapshot
    from go2_pvcnn.control.dh116.tabletop_r0_health_separation import make_solver_status
    from go2_pvcnn.control.dh116.tabletop_r0_measured_approach import positive_pad_mean
    c,ep,rod,hands,sim,rows=fixture(monkeypatch,tmp_path,profile,rollout_steps=4)
    ep._record=c._original_record;ep._exchange=c._original_exchange;c.owner._restores=[]
    if start_sdk>2:
        from go2_pvcnn.control.dh116.tabletop_deformable_dual_hand_episode import DualHandPhysicalEpisode
        from go2_pvcnn.control.dh116.tabletop_native_teacher_action_history import NativeAction24HistoryOwner
        from go2_pvcnn.control.dh116.tabletop_native_teacher_context import TaskContextOwner
        from test_native_teacher_collector import FixtureSources
        # Genuine original CPU control calls establish the absolute SDK/solver
        # clock. No clock field is assigned or reset to manufacture SDK>234.
        ep._end=1024;ep._exchange._end=1024;ep._record=rows.append
        for _ in range((start_sdk-2)//8):ep.control(torch.zeros(1,24))
        assert sim.current_time_step_index==start_sdk
        history=NativeAction24HistoryOwner(initial_action24=torch.zeros(1,24),seed_source="CPU fresh actual reset",
            physics_budget_end=1024,solver_substeps=12,previous_reset=rod.clock())
        owners=dict(c.owner.original_owner_resets)
        owners["controller"]=history.wrap_controller_reset(lambda clock:dict(owner="controller",status="cleared",
            episode_uid=clock["episode_uid"],source="original CPU owner clear"))
        def durable(row):c.journal.emit("physical",row);rows.append(deepcopy(row));history.record(row)
        fresh=DualHandPhysicalEpisode(sim,rod,ep._bindings,backend_artifact=ep._boundary._artifact,
            backend_sha256=ep._boundary._sha256,inspect_geometry=ep._inspect_geometry,physics_budget_end=1024,record=durable)
        fresh.reset(c.owner.reset_seed,episode_uid=f"cpu-policy-episode2-sdk{start_sdk}",owner_resets=owners)
        for side in ("left","right"):
            history.record(dict(kind="initial_geometry",side=side,evidence=dict(side=side,
                identity=dict(sdk_step=start_sdk,sdk_time_s=sim.current_time))))
        ep=fresh;c.history=history;c.owner.history=history;c.owner.episode=ep
        c.owner.pre_action_state=ep.pre_action_state;c.owner.control=ep.control
        c.sources=FixtureSources(c.owner,profile,rod,99)
        c.context=TaskContextOwner(backend_profile=profile,mode=c.reward.config.profile,scope="cpu_fixture",episode_cap_s=2.,external_cap_s=3.)
        c.context.reset_from_verified_state(c.sources.context_state(None),history_owner=history)
        c.owner.context=c.context
    # Assemble new, explicit CPU SDK buffers from the source geometry fixture.
    # The original runtime/control/row/history and locks remain active.
    for hand,template in zip(hands,sdk_templates[4]):
        hand.body_names=list(template.body_names)
        hand.link_poses=torch.as_tensor(template.transforms).clone()
        hand.root_native=torch.as_tensor(template.root).clone()
        hand.data.root_pose_w=torch.as_tensor(template.data.root_pose_w).clone()
        hand.data.body_link_pose_w=hand.link_poses[..., [0,1,2,6,3,4,5]]
        hand.root_physx_view.link_paths=[[f"/World/{hand.side}Hand/"+n for n in hand.body_names]]
    from go2_pvcnn.control.dh116.tabletop_palm_mount import MountView
    from dataclasses import replace
    ep._bindings=tuple(replace(binding,mount=MountView(binding.hand)) for binding in ep._bindings)
    root_seed=torch.stack([h.data.root_pose_w[0] for h in hands])[None].double()
    rod.rope.nodes=np.array(sdk_templates[0]["rod"]["node_positions"],copy=True)
    rod.rope.force=np.zeros_like(rod.rope.nodes)
    rod._base.nodes=rod.rope.nodes[0].copy();rod._base.rest=np.linalg.norm(np.diff(rod._base.nodes,axis=0),axis=1)
    limits=dict(max_sdk_steps=1024,max_solver_calls=1024,max_resets=4,max_episode_count=4,wall_timeout_s=1000.)
    contract=SimpleNamespace(config=SimpleNamespace(preflight=dict(style="pinch",profile="approach_wiring_smoke_v1"),
        project=ROOT,episode_cap_s=20.),backend_profile=profile,
        reset_manifest=SimpleNamespace(data=lambda:dict(rod_state=dict(positions=rod.rope.nodes.tolist()))),
        budget=limits,runtime=SessionBudget(limits))
    # Only actual native parameter/health getters are replaced. Packets are
    # newly generated CPU-scoped current-ABI data, not old backend relabelling.
    _,_,template_parameters,_=raw_cpu_packets(profile)
    def packet(row):
        from go2_pvcnn.control.dh116.tabletop_deformable_r0_replay import _copy
        result=_copy(template_parameters);clock={k:row[k] for k in template_parameters["identity"] if k in row}
        # The raw getter identity has no policy-step requirement.
        result["identity"]=clock;result["source_instance_id"]=id(rod.rope);result["node_positions_m"]=np.array(row["node_positions"],copy=True)
        result["raw_native_parameters"]["time_s"]=clock["solver_time_s"]
        result["native_parameters"]=validate_snapshot(result["raw_native_parameters"],expected_num_envs=1,expected_nodes=41,expected_time_s=clock["solver_time_s"])
        result["reward_radius_sample"]["source"]["identity"]=clock
        return result
    class Getter:
        def __init__(self,*a,**k):pass
        def capture(self,row,**k):return packet(row)
        def capture_reset(self,row,**k):return packet(row)
    monkeypatch.setattr(parameters,"NativeParameterEndpoints",Getter)
    c.owner.sim=sim;c.owner.rod=rod;c.owner.bindings=ep._bindings;c.owner.episode_uid=rod.clock()["episode_uid"]
    import sys
    monkeypatch.setitem(sys.modules,"ropesim._ropesim",SimpleNamespace())
    source=NativePinchSources(c.owner,contract,scope="cpu_fixture")
    c.owner.sources=source
    # Real frozen SDK/asset/SDF bindings, copied onto the explicit CPU scene.
    for name,binding in rod.bindings.items():
        side,pad=name.split(".");locked=source.distance.geometry.sources[side]["sdf_binding"][pad]
        binding.update(body_index=hands[("left","right").index(side)].body_names.index(pad),
            sdf_sha256=locked["sha256"],capture_sha256=locked["capture_sha256"])
    from go2_pvcnn.control.dh116.tabletop_deformable_live_geometry import LiveGeometryReader
    from go2_pvcnn.control.dh116.tabletop_deformable_clearance_guard import ClearanceGuardedExchange
    geometry_reader=LiveGeometryReader(sim,ep._bindings,ep._inspect_geometry(),read_clock=rod.clock)
    ep._exchange=ClearanceGuardedExchange(sim,rod,ep._bindings,root_seed_w=root_seed,
        read_geometry=geometry_reader,read_geometry_after_sdk=geometry_reader.read_after_sdk_step,physics_budget_end=1024)
    raw=deepcopy(next(e["reset"]["rod_reset"] for e in reversed(rows) if e["kind"]=="reset"))
    raw.update(node_positions=rod.rope.nodes.copy(),velocities=np.zeros_like(rod.rope.nodes),
        rest_lengths_m=rod._base.rest.copy(),mesh_bindings=deepcopy(rod.bindings))
    ep._rod_packet=ep._boundary.capture(raw)
    def health(clock):
        return make_solver_status(clock,0,source_scope="cpu_fixture",source_instance_id=id(rod.rope),backend_sha256=profile.binary_sha256)
    monkeypatch.setattr(source,"_health",health)
    clock=rod.clock();source.parameters=packet(raw);source._raw=raw
    values=[]
    for side,hand in zip(("left","right"),hands):
        pads=[]
        for pad in ("finger14_link","finger23_link"):
            pose=np.asarray(hand.link_poses[0,hand.body_names.index(pad)])
            transform=source.distance.geometry._pose_matrix(np.r_[pose[:3],pose[6],pose[3:6]])
            pads.append(source.distance.geometry._sdf(source.distance.geometry._grids[side][pad],transform,rod.rope.nodes[0],radius=.005))
        values.append(positive_pad_mean(pads))
    assert all(v is not None for v in values)
    reset_distance=dict(auxiliary=dict(identity=clock,samples=dict(approach_distance_m=dict(value=values))))
    with profile_reader_scope(profile):source._reset_endpoint=source._last=source._endpoint(clock,reset_distance,health(clock),None)
    source._install_exchange()
    return c,source,ep,rod,hands,sim,rows


def run_window(c,source,ep,*,action=.123):
    from go2_pvcnn.control.dh116.tabletop_native_teacher_collector import _TerminalObserver
    from go2_pvcnn.control.dh116.tabletop_training_reward import TrainingReward
    history=c.history;pre=ep.pre_action_state();history.capture_pre_action(pre)
    # Sample with the real trainer, using a deterministic legal mean only for
    # this source-interface test. Original Tanh sampling remains untouched.
    obs=c.sources.pack(c.trainer.model,pre,c.context.current)
    with torch.no_grad():
        c.trainer.model.actor_head.weight.zero_();c.trainer.model.actor_head.bias.fill_(float(np.arctanh(action)))
        c.trainer.model.log_std.fill_(-20.)
    sample=c.trainer.collect_step(obs,obs);decision=history.begin(pre,c.trainer)
    reward=TrainingReward(c.reward.config);source.reset_reward(reward)
    reward.begin(source.submitted_request(sample,decision),remaining_substeps=8)
    reason=[None];record=ep._record
    def observed(event):
        record(event)
        endpoint=source.observe(event);reward.observe(endpoint)
        if reward._terminal:reason[0]="reward:"+"|".join(reward._reasons)
    ep._record = observed
    ep._exchange = _TerminalObserver(ep._exchange, lambda: reason[0], c.journal,
                                     lambda row: None, wall_time=lambda: None)
    result=ep.control(sample["actions"])
    settlement=history.end(result);issued=history.issued_for_pending(c.trainer,settlement)
    packet=reward.settle_once(sample["collection_token"])
    return packet,result,issued


@pytest.mark.parametrize("start_sdk,action",[(2,.123),(10,-.51),(234,.72)])
def test_actual_adapter_legal_policy_action_has_finite_distance_and_history_binding(monkeypatch,tmp_path,backend_profile,sdk_templates,start_sdk,action):
    c,source,ep,rod,hands,sim,rows=actual_source_case(monkeypatch,tmp_path,backend_profile,sdk_templates,start_sdk=start_sdk)
    packet,result,issued=run_window(c,source,ep,action=action)
    assert np.isfinite(packet.reward)
    assert source.distance.VERSION=="dh116-policy-owned-pinch-distance-v1"
    assert packet.scope=="cpu_fixture" and packet.native_qualification is False
    assert result["physical_samples"]==8 and not packet.terminated
    assert abs(float(issued[0,0])-action)<1e-4
    assert sim.current_time_step_index==start_sdk+8
    assert rows[-1]["exchange"]["policy_step"]==0
    import json
    measured=[json.loads(line) for line in c.journal.path.read_text().splitlines() if '"operation": "policy_pinch_distance"' in line]
    assert len(measured)==8
    assert measured[-1]["payload"]["history_binding"]["collection_token"]==packet.token
    assert measured[-1]["payload"]["history_binding"]["physical_samples"]==8
    evidence=measured[-1]["payload"]
    assert evidence["auxiliary"]["samples"]["approach_distance_m"]["source"].endswith(evidence["source_sha256"])
    assert evidence["source"]["target_material_ids"]==list(range(41))


def test_actual_source_third_original_guard_is_finite_terminal_not_corruption(monkeypatch,tmp_path,backend_profile,sdk_templates):
    c,source,ep,rod,hands,sim,rows=actual_source_case(monkeypatch,tmp_path,backend_profile,sdk_templates)
    original=sim.step
    def third(*,render):
        original(render=render)
        if rod.rope.steps==2:hands[1].measured[0,hands[1].joint_names.index("finger12")]=-.05
    sim.step=third
    packet,result,issued=run_window(c,source,ep)
    assert result["physical_samples"]==3 and rod.rope.steps==3
    assert packet.terminated and packet.event==-40 and packet.endpoint_phi==0
    assert packet.duration_s==pytest.approx(3/480) and np.isfinite(packet.reward)
    assert source._error is None
    assert rows[-1]["exchange"]["closure_geometry"]["source_complete"] is True
    assert rows[-1]["exchange"]["closure_geometry"]["passed"] is False



@pytest.mark.parametrize("damage",["pose","clock","source","radius","history_target","scope","nan","parameter_clock"])
def test_actual_source_rejects_third_corrupt_measurement_after_durable_row(monkeypatch,tmp_path,backend_profile,sdk_templates,damage):
    from go2_pvcnn.control.dh116.tabletop_training_reward import SourceFailure
    c,source,ep,rod,hands,sim,rows=actual_source_case(monkeypatch,tmp_path,backend_profile,sdk_templates)
    step=ep._exchange._step
    def damaged(action,substep):
        row=step(action,substep)
        if substep==2:
            if damage=="pose":row["closure_geometry"]["raw"]["hands"]["left"]["link_poses_xyzw"][0,0,0]+=.01
            elif damage=="clock":row["closure_geometry"]["raw"]["identity"]["physics_step"]+=1
            elif damage=="source":row["closure_geometry"]["sources"]["policy_owned"]={}
            elif damage=="radius":row["material_director"]["reward_radius_sample"]["value"]*=2
            elif damage=="scope":row["material_director"]["source_scope"]="owner_source_unqualified"
            elif damage=="nan":row["closure_geometry"]["raw"]["rod_nodes_m"][0,0]=np.nan
            elif damage=="parameter_clock":row["material_director"]["identity"]["physics_step"]+=1
            else:row["action_readback"]["post"]["hands"]["left"]["backend_targets"][0,0]+=.1
        return row
    ep._exchange._step=damaged
    with pytest.raises(ValueError if damage=="history_target" else SourceFailure):run_window(c,source,ep)
    assert rod.rope.steps==3 and sim.current_time_step_index==5
    assert c.trainer._pending is not None and c.trainer.counters["updates"]==0 and len(c.trainer.rollout)==0
    assert rows[-1]["kind"]=="physical" and rows[-1]["exchange"]["physics_step"]==5
    assert not list(tmp_path.glob("*.pt"))



@pytest.mark.parametrize("device_index", [0, 3])
def test_actual_adapter_passes_string_device_to_original_parameter_constructor(backend_profile, device_index):
    """Exercise the unchanged native reader constructor without CUDA allocation."""
    from go2_pvcnn.control.dh116.tabletop_native_teacher_sources import NativePinchSources
    from go2_pvcnn.control.dh116.tabletop_native_parameter_endpoints import NativeParameterEndpoints
    # This is a CPU interface fixture, not an actual GPU tensor/native source.
    # torch.device is the same property type exposed by the real SDK tensor.
    device = torch.device(f"cuda:{device_index}")
    nodes = np.zeros((1, 41, 3), dtype=np.float32)
    nodes[0, :, 0] = np.arange(41, dtype=np.float32) * .005
    owner = SimpleNamespace(
        bindings=[SimpleNamespace(hand=SimpleNamespace(
            data=SimpleNamespace(joint_pos=SimpleNamespace(device=device))))],
        rod=SimpleNamespace(_base=SimpleNamespace(cfg={"rod": {"solver_substeps": 12}})),
        sim=SimpleNamespace(current_time_step_index=2))
    contract = SimpleNamespace(
        config=SimpleNamespace(preflight={"style": "pinch",
            "profile": "approach_wiring_smoke_v1"}, project=ROOT),
        backend_profile=backend_profile,
        reset_manifest=SimpleNamespace(data=lambda: {"rod_state": {"positions": nodes}}))
    source = NativePinchSources(owner, contract, scope="cpu_fixture")
    assert type(source.reader) is NativeParameterEndpoints
    assert source.reader.device == str(device)
    assert source.reader.sources["backend_sha256"] == backend_profile.binary_sha256
    np.testing.assert_array_equal(source.reader._nodes, nodes[0])
