"""Independent common66 + material41x13 + fixed-S0 context51 Teacher.

The original Tanh distribution, material encoder and PPO remain unchanged.
Only explicit CPU fixtures can pack inputs until W4 supplies a real source
adapter and same-owner witness. No native producer is implemented here.
"""
from copy import deepcopy
from pathlib import Path
import json
import math

import numpy as np
import torch
from torch import nn

from .tabletop_native_teacher_profile_v2 import (
    MODEL_SCHEMA_VERSION_V2, MODEL_SCHEMA_SHA256_V2,
    INPUT_SCHEMA_VERSION_V2, INPUT_SCHEMA_SHA256_V2,
    CONTEXT_SCHEMA_VERSION_V2, CONTEXT_SCHEMA_SHA256_V2,
    PHASE_SCHEMA_VERSION_V2, PHASE_SCHEMA_SHA256_V2, PHASE_CONTRACT_SHA256_V2,
    INPUT_KEYS_V2, CONTEXT_WIDTH_V2, CPU_FIXTURE_SCHEMA_VERSION_V2,
    validate_backend_profile_v2, digest,
)
from .tabletop_native_teacher_context_v2 import (
    ContextIdentity, TaskContextPacketV2, encode_context_v2,
    validate_context_v2, validate_context_tensors_v2,
)
from .tabletop_deformable_privileged_teacher import (
    DeformablePrivilegedTeacher, ROD_FEATURES, _check_rod_identity, _features, _MaterialEncoder,
)
from .tabletop_flexible_measured_common_teacher import common_selection
from .tabletop_privileged_teacher import PrivilegedTeacher
from .tabletop_stereo_contract import file_sha256

VERSION = MODEL_SCHEMA_VERSION_V2


def _profile(profile):
    return validate_backend_profile_v2(profile)


def _common(ledger,valid):
    if (not isinstance(ledger,torch.Tensor) or ledger.dtype != torch.float32 or ledger.ndim != 2
            or ledger.shape[1] != 371 or not len(ledger) or not isinstance(valid,torch.Tensor)
            or valid.dtype != torch.bool or valid.shape != ledger.shape or valid.device != ledger.device):
        raise ValueError("raw common371 requires float32[N,371] and boolean validity")
    wanted=torch.zeros_like(valid)
    wanted[:,common_selection()]=True
    if (not torch.equal(wanted,valid) or not bool(torch.isfinite(ledger[valid]).all())
            or not bool(torch.isnan(ledger[~valid]).all())):
        raise ValueError("common fixed measured66 mask/finite and 305 raw NaN required")
    return ledger[:,common_selection()]


def _all_nodes(mask,batch,device):
    if (not isinstance(mask,torch.Tensor) or mask.shape != (batch,41) or mask.dtype != torch.bool
            or mask.device != device or not bool(mask.all())):
        raise ValueError("all 41 original material nodes required; no mask may hide a bad node")
    return mask


def _context_clock(packet,clock):
    for i,identity in enumerate(packet.metadata.identities):
        expected=dict(env_ids=identity.env_id,episode_uids=identity.episode_uid,
            reset_ids=identity.reset_id,rod_uids=identity.rod_uid)
        if (any(clock[k][i] != v for k,v in expected.items())
                or any(clock[k] != getattr(identity,k) for k in ("physics_step","rod_tick","timestamp_s"))):
            raise ValueError("context and rod/common time/episode identity differs")


def _pack_v2(ledger,valid,snapshot,clock,mask,packet,profile,mode):
    common=_common(ledger,valid)
    _all_nodes(mask,len(common),common.device)
    _check_rod_identity(snapshot,clock,profile.binary_sha256,len(common))
    if (snapshot["node_positions"].shape != (len(common),41,3)
            or packet.metadata.mode != mode):
        raise ValueError("exact rod41 and task context profile required")
    validate_context_v2(packet,backend_profile=profile)
    if (len(packet.values) != len(common) or len(packet.valid) != len(common)
            or len(packet.metadata.identities) != len(common)):
        raise ValueError("context/common/rod batch differs")
    _context_clock(packet,clock)
    if packet.values.device != common.device:
        raise ValueError("common/context device differs")
    features=_features(snapshot,common,mask)
    return dict(common371=ledger.clone(),common_valid=valid.clone(),rod_nodes=features.clone(),
        rod_mask=mask.clone(),task_context=packet.values.clone(),task_context_valid=packet.valid.clone())


def model_contract_v2(config, backend_profile, context_mode):
    profile = _profile(backend_profile)
    model = config.get("model",{})
    expected = dict(common_size=371,action_size=24,hidden=[256,256],rod_feature_size=13,rod_hidden=64)
    if (config.get("profile") != "fixed_s0_pinch_v2" or config.get("grasp") != "pinch"
            or config.get("task_kind") != "deformable_rod"
            or config.get("native_ready") is not False or config.get("use_cameras") is not False
            or config.get("checkpoint_adapter_ready") is not False
            or config.get("resolved_reset_manifest") is not None
            or set(model) != set(expected)|{"initial_std"}
            or any(model[k] != v for k,v in expected.items())
            or context_mode != "fixed_s0_v2"
            or any(type(config.get(k)) not in (int,float) or not math.isfinite(config[k])
                   or config[k] <= 0 for k in ("prelift_stable_s","prelift_scale_s"))
            or type(model.get("initial_std")) not in (int,float)
            or not math.isfinite(model["initial_std"]) or model["initial_std"] <= 0):
        raise ValueError("explicit camera-free fixed S0 v2 CPU candidate config required; unresolved S0 grants no production admission")
    parent = Path(__file__).parent
    identity = dict(policy_version=VERSION,task_kind="deformable_rod",grasp=config["grasp"],
        checkpoint_namespace="native_fixed_s0_teacher_pinch_cpu_v2",
        config_semantic_sha256=digest(config),context_mode=context_mode,common_size=371,
        common_indices=list(common_selection()),raw_common_unknown="NaN_false_305",
        action_size=24,distribution="tanh_latent_gaussian",rod_features=list(ROD_FEATURES),
        rod_schema=profile.rod_schema,model_schema=MODEL_SCHEMA_VERSION_V2,
        model_schema_sha256=MODEL_SCHEMA_SHA256_V2,
        input_schema=INPUT_SCHEMA_VERSION_V2,input_schema_sha256=INPUT_SCHEMA_SHA256_V2,
        input_keys=list(INPUT_KEYS_V2),context_schema=CONTEXT_SCHEMA_VERSION_V2,
        context_schema_sha256=CONTEXT_SCHEMA_SHA256_V2,context_size=CONTEXT_WIDTH_V2,
        context_encoded_size=2*CONTEXT_WIDTH_V2,
        phase_schema=PHASE_SCHEMA_VERSION_V2,phase_schema_sha256=PHASE_SCHEMA_SHA256_V2,
        phase_contract_sha256=PHASE_CONTRACT_SHA256_V2,phase_source="execution_phase_owner_v2",
        fixed_s0_manifest_sha256=None,prelift_stable_s=float(config["prelift_stable_s"]),
        prelift_scale_s=float(config["prelift_scale_s"]),
        cpu_fixture_schema=CPU_FIXTURE_SCHEMA_VERSION_V2,
        backend_sha256=profile.binary_sha256,backend_source_tree_sha256=profile.source_tree_sha256,
        backend_profile_sha256=profile.digest,backend_inventory_sha256=profile.inventory_sha256,
        backend_patch_sha256=profile.patch_sha256,
        architecture=dict(common=[66,256,256],rod=[13,64,64],rod_pool="mean+max",
            context=[102,64,64],fusion=448,actor=24,critic=1,independent_branches=True),
        dependency_source_sha256={name:file_sha256(parent/name) for name in (
            Path(__file__).name,"tabletop_native_teacher_profile_v2.py",
            "tabletop_native_teacher_context_v2.py","tabletop_native_phase_contract_v2.py",
            "tabletop_native_teacher_context.py","tabletop_native_teacher_profile.py",
            "tabletop_privileged_teacher.py","tabletop_deformable_privileged_teacher.py",
            "tabletop_squashed_gaussian.py","tabletop_flexible_measured_common_teacher.py",
            "tabletop_deformable_grasp_task.py","tabletop_action24_history_owner.py",
            "tabletop_mesh_observation_source.py","tabletop_deformx_parameter_snapshot.py",
            "tabletop_deformable_rod_boundary.py","tabletop_teacher_mounted_wrist_source.py",
            "tabletop_deformable_grasp_runtime.py","tabletop_deformable_r0_replay.py")},
        native_ready=False,checkpoint_adapter_ready=False,
        training_reward_qualified=False,context_owner_native_connected=False)
    return dict(identity,model_identity_sha256=digest(identity))


def verify_native_teacher_identity_v2(observed, expected):
    required = dict(policy_version=VERSION,model_schema=MODEL_SCHEMA_VERSION_V2,
        model_schema_sha256=MODEL_SCHEMA_SHA256_V2,input_schema=INPUT_SCHEMA_VERSION_V2,
        input_schema_sha256=INPUT_SCHEMA_SHA256_V2,context_schema=CONTEXT_SCHEMA_VERSION_V2,
        context_schema_sha256=CONTEXT_SCHEMA_SHA256_V2,phase_schema=PHASE_SCHEMA_VERSION_V2,
        phase_schema_sha256=PHASE_SCHEMA_SHA256_V2,phase_contract_sha256=PHASE_CONTRACT_SHA256_V2)
    if (not isinstance(expected,dict) or any(expected.get(k) != v for k,v in required.items())
            or expected.get("native_ready") is not False
            or expected.get("checkpoint_adapter_ready") is not False
            or expected.get("model_identity_sha256") != digest({
                k:v for k,v in expected.items() if k != "model_identity_sha256"})
            or observed != expected):
        raise ValueError("fixed S0 v2 Teacher model identity differs")


class NativeContextTeacherV2(DeformablePrivilegedTeacher):
    """Independent weights/identity, with the original three-part PPO protocol."""
    def __init__(self,config,*,backend_profile,context_mode="fixed_s0_v2"):
        identity=model_contract_v2(config,backend_profile,context_mode)
        # Initialize layers/distribution without constructing a historical
        # backend identity and then relabelling it as current native proof.
        PrivilegedTeacher.__init__(self,initial_std=config["model"]["initial_std"])
        self._model_identity=identity
        self._backend_profile=backend_profile
        self.backend_sha256=backend_profile.binary_sha256
        self.context_mode=context_mode
        def common_encoder():
            return nn.Sequential(nn.Linear(66,256),nn.ELU(),nn.Linear(256,256),nn.ELU())
        def context_encoder():
            return nn.Sequential(nn.Linear(102,64),nn.ELU(),nn.Linear(64,64),nn.ELU())
        self.actor,self.critic=common_encoder(),common_encoder()
        self.actor_rod,self.critic_rod=_MaterialEncoder(64),_MaterialEncoder(64)
        self.actor_context,self.critic_context=context_encoder(),context_encoder()
        self.actor_head,self.critic_head=nn.Linear(448,24),nn.Linear(448,1)

    def pack(self,*args,**kwargs):
        raise ValueError("new model requires pack_native_inputs_v2 or explicit pack_cpu_fixture_v2")

    def pack_cpu_fixture_v2(self,ledger,valid,rod_snapshot,common_identity,node_valid,task_context):
        if (type(task_context) is not TaskContextPacketV2
                or task_context.metadata.grasp != self._model_identity["grasp"]
                or task_context.metadata.prelift_scale_s != self._model_identity["prelift_scale_s"]):
            raise ValueError("task context v2 grasp/prelift scale differs from model")
        return pack_cpu_fixture_v2(ledger,valid,rod_snapshot,common_identity,node_valid,task_context,
            backend_profile=self._backend_profile,context_mode=self.context_mode)

    def pack_native_inputs_v2(self,pre_action,raw_rod,parameters,task_context,*,backend_profile,
                              source_adapter,measurement_snapshot,phase_receipt):
        if backend_profile != self._backend_profile:
            raise ValueError("model backend profile differs")
        return pack_native_inputs_v2(pre_action,raw_rod,parameters,task_context,
            backend_profile=backend_profile,source_adapter=source_adapter,
            measurement_snapshot=measurement_snapshot,phase_receipt=phase_receipt)

    def _parts(self,observations):
        if not isinstance(observations,dict) or set(observations) != set(INPUT_KEYS_V2):
            raise ValueError("new Teacher requires exact six input keys")
        common=_common(observations["common371"],observations["common_valid"])
        nodes=observations["rod_nodes"]
        if (not isinstance(nodes,torch.Tensor) or nodes.shape != (len(common),41,13)
                or nodes.dtype != common.dtype or nodes.device != common.device
                or not bool(torch.isfinite(nodes).all())):
            raise ValueError("all 41x13 packed material features must be finite")
        mask=_all_nodes(observations["rod_mask"],len(common),common.device)
        context=observations["task_context"]
        validate_context_tensors_v2(context,observations["task_context_valid"],mode=self.context_mode)
        if context.device != common.device or len(context) != len(common):
            raise ValueError("context/common batch/device differs")
        return common,nodes,mask

    def latent_mean(self,observations):
        common,nodes,mask=self._parts(observations)
        context=encode_context_v2(observations["task_context"],observations["task_context_valid"])
        return self.actor_head(torch.cat((self.actor(common),self.actor_rod(nodes,mask),
            self.actor_context(context)),dim=-1))

    def value(self,observations):
        common,nodes,mask=self._parts(observations)
        context=encode_context_v2(observations["task_context"],observations["task_context_valid"])
        return self.critic_head(torch.cat((self.critic(common),self.critic_rod(nodes,mask),
            self.critic_context(context)),dim=-1)).squeeze(-1)

    def get_extra_state(self):
        return self.model_identity

    def set_extra_state(self,state):
        verify_native_teacher_identity_v2(state,self._model_identity)

    def load_state_dict(self,state_dict,strict=True,assign=False):
        verify_native_teacher_identity_v2(state_dict.get("_extra_state"),self._model_identity)
        return super().load_state_dict(state_dict,strict=strict,assign=assign)


class _OwnedClock:
    def __init__(self,clock):
        self._clock=deepcopy(clock)
    def clock(self):
        return deepcopy(self._clock)


def _pack_raw_inputs_v2(pre_action,raw_rod,parameters,task_context,*,backend_profile):
    """Private CPU-fixture ABI verification, kept separate from production gating.

    It preserves current raw parameter, units, geometry, free-rod and clock
    checks. This is not a live-source adapter and accepts only fixture scope.
    """
    from . import tabletop_mesh_observation_source as source
    from .tabletop_deformable_rod_boundary import RodBoundaryState
    from .tabletop_deformable_r0_replay import _digest
    from .tabletop_deformx_parameter_snapshot import validate_snapshot
    from .tabletop_teacher_mounted_wrist_source import validate_mounted_observation, _clock
    profile=_profile(backend_profile)
    if (not isinstance(pre_action,dict) or not isinstance(pre_action.get("common"),dict)
            or type(task_context) is not TaskContextPacketV2
            or task_context.metadata.scope != "cpu_fixture"
            or not isinstance(raw_rod,dict) or not isinstance(parameters,dict)
            or raw_rod.get("source_scope") != "cpu_fixture"
            or parameters.get("source_scope") != "cpu_fixture"):
        raise ValueError("explicit v2 CPU raw/context fixture scopes required")
    profile.verify()
    common=pre_action["common"]
    validate_mounted_observation(common)
    clock=_clock(common["identity"])
    keys=("run_uuid","env_id","episode_uid","reset_id","rod_uid","physics_step","rod_tick",
          "timestamp_s","solver_time_s","clock_valid")
    if not isinstance(raw_rod,dict) or any(raw_rod.get(k) != clock[k] for k in keys):
        raise ValueError("pre-action and raw material node clock differ")
    if (not isinstance(parameters,dict) or parameters.get("version") != source.PARAMETER_VERSION
            or parameters.get("sources") != source.source_basis()
            or parameters.get("new_sdk_steps") != 0 or parameters.get("new_solver_steps") != 0
            or parameters.get("teacher_admission") is not False
            or parameters.get("task_physics_qualified") is not False):
        raise ValueError("current actual parameter source/profile required; historical snapshot rejected")
    identity=parameters.get("identity",{})
    if any(identity.get(k) != clock[k] for k in keys if k != "clock_valid"):
        raise ValueError("parameter and raw node clock differ")
    raw=parameters["raw_native_parameters"]
    verified=validate_snapshot(raw,expected_num_envs=1,expected_nodes=41,expected_time_s=clock["solver_time_s"])
    if (_digest(verified) != _digest(parameters["native_parameters"])
            or not np.array_equal(parameters["node_positions_m"],raw_rod["node_positions"])
            or not np.array_equal(raw["rest_lengths_m"],np.asarray(raw_rod["rest_lengths_m"])[None])):
        raise ValueError("raw/derived parameter geometry differs")
    # The source profile seals this unchanged physical config. Parameter
    # consistency alone must not admit a different gravity/material/free-root.
    config=json.loads((source.ROOT/source.CONFIG).read_text())["rod"]
    expected=dict(contact_radius_m=config["radius_m"],gravity_world_m_s2=config["gravity_world_m_s2"],
        contact_mu_static=config["mu_s"],contact_mu_kinetic=config["mu_k"],
        contact_v_slip_m_s=config["v_slip_m_s"],contact_kappa=config["kappa"],
        stretch_ea_n=np.full((1,40),config["axial_stiffness_n"]),
        bend_ei_n_m2=np.full((1,40),config["bend_stiffness_n_m2"]))
    if verified["fixed_node_indices"] or any(not np.allclose(raw[k],value,rtol=2e-6,atol=1e-9)
            for k,value in expected.items()):
        raise ValueError("actual parameters differ from locked physical profile/free rod")
    rest=np.asarray(raw["rest_lengths_m"])[0]
    expected_masses=np.r_[rest[:1]/2,(rest[:-1]+rest[1:])/2,rest[-1:]/2]*config["linear_density_kg_m"]
    from .tabletop_deformable_grasp_runtime import verify_mass_readback
    verify_mass_readback(np.asarray(raw["inverse_masses_kg_inv"]),expected_masses)
    radius=dict(value=float(raw["contact_radius_m"]),source=dict(
        api="RopeBatch.parameter_snapshot.contact_radius_m",provenance="native_host_metadata",
        backend_sha256=profile.binary_sha256,identity=identity,uncertainty=None,
        hardware_measured=False,task_physics_qualified=False))
    if parameters["reward_radius_sample"] != radius:
        raise ValueError("actual parameter radius provenance differs")
    rebuilt=RodBoundaryState(_OwnedClock({k:raw_rod[k] for k in keys}),
        profile.binary_path,profile.binary_sha256).capture(raw_rod)
    if _digest(pre_action.get("rod_boundary")) != _digest(rebuilt):
        raise ValueError("pre-action rod boundary differs from raw current-backend reconstruction")
    validate_context_v2(task_context,backend_profile=profile,identities=(ContextIdentity.from_clock(clock),))
    ledger=torch.as_tensor(np.array(common["values"],copy=True),dtype=torch.float32)[None]
    valid=torch.as_tensor(np.array(common["valid"],copy=True),dtype=torch.bool)[None]
    snapshot=deepcopy(rebuilt["rod_snapshot"])
    for key,value in snapshot.items():
        if isinstance(value,(np.ndarray,torch.Tensor)):
            tensor=torch.as_tensor(value).clone()
            snapshot[key]=tensor.float() if tensor.is_floating_point() else tensor
    common_identity={key:deepcopy(snapshot[key]) for key in (
        "env_ids","episode_uids","reset_ids","rod_uids","physics_step","rod_tick","timestamp_s")}
    mask=torch.ones((1,41),dtype=torch.bool)
    return _pack_v2(ledger,valid,snapshot,common_identity,mask,task_context,profile,task_context.metadata.mode)


def pack_cpu_fixture_v2(ledger,valid,rod_snapshot,common_identity,node_valid,task_context,*,
                        backend_profile,context_mode="fixed_s0_v2"):
    """Pack owned six tensors with exact v2 fixture metadata; never issue a witness."""
    profile = _profile(backend_profile)
    if (not isinstance(rod_snapshot,dict) or rod_snapshot.get("source_scope") != "cpu_fixture"
            or type(task_context) is not TaskContextPacketV2
            or task_context.metadata.scope != "cpu_fixture"):
        raise ValueError("explicit synthetic v2 CPU fixture required; native source cannot be retagged")
    return _pack_v2(ledger,valid,rod_snapshot,common_identity,node_valid,task_context,
        profile,context_mode)


def pack_native_inputs_v2(pre_action,raw_rod,parameters,task_context,*,backend_profile,
                          source_adapter,measurement_snapshot,phase_receipt):
    """Production entry remains closed until the real W4 source contract exists.

    No registered SourceAdapterV2, owned SnapshotV2, PhaseSourceWitnessV2 or
    validate_phase_source_witness_v2 exists in this version. ContextMetadataV2
    has no issued-witness field yet. Consequently no caller object, bool,
    fixture receipt, duck-typed validator or source_scope can grant admission.
    W4/V2-I01 must add the actual validator before invoking raw packing here.
    """
    _profile(backend_profile)
    if source_adapter is None:
        raise ValueError("full_v2_phase_source_adapter_missing")
    # Any supplied caller object still lacks an actual same-owner signed
    # witness. Do not call its methods or manufacture a replacement witness.
    raise ValueError("full_v2_phase_source_witness_missing")
