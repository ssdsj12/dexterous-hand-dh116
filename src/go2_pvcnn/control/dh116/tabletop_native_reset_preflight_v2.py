"""Source-loaded S0 identity and current v2 Reset evidence are separate checks.

This module never supplies a native worker or upgrades CPU fixture evidence.
The original manifest loader validates only its original historical S0 sources.
"""
from .tabletop_native_teacher_preflight import (
    ResolvedResetManifest, validate_live_mounted)
from .tabletop_native_teacher_profile import BackendProfile, digest
from .tabletop_native_training_clock import read_clock_v2
from .tabletop_mesh_reset_readback_v2 import (
    validate_reset_readback_v2,capture_mesh_state_v2,validate_capture_v2)
from .tabletop_mesh_reset_numeric_state_v2 import (
    validate_numeric_reset_audit_v2,capture_numeric_state_v2,validate_numeric_sample_v2)
from .tabletop_deformable_r0_replay import decode_owned,encode_owned
from .tabletop_dual_hand_reset import _hands
import numpy as np


def _owner(owner):
    from .tabletop_native_episode_v2 import FixedS0PhysicalEpisodeV2
    episode=getattr(owner,"episode",None)
    if (type(episode) is not FixedS0PhysicalEpisodeV2
            or getattr(owner,"sim",None) is not episode._sim
            or getattr(owner,"rod",None) is not episode._rod
            or tuple(getattr(owner,"bindings",())) != episode._bindings):
        raise ValueError("same borrowed explicit v2 episode owner required")
    return episode


class ResetPreflightV2:
    @staticmethod
    def require_reset_admitted(owner, *, clock_policy, manifest):
        if type(manifest) is not ResolvedResetManifest:
            raise ValueError("actual source-loaded ResolvedResetManifest required")
        # Includes the original loader's 52MB source authentication; no cache.
        manifest.verify()
        episode=_owner(owner)
        if episode.clock_policy != clock_policy:
            raise ValueError("same explicit v2 clock policy required")
        from .tabletop_native_episode_v2 import _verify_factories
        inventory=_verify_factories(episode._reset_factory,episode._exchange_factory)
        if inventory!=episode.source_inventory:
            raise ValueError("owner v2 factory source identity differs")
        clock=read_clock_v2(owner.sim,owner.rod,policy=clock_policy)
        return dict(schema="dh116-native-reset-preflight-v2",
            clock_policy_sha256=clock_policy.sha256,actual_reset_sdk_step=clock["sdk_physics_step"],
            actual_sdk_timestamp_s=clock["sdk_timestamp_s"],
            reset_epoch_before=clock["rod"]["reset_id"],
            reset_epoch_next=clock["rod"]["reset_id"]+1,
            max_reset_sdk_step=clock_policy.max_sdk_step,
            resolved_reset_manifest_sha256=manifest.sha256,source_inventory=inventory,
            reset_clock_admitted=True,native_ready=False,native_qualification=False)


def validate_live_reset_v2(*,owner,receipt,numeric_audit,mesh_plan,manifest,clock_policy):
    """Re-read current mounted/rod/backend state and independent v2 evidence."""
    admission=ResetPreflightV2.require_reset_admitted(owner,
        clock_policy=clock_policy,manifest=manifest)
    episode=_owner(owner)
    profile=getattr(owner,"backend_profile",None)
    if type(profile) is not BackendProfile:
        raise ValueError("actual immutable BackendProfile required for live Reset")
    profile.verify()
    from . import tabletop_mesh_observation_source as source
    source.verify_runtime(owner.rod)
    data=manifest.data()
    receipt=decode_owned(receipt)
    audit=decode_owned(numeric_audit)
    clock=read_clock_v2(owner.sim,owner.rod,policy=clock_policy)
    if (receipt["after_actual_clocks"]!=clock or receipt["after_clock"]!=clock["rod"]
            or receipt["reset_epoch"]!=clock["rod"]["reset_id"]
            or receipt["seed"]!=data["reset_seed"] or episode.mesh_plan is not mesh_plan):
        raise ValueError("live SDK/solver/reset UID/seed/mesh differs from immutable S0")
    numeric=validate_numeric_reset_audit_v2(audit,reset_receipt=receipt,clock_policy=clock_policy)
    if numeric["source_instance_id"]!=id(owner.rod.rope):
        raise ValueError("current live numeric owner differs")
    if digest(encode_owned(_hands(owner.bindings,owner.rod,finite=True)))!=digest(encode_owned(receipt["hands"])):
        raise ValueError("current SDK q/qd/targets/body poses differ from actual Reset")
    sample=capture_numeric_state_v2(owner.rod.rope,owner.sim,owner.rod,clock_policy=clock_policy)
    current=validate_numeric_sample_v2(sample,clock_policy=clock_policy)
    for name in ("positions","velocities","quaternions"):
        if (not np.array_equal(current[name],audit["samples"][1]["state"][name])
                or not np.array_equal(current[name],np.asarray(data["rod_state"][name],np.float32))):
            raise ValueError("current rod position/velocity/director differs from resolved S0")
    if digest(encode_owned(receipt["rod_reset"]["mesh_bindings"]))!=data["mesh_bindings_sha256"]:
        raise ValueError("current mesh bindings differ from resolved S0")
    mesh=validate_reset_readback_v2(receipt["mesh_state_readback"],
        reset_receipt=receipt,mesh_plan=mesh_plan,clock_policy=clock_policy)
    fresh=capture_mesh_state_v2(owner.sim,owner.rod,clock_policy=clock_policy)
    validate_capture_v2(fresh,clock_policy=clock_policy)
    saved=receipt["mesh_state_readback"]["after"]
    if (fresh["source_instance_id"]!=saved["source_instance_id"]
            or fresh["source_instance_id"]!=id(owner.rod.rope)
            or fresh["clock_before"]!=saved["clock_after"]
            or digest(encode_owned(fresh["snapshot"]))!=digest(encode_owned(saved["snapshot"]))):
        raise ValueError("current mesh owner/clock/current/target differs from actual Reset")
    observation=episode.pre_action_state()["common"]
    common=validate_live_mounted(owner.bindings,observation,read_identity=episode._read_state_clock)
    if common["valid_channels"]!=66 or common["unknown_channels"]!=305:
        raise ValueError("actual mounted common66 ledger differs")
    if read_clock_v2(owner.sim,owner.rod,policy=clock_policy)!=clock:
        raise ValueError("clock advanced during live v2 Reset verification")
    return dict(schema="dh116-live-reset-v2",live_reset_verified=True,
        reset_identity=clock["rod"],numeric=numeric,mesh=mesh,
        backend_profile_digest=profile.digest,source_inventory=admission["source_inventory"],
        clock_policy_sha256=clock_policy.sha256,resolved_reset_manifest_sha256=manifest.sha256,
        native_ready=False,native_qualification=False,teacher_admission=False,new_native_steps=0)
