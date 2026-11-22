"""Compose the original camera-free PPO components for a granted C04 session.

The provider supplies the episode builder and source-pack observer. This module
does not create Isaac, issue a worker lease, or grant native qualification.
"""

import json
from weakref import ref

from .tabletop_deformable_privileged_ppo import DeformablePPOTrainer
from .tabletop_native_teacher_actual_process import ActualWorkerLease
from .tabletop_native_teacher_context import TaskContextOwner,CONTACT_PRELIFT
from .tabletop_native_teacher_contact_context import PrimedContactContextOwner
from .tabletop_native_teacher_model import NativeContextTeacher
from .tabletop_native_teacher_session import ModelContract, NativeEpisodeOwner, NativeFactories, StaticContract
from .tabletop_native_teacher_training import VERSION
from .tabletop_training_reward import TrainingReward


# Fixed diagnostic seeds are bound by the A-fresh source closure, independent
# of the per-attempt nonce and of any RNG draws during scene construction.
FRESH_POLICY_INITIALIZATION_SEED = 2026112001
FRESH_POLICY_EXPLORATION_SEED = 2026112002


def build_actual_factories(*, lease, episode_factory, prepare_source_pack, resume_contract=None):
    """Return factories in the order consumed by create_native_collector.

    prepare_source_pack verifies the live source capability and returns the
    provider's binder for the real model. Both actions precede PPO creation.
    """
    if type(lease) is not ActualWorkerLease or not callable(episode_factory) or not callable(prepare_source_pack):
        raise TypeError("actual worker lease, episode factory and source-pack binding required")
    origin = lease.resume_origin
    if origin is None and resume_contract is not None or origin is not None and not callable(resume_contract):
        raise ValueError("checkpoint restore requires both a leased A origin and a live training contract")
    owner_ref = policy_ref = None

    def episode(static):
        nonlocal owner_ref, policy_ref
        lease.revalidate()
        if type(static) is not StaticContract:
            raise ValueError("verified static Teacher contract required")
        config = static.config
        expected = dict(schema=VERSION, preflight=config.preflight, model_config=config.model_config,
                        episode_cap_s=config.episode_cap_s, rollout_steps=config.rollout_steps)
        if (json.dumps(expected, sort_keys=True, allow_nan=False) !=
                json.dumps(lease.process_config, sort_keys=True, allow_nan=False)):
            raise ValueError("C04 Teacher contract differs from the issued process config")
        if (config.preflight["style"] not in ("pinch","tripod")
                or (config.preflight["profile"] != "approach_wiring_smoke_v1"
                    and (config.preflight["profile"] != CONTACT_PRELIFT or config.preflight["style"] != "tripod"))
                or type(config.rollout_steps) is not int or config.rollout_steps not in (8, 32)):
            raise ValueError("approved independent grasp approach smoke and bounded horizon required")
        static.runtime.bind_parent_deadline(lease.deadline_monotonic_s)
        owner = episode_factory(static)
        if (type(owner) is not NativeEpisodeOwner or owner.contract is not static
                or owner._closed or owner.episode._state != "new"):
            raise ValueError("fresh original NativeEpisodeOwner for this static contract required")
        owner_ref, policy_ref = ref(owner), None
        return owner

    def live_owner():
        lease.revalidate()
        owner = owner_ref() if owner_ref is not None else None
        context = getattr(owner, "context", None)
        mode=owner.contract.config.preflight["profile"] if owner is not None else None
        expected_context=PrimedContactContextOwner if mode == CONTACT_PRELIFT else TaskContextOwner
        if (owner is None or owner._closed or owner.episode._state != "active"
                or type(context) is not expected_context or context._current is None or context._closed
                or mode == CONTACT_PRELIFT and context._contact_failed
                or context.profile is not owner.backend_profile
                or context.scope != owner.sources.scope or context.scope != "owner_source_unqualified"
                or context.mode != owner.contract.config.preflight["profile"]
                or context.grasp != owner.contract.config.preflight["style"]):
            raise ValueError("same live episode and verified Reset context required")
        if mode == CONTACT_PRELIFT:
            try:
                measured=context.contact_readback
                if (measured["physical_observations"] != 1
                        or context.current.metadata.identities[0] != context._contact_identity):
                    raise ValueError("fresh primed physical contact required")
            except (ValueError,RuntimeError,AttributeError) as error:
                raise ValueError("same live episode and verified contact context required") from error
        return owner

    def policy(contract):
        nonlocal policy_ref
        owner = live_owner()
        if (type(contract) is not ModelContract or contract.config is not owner.contract.config.model_config
                or contract.backend_profile is not owner.backend_profile
                or contract.context_mode != owner.context.mode):
            raise ValueError("C04 live model contract differs")
        bind_source_pack = prepare_source_pack(owner, contract)
        if not callable(bind_source_pack):
            raise ValueError("verified source-pack binder required before constructing the policy")
        import torch

        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.default_generator.manual_seed(FRESH_POLICY_INITIALIZATION_SEED)
            model = NativeContextTeacher(contract.config, backend_profile=contract.backend_profile,
                                         context_mode=contract.context_mode)
        bind_source_pack(model)
        policy_ref = ref(model)
        return model

    def trainer(model, ppo):
        owner = live_owner()
        if policy_ref is None or model is not policy_ref():
            raise ValueError("same policy constructed for this live episode required")
        if ppo is not owner.contract.ppo_config:
            raise ValueError("same approved PPO configuration required")
        def new_trainer():
            return DeformablePPOTrainer(model, ppo, rollout_steps=owner.contract.config.rollout_steps,
                                        num_envs=1)

        if origin is None:
            return new_trainer()
        from .tabletop_native_teacher_checkpoint import checkpoint_trainer_factory

        restore = checkpoint_trainer_factory(new_trainer, origin["checkpoint_path"],
            lambda trainer: resume_contract(trainer, owner), expected_sha256=origin["checkpoint_sha256"])
        return restore()

    def reward(contract, *, contact_owner=None):
        owner = live_owner()
        if contract is not owner.contract.reward_config:
            raise ValueError("same file-backed approved reward configuration required")
        if contract.profile == CONTACT_PRELIFT:
            if contact_owner is not owner.context:
                raise ValueError("same actual primed contact context required")
            return TrainingReward(contract,contact_owner=owner.context)
        if contact_owner is not None:
            raise ValueError("contact context is excluded from old smoke reward")
        return TrainingReward(contract)

    return NativeFactories(episode, policy, trainer, reward)
