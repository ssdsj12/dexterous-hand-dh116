"""Exact indexed sensor storage and the existing on-policy PPO equations."""
import copy
import importlib
import importlib.util

import pytest
import torch
from torch import nn

from test_dh116_stereo_estimator import history, sensor, module


def storage(**kwargs):
    target='go2_pvcnn.control.dh116.tabletop_stereo_rollout_storage'
    assert importlib.util.find_spec(target), 'indexed sensor-frame storage is required; full history clones exhaust VGA rollout memory'
    return importlib.import_module(target).IndexedStereoRolloutStorage(**kwargs)


def byte_equal(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def quantized_history(batch=2):
    h=history(batch)
    h.left_rgb.copy_(torch.randint(0,256,h.left_rgb.shape).float()/255)
    h.right_rgb.copy_(torch.randint(0,256,h.right_rgb.shape).float()/255)
    return h


def test_student_default_pending_observations_are_compact_references():
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_student_trainer,history_to_tree
    trainer=make_student_trainer({'network':{'embed_dim':32,'num_layers':1},'ppo':{'epochs':1,'minibatches':1}},1,rollout_steps=1)
    trainer.collect_step(history_to_tree(history(1)),torch.zeros(1,371))
    assert isinstance(trainer._pending['actor_obs'],torch.Tensor), 'pending actor input must be compact frame references, not cloned histories'
    assert trainer._pending['actor_obs'].dtype==torch.int64 and trainer._pending['actor_obs'].numel()==1


def test_exact_history_roundtrip_owned_independent_and_uint8_only_when_lossless():
    store=storage(max_visual_batch=1)
    h=quantized_history();original={k:v.clone() for k,v in h.as_dict().items()}
    refs=store.capture(h)
    assert refs.shape==(2,) and refs.dtype==torch.int64 and refs.device.type=='cpu'
    restored=store.resolve(refs,'cpu')
    for name,value in original.items(): assert byte_equal(value,restored[name]),name
    assert store.stats()['rgb_uint8_count']>0 and store.stats()['rgb_float32_count']==0
    h.left_rgb.zero_();restored['proprio'].fill_(9.)
    again=store.resolve(refs,'cpu')
    assert byte_equal(again['left_rgb'],original['left_rgb'])
    assert byte_equal(again['proprio'],original['proprio'])
    exact=history(1);exact.left_rgb[:]=.12345679;exact.right_rgb[:]=.7654321
    refs2=store.capture(exact)
    assert byte_equal(store.resolve(refs2,'cpu')['left_rgb'],exact.left_rgb)
    assert store.stats()['rgb_float32_count']>0


def test_overlapping_histories_deduplicate_images_independently_of_proprio_depth():
    observation=module('observation');buffer=observation.StereoHistory(1,history=4)
    store=storage()
    frame=sensor(1); frame.left_rgb[:]=.2;frame.right_rgb[:]=.8
    first=buffer.update(frame);refs1=store.capture(first)
    first_stats=store.stats()
    frame.proprio[:,0]=.2;frame.depth_m[:]=2.
    second=buffer.update(frame);refs2=store.capture(second)
    stats=store.stats()
    assert stats['image_count']==first_stats['image_count']
    assert stats['frame_count']==first_stats['frame_count']+1
    assert stats['sample_count']==2
    assert byte_equal(store.resolve(refs1,'cpu')['depth_m'],first.depth_m)
    assert byte_equal(store.resolve(refs2,'cpu')['proprio'],second.proprio)
    before=stats['storage_bytes'];store.capture(second)
    assert store.stats()['image_count']==stats['image_count']
    assert store.stats()['storage_bytes']-before < 128


def test_depth_na_masks_and_palm_local_fields_keep_exact_bytes():
    store=storage();h=history(1)
    h.depth_valid[0,-1,0,0]=False;h.depth_m[0,-1,0,0]=float('nan')
    h.depth_confidence[0,-1,0,0]=float('nan')
    h.point_valid[0,-1,0]=False;h.points_world[0,-1,0]=float('nan')
    h.point_valid_left[0,-1,0]=False;h.points_left_palm[0,-1,0]=float('nan')
    refs=store.capture(h);result=store.resolve(refs,'cpu')
    for name,value in h.as_dict().items(): assert byte_equal(value,result[name]),name
    store.reset()
    assert store.stats()['storage_bytes']==0
    with pytest.raises(ValueError,match='stale|reference|generation'):store.resolve(refs,'cpu')


def test_noncontiguous_input_never_stages_a_whole_history_for_pool_capture(monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_stereo_observation import StereoHistoryBatch

    original = quantized_history(1).as_dict()
    expanded = {name: tensor.expand(2, *tensor.shape[1:]) for name, tensor in original.items()}
    history_batch = StereoHistoryBatch.from_mapping(expanded)
    whole_history_copies = []
    contiguous = torch.Tensor.contiguous

    def observe_contiguous(value, *args, **kwargs):
        if value.ndim == 5 and not value.is_contiguous():
            whole_history_copies.append(tuple(value.shape))
        return contiguous(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'contiguous', observe_contiguous)
    pool = storage()
    restored = pool.resolve(pool.capture(history_batch))
    assert whole_history_copies == [], 'staging must transfer individual frames rather than contiguous histories'
    for name, value in expanded.items():
        torch.testing.assert_close(restored[name], value, atol=0, rtol=0, equal_nan=True)


def test_signed_zero_rgb_and_foreign_pool_reference_are_not_silently_reinterpreted():
    store=storage();h=history(1)
    h.left_rgb.zero_();h.left_rgb[0,-1,0,0,0]=-0.
    refs=store.capture(h)
    assert byte_equal(store.resolve(refs,'cpu')['left_rgb'],h.left_rgb)
    other=storage();other.capture(h)
    with pytest.raises(ValueError,match='stale|foreign|reference|generation'):other.resolve(refs,'cpu')


def test_strict_sensor_only_storage_no_latent_critic_or_unknown_mapping():
    store=storage();h=history(1)
    for key in ('critic_obs','privileged_state','latent','teacher_action'):
        with pytest.raises((ValueError,TypeError),match='sensor|privileged|unknown'):
            store.capture({**h.as_dict(),key:torch.zeros(1,371)})


class PixelPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Linear(3,24)
        self.critic=nn.Linear(371,1)
        self.log_std=nn.Parameter(torch.full((24,),-2.3))

    def latent_mean(self,tree):
        return self.encoder(tree['left_rgb'][:,-1].mean((-1,-2)))

    def value(self,critic):return self.critic(critic).squeeze(-1)


def test_existing_ppo_math_matches_legacy_with_bounded_visual_reconstruction():
    from go2_pvcnn.control.dh116.tabletop_privileged_ppo import PPOConfig,PPOTrainer
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import history_to_tree
    torch.manual_seed(91);model=PixelPolicy()
    config=PPOConfig(epochs=2,minibatches=1)
    legacy=PPOTrainer(copy.deepcopy(model),config,rollout_steps=2,num_envs=3)
    indexed=PPOTrainer(copy.deepcopy(model),config,rollout_steps=2,num_envs=3,actor_storage=storage(max_visual_batch=2))
    observed=[]
    hook=indexed.model.encoder.register_forward_pre_hook(lambda layer,args:observed.append(args[0].shape[0]))
    for step in range(2):
        h=quantized_history(3);critic=torch.randn(3,371)
        expected=None
        for trainer in (legacy,indexed):
            torch.manual_seed(100+step)
            sampled=trainer.collect_step(history_to_tree(h),critic)
            if expected is not None:
                torch.testing.assert_close(sampled['latent_mean'],expected['latent_mean'],atol=2e-6,rtol=2e-6)
                torch.testing.assert_close(sampled['log_prob'],expected['log_prob'],atol=2e-5,rtol=2e-5)
            expected=sampled
            trainer.record_step(torch.tensor([.2,.4,.8]),torch.zeros(3,dtype=torch.bool),torch.zeros(3),
                executed_actions=sampled['actions'],collection_token=sampled['collection_token'])
    batch=indexed.rollout.batch(resolve_actor=False)
    assert batch['actor_obs'].shape==(6,) and batch['actor_obs'].dtype==torch.int64
    torch.manual_seed(777);a=legacy.update()
    torch.manual_seed(777);b=indexed.update()
    hook.remove()
    for name,p in legacy.model.state_dict().items():
        torch.testing.assert_close(p,indexed.model.state_dict()[name],atol=2e-6,rtol=2e-6)
    for name in ('policy_loss','value_loss','entropy','kl'):
        assert a[name]==pytest.approx(b[name],abs=2e-5,rel=2e-5)
    assert a['minibatch_updates']==b['minibatch_updates']==2
    assert max(observed)<=2
    assert indexed.model.encoder.weight.grad.abs().sum()>0
    assert indexed.actor_storage.stats()['storage_bytes']==0


def test_indexed_on_policy_ownership_tokens_and_full_encoder_gradient():
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_student_trainer
    trainer=make_student_trainer({'network':{'embed_dim':32,'num_layers':1},'max_visual_batch':1,
        'ppo':{'epochs':1,'minibatches':1}},2,rollout_steps=1)
    h=history(2);sample=trainer.collect_step(h,torch.zeros(2,371))
    with pytest.raises(ValueError,match='token'):
        trainer.record_step(torch.ones(2),torch.zeros(2,dtype=torch.bool),torch.zeros(2),collection_token='foreign')
    with pytest.raises(ValueError,match='owner'):
        trainer.record_step(torch.ones(2),torch.zeros(2,dtype=torch.bool),torch.zeros(2),action_owner='teacher')
    trainer.record_step(torch.tensor([.1,1.]),torch.zeros(2,dtype=torch.bool),torch.zeros(2),
        executed_actions=sample['actions'],collection_token=sample['collection_token'])
    batch=trainer.rollout.batch(resolve_actor=True)
    assert set(batch['actor_obs'])==set(h.as_dict())
    assert 'privileged_state' not in batch['actor_obs']
    h.left_rgb.zero_();batch['actions'].zero_()
    trainer.update()
    parameter=next(trainer.model.actor.estimator.encoder.backbone.parameters())
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0


def test_explicit_legacy_trainer_stays_available_without_actor_storage():
    from go2_pvcnn.control.dh116.tabletop_stereo_student_ppo import make_student_trainer,history_to_tree
    trainer=make_student_trainer({'network':{'embed_dim':32,'num_layers':1},'rollout_storage':'legacy'},1,rollout_steps=1)
    trainer.collect_step(history_to_tree(history(1)),torch.zeros(1,371))
    assert trainer.actor_storage is None and isinstance(trainer._pending['actor_obs'],dict)
