"""CPU-only V4 fixtures. No fixture result counts as a native success."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from test_bimanual_checkpoint_contract import setup as binding_setup, derive, sha, rehash
from test_vision_action_derivation import source_checkpoint, observations
from test_vision_evaluation_trials import manifest
from go2_pvcnn.control.dh116.linear_coupling import ACTIVE, JOINTS
from go2_pvcnn.control.dh116 import tabletop_bimanual_checkpoint_contract as binding
from go2_pvcnn.control.dh116 import tabletop_vision_inference as inference
from go2_pvcnn.control.dh116.tabletop_vision_action_derivation import derive_action_checkpoint
from go2_pvcnn.control.dh116.tabletop_vision_evaluation_contract import (
    evaluation_contract, evaluation_physics_sha256, validate_vision_evaluation_options,
)


@pytest.fixture
def qualified(tmp_path):
    context = binding_setup.__wrapped__(tmp_path)
    source, saved = source_checkpoint(tmp_path)
    saved['contract'].update(camera_mode='fixed_per_env', camera_batch_size=2,
                             camera_resolution_wh=[224, 96],
                             camera_batching='fixed_per_environment_real_RGBD')
    torch.save(saved, source)
    v3 = tmp_path / 'real_actor_v3.pt'
    derive_action_checkpoint(source, v3)
    context.source = v3
    context.before = torch.load(v3, map_location='cpu', weights_only=False)
    derive(context)
    context.saved = torch.load(context.output, map_location='cpu', weights_only=False)
    context.current = evaluation_contract(context.before['contract'], 2)
    return context


def test_v4_requires_explicit_load_and_actual_runtime_verification(qualified):
    with pytest.raises(ValueError, match='explicit opt-in'):
        inference.load_vision_inference(qualified.output, qualified.current)
    policy = inference.load_vision_inference(qualified.output, qualified.current, allow_physical_adapter=True)
    assert policy.requires_physical_adapter_verification
    assert policy.source_version == binding.VERSION
    assert policy.provenance['source_checkpoint_version'] == binding.VERSION
    assert policy.provenance['physical_adapter_native_qualification'] == qualified.saved['contract'][binding.QUALIFICATION_KEY]
    assert not policy.provenance['physical_action_adapter_runtime_verified']
    with pytest.raises(RuntimeError, match='verification before act'):
        policy.act(observations())
    mismatch = copy.deepcopy(qualified.adapter)
    mismatch['source_asset_sha256']['right'] = '0' * 64
    rehash(mismatch)
    with pytest.raises(ValueError, match='actual runtime'):
        policy.verify_physical_adapter(mismatch)
    with pytest.raises(RuntimeError, match='verification before act'):
        policy.act(observations())


def test_verified_v4_is_bitexact_v3_actor_without_critic_adam_rng_or_source_changes(qualified, monkeypatch):
    obs = observations()
    original_bytes = qualified.output.read_bytes()
    import go2_pvcnn.control.dh116.tabletop_vision_student as student
    monkeypatch.setattr(student, 'make_vision_algorithm', lambda *a, **kw: pytest.fail('PPO constructed'))
    monkeypatch.setattr(torch.optim, 'Adam', lambda *a, **kw: pytest.fail('Adam constructed'))
    rng = torch.get_rng_state().clone()
    original = inference.load_vision_inference(qualified.source, qualified.current)
    policy = inference.load_vision_inference(qualified.output, qualified.current, allow_physical_adapter=True)
    policy.verify_physical_adapter(qualified.adapter)
    assert torch.equal(policy.act(obs), original.act(obs))
    bad_critic = obs.clone()
    bad_critic['critic'].fill_(float('nan'))
    assert torch.equal(policy.act(obs), policy.act(bad_critic))
    assert torch.equal(torch.get_rng_state(), rng)
    assert qualified.output.read_bytes() == original_bytes
    assert not hasattr(policy, 'optimizer') and not hasattr(policy, 'storage')
    assert not hasattr(policy, 'critic') and not hasattr(policy, 'saved')
    assert not any('critic' in name for name, _ in policy.actor.named_parameters())
    assert policy.provenance['physical_action_adapter_runtime_verified']
    # A failed subsequent verification cannot leave stale permission to act.
    mismatch = copy.deepcopy(qualified.adapter)
    mismatch['adapter_source_sha256'] = '0' * 64
    rehash(mismatch)
    with pytest.raises(ValueError): policy.verify_physical_adapter(mismatch)
    with pytest.raises(RuntimeError): policy.act(obs)


def test_allowing_v4_does_not_auto_enable_adapter_for_an_original_v3(qualified):
    policy = inference.load_vision_inference(qualified.source, qualified.current, allow_physical_adapter=True)
    assert not policy.requires_physical_adapter_verification
    assert policy.act(observations()).shape == (2, 24)
    with pytest.raises(ValueError, match='no physical adapter binding'):
        policy.verify_physical_adapter(qualified.adapter)


@pytest.mark.parametrize('version', [binding.SQUASHED_VERSION, 'dh116-vision-student-ppo-v2-bounded'])
def test_relabelled_physical_metadata_cannot_bypass_v4_runtime_gate(qualified, version):
    saved = copy.deepcopy(qualified.saved)
    saved['version'] = saved['contract']['version'] = version
    bad = qualified.output.parent/'relabelled.pt'
    torch.save(saved, bad)
    with pytest.raises(ValueError, match='independent V4 version'):
        inference.load_vision_inference(bad, qualified.current)
    args = options(qualified)
    args.bimanual_checkpoint_adapter = False
    args.vision_evaluation_checkpoint = bad
    with pytest.raises(ValueError, match='independent V4 version'):
        validate_vision_evaluation_options(args)


def runtime_stub(qualified):
    sources = json.loads(qualified.mapping_path.read_text())['source_assets']
    hands = [SimpleNamespace(joint_names=list(JOINTS), body_names=['base_link']) for _ in range(2)]
    native = SimpleNamespace(
        source={'hands': [SimpleNamespace(cfg=SimpleNamespace(spawn=SimpleNamespace(usd_path=sources[side])))
                          for side in ('left', 'right')]},
        ids=[[list(JOINTS).index(name) for name in ACTIVE] for _ in range(2)])
    return SimpleNamespace(w=native, hands=hands)


def test_native_verifier_uses_actual_spawn_assets_and_sdk_names_without_substep_hook(qualified, monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_vision_evaluation import _verify_native_physical_adapter
    from go2_pvcnn.control.dh116 import tabletop_bimanual_physical_action_adapter as physical
    policy = inference.load_vision_inference(qualified.output, qualified.current, allow_physical_adapter=True)
    world = runtime_stub(qualified)
    opened = []
    def source_model(cls, path, side):
        opened.append((side, str(path)))
        return SimpleNamespace(side=side, source_sha256=sha(path))
    monkeypatch.setattr(physical.HandPhysicalKinematics, 'from_usd', classmethod(source_model))
    _verify_native_physical_adapter(policy, world)
    assert policy.provenance['physical_action_adapter_runtime_verified']
    assert [side for side, _ in opened] == ['left', 'right']
    assert all(Path(path).name.endswith('.fixture.usd') for _, path in opened)
    runtime = policy.provenance['physical_action_adapter_runtime']
    assert runtime['execution'] == 'original_SDK24_identity_mapping'
    assert runtime['additional_per_substep_hooks'] == 0
    assert not hasattr(world, 'bimanual_action_adapter')
    assert policy.act(observations()).shape == (2, 24)


def test_native_sdk_joint_order_mismatch_rejects_before_actor(qualified):
    from go2_pvcnn.control.dh116.tabletop_vision_evaluation import _verify_native_physical_adapter
    policy = inference.load_vision_inference(qualified.output, qualified.current, allow_physical_adapter=True)
    world = runtime_stub(qualified)
    world.w.ids[0] = list(reversed(world.w.ids[0]))
    with pytest.raises(ValueError, match='SDK active joint order'):
        _verify_native_physical_adapter(policy, world)
    with pytest.raises(RuntimeError): policy.act(observations())


def options(qualified):
    frozen = qualified.output.parent / 'manifest.json'
    value = manifest(100, 'validation')
    source = qualified.before['contract']
    value['metadata']['camera_sha256'] = source['camera_config_sha256']
    value['metadata']['physics_sha256'] = evaluation_physics_sha256(source['privileged_critic_contract'])
    frozen.write_text(json.dumps(value))
    return SimpleNamespace(vision_evaluation_checkpoint=qualified.output,
        vision_evaluation_manifest=frozen, vision_evaluation_max_control_steps=600,
        output=qualified.output.parent/'new_eval', vision_student=True, tensor_pipeline=True,
        headless=True, num_envs=200, vision_camera_mode='fixed', vision_width=224, vision_height=96,
        vision_visibility_mode='all', vision_action_distribution='tanh_gaussian', resume=None,
        rollout_steps=2, bimanual_checkpoint_adapter=True)


def test_prekit_v4_structural_preflight_remains_readonly_and_explicit(qualified):
    args = options(qualified)
    before = qualified.output.read_bytes()
    validate_vision_evaluation_options(args)
    assert not args.output.exists() and qualified.output.read_bytes() == before
    args.bimanual_checkpoint_adapter = False
    with pytest.raises(ValueError, match='explicit opt-in'):
        validate_vision_evaluation_options(args)


@pytest.mark.parametrize('flag,value', [('paired_bias_arm', 'm0'), ('vision_m2_replay', 'source'),
    ('vision_teacher_source', 'source'), ('bimanual_explicit_sdk_adapter', True),
    ('vision_mirror_preprocessing', 'm2_paired_canonical_v1'), ('vision_evaluation_checkpoint', None)])
def test_v4_cli_mode_excludes_paired_replay_teacher_and_training(qualified, flag, value):
    args = options(qualified)
    setattr(args, flag, value)
    with pytest.raises(ValueError, match='plain M0 evaluation'):
        validate_vision_evaluation_options(args)
    assert not args.output.exists()


@pytest.mark.parametrize('wrapped', [False, True])
def test_offline_derivation_cli_accepts_both_contract_json_forms_without_kit(qualified, wrapped):
    root = Path(__file__).resolve().parents[1]
    contract_path = qualified.output.parent/'cli_adapter.json'
    value = {'physical_action_adapter': qualified.adapter} if wrapped else qualified.adapter
    contract_path.write_text(json.dumps(value))
    output = qualified.output.parent/'cli_v4.pt'
    result = subprocess.run([sys.executable, str(root/'scripts/derive_dh116_bimanual_checkpoint.py'),
        '--input', str(qualified.source), '--output', str(output), '--source-sha256', sha(qualified.source),
        '--adapter-contract', str(contract_path), '--teacher-report', str(qualified.teacher_path),
        '--mapping-report', str(qualified.mapping_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['status'] == 'derived_offline_metadata_only'
    assert output.exists() and output.with_suffix('.json').exists()
