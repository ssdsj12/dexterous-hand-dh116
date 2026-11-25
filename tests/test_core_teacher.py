"""Production teacher contracts and frozen pre-consolidation numerical oracle.

The oracle was captured from the actual original network/device trainer, before
removing the historical wrappers (PyTorch 2.7.0, CPU with two math threads).
No archived Python source is imported or executed by this suite.
"""
import ast
from pathlib import Path
import pytest
from copy import deepcopy
from dataclasses import replace
import hashlib, json, random
import numpy as np
import torch
from dh116.rl.teacher import TensorNativeTeacher
from dh116.physics.solver import TensorBackendProfile
from dh116.rl.teacher import pack_shared_tensor_inputs
from dh116.rl.device import DeviceDeformablePPOTrainer as DeviceDeformablePPOTrainer
from dh116.rl.ppo import PPOConfig
from dh116.rl.reward import load_config
from dh116.env.contracts import JOINTS
from dh116.hand.kinematics import WRIST_LINEAR
from dh116.hand.kinematics import WRIST_ANGULAR
from dh116.env.contracts import runti_io__ROOT as ROOT

def seed(v):
    torch.manual_seed(v)
    random.seed(v)
    np.random.seed(v)

def digest_tree(tree):
    h = hashlib.sha256()

    def visit(v):
        if isinstance(v, torch.Tensor):
            h.update(str(v.dtype).encode())
            h.update(str(tuple(v.shape)).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, np.ndarray):
            visit(torch.from_numpy(v.copy()))
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode())
                visit(v[k])
        elif isinstance(v, (tuple, list)):
            for x in v:
                visit(x)
        else:
            h.update(repr(v).encode())
    visit(tree)
    return h.hexdigest()

def rng():
    return (torch.get_rng_state(), random.getstate(), np.random.get_state())

def teacher_data(monkeypatch):
    from dh116.rl.teacher import SharedSourceIdentity as SharedSourceIdentity
    from dh116.rl.teacher import ContextIdentity as ContextIdentity
    from dh116.rl.teacher import tenso_teach_schem__SMOKE_EXCLUDED as SMOKE_EXCLUDED
    p = TensorBackendProfile('cpu-fixture', 'a' * 64, 'b' * 64, 'c' * 64, 'd' * 64, 'cpu-fixture')
    monkeypatch.setattr(TensorBackendProfile, 'verify', lambda self: self)
    c = dict(grasp='pinch', task_kind='deformable_rod', use_cameras=False, model=dict(common_size=371, action_size=24, hidden=[256, 256], rod_feature_size=13, rod_hidden=64, initial_std=0.1))
    reward = load_config(ROOT / 'configs/training/reward_pinch.json', ppo_gamma=0.995)
    seed(18)
    model = TensorNativeTeacher(c, backend_profile=p, num_envs=2, reward_config=reward)
    context = torch.full((2, 50), torch.nan)
    valid = torch.zeros((2, 50), dtype=torch.bool)
    required = tuple((i for i in range(50) if i not in (*SMOKE_EXCLUDED, 8, 9, 46)))
    valid[:, required] = True
    context[valid] = 0.0
    context[:, 0] = 1.0
    context[:, 36:38] = 0.1
    context[:, 38:40] = 0.02
    context[:, 40:43] = torch.tensor([0.03, 5.0, 0.0005])
    context[:, 44] = 1.0
    source = SharedSourceIdentity(1, 2, (0, 1), 'fixture-run', ('ep0', 'ep1'), ('rod0', 'rod1'), (1, 1), (0, 0), 0, 0.0, 1 / 480, 1 / 480, 0.0, 0.0, p.digest, 'cpu_fixture')
    identities = tuple((ContextIdentity('fixture-run', i, 'ep' + str(i), 1, 'rod' + str(i), 0, 0, 0.0, 0.0) for i in range(2)))
    q = torch.zeros(2, 2, 17)
    qd = torch.arange(68).reshape(2, 2, 17).float() * 0.0001
    pose = torch.zeros(2, 2, 7)
    pose[:, :, 3] = 1.0
    origins = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    nodes = torch.zeros(2, 41, 3)
    nodes[:, :, 0] = torch.arange(41) * 0.005
    nodes[:, :, 2] = 0.1
    nodes += origins[:, None]
    kw = dict(joint_names=(JOINTS + WRIST_LINEAR + WRIST_ANGULAR,) * 2, body_link_pose_w=pose, body_com_velocity_w=torch.zeros(2, 2, 6), root_seed_w=pose.clone(), env_origins=origins, rod_positions_w=nodes, rod_velocities_w=torch.zeros_like(nodes), rest_lengths_m=torch.full((2, 40), 0.005), task_context=context, task_context_valid=valid, source_identity=source, context_identities=identities, backend_profile_sha256=p.digest, grasp='pinch', scope='cpu_fixture')
    packet = pack_shared_tensor_inputs(q, qd, **kw)
    return (model, packet.inputs, p, c, (q, qd, kw))

def collect(trainer, obs):
    for t in range(3):
        sample = trainer.collect_step(obs, obs)
        done = torch.zeros(2, dtype=torch.bool)
        trunc = torch.zeros_like(done)
        boot = ~done
        if t == 1:
            done[:] = True
            trunc[1] = True
            boot[0] = False
        trainer.record_step(torch.arange(2).float() + 1 + t * 0.25, done, torch.ones(2) * (t + 0.5), truncated=trunc, bootstrap_mask=boot, executed_actions=sample['actions'], collection_token=sample['collection_token'])

def parameter_digest(model):
    return digest_tree({k: v for (k, v) in model.state_dict().items() if isinstance(v, torch.Tensor)})
BASELINE = {'torch_version': '2.7.0+cu128', 'init_parameters': '7e7fc72d3294d19352923dc891023152cb104867e5873108496ad41e7044d4a8', 'init_rng': 'd9c5fb937a97022006c107bbcc31598b70a20584ad99a3851b60c563a78f9ad3', 'input_digest': 'd9e13f07c3878185391c7d5295954864aaf40315f1e1c54c0f60ee5602de971d', 'mu': [[0.026013530790805817, 0.008120089769363403, -0.11657771468162537, 0.008960969746112823, 0.15724068880081177, 0.08951552212238312, -0.15465174615383148, -0.03359898552298546, 0.09351304173469543, 0.10686421394348145, 0.08580714464187622, -0.010622968897223473, 0.05928798019886017, -0.23336084187030792, 0.113273024559021, 0.07785774767398834, 0.042168937623500824, 0.05792419984936714, -0.12382432818412781, 0.06507298350334167, 0.06332023441791534, -0.10311490297317505, -0.07263704389333725, 0.138200044631958], [0.026677187532186508, 0.006716568022966385, -0.11631037294864655, 0.008920863270759583, 0.15731436014175415, 0.089679054915905, -0.15452128648757935, -0.03344159945845604, 0.09320004284381866, 0.10735692083835602, 0.08602764457464218, -0.01068066619336605, 0.06011701002717018, -0.23340308666229248, 0.11354681104421616, 0.07789236307144165, 0.042199745774269104, 0.05784523859620094, -0.12399955093860626, 0.0657428503036499, 0.0633164495229721, -0.10260745882987976, -0.07215286791324615, 0.13852420449256897]], 'value': [0.1802477240562439, 0.17858991026878357], 'updates': {'False': [{'batch': '5c3d8a5a4923e11cd6541dce0cbf85c3d67e57971556784d4957dfa0b4956980', 'gae': '478f7ba25da32d148c2c8fa0efb618e8aac1b60e783fb083e84ace91d3eef35b', 'parameters': 'bd0c4a25318cd6a493497b09383308e3ffce4173b7f6321dc12534f56ce404b2', 'adam': 'c2a945d222c950536f103f9e5810bcc4156ee53217e3ab4782568fa5c16eee92', 'metrics': '18339967371841be6fcaa03740e3e90d86b118e369aaa58af13124f0a2824f50', 'rng': '3e84d5306924a4e17bb6a7b63f345a6f3e68d031612813a1ffb5cf1ce4b8ce0d', 'counters': {'environment_steps': 6, 'updates': 1}}, {'batch': '9b7ce81c59d2a2bdc77c4a35628a836658cf01e78d82c5d27d392ef060a03104', 'gae': '0f86bd532e61eb3bba4ac1703cbf5e8eae89f3a886b7bc07404491b1f1ed2c16', 'parameters': 'ba507a16985bd82d7d8099170eb56f3403647965d5198bd8bf84b113506e13ed', 'adam': 'fbfc9d2e20c593299b4673ca6ab07afea638a84c0caae642b9da45fb0d5a7526', 'metrics': '5928da328a6f2a43cd32e624ff931af69022642479c38022d86bf002245cfabc', 'rng': 'b2f67273b5e7098df2170fd1d3fd3b69eb0533b21a59018db7fc00c7efe30877', 'counters': {'environment_steps': 12, 'updates': 2}}], 'True': [{'batch': '5c3d8a5a4923e11cd6541dce0cbf85c3d67e57971556784d4957dfa0b4956980', 'gae': '478f7ba25da32d148c2c8fa0efb618e8aac1b60e783fb083e84ace91d3eef35b', 'parameters': 'e48bf8025c30cc10aa0972393ba118eec6e6dffdb179c25ee708d7f7dbe755d1', 'adam': 'b415c26a4164dcd6fbea18d5d9a20d825352fa7a0f89f1b5b3414d7dacc3eb0c', 'metrics': 'e1879e37b738b294c6a264119d064d447065e58015517a1ab516167a5de70efd', 'rng': '3e84d5306924a4e17bb6a7b63f345a6f3e68d031612813a1ffb5cf1ce4b8ce0d', 'counters': {'environment_steps': 6, 'updates': 1}}, {'batch': '8c9cd84f61e9c5f7e05199c3e2e6657760c1a526c9fff55f47e10cc447bb21c4', 'gae': '67645f71026d1a310701f01e48172440d5ac1d40470a5aea6f71948c385f340f', 'parameters': 'df503f894ddc572c79193663f257777e6e245cec032b0c82a2f2edddcaf73c1c', 'adam': 'ae81670d9251d3e866c2a3687e90b8cac1b65099b077fc06924b86f5b2c07263', 'metrics': '1cbce7c4059f20145d497452e1e7b91f33ddb939c3e89553106bb23a33ee8294', 'rng': 'b2f67273b5e7098df2170fd1d3fd3b69eb0533b21a59018db7fc00c7efe30877', 'counters': {'environment_steps': 12, 'updates': 2}}]}}

@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)

def test_production_teacher_has_no_historical_model_or_source_dependency():
    from dh116.rl.ppo import PrivilegedTeacher
    from dh116.env.contracts import EagerRuntimePolicy
    assert TensorNativeTeacher.__bases__ == (EagerRuntimePolicy, PrivilegedTeacher)
    forbidden = ('shared_teacher_source', 'shared_native_teacher', 'native_teacher_model', 'native_teacher_context', 'deformable_privileged_teacher', 'flexible_measured_common_teacher', 'native_teacher_profile')
    for name in ('rl/teacher.py',):
        path = ROOT / 'src/dh116' / name
        imports = [n.module or '' for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.ImportFrom)]
        assert not [module for module in imports if any((module.endswith('tabletop_' + old) for old in forbidden))]

def test_original_network_parameters_outputs_rng_and_input_math_are_exact(monkeypatch):
    (model, obs, profile, cfg, raw) = teacher_data(monkeypatch)
    assert parameter_digest(model) == BASELINE['init_parameters']
    assert digest_tree(rng()) == BASELINE['init_rng']
    assert digest_tree(obs) == BASELINE['input_digest']
    torch.testing.assert_close(model.latent_mean(obs), torch.tensor(BASELINE['mu']), atol=0, rtol=0)
    torch.testing.assert_close(model.value(obs), torch.tensor(BASELINE['value']), atol=0, rtol=0)
    assert list(model._parameters) == ['log_std']
    assert list(model._modules) == ['actor', 'critic', 'actor_rod', 'critic_rod', 'actor_context', 'critic_context', 'actor_head', 'critic_head']
    assert torch.isnan(obs['common371'][~obs['common_valid']]).all()
    assert obs['common_valid'].sum(1).tolist() == [66, 66]
    assert obs['rod_mask'].all() and obs['rod_nodes'].shape == (2, 41, 13)

def test_owned_identity_checkpoint_rejection_and_all_nodes_are_required(monkeypatch):
    (model, obs, *_) = teacher_data(monkeypatch)
    identity = model.model_identity
    identity['architecture']['actor'] = 99
    assert model.model_identity['architecture']['actor'] == 24
    state = deepcopy(model.state_dict())
    state['_extra_state']['grasp'] = 'power'
    before = parameter_digest(model)
    with pytest.raises(ValueError, match='identity'):
        model.load_state_dict(state)
    assert parameter_digest(model) == before
    valid_state = deepcopy(model.state_dict())
    model.load_state_dict(valid_state)
    valid_state['_extra_state']['architecture']['actor'] = 99
    assert model.model_identity['architecture']['actor'] == 24
    for (key, damage) in [('rod_mask', False), ('common_valid', False)]:
        bad = deepcopy(obs)
        bad[key][0, 0] = damage
        with pytest.raises(ValueError):
            model.latent_mean(bad)
    bad = deepcopy(obs)
    bad['rod_nodes'][0, 40, 0] = torch.nan
    with pytest.raises(ValueError):
        model.value(bad)
    bad = deepcopy(obs)
    bad['legacy'] = bad['rod_mask']
    with pytest.raises(ValueError, match='six'):
        model.value(bad)

def test_tensor_packet_requires_current_tensor_teacher_and_fresh_source(monkeypatch):
    from dh116.rl.teacher import pack_shared_tensor_inputs
    (model, obs, profile, cfg, (q, qd, kw)) = teacher_data(monkeypatch)
    packet = pack_shared_tensor_inputs(q, qd, **kw)
    packed = packet.for_teacher(model, current_identity=packet.source)
    packed['rod_nodes'].zero_()
    assert digest_tree(packet.inputs) == BASELINE['input_digest']
    with pytest.raises(ValueError):
        packet.for_teacher(object(), current_identity=packet.source)
    with pytest.raises(ValueError):
        packet.for_teacher(model, current_identity=replace(packet.source, sdk_step=1))

def test_source_owner_retains_nominal_fp32_clock_health_material_and_mass_guards():
    from types import SimpleNamespace
    from dh116.rl.teacher import TensorSourceOwner
    from dh116.rl.teacher import ContextIdentity as ContextIdentity
    assert not hasattr(ContextIdentity, 'from_clock')
    owner = object.__new__(TensorSourceOwner)
    solver = SimpleNamespace(num_envs=2, last_error=0, time=0.25)
    owner.runtime = SimpleNamespace(solver=solver, _failed=False, _time=0.25, physics_step=100, run_uuid='run', episodes=('ep0', 'ep1'), resets=[1, 1], ticks=[100, 100])
    owner.n = 2
    owner._parent_solver_id = id(solver)
    owner.profile = SimpleNamespace(digest='d' * 64)
    owner._profile_sha256 = 'd' * 64
    owner.rod_uids = ('rod0', 'rod1')
    owner.scope = 'cpu_fixture'
    owner.device = torch.device('cpu')
    owner.grasp = 'pinch'
    owner.sim = SimpleNamespace(current_time_step_index=100, current_time=100 * float(np.float32(1 / 480)), get_physics_dt=lambda : 1 / 480)
    boundary = owner._boundary()
    assert boundary.sdk_dt_s == 1 / 480 and boundary.sdk_timer_dt_s == float(np.float32(1 / 480))
    owner.sim.get_physics_dt = lambda : float(np.float32(1 / 480))
    assert owner._boundary().sdk_dt_s == float(np.float32(1 / 480))
    owner.sim.current_time += 1e-05
    with pytest.raises(ValueError, match='boundary'):
        owner._boundary()
    owner.sim.current_time = boundary.sdk_time_s
    owner.runtime._failed = True
    with pytest.raises(ValueError, match='boundary'):
        owner._boundary()
    owner.runtime._failed = False
    cfg = json.loads((ROOT / 'configs/tasks/pinch.json').read_text())['rod']
    rest = np.full((2, 40), 0.005)
    ea = np.full_like(rest, cfg['axial_stiffness_n'])
    ei = np.full_like(rest, cfg['bend_stiffness_n_m2'])
    mass = np.concatenate((rest[:, :1] / 2, (rest[:, :-1] + rest[:, 1:]) / 2, rest[:, -1:] / 2), 1) * cfg['linear_density_kg_m']
    raw = dict(schema='dh116.native_parameter_snapshot.v1', metadata_source='native_host_metadata', segment_source='cuda_device_copy', hardware_measured=False, num_envs=2, nodes_per_env=41, time_s=0.25, contact_radius_m=cfg['radius_m'], gravity_world_m_s2=cfg['gravity_world_m_s2'], contact_mu_static=cfg['mu_s'], contact_mu_kinetic=cfg['mu_k'], contact_v_slip_m_s=cfg['v_slip_m_s'], contact_kappa=cfg['kappa'], rest_lengths_m=rest, stored_k_stretch_n_m=ea * rest, rest_quaternion_norm_n_m=4 * ei / rest, stretch_ea_n=ea, bend_ei_n_m2=ei, inverse_masses_kg_inv=1 / mass)
    torch.testing.assert_close(owner._parameters(raw, boundary), torch.tensor(rest, dtype=torch.float32), atol=0, rtol=0)
    for (field, bad) in [('inverse_masses_kg_inv', 1 / (mass * 2)), ('contact_radius_m', cfg['radius_m'] * 2), ('stretch_ea_n', ea * 2)]:
        changed = deepcopy(raw)
        changed[field] = bad
        with pytest.raises(ValueError):
            owner._parameters(changed, boundary)
    changed = deepcopy(raw)
    changed['inverse_masses_kg_inv'][0, 0] = 0
    with pytest.raises(ValueError, match='free-root'):
        owner._parameters(changed, boundary)

def test_source_owner_verifies_actual_v2_profile_and_rejects_duck_typed_profile():
    from types import SimpleNamespace
    from dh116.rl.teacher import TensorSourceOwner
    profile = TensorBackendProfile.load()
    owner = object.__new__(TensorSourceOwner)
    owner.profile = profile
    owner._profile_sha256 = profile.digest
    owner._manifest_path = profile.manifest_path
    owner._verify_profile()
    owner.profile = SimpleNamespace(digest=profile.digest, manifest_path=profile.manifest_path, verify=lambda : None)
    with pytest.raises(ValueError):
        owner._verify_profile()


def full_context_fixture(kw):
    values = kw['task_context'].clone()
    valid = kw['task_context_valid'].clone()
    valid[:, (4, 5, 6, 7, 47)] = True
    values[:, 4:8] = torch.tensor([[1.0, 1.0, 0.8, 0.2], [0.25, 0.0, 0.0, 0.0]])
    values[:, 47] = 0.0
    values[0, :4] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    values[0, 8:12] = torch.tensor([0.3, 0.75, 1.0, 1.0])
    valid[0, 8:10] = True
    return values, valid


def test_full_context_applicability_tracks_anchors_in_each_environment(monkeypatch):
    from dh116.rl.teacher import CONTEXT_FIELDS, ContextMetadata, TaskContextPacket, validate_context

    _, _, profile, _, (_, _, kw) = teacher_data(monkeypatch)
    values, valid = full_context_fixture(kw)
    applicable = tuple(tuple(row) for row in valid.tolist())
    reasons = tuple(tuple(None if field else 'unavailable_in_this_row' for field in row) for row in applicable)
    metadata = ContextMetadata(
        'owner', kw['context_identities'], profile.digest, 'cpu_fixture', 'flexible_train_r1_full',
        tuple(field['source'] for field in CONTEXT_FIELDS), applicable, reasons, (('reset', 'reset_fixture'),), 'request',
    )
    context = TaskContextPacket(values, valid, metadata)
    assert validate_context(context, backend_profile=profile, identities=kw['context_identities']) is context
    with pytest.raises(ValueError, match='applicability'):
        wrong = replace(metadata, applicable=metadata.applicable[::-1], reasons=metadata.reasons[::-1])
        validate_context(replace(context, metadata=wrong), backend_profile=profile)
    with pytest.raises(ValueError, match='reason'):
        wrong = replace(metadata, reasons=(reasons[0], (None,) * 50))
        validate_context(replace(context, metadata=wrong), backend_profile=profile)
    with pytest.raises(ValueError, match='identity'):
        validate_context(context, backend_profile=profile, identities=kw['context_identities'][::-1])


def test_full_tensor_packet_requires_profile_and_row_matched_context(monkeypatch):
    smoke, _, backend, cfg, (q, qd, kw) = teacher_data(monkeypatch)
    values, valid = full_context_fixture(kw)
    kw = dict(kw, task_context=values, task_context_valid=valid, context_mode='flexible_train_r1_full')
    packet = pack_shared_tensor_inputs(q, qd, **kw)
    assert packet.inputs['common_valid'].sum(1).tolist() == [66, 66]
    assert packet.inputs['task_context_valid'][:, 8:10].tolist() == [[True, True], [False, False]]
    assert not packet.native_qualified
    with pytest.raises(ValueError, match='contract'):
        packet.for_teacher(smoke, current_identity=packet.source)
    with pytest.raises(ValueError, match='row/time/episode'):
        pack_shared_tensor_inputs(q, qd, **dict(kw, context_identities=kw['context_identities'][::-1]))
    with pytest.raises(ValueError, match='mode'):
        pack_shared_tensor_inputs(q, qd, **dict(kw, context_mode='made-up-mode'))
    with pytest.raises(ValueError, match='smoke'):
        pack_shared_tensor_inputs(q, qd, **dict(kw, context_mode='approach_wiring_smoke_v1'))
    with pytest.raises(ValueError, match='anchor'):
        wrong = values.clone()
        wrong[1, 10] = 1.0
        pack_shared_tensor_inputs(q, qd, **dict(kw, task_context=wrong))


def test_full_teacher_packs_all_1024_source_rows(monkeypatch):
    from dh116.rl.teacher import ContextIdentity

    _, _, backend, cfg, (q, qd, kw) = teacher_data(monkeypatch)
    values, valid = full_context_fixture(kw)
    count = 1024
    rows = torch.arange(count) % 2
    source = replace(kw['source_identity'], parent_num_envs=count, env_rows=tuple(range(count)),
                     episode_uids=tuple(f'ep{i}' for i in range(count)),
                     rod_uids=tuple(f'rod{i}' for i in range(count)),
                     reset_ids=(1,) * count, rod_ticks=(0,) * count)
    identities = tuple(ContextIdentity(source.run_uuid, i, source.episode_uids[i], 1,
                                       source.rod_uids[i], 0, 0, 0.0, 0.0) for i in range(count))
    batched = {k: v[rows] if isinstance(v, torch.Tensor) else v for k, v in kw.items()}
    reward = load_config(ROOT / 'configs/training/reward_pinch.json', full_grasp=True)
    model = TensorNativeTeacher(cfg, backend_profile=backend, num_envs=count, reward_config=reward)
    batched.update(source_identity=source, context_identities=identities,
                   task_context=values[rows], task_context_valid=valid[rows],
                   context_mode=model.context_mode)

    packet = pack_shared_tensor_inputs(q[rows], qd[rows], **batched)
    observations = packet.for_teacher(model, current_identity=packet.source)
    assert observations['common371'].shape == (count, 371)
    assert model.latent_mean(observations).shape == (count, 24)
    assert observations['task_context_valid'][-2:, 8:10].tolist() == [[True, True], [False, False]]


def test_full_teacher_uses_exact_goal_reward_profile(monkeypatch):
    smoke, smoke_inputs, backend, cfg, (q, qd, kw) = teacher_data(monkeypatch)
    reward = load_config(ROOT / 'configs/training/reward_pinch.json', ppo_gamma=0.995, full_grasp=True)
    model = TensorNativeTeacher(cfg, backend_profile=backend, num_envs=2, reward_config=reward)
    values, valid = full_context_fixture(kw)
    packet = pack_shared_tensor_inputs(q, qd, **dict(
        kw, context_mode=model.context_mode, task_context=values, task_context_valid=valid,
    ))
    obs = packet.for_teacher(model, current_identity=packet.source)
    assert model.model_identity['context_mode'] == 'flexible_train_r1_full'
    assert model.model_identity['reward_config_sha256'] == reward.config_sha256
    assert model.latent_mean(obs).shape == (2, 24)
    assert torch.isfinite(model.value(obs)).all()
    with pytest.raises(ValueError, match='qualified contact/support'):
        model.latent_mean(smoke_inputs)
    with pytest.raises(ValueError, match='identity'):
        model.load_state_dict(smoke.state_dict())


def test_full_source_context_uses_current_device_history_and_fixed_stations(monkeypatch):
    from dh116.rl.teacher import FullTaskContext, TensorSourceOwner

    _, _, _, _, (_, _, kw) = teacher_data(monkeypatch)
    stamp = kw['source_identity']
    owner = object.__new__(TensorSourceOwner)
    owner.context_mode = 'flexible_train_r1_full'
    owner.n, owner.device = 2, torch.device('cpu')
    owner._boundary = lambda: stamp
    owner._reset_identities = kw['context_identities']
    owner._reset_z = torch.tensor([[0.2, 0.1], [0.2, 0.1]])
    owner._reset_distance = torch.tensor([0.003, 0.02])
    owner._reference_policy_time = torch.zeros(2)
    owner._rest_total = torch.tensor([0.2, 0.25], dtype=torch.float64)
    owner._previous = torch.zeros(2, 24)
    owner.episode_cap_s = owner.wall_cap_s = 10.0
    owner.started = 0.0
    full = FullTaskContext(
        source=stamp, phase=torch.tensor([3, 0]), contact_fraction=torch.tensor([[1, 1], [0.25, 0.0]]),
        contact_duration_s=torch.tensor([0.2, 0.0]), hold_duration_s=torch.tensor([1.0, 0.0]),
        anchor_station_m=torch.tensor([[0.06, 0.15], [torch.nan, torch.nan]], dtype=torch.float64),
        anchor_valid=torch.tensor([[True, True], [False, False]]), prohibited_support_latched=torch.tensor([False, True]),
        success_event_emitted=torch.tensor([True, False]), trusted_failure_event_emitted=torch.tensor([False, True]),
    )
    values, valid, identities = owner.contexts(full=full)
    expected, _ = full_context_fixture(kw)
    torch.testing.assert_close(values[:, :12], expected[:, :12], equal_nan=True)
    torch.testing.assert_close(values[:, 47:50], torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.float32))
    assert valid[:, 47:50].all()
    assert torch.isnan(values[1, 8:10]).all() and not valid[1, 8:10].any()
    assert identities == kw['context_identities']
    with pytest.raises(ValueError, match='boundary'):
        owner.contexts(full=replace(full, source=replace(stamp, sdk_step=1)))
    with pytest.raises(ValueError, match='full context'):
        owner.contexts()
    with pytest.raises(ValueError, match='full context'):
        owner.contexts(full=replace(full, contact_fraction=full.contact_fraction.fill_(torch.nan)))
