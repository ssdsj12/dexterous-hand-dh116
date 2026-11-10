"""CPU candidate/gate fixtures, excluded from native success counts."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_bimanual_coordination import state as coordination_state
from test_bimanual_tracking_guard import state as tracking_state
from test_stage5_teacher_regression import proof_fixture
from go2_pvcnn.control.dh116 import tabletop_bimanual_teacher_coordination as api


def proof(tmp_path):
    value = proof_fixture(tmp_path)
    path = tmp_path/'proof.json'
    path.write_text(json.dumps(value))
    return path, value


def setup(tmp_path, monkeypatch):
    n = 2
    task = SimpleNamespace(m={'bottom': torch.tensor([.36, .36])},
        palm_targets=torch.full((n, 2, 3), .02, dtype=torch.float64),
        rotation_targets=torch.full((n, 2, 3), .2, dtype=torch.float64))
    world = SimpleNamespace(n=n, device=torch.device('cpu'), dt=1/480,
        palms=torch.full((n, 2, 3), .01, dtype=torch.float64),
        wrist_angles=torch.full((n, 2, 3), .1, dtype=torch.float64),
        _physics_steps=12, w=SimpleNamespace(sim=SimpleNamespace(current_time=.025)))
    previous_telemetry = SimpleNamespace(callback=lambda packet: None)
    env = SimpleNamespace(task=task, diagnostic_episode_ids=torch.tensor([0, 0]),
        _diagnostic_policy_step=5, evaluation_telemetry=previous_telemetry,
        cache_bimanual_slip=False, _bimanual_slip_packet={'original': True})
    co, track = coordination_state(n), tracking_state(n)
    co.opposed[1, 1] = False
    track.active_tracking_error_rad[1, 1] = .04
    track.measured_palm_offset_m[1] = .004
    monkeypatch.setattr(api, 'coordination_state_from_native', lambda *a: co)
    monkeypatch.setattr(api, 'tracking_state_from_native', lambda *a: track)
    candidate = api.TripodTeacherCoordination(env, world, tmp_path,
        ['pinch_thumb_index', 'tripod_thumb_index_middle'], {'sha256': 'a'*64})
    return env, world, candidate, co, track, previous_telemetry


def test_pinch_action_and_every_target_remain_bitwise_untouched(tmp_path, monkeypatch):
    env, world, candidate, _, _, original_telemetry = setup(tmp_path, monkeypatch)
    tensors = [world.palms, world.wrist_angles, env.task.palm_targets, env.task.rotation_targets]
    originals = [value[0].numpy().tobytes() for value in tensors]
    source = torch.linspace(-.8, .8, 48, dtype=torch.float64).reshape(2, 24)
    before = source.clone()
    issued = candidate.project(source)
    assert issued[0].numpy().tobytes() == source[0].numpy().tobytes()
    assert [value[0].numpy().tobytes() for value in tensors] == originals
    assert torch.equal(issued[:, :12], source[:, :12]) and torch.equal(before, source)
    assert issued[1, 12:].eq(0).all()
    assert world.palms[1].eq(.004).all() and env.task.palm_targets[1].eq(.004).all()
    assert world.wrist_angles[1].eq(0).all() and env.task.rotation_targets[1].eq(0).all()
    candidate.observe(env, {}, torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
    candidate.close()
    assert env.evaluation_telemetry is original_telemetry
    assert env.cache_bimanual_slip is False and env._bimanual_slip_packet == {'original': True}
    assert not hasattr(env, 'bimanual_action_adapter') and not hasattr(env, 'joint_replay_observer')
    data = np.load(tmp_path/'tripod_teacher_coordination.npz', allow_pickle=False)
    assert np.array_equal(data['source_requests'][0], source.numpy())
    assert np.array_equal(data['executed_requests'][0], issued.numpy())
    assert not np.array_equal(data['executed_requests'][0, 1], data['source_requests'][0, 1])
    contract = json.loads((tmp_path/'tripod_teacher_coordination_contract.json').read_text())
    assert contract['tripod_supervision_qualified'] is False
    assert contract['supervision_eligible_env_ids'] == [0]


def test_contact_loss_cancels_both_queued_ascents_without_touching_pinch(tmp_path, monkeypatch):
    env, world, candidate, _, track, _ = setup(tmp_path, monkeypatch)
    track.active_tracking_error_rad.zero_()
    source = torch.zeros(2, 24, dtype=torch.float64)
    source[:, [14, 17]] = .7
    issued = candidate.project(source)
    assert torch.equal(issued[0], source[0])
    assert (issued[1, [14, 17]] <= 0).all()
    assert world.palms[1, :, 2].eq(.004).all()
    assert env.task.palm_targets[1, :, 2].eq(.004).all()
    assert world.palms[0, :, 2].eq(.01).all()
    candidate.close(error='fixture stopped')


def test_unready_healthy_tripod_retains_bounded_source_downward_acquisition(tmp_path, monkeypatch):
    env, world, candidate, _, track, _ = setup(tmp_path, monkeypatch)
    track.active_tracking_error_rad.zero_()
    source = torch.zeros(2,24, dtype=torch.float64)
    source[:, [14,17]] = torch.tensor([-.2, -.3], dtype=torch.float64)
    issued = candidate.project(source)
    assert torch.equal(issued[:, [14,17]], source[:, [14,17]])
    candidate.observe(env, {}, torch.ones(2,dtype=torch.bool), torch.zeros(2,dtype=torch.bool))
    candidate.close()
    data = np.load(tmp_path/'tripod_teacher_coordination.npz')
    fields = data['fields'].tolist()
    assert data['samples'][0,1,fields.index('source_downward_admitted_left')] == 1
    assert data['samples'][0,1,fields.index('source_downward_admitted_right')] == 1
    assert data['samples'][0,0,fields.index('source_downward_admitted_left')] == 0


def test_source_acquisition_floor_restricts_descent_without_increasing_amplitude(tmp_path, monkeypatch):
    from go2_pvcnn.control.dh116.tabletop_policy_control import CONTRACT
    env, world, candidate, co, track, _ = setup(tmp_path, monkeypatch)
    track.active_tracking_error_rad.zero_()
    co.clearance_m[1] = CONTRACT['palm_descent_clearance_m'] + .1*.01*world.dt
    source = torch.zeros(2,24, dtype=torch.float64)
    source[:, [14,17]] = -.4
    issued = candidate.project(source)
    torch.testing.assert_close(issued[1,[14,17]], torch.full((2,),-.1,dtype=source.dtype), atol=1e-12,rtol=0)
    assert issued[1,[14,17]].abs().le(source[1,[14,17]].abs()).all()
    assert torch.equal(issued[0],source[0])
    candidate.close()


def test_tracking_guard_remains_last_and_can_veto_source_descent(tmp_path, monkeypatch):
    env, _, candidate, _, _, _ = setup(tmp_path, monkeypatch)
    source = torch.zeros(2,24,dtype=torch.float64)
    source[:,[14,17]] = -.4
    issued = candidate.project(source)
    assert issued[1,12:].eq(0).all()
    assert torch.equal(issued[0],source[0])
    candidate.close()


def test_pinched_success_cannot_promote_tripod_source_requests_to_bc():
    assert api.supervision_eligibility(['pinch_thumb_index', 'tripod_thumb_index_middle']) == [True, False]
    assert api.supervision_eligibility(['tripod_thumb_index_middle']) == [False]


@pytest.mark.parametrize('new_episode', [False, True])
def test_terminal_remainder_and_later_episode_never_reproject_or_write_targets(tmp_path, monkeypatch, new_episode):
    env, world, candidate, co, _, _ = setup(tmp_path, monkeypatch)
    source = torch.full((2, 24), .4, dtype=torch.float64)
    candidate.project(source)
    candidate.observe(env, {}, torch.ones(2, dtype=torch.bool), torch.tensor([False, True]))
    co.live[1] = new_episode
    if new_episode:
        env.diagnostic_episode_ids[1] += 1
        co.episode_id[1] += 1
    targets = [world.palms, world.wrist_angles, env.task.palm_targets, env.task.rotation_targets]
    for target in targets: target[1] = .017
    before = [target[1].numpy().tobytes() for target in targets]
    issued = candidate.project(source)
    assert issued[1].numpy().tobytes() == source[1].numpy().tobytes()
    assert [target[1].numpy().tobytes() for target in targets] == before
    candidate.close()


def test_gate_reuses_t5_native_inputs_and_freezes_proof_sha(tmp_path):
    path, value = proof(tmp_path)
    qualified = api.read_tripod_coordination_proof(path)
    assert qualified['validation']['verified']
    assert qualified['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(json.dumps(dict(value, counts_as_student_success=True)))
    with pytest.raises(ValueError, match='changed after pre-Kit'):
        api.read_tripod_coordination_proof(path, expected_sha256=qualified['sha256'])


def test_gate_rejects_replaced_bound_t5_evidence(tmp_path):
    path, value = proof(tmp_path)
    folder = value['input_directories']['reference']
    from pathlib import Path
    (Path(folder)/'teacher_collection_report.json').write_text('changed')
    with pytest.raises(ValueError, match='SHA mismatch'):
        api.read_tripod_coordination_proof(path)


def test_options_reject_false_gate_sdk_combo_and_changed_teacher_source(tmp_path):
    path, value = proof(tmp_path)
    source = tmp_path/'reference.pt'
    source.write_bytes(b'synthetic reference')
    value['teacher_source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_text(json.dumps(value))
    args = SimpleNamespace(bimanual_tripod_teacher_coordination_proof=path,
        vision_teacher_source=source, seed=19)
    assert api.validate_tripod_coordination_options(args)['validation']['verified']
    args.bimanual_explicit_sdk_adapter = True
    with pytest.raises(ValueError, match='excludes explicit SDK'):
        api.validate_tripod_coordination_options(args)
    args.bimanual_explicit_sdk_adapter = False
    source.write_bytes(b'changed source')
    with pytest.raises(ValueError, match='Teacher source differs'):
        api.validate_tripod_coordination_options(args)


def test_collection_excludes_candidate_tripod_rows_from_supervision_pool(tmp_path, monkeypatch):
    from test_vision_training import fixture, Camera
    from test_vision_teacher_collection import options
    from go2_pvcnn.control.dh116 import tabletop_vision_teacher_collection as collection
    from go2_pvcnn.control.dh116 import tabletop_vision_supervision_dataset as dataset
    from go2_pvcnn.control.dh116 import tabletop_bimanual_teacher_audit as audit
    dw, task, env, args = fixture(tmp_path)
    for key, value in vars(options(num_envs=2)).items(): setattr(args, key, value)
    args.bimanual_tripod_teacher_coordination_proof = tmp_path/'fixture_proof.json'
    monkeypatch.setattr(api, 'validate_tripod_coordination_options',
        lambda args: setattr(args, '_bimanual_tripod_proof_sha256', 'a'*64))
    fake = SimpleNamespace(supervision_eligible=[True, False], mask=torch.tensor([False, True]),
        proof={'sha256': 'a'*64}, paths=[], close=lambda **kw: None)
    monkeypatch.setattr(api, 'install_tripod_teacher_coordination', lambda *a, **kw: fake)
    monkeypatch.setattr(audit, 'install_teacher_physics_audit', lambda *a, **kw:
        SimpleNamespace(paths=[], submit=lambda *a, **kw: None, finish=lambda **kw: None))
    styles = ['pinch_thumb_index', 'tripod_thumb_index_middle']
    refs = [dict(source_episode_id=i, grasp_style=style, source_trajectory_id=style,
                 actions=torch.full((2, 24), .1)) for i, style in enumerate(styles)]
    monkeypatch.setattr(dataset, 'load_teacher_reference_dataset', lambda *a:
        dict(source_sha256='a'*64, episodes=refs, references={}))
    sim = SimpleNamespace(current_time=0., current_time_step_index=0)
    class Capture(Camera):
        def set_raw_frame_sink(self, sink): self.sink = sink
        def sample(self):
            frame = sim.current_time_step_index + 1
            self.sink(dict(rgb=torch.zeros(2,2,2,3,dtype=torch.uint8), depth=torch.ones(2,2,2,1),
                frame_id=frame, simulation_time=sim.current_time, rendered_simulation_time=sim.current_time,
                physics_step=sim.current_time_step_index))
            return dict(super().sample(), frame_id=frame, simulation_time=sim.current_time)
    def step(action):
        env.tick += 1
        sim.current_time += 1/60
        sim.current_time_step_index += 8
        done = torch.full((2,), env.tick == 2, dtype=torch.bool)
        return torch.zeros(2,371), torch.zeros(2), done, dict(success=done.clone(),
            failure=torch.zeros(2,dtype=torch.long), recovered=torch.zeros(2,dtype=torch.bool))
    env.step = step
    report = collection.run_teacher_collection(SimpleNamespace(sim=sim), dw, task, env, args,
                                               {'observation_size': 371}, camera=Capture(2))
    pool = torch.load(args.output/'all_executed_supervision.pt', weights_only=False)
    assert set(pool['episodes']) == {'0'}
    assert all(row['env_id'] == 0 for row in pool['sample_metadata'])
    assert report['episodes']['1']['success'] is False
    assert report['tripod_coordination_candidate']['supervision_qualified'] is False
    assert report['tripod_coordination_candidate']['native_physical_successes'] == 0
