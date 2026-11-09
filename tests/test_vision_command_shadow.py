"""CPU checks use the actual native controller kernels, without Isaac."""
import importlib
import importlib.util
import inspect

import pytest
import torch


def shadow():
    name = 'go2_pvcnn.control.dh116.tabletop_vision_command_shadow'
    assert importlib.util.find_spec(name), 'conditional command shadow is missing'
    return importlib.import_module(name)


def fixture(n=3):
    from test_free_wrist_grasp import setup
    from test_dh116_tensor_world import palm_world
    t, _ = setup(n)
    w, _ = palm_world()
    assert n == w.n
    t.free_wrist_world = w
    t.surface_force_sharing = True
    t.surface_policy_owned_targets = True
    t.surface_redistribution = False
    w.wrist_angles = torch.zeros(n, 2, 3, dtype=torch.float64)
    t.command[:] = (t.low + t.high) / 2
    t.m['q'][:] = t.command
    t.m['qd'] = torch.zeros_like(t.command)
    w.commands = t.command.reshape(n, 2, 6).clone()
    w.palms[:, :, 2] = -.002
    w.roots[:, :, 2] -= .001
    w.clearances[:, 0] = -.00036
    t.offset[:] = 0
    return t, w


def test_teacher_matches_real_control_and_transport_float64_and_cast32():
    m = shadow()
    t, w = fixture()
    contract = m.controller_contract(t, w)
    state = m.command_state(t, w)
    action = torch.linspace(-1., 1., 72, dtype=torch.float64).reshape(3, 24)
    live = torch.tensor([True, True, False])
    context = m.measured_context(t, w, live)
    allowed = t.control(action, live, w.dt)
    context['allowed'] = allowed
    w.advance(t, allowed, live)
    new, diagnostics = m.shadow_step(state, action, context, contract)
    for key, expected in dict(joint_command_rad=t.command,
            palm_command_offset_m=w.palms, wrist_command_angle_rad=w.wrist_angles).items():
        torch.testing.assert_close(new[key], expected, rtol=0, atol=0)
    torch.testing.assert_close(diagnostics['root_pose'], w.root_targets, rtol=0, atol=0)
    assert torch.equal(diagnostics['sdk_float32']['finger'], w.commands.flatten(1).float())
    assert diagnostics['safe_z_m'][0, 0] > -.001


def test_sequence_keeps_student_commands_and_gradients_across_all_eight_substeps():
    m = shadow(); t, w = fixture()
    state = m.command_state(t, w); w.clearances.fill_(.01)
    context = m.measured_context(t, w, torch.ones(3, dtype=torch.bool))
    context['allowed'] = torch.ones(3, dtype=torch.bool)
    contexts = {k: (v[:, None, None].expand(3, 2, 8, *v.shape[1:]).clone()
                   if isinstance(v, torch.Tensor) else v) for k, v in context.items()}
    actions = torch.full((3, 2, 24), .2, dtype=torch.float64, requires_grad=True)
    out = m.shadow_sequence(state, actions, contexts, m.controller_contract(t, w))
    assert out['finger'].shape == (3, 2, 8, 12)
    out['finger'][:, -1, -1].sum().backward()
    assert (actions.grad[:, 0, :12] > 0).all()
    assert (actions.grad[:, 1, :12] > 0).all()
    assert not actions.grad[:, :, 12:].any()


def test_shadow_refuses_unimplemented_callbacks_and_source_invalid():
    m = shadow(); t, w = fixture()
    t.surface_redistribution = object()
    with pytest.raises(ValueError, match='redistribution'):
        m.controller_contract(t, w)
    t.surface_redistribution = False
    t.tripod_recovery_runtime = object()
    with pytest.raises(ValueError, match='tripod_recovery_runtime'):
        m.controller_contract(t, w)
    del t.tripod_recovery_runtime
    context = m.measured_context(t, w, torch.ones(3, dtype=torch.bool))
    context['source_valid'][0] = False
    with pytest.raises(ValueError, match='source_valid'):
        m.shadow_step(m.command_state(t, w), torch.zeros(3, 24), context, m.controller_contract(t, w))


def test_collection_sink_is_default_disabled():
    from go2_pvcnn.control.dh116.tabletop_vision_teacher_collection import run_teacher_collection
    assert inspect.signature(run_teacher_collection).parameters['sequence_context_sink'].default is None


def test_capture_owns_precontrol_feedback_targets_and_restores_methods(tmp_path):
    name = 'go2_pvcnn.control.dh116.tabletop_vision_sequence_capture'
    assert importlib.util.find_spec(name), 'opt-in native sequence capture is missing'
    m = importlib.import_module(name)
    t, w = fixture(); records = []
    capture = m.SequenceContextCapture(t, w, records.append)
    action = torch.full((3, 24), .2)
    metadata = [dict(run_uuid='run', env_id=i, episode_uid=f'ep{i}', tick=0,
        policy_timestamp_s=0., first_physics_step=0, terminal=False,
        previous_action=[0.]*24) for i in range(3)]
    capture.begin_tick(metadata, action)
    q = t.m['q'].clone()
    for step in range(8):
        live = torch.tensor([True, step < 3, False])
        allowed = t.control(action, live, w.dt)
        w.advance(t, allowed, live)
    capture.finish_tick(torch.tensor([False, True, False]))
    capture.close()
    assert 'control' not in vars(t) and 'advance' not in vars(w)
    assert len(records) == 1
    raw = capture.write_unqualified(tmp_path/'raw_native_context.pt')
    raw_value = torch.load(raw['path'], weights_only=False)
    assert raw_value['qualified'] is False and len(raw_value['policy_records']) == 1
    assert raw_value['incomplete_policy_context'] is None
    record = records[0]
    assert record['physics_context']['measured_q_rad'].shape == (3, 8, 12)
    assert torch.equal(record['physics_context']['measured_q_rad'][:, 0], q)
    assert torch.equal(record['teacher_targets']['finger'][:, -1], t.command)
    assert record['physics_context']['live'][1].tolist() == [True]*3+[False]*5
    assert record['samples']['valid'][1]  # executed terminal prefix remains usable
    assert record['native_backend_targets'][0]['hands'][0]['backend']['available'] is False
    path = tmp_path/'dataset.pt'; path.write_bytes(b'finalized pool bytes')
    result = capture.finalize(path, metadata[:1], tmp_path/'sidecar.pt')
    sidecar = torch.load(result['path'], weights_only=False)
    assert sidecar['samples']['env_id'] == [0]
    assert sidecar['teacher_targets']['finger'].shape == (1, 8, 12)
    assert result['teacher_shadow_parity']['passed']
    assert result['teacher_shadow_parity']['max_float64_error'] == 0
    assert sidecar['dataset_sha256'] == __import__('hashlib').sha256(path.read_bytes()).hexdigest()
    from go2_pvcnn.control.dh116.tabletop_vision_supervision_dataset import validate_sequence_context
    validate_sequence_context(sidecar, dataset_sha256=result['dataset_sha256'])
    t.m['q'].zero_()
    assert torch.equal(record['physics_context']['measured_q_rad'][:, 0], q)
    capture.require_native_readback = True
    with pytest.raises(ValueError, match='native.*readback'):
        capture.finalize(path, metadata[:1], tmp_path/'cannot_qualify.pt')


def test_inactive_missing_context_freezes_state_without_fabricating_feedback():
    m = shadow(); t, w = fixture()
    context = m.measured_context(t, w, torch.tensor([True, False, False]))
    context['source_valid'][1:] = False
    context['measured_q_rad'][1:] = float('nan')
    context['dt_s'][1:] = 0.
    state = m.command_state(t, w)
    new, out = m.shadow_step(state, torch.ones(3, 24), context, m.controller_contract(t, w))
    for key in state:
        assert torch.equal(new[key][1:], state[key][1:])
    assert torch.isnan(context['measured_q_rad'][1:]).all()
    assert out['physics_valid'].tolist() == [True, False, False]


def test_contact_event_uses_original_opposed_predicate_before_lift():
    from go2_pvcnn.control.dh116.tabletop_vision_sequence_capture import SequenceContextCapture
    from types import SimpleNamespace
    t, w = fixture()
    t.surface_lift_success = SimpleNamespace(_paired=torch.zeros(3, 2, dtype=torch.bool),
        window_good=torch.zeros(3, dtype=torch.bool))
    t.m['effective_pad_loads_n'] = torch.tensor([[[2., 2., 0., 0., 0.]]]*3,
        dtype=torch.float64).expand(3, 2, 5).clone()
    t.m['effective_pad_vectors_w'] = torch.zeros(3, 2, 5, 3, dtype=torch.float64)
    t.m['effective_pad_vectors_w'][:, :, 0, 0] = 1
    t.m['effective_pad_vectors_w'][:, :, 1, 0] = -1
    t.m['effective_pad_valid'] = t.m['effective_pad_loads_n'] > 0
    capture = SequenceContextCapture(t, w)
    capture.begin_tick([], torch.zeros(3, 24))
    assert capture.active['events']['opposed_contact'].all()
    assert not capture.active['events']['native_hold_qualified'].any()
    capture.close()


def test_all_command_groups_backpropagate_through_clearance_projection():
    m = shadow(); t, w = fixture()
    w.clearances.fill_(.01)
    context = m.measured_context(t, w, torch.ones(3, dtype=torch.bool))
    contexts = {key: value[:, None, None].expand(3, 2, 8, *value.shape[1:]).clone()
                for key, value in context.items()}
    actions = torch.full((3, 2, 24), .2, dtype=torch.float64, requires_grad=True)
    output = m.shadow_sequence(m.command_state(t, w), actions, contexts, m.controller_contract(t, w))
    sum(value[:, -1, -1].sum() for value in output.values()).backward()
    assert torch.isfinite(actions.grad).all()
    for low, high in ((0, 12), (12, 18), (18, 24)):
        assert (actions.grad[:, 0, low:high] > 0).all()
        assert (actions.grad[:, 1, low:high] > 0).all()


def test_native_capture_refuses_missing_measured_velocity_event_source():
    from go2_pvcnn.control.dh116.tabletop_vision_sequence_capture import SequenceContextCapture
    t, w = fixture()
    capture = SequenceContextCapture(t, w, require_native_readback=True)
    del t.m['wrist_velocity']
    try:
        with pytest.raises(ValueError, match='wrist_velocity'):
            capture.begin_tick([], torch.zeros(3, 24))
    finally:
        capture.close()


def test_terminal_reset_targets_are_retained_but_excluded_from_shadow_parity(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_vision_sequence_capture import SequenceContextCapture
    t, w = fixture(); capture = SequenceContextCapture(t, w)
    action = torch.full((3, 24), .2)
    metadata = [dict(run_uuid='run', env_id=1, episode_uid='ep1', tick=0,
        policy_timestamp_s=0., first_physics_step=0, terminal=True, previous_action=[0.]*24)]
    capture.begin_tick(metadata, action)
    for substep in range(8):
        live = torch.tensor([True, substep < 3, False])
        allowed = t.control(action, live, w.dt)
        w.advance(t, allowed, live)
        if substep == 2:
            # TensorEnv resets physical command caches immediately after the
            # first terminal, while retaining the task snapshot until tick end.
            w.commands[1] = 0
            w.palms[1] = 0
            w.wrist_angles[1] = 0
            w.roots[1] = w.root_seed[1]
            w.root_targets[1] = w.root_seed[1]
    record = capture.finish_tick(torch.tensor([False, True, False]))
    capture.close()
    assert record['teacher_targets']['finger'][1, 3:].eq(0).all()
    assert t.command[1].abs().sum() > 0
    path = tmp_path/'pool.pt'; path.write_bytes(b'pool')
    result = capture.finalize(path, metadata, tmp_path/'sidecar.pt')
    assert result['teacher_shadow_parity']['passed']
    assert result['teacher_shadow_parity']['checked_physics_substeps'] == 3


@pytest.mark.parametrize('clock_drift', [False, True])
def test_postphysics_feedback_is_owned_and_matches_backend_readback_clock(monkeypatch, clock_drift):
    from go2_pvcnn.control.dh116 import tabletop_vision_sequence_capture as capture_module
    t, w = fixture(); sim = w.w.sim
    sim.current_time_step_index = 20
    refresh = w.refresh
    w.refresh = lambda: dict(refresh(), q=t.m['q']+.005, qd=t.m['qd']+.25)
    original_step = sim.step
    def step(**kwargs):
        original_step(**kwargs)
        sim.current_time_step_index += 1
    sim.step = step
    if clock_drift:
        reader = capture_module.read_joint_target_evidence
        def drifting_reader(hand):
            result = reader(hand); sim.current_time_step_index += 1
            return result
        monkeypatch.setattr(capture_module, 'read_joint_target_evidence', drifting_reader)
    capture = capture_module.SequenceContextCapture(t, w)
    capture.begin_tick([], torch.zeros(3, 24))
    before = t.m['q'].clone()
    allowed = t.control(torch.zeros(3, 24), torch.ones(3, dtype=torch.bool), w.dt)
    returned = w.advance(t, allowed, torch.ones(3, dtype=torch.bool))
    post = capture.active['post_physics_feedback'][0]
    assert torch.equal(post['measured_q_rad'], before+.005)
    assert not torch.equal(post['measured_q_rad'], capture.active['contexts'][0]['measured_q_rad'])
    assert post['sdk_physics_step'] == 21
    assert post['clock_aligned'] is (not clock_drift)
    assert post['source_valid'].tolist() == [not clock_drift]*3
    returned['q'].zero_()
    assert torch.equal(post['measured_q_rad'], before+.005)
    capture.close()
