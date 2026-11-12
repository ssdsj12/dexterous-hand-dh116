from types import SimpleNamespace

import pytest
import torch


def _task():
    n = 2

    def value(width, raw):
        return torch.full((n, width) if width else (n,), raw, dtype=torch.float64)

    t = SimpleNamespace(surface_force_sharing=True, free_wrist_grasp=True,
        contact_reward_window='close_start', force_reward_start='downward_after_close',
        reference=value(12, .125), low=value(12, -1)[0], high=value(12, 1)[0],
        elapsed=value(None, 1.5), contact_time=value(None, 3), success_time=value(None, 4.5),
        close_progress=value(None, .25), descent_stop=2, grip=value(3, .025),
        grip_valid=torch.tensor([True, False]), filtered_residual=value(12, .0125),
        f_last_slip=value(None, .004), contact_reward_clock=value(None, .25),
        grasp_ready_awarded=torch.tensor([True, False]),
        force_scoring_started=torch.tensor([False, True]),
        rotation_targets=value(6, .25*torch.pi).reshape(n, 2, 3),
        surface_feedback=SimpleNamespace(_started=False,
            _sum=value(20, 0), _elapsed=value(None, 0), _depth=value(None, 0),
            _drift=value(None, 0), _hold=value(None, 0), _start_z=value(2, 0),
            _end_z=value(2, 0), _window_good=torch.zeros(n,dtype=torch.bool),
            _valid=torch.zeros(n,dtype=torch.bool), _open=torch.zeros(n,dtype=torch.bool),
            _ok=torch.zeros(n,dtype=torch.bool)))
    for key in ('f_offset', 'f_filtered', 'f_target', 'f_boost', 'f_requested'):
        setattr(t, key, value(2, .5))
    t.parameters=SimpleNamespace(max_force_n=10.)
    return t


def _observation():
    obs=torch.zeros((2,371),dtype=torch.float32)
    obs[:,36:48]=.125
    obs[:,52:55]=torch.tensor([.1,.2,.3])
    obs[:,56]=.25
    obs[:,59]=1
    obs[:,74:77]=.5
    obs[0,77]=1
    obs[:,88:100]=.125
    obs[:,131]=.2
    obs[:,144]=.25
    obs[0,145]=1
    obs[1,146]=1
    obs[:,185:191]=.5
    return obs


def test_independent_state_values_are_cloned_and_force_is_kept_separate():
    from go2_pvcnn.control.dh116.tabletop_mpc_internal_source_capture import capture_internal_sources, internal_source_masks

    task = _task()
    snapshot = capture_internal_sources(task, 2, after_reset=True)
    task.reference.fill_(1)
    obs = _observation()
    obs[0, 36] = .375
    proof = internal_source_masks(obs, snapshot)
    values, reset = proof['internal_state_value_valid'], proof['reset_feedback_initialization_value_valid']
    assert values.sum(1).tolist() == [44, 42]
    assert not values[0,36] and values[1,36]
    assert values[0,74:78].all() and values[1,77] and not values[1,74:77].any()
    assert not values[:,121:131].any()
    assert snapshot['raw_task']['f_filtered'][0,0] == .5
    assert reset[:,367:371].all() and not values[:,367:371].any()


@pytest.mark.parametrize('fault', ['step_event', 'feedback_valid', 'accumulator', 'finished', 'echo_nonzero'])
def test_reset_sentinel_requires_actual_reset_accumulators_and_separate_mask(fault):
    from go2_pvcnn.control.dh116.tabletop_mpc_internal_source_capture import capture_internal_sources, internal_source_masks

    task=_task()
    obs=_observation()
    if fault=='feedback_valid':
        task.surface_feedback._valid.fill_(True)
    elif fault=='accumulator':
        task.surface_feedback._sum[:,17]=.001
    elif fault=='finished':
        task.surface_feedback._started=True
    elif fault=='echo_nonzero':
        obs[:,367:371]=.25
    snapshot=capture_internal_sources(task,2,after_reset=fault!='step_event')
    result=internal_source_masks(obs,snapshot)
    assert not result['reset_feedback_initialization_value_valid'].any()
    assert not result['internal_state_value_valid'][:,351:].any()


def test_optional_capture_reads_original_pre_action_boundaries_and_restores_hooks():
    from go2_pvcnn.control.dh116.tabletop_mpc_privileged_capture import PreActionCapture, capture_environment

    class Env:
        def __init__(self):
            self.task = _task()
            self.task.m=dict(q=torch.zeros(2,12),qd=torch.zeros(2,12))
            self.step_id=0
        def clock(self):
            step=self.step_id
            return dict(world_step=step, sdk_step=18+step, world_time_s=step/480, sdk_time_s=(18+step)/480)
        def reset(self):
            return _observation()
        def step(self, action):
            self.step_id+=8
            self.task.reference+=.125
            obs=_observation()
            obs[:,36:48]=.25
            return obs, None, None, {}
    env=Env()
    recorder=PreActionCapture(2,2,capture_internal=True)
    issued=torch.zeros(2,24)
    with capture_environment(env,recorder,env.clock):
        env.reset()
        env.step(issued)
        env.step(issued)
    assert len(recorder.samples)==2 and env.step_id==16
    first, second=recorder.samples
    assert first.internal_source['after_reset'] and not second.internal_source['after_reset']
    assert first.internal_source['raw_task']['reference'][0,0] == .125
    assert second.internal_source['raw_task']['reference'][0,0] == .25
    assert first.action.equal(issued) and second.action.equal(issued)
    assert 'reset' not in vars(env) and 'step' not in vars(env)


def test_incompatible_task_contract_is_rejected_before_instrumentation():
    from go2_pvcnn.control.dh116.tabletop_mpc_internal_source_capture import capture_internal_sources

    task=_task()
    task.free_wrist_grasp=False
    with pytest.raises(ValueError,match='canonical'):
        capture_internal_sources(task,2,after_reset=True)


def test_371_native_layout_uses_same_internal_sources_without_force_admission(tmp_path):
    from phase0_state_contract_fixture import cpu_task
    from test_mpc_privileged_capture import clock, make_evidence
    from go2_pvcnn.control.dh116.tabletop_mpc_privileged_capture import PreActionCapture, join_execution

    task = cpu_task()
    task.surface_feedback.reset(torch.arange(task.n))
    obs = task.observe()
    assert obs.shape == (2,371)
    recorder = PreActionCapture(2,1,capture_internal=True)
    recorder.observe(obs,task,clock(16),after_reset=True)
    action=torch.full((2,24), .125)
    sample=recorder.prepare(action,clock(16))
    path=tmp_path/'states_480hz.jsonl.gz'
    make_evidence(path,sample.clock,action)
    visual=dict(actions=action[:1],reference_actions=torch.zeros((1,24)),
        sample_metadata=[dict(env_id=0, episode_id=0, tick=0, frame_id=7, physics_step=18,
                              simulation_time_s=18/480, grasp_style='pinch_thumb_index')])
    archive=join_execution([sample],visual,path,run_uuid='recorded-run')
    assert archive['observations'][0].equal(obs[0])
    assert archive['privileged_valid'][0].sum()==24
    assert archive['internal_state_value_valid'][0].sum()==45
    assert archive['reset_feedback_initialization_value_valid'][0,367:371].all()
    assert archive['internal_source_records'][0]['raw_task']['reference'].equal(task.reference[0])
    assert archive['force_contact_qualification'] is False and archive['qualified_for_bc'] is False
