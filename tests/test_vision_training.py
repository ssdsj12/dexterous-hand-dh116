from types import SimpleNamespace
import json
import torch
from go2_pvcnn.control.dh116.tabletop_vision_training import run_vision_training


class Camera:
    def __init__(self, n): self.n=n
    def sample(self):
        return dict(points=torch.ones(self.n,2,128,3)*.05,
            point_valid=torch.ones(self.n,128,dtype=torch.bool),
            image_valid=torch.ones(self.n,dtype=torch.bool),
            object_valid=torch.ones(self.n,dtype=torch.bool),
            diagnostics={'visible_points_mean':128.})


def fixture(tmp_path, maximum=2, resume=None):
    n=2
    roots=torch.zeros(n,2,7,dtype=torch.float64);roots[...,3]=1.
    dw=SimpleNamespace(n=n,device=torch.device('cpu'),q=torch.zeros(n,2,6,dtype=torch.float64),
        qd=torch.zeros(n,2,6,dtype=torch.float64),roots=roots,
        origins=torch.zeros(n,3,dtype=torch.float64),
        w=SimpleNamespace(ids=[list(range(6)),list(range(6))]))
    task=SimpleNamespace(m={'bottom':torch.ones(n,dtype=torch.float64)*.36},
        surface_reference_bottom_m=torch.ones(n,dtype=torch.float64)*.36)
    class Env:
        def __init__(self):
            self.episodes=torch.zeros(n,dtype=torch.long)
            self.successes=torch.zeros(n,dtype=torch.long)
            self.tick=0
        def reset(self): return torch.zeros(n,371)
        def step(self,action):
            self.tick+=1
            done=torch.full((n,),self.tick%3==0,dtype=torch.bool)
            self.episodes+=done.long()
            return torch.ones(n,371)*self.tick,1-action.square().mean(-1),done,{'failure':torch.zeros(n,dtype=torch.long)}
    cfg=tmp_path/'camera.json';cfg.write_text(json.dumps({'test':'calibration'}))
    seed=tmp_path/'seed.json';seed.write_text('{}')
    args=SimpleNamespace(output=tmp_path/'out',num_envs=n,device='cpu',rollout_steps=2,
        max_iterations=maximum,resume=resume,seed=42,depth_camera_config=cfg,
        grasp_initialization=seed,mixed_grasp_initialization=seed,
        vision_width=160,vision_height=120,surface_reward_profile='support_deficit')
    args.output.mkdir(exist_ok=True)
    return dw,task,Env(),args


def test_visual_training_has_distinct_policy_contract_and_atomic_checkpoint(tmp_path):
    torch.set_num_threads(1)
    dw,task,env,args=fixture(tmp_path)
    result=run_vision_training(SimpleNamespace(),dw,task,env,args,{'observation_size':371},camera=Camera(2))
    assert result['status']=='training_budget_complete'
    saved=torch.load(args.output/'vision_student_ppo.pt',weights_only=False)
    assert saved['iteration']==2
    assert saved['contract']['actor_observation_size']==3844
    assert saved['contract']['critic_observation_size']==371
    assert saved['contract']['teacher_distillation']=='disabled_without_synchronized_visual_labels'
    assert saved['contract']['actor_uses_privileged_state'] is False
    assert not (args.output/'vision_student_ppo.pt.tmp').exists()
    progress=json.loads((args.output/'collection_status.json').read_text())
    assert progress['iteration']==2 and progress['rollout_step']==2
    assert progress['phase']=='optimizer_complete'



def test_visual_resume_retains_iteration_and_rejects_legacy_checkpoint(tmp_path):
    torch.set_num_threads(1)
    dw,task,env,args=fixture(tmp_path)
    run_vision_training(SimpleNamespace(),dw,task,env,args,{'observation_size':371},camera=Camera(2))
    checkpoint=args.output/'vision_student_ppo.pt'
    args.resume=checkpoint;args.max_iterations=3
    args.output=tmp_path/'resume';args.output.mkdir()
    result=run_vision_training(SimpleNamespace(),dw,task,env,args,{'observation_size':371},camera=Camera(2))
    assert result['iteration']==3

class SensorOnlyWorld:
    """Reject accidental native object/contact lookups from the actor adapter."""
    def __init__(self, reference):
        object.__setattr__(self, '_values', {
            name: getattr(reference, name)
            for name in ('n', 'q', 'qd', 'roots', 'origins', 'w')
        })

    def __getattribute__(self, name):
        values = object.__getattribute__(self, '_values')
        if name not in values:
            raise AssertionError(f'actor adapter requested non-sensor native field {name}')
        return values[name]


def test_observation_adapter_reads_only_encoders_fk_and_ignores_privileged_critic(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_vision_student import SensorHistory
    from go2_pvcnn.control.dh116.tabletop_vision_training import _observations

    dw, task, env, args = fixture(tmp_path)
    sensor_world = SensorOnlyWorld(dw)
    packet = Camera(2).sample()
    previous = torch.zeros(2, 24)
    first = _observations(packet, sensor_world, previous, SensorHistory(2),
                          torch.zeros(2, 371))
    changed = _observations(packet, sensor_world, previous, SensorHistory(2),
                            torch.randn(2, 371) * 1000.)
    torch.testing.assert_close(first['policy'], changed['policy'], atol=0., rtol=0.)
    assert not torch.equal(first['critic'], changed['critic'])


def test_visual_resume_rejects_a_legacy_privileged_policy_checkpoint(tmp_path):
    import pytest

    dw, task, env, args = fixture(tmp_path)
    legacy = tmp_path / 'legacy.pt'
    torch.save(dict(version='dh116-rsl-tensor-ppo-v1', contract={}), legacy)
    args.resume = legacy
    with pytest.raises(ValueError, match='visual resume requires'):
        run_vision_training(SimpleNamespace(), dw, task, env, args,
                            {'observation_size': 371}, camera=Camera(2))


def test_group_lift_retains_pre_reset_peak_from_surface_diagnostics(tmp_path):
    import pytest

    torch.set_num_threads(1)
    dw, task, env, args = fixture(tmp_path, maximum=1)
    def step(action):
        env.tick += 1
        done = torch.ones(2, dtype=torch.bool)
        env.episodes += 1
        # Terminal-step diagnostics precede the native physical/task reset.
        net = torch.tensor([.042, .063]) if env.tick == 1 else torch.tensor([.002, .003])
        task.m['bottom'] = task.surface_reference_bottom_m.clone()
        return (torch.zeros(2, 371), torch.ones(2), done,
                dict(failure=torch.zeros(2, dtype=torch.long),
                     surface_diagnostics=dict(net_lift_m=net)))
    env.step = step
    run_vision_training(SimpleNamespace(), dw, task, env, args,
                        {'observation_size': 371}, camera=Camera(2))
    first = json.loads((args.output / 'iterations.jsonl').read_text().splitlines()[0])
    groups = first['grasp_styles']
    assert groups['pinch_thumb_index']['net_rod_lift_max_m'] == pytest.approx(.042)
    assert groups['tripod_thumb_index_middle']['net_rod_lift_max_m'] == pytest.approx(.063)


def test_resume_at_completed_budget_publishes_a_real_checkpoint(tmp_path):
    from pathlib import Path

    torch.set_num_threads(1)
    dw, task, env, args = fixture(tmp_path, maximum=1)
    run_vision_training(SimpleNamespace(), dw, task, env, args,
                        {'observation_size': 371}, camera=Camera(2))
    source = args.output / 'vision_student_ppo.pt'
    args.resume = source
    args.output = tmp_path / 'already_complete'
    args.output.mkdir()
    result = run_vision_training(SimpleNamespace(), dw, task, env, args,
                                {'observation_size': 371}, camera=Camera(2))
    assert result['status'] == 'training_budget_complete'
    assert Path(result['checkpoint']).is_file()
    saved = torch.load(result['checkpoint'], weights_only=True)
    assert saved['iteration'] == 1

@__import__('pytest').mark.parametrize('requested,resolved', [(None, 24), (32, 32)])
def test_visual_rollout_default_is_resolved_without_mutating_caller(tmp_path, requested, resolved, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    torch.set_num_threads(1)
    dw, task, env, args = fixture(tmp_path, maximum=1)
    args.rollout_steps = requested
    result = run_vision_training(SimpleNamespace(), dw, task, env, args,
                                {'observation_size': 371}, camera=Camera(2))
    assert args.rollout_steps is requested
    assert result['timesteps'] == args.num_envs * resolved
    saved = torch.load(args.output / 'vision_student_ppo.pt', weights_only=True)
    assert saved['contract']['rollout_steps'] == resolved
    row = json.loads((args.output / 'monitor.json').read_text())
    assert row['rollout_steps'] == resolved


def test_native_initial_geometry_audit_uses_configured_cylinder_dimensions(tmp_path, monkeypatch):
    from go2_pvcnn.control.dh116 import tabletop_vision_audit as audit
    from go2_pvcnn.control.dh116 import tabletop_vision_batched_camera as batched
    from go2_pvcnn.control.dh116 import tabletop_vision_memory as memory

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    torch.set_num_threads(1)
    dw, task, env, args = fixture(tmp_path, maximum=1)
    args.solid_cylinder_radius_m = .013
    calls = []
    def audited(packet, device_world, **kwargs):
        assert device_world is dw
        assert packet['points'].shape == (2, 2, 128, 3)
        calls.append(kwargs)
        return dict(validation='test_native_audit_hook', **kwargs)
    monkeypatch.setattr(batched, 'BoundedVisionCamera', lambda *unused, **kwargs: Camera(2))
    monkeypatch.setattr(memory, 'memory_snapshot', lambda *unused, **kwargs: None)
    monkeypatch.setattr(audit, 'audit_initial_points', audited)
    run_vision_training(SimpleNamespace(), dw, task, env, args,
                        {'observation_size': 371})
    assert calls == [dict(radius=.013, length=.5)]
    report = json.loads((args.output / 'initial_camera_geometry_validation.json').read_text())
    assert report['radius'] == .013 and report['length'] == .5
