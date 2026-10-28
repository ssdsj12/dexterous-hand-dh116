"""Inference-only playback using the unchanged training environment."""
import json
import time
from pathlib import Path

import numpy as np

from .tabletop_training import canonical_contract, load_checkpoint, write_json


def load_playback_policy(path, env, contract, *, device='cpu', allow_material_change=False):
    """Only the displayed environment count may differ from the saved run."""
    previous = json.loads(Path(path).with_suffix('.json').read_text())['contract']
    previous, current = canonical_contract(previous), canonical_contract(contract)
    if env.num_envs != 1 or current.get('num_envs') != 1:
        raise ValueError('policy playback requires exactly one environment')
    comparison = dict(current)
    override = None
    if allow_material_change:
        from .tabletop_material import material_contract
        requested = current.get('object_contact_material')
        supported = [material_contract(p).get('object_contact_material')
                     for p in ('default', 'high_friction', 'rough_rod', 'extreme_grip')]
        if requested not in supported:
            raise ValueError('unsupported playback material contract')
        override = dict(source=previous.get('object_contact_material'), current=requested,
                        scope='inference_comparison_only_not_training_resume')
        comparison.pop('object_contact_material', None)
        if 'object_contact_material' in previous:
            comparison['object_contact_material'] = previous['object_contact_material']
    if dict(previous, num_envs=1) != comparison:
        raise ValueError('playback observation/action/reward/physics contract differs')
    model, metadata = load_checkpoint(path, None, previous, device=device)
    from stable_baselines3.common.utils import check_for_correct_spaces
    check_for_correct_spaces(env, model.observation_space, model.action_space)
    if override is not None:
        metadata = dict(metadata, playback_material_override=override)
    return model, metadata


def play_policy(env, app, checkpoint, contract, output, *, device='cuda:0', seed=42,
                episodes=0, deterministic=True, allow_material_change=False):
    model, metadata = load_playback_policy(checkpoint, env, contract, device=device, allow_material_change=allow_material_change)
    env.env_method('set_mode', 'evaluation')
    env.seed(seed)
    observation = env.reset()
    world = env.world
    initial_q = world.q.copy()
    started = time.monotonic()
    report = dict(status='policy_playback', policy_executed=True, training=False,
        checkpoint=str(checkpoint), checkpoint_timesteps=metadata['timesteps'],
        checkpoint_updates=metadata['updates'], deterministic=deterministic,
        num_envs=1, policy_steps=0, completed_episodes=0, successes=0,
        max_joint_change_rad=0., episode_results=[], grasp_qualified=False)
    if 'playback_material_override' in metadata:
        report['playback_material_override'] = metadata['playback_material_override']
    from .tabletop_playback_panel import PlaybackPanel
    panel = PlaybackPanel()
    report.update(playback_paused=False, manual_replays=0, controls_window='Playback Controls')
    write_json(Path(output)/'monitor.json', report)
    while app.is_running() and (episodes == 0 or report['completed_episodes'] < episodes):
        if panel.take_replay():
            world.sim.play()
            env.seed(seed)
            observation = env.reset()
            report['manual_replays'] += 1
            report['phase'] = 'close'
            write_json(Path(output)/'monitor.json', report)
        if panel.paused != report['playback_paused']:
            report['playback_paused'] = panel.paused
            write_json(Path(output)/'monitor.json', report)
        panel.update(report.get('phase', 'close'), report['policy_steps'])
        if panel.paused:
            if world.sim.is_playing():
                world.sim.pause()
            app.update()
            time.sleep(.01)
            continue
        if not world.sim.is_playing():
            world.sim.play()
        action, _ = model.predict(observation, deterministic=deterministic)
        observation, rewards, dones, infos = env.step(action)
        info = infos[0]
        report['policy_steps'] += 1
        report.update(phase=info.get('phase'), physical=info.get('physical'),
            force_feedback=info.get('force_feedback'), last_action=action[0].tolist(),
            last_reward=float(rewards[0]), elapsed_seconds=time.monotonic()-started,
            max_joint_change_rad=max(report['max_joint_change_rad'],
                                     float(np.max(np.abs(world.q-initial_q)))))
        if info.get('safety_failure') or not np.isfinite(observation).all():
            report.update(status='playback_safety_stop', safety_stop=info.get('safety_failure') or 'nonfinite_observation')
            write_json(Path(output)/'monitor.json', report)
            raise RuntimeError(str(report['safety_stop']))
        if dones[0]:
            report['completed_episodes'] += 1
            report['successes'] += int(bool(info.get('is_success')))
            report['episode_results'].append(dict(episode=report['completed_episodes'],
                success=bool(info.get('is_success')), failure=info.get('failure'),
                reached_phases=info.get('reached_phases'), physical=info.get('physical')))
            report['episode_results'] = report['episode_results'][-20:]
        if report['policy_steps'] % 5 == 0 or dones[0]:
            write_json(Path(output)/'monitor.json', report)
    panel.close()
    report['status'] = 'playback_closed'
    write_json(Path(output)/'monitor.json', report)
    return report
