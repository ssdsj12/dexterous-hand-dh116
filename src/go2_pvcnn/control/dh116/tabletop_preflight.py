"""Separate safe learning admission from explicit contact diagnostics."""
import time
import numpy as np


def contact_preflight(vec, world, *, direct, progress=None):
    count = vec.num_envs
    ticks = 2 if direct else 120
    peak = np.zeros((count,2))
    started = time.monotonic()
    for tick in range(ticks):
        obs,rewards,dones,infos = vec.step(np.zeros((count,20),dtype=np.float32))
        if (not np.isfinite(obs).all() or not np.isfinite(rewards).all()
                or not np.isfinite(world.force).all()
                or any(info.get('safety_failure') for info in infos)):
            raise ValueError('GPU contact preflight physical failure')
        peak = np.maximum(peak,world.force.reshape(count,2,2).sum(axis=2))
        if progress is not None and (tick % 4 == 0 or tick + 1 == ticks):
            progress(dict(status='startup_preflight' if direct else 'contact_preflight',
                num_envs=count,timesteps=0,updates=0,preflight_ticks=tick+1,
                preflight_total_ticks=ticks,
                simulation_transitions_per_second=(tick+1)*count/(time.monotonic()-started),
                hands_with_contact=int(np.sum(peak>.01)),total_hands=2*count,safety_stop=None))
    report = dict(elapsed_seconds=time.monotonic()-started,vector_steps=ticks,
        mode='finite_bounded_exploration_contact_not_required' if direct else 'fixed_120_steps',
        simulation_transitions_per_second=ticks*count/(time.monotonic()-started),
        peak_hand_load_n=peak.tolist(),all_hands_detect_contact=bool(np.all(peak>.01)))
    if not direct and not report['all_hands_detect_contact']:
        raise ValueError('GPU contact preflight failed: missing measured hand/object reaction')
    return report
