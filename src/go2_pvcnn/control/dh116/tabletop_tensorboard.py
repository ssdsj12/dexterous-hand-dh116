"""Live measured training summaries, separate from SB3's rollout logger."""
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class LiveTrainingWriter:
    def __init__(self, directory):
        self.writer = SummaryWriter(str(directory), flush_secs=5)
        self.last_write = -float('inf')

    def write(self, step, infos, rewards, per_env, failure, *, force=False):
        now = time.monotonic()
        if not force and now-self.last_write < 2.:
            return
        self.last_write = now
        values = {'safety_stop': float(bool(failure)),
                  'completed_episodes_total': sum(r['episodes'] for r in per_env)}
        completed = values['completed_episodes_total']
        if completed:
            values['success_rate'] = sum(r['successes'] for r in per_env)/completed
            values['episode_return_mean'] = sum(r['return_sum'] for r in per_env)/completed
        if rewards is not None:
            rewards = np.asarray(rewards, dtype=float)
            valid = np.isfinite(rewards)
            values['rewards_finite_fraction'] = float(valid.mean())
            if valid.any(): values['reward_mean'] = float(rewards[valid].mean())
        if infos:
            for phase in ('close', 'hold', 'lift'):
                values[f'phase_{phase}_fraction'] = sum(i.get('phase') == phase for i in infos)/len(infos)
            loads = np.asarray([i.get('force_feedback', {}).get('measured_hand_load_n', [np.nan, np.nan]) for i in infos], dtype=float)
            if loads.shape == (len(infos), 2):
                values['hand_load_finite_fraction'] = float(np.isfinite(loads).mean())
                for column, hand in enumerate(('left', 'right')):
                    valid = np.isfinite(loads[:, column])
                    if valid.any(): values[f'hand_load_{hand}_n'] = float(loads[valid, column].mean())
        for name, value in values.items():
            if np.isfinite(value): self.writer.add_scalar('live/'+name, value, step)
        self.writer.flush()

    def close(self):
        self.writer.close()


def finite_diagnostics(value):
    """Keep absent/non-finite diagnostics explicit without invalid JSON."""
    if isinstance(value, dict): return {k: finite_diagnostics(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [finite_diagnostics(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
    if isinstance(value, np.generic): return value.item()
    return value
