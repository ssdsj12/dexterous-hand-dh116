"""SB3 adapter for batched physics and equivalent batched host task logic."""
from contextlib import contextmanager

from .tabletop_vec_env import SharedPhysicsVecEnv
from .tabletop_batched_task import BatchTask


class GpuTaskVecEnv(SharedPhysicsVecEnv):
    def __init__(self,envs,world):
        self.world=world
        super().__init__(envs,world.advance)

    @contextmanager
    def _batch_reset(self, indices):
        """Prepare deterministic physical resets; retain normal task/RNG reset.

        Flush old episode records before changing physical state. Each backend
        consumes its prepared reset once, and cleanup also covers failed resets.
        Jittered resets retain their original per-slot RNG/physics ordering.
        """
        indices = list(indices)
        if not indices or any(self.envs[i].reset_jitter != 0 for i in indices):
            yield
            return
        for i in indices:
            self.envs[i].flush_episode()
        offsets = [self.envs[i].initial_offset for i in indices]
        self.world.reset_slots(indices, offsets)
        try:
            for i, offset in zip(indices, offsets):
                self.envs[i].backend._prepared_reset_offset = offset
            yield
        finally:
            for i in indices:
                self.envs[i].backend.__dict__.pop('_prepared_reset_offset', None)

    def reset(self):
        with self._batch_reset(range(self.num_envs)):
            return super().reset()

    def finish_results(self, results):
        indices = [i for i, result in enumerate(results) if result[2] or result[3]]
        # Physical writes leave each env's captured terminal measurement and
        # task state intact; SharedPhysicsVecEnv records them before env.reset.
        with self._batch_reset(indices):
            return super().finish_results(results)

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError('no vector step is pending')
        actions,self._actions=self._actions,None
        task=BatchTask(self.envs)
        def advance(offset,command,allowed,live):
            self.world.advance_batch(offset,command,allowed,live,task.palm_actions)
            return self.world.measure_batch()
        results=task.step(actions,advance)
        for i,b in enumerate(self.world.backends):
            b.adaptive_palm_offset=self.world.palms[i].copy()
            b.grip_stiffness=self.world.gains[i].copy()
        return self.finish_results(results)
