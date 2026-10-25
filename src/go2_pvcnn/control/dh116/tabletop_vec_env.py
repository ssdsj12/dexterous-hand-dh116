"""Synchronous SB3 environments sharing one physical scene and one clock.

No subprocesses or per-environment physics stepping are used. ``advance_all``
accepts a list of ``(offset, command, kwargs)`` requests (or None for a slot
already done), applies commands, steps the world once, and refreshes all
backend measurements. Each backend reset must only reset its own slot and
must never step the world; reset/settling belongs to the scene initializer.
"""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.vec_env import VecEnv


class SharedPhysicsVecEnv(VecEnv):
    def __init__(self, envs, advance_all):
        self.envs = list(envs)
        if not self.envs or len({id(env) for env in self.envs}) != len(self.envs):
            raise ValueError('at least one distinct environment per slot required')
        first = self.envs[0]
        self.decimation = first.backend.decimation
        if (not isinstance(self.decimation, int) or self.decimation < 1
                or any(env.backend.dt != first.backend.dt or
                       env.backend.decimation != self.decimation for env in self.envs)):
            raise ValueError('all environments must share one physics clock')
        if any(env.observation_space != first.observation_space or
               env.action_space != first.action_space for env in self.envs):
            raise ValueError('all environments must share observation and action spaces')
        self.advance_all = advance_all
        self._actions = None
        self._closed = False
        super().__init__(len(self.envs), first.observation_space, first.action_space)

    def reset(self):
        self._actions = None
        observations = []
        for i, env in enumerate(self.envs):
            obs, self.reset_infos[i] = env.reset(seed=self._seeds[i], options=self._options[i])
            observations.append(obs)
        self._reset_seeds()
        self._reset_options()
        return np.stack(observations)

    def step_async(self, actions):
        if self._actions is not None:
            raise RuntimeError('a vector step is already pending')
        actions = np.asarray(actions)
        expected = (self.num_envs,) + self.action_space.shape
        if actions.shape != expected:
            raise ValueError(f'action shape must be {expected}, got {actions.shape}')
        self._actions = actions.copy()

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError('no vector step is pending')
        actions, self._actions = self._actions, None
        iterators = [env.step_iter(action) for env, action in zip(self.envs, actions)]
        results = [None] * self.num_envs
        requests = [None] * self.num_envs

        def resume(i):
            try:
                requests[i] = next(iterators[i])
            except StopIteration as ended:
                results[i] = ended.value
                requests[i] = None

        for i in range(self.num_envs):
            resume(i)
        for _ in range(self.decimation):
            # Completed slots keep their terminal samples, even as the world
            # continues stepping for other slots. Reset only at tick end.
            self.advance_all(requests)
            for i in range(self.num_envs):
                if results[i] is None:
                    resume(i)
        if any(result is None for result in results):
            raise RuntimeError('environment requested more than one control tick of physics')

        return self.finish_results(results)

    def finish_results(self,results):
        """Capture terminal state and reset only after the shared physics tick."""
        observations, rewards, dones, infos = [], [], [], []
        for i, result in enumerate(results):
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
            info = dict(info)
            info['TimeLimit.truncated'] = bool(truncated and not terminated)
            env = self.envs[i]
            # These are the measured transition metrics, including the early
            # terminal substep. Capture them before auto-reset replaces state.
            info['physical'] = dict(
                object_bottom_m=float(env.metrics['object_bottom_m']),
                slip_m=float(env.metrics['slip_m']),
                max_force_n=float(env.metrics['max_force_n']),
                max_mimic_error_rad=float(np.max(np.abs(
                    env.measurement.get('mimic_errors', np.zeros(10))))))
            if done:
                info['terminal_observation'] = obs.copy()
                obs, self.reset_infos[i] = self.envs[i].reset()
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return (np.stack(observations), np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=bool), infos)

    def close(self):
        if not self._closed:
            for env in self.envs:
                env.flush_episode()
                env.close()
            self._closed = True

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.envs[i], attr_name) for i in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        for i in self._get_indices(indices):
            setattr(self.envs[i], attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self.envs[i], method_name)(*method_args, **method_kwargs)
                for i in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [is_wrapped(self.envs[i], wrapper_class) for i in self._get_indices(indices)]
