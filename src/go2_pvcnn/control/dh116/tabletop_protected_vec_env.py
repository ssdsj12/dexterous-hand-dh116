"""Opt-in candidate for guarded mimic-fault episode resets in a shared world.

The default SharedPhysicsVecEnv is unchanged. A failure still terminates its
original episode with its original reward, failure, and terminal observation.
Only a bounded mimic fault with all other guards passing may reset locally.
Remaining shared physics steps are settling after that reset: the new episode
clock stays zero and its returned observation uses the latest measured state.
Validation with zero remaining steps covers only the immediate reset state,
not a subsequent physical step. Its first subsequent physics step must pass
all guards before another mimic reset can be considered. No physics guard or
success criterion changes.
"""
from copy import deepcopy
import numbers
import numpy as np
from .tabletop_vec_env import SharedPhysicsVecEnv as _SharedPhysicsVecEnv
from .tabletop_training import PHYSICAL_FAILURES
from .tabletop_task import joint_safety_failure
from .linear_coupling import MIMIC


class UnrecoverablePhysicsError(RuntimeError):
    def __init__(self, failure, info, env_index):
        self.failure = str(failure)
        self.info = dict(info, global_stop=True, global_stop_failure=self.failure)
        self.env_index = int(env_index)
        self.global_stop = True
        super().__init__(f'environment {env_index}: {failure}; shared physics stopped')


def make_joint_guard(scenes):
    """Check each real hand independently, using the same original guard."""
    def guard(index):
        failures = []
        for hand in scenes[index]['hands']:
            q = hand.data.joint_pos[0].detach().cpu().numpy()
            named = dict(zip(hand.joint_names, q))
            errors = [named[f] - ratio*named[m] - offset for f,m,ratio,offset in MIMIC]
            failures.append(joint_safety_failure(q,
                hand.data.joint_pos_limits[0].detach().cpu().numpy(), errors))
        return failures
    return guard


class ProtectedMimicVecEnv(_SharedPhysicsVecEnv):
    def __init__(self, envs, advance_all, *, joint_guard=None):
        self.joint_guard = joint_guard
        self.global_stop = False
        super().__init__(envs, advance_all)
        self._pending_reset_info = [None] * self.num_envs

    @staticmethod
    def _finite(value):
        if isinstance(value, dict):
            return all(ProtectedMimicVecEnv._finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(ProtectedMimicVecEnv._finite(v) for v in value)
        if isinstance(value, (numbers.Number, np.ndarray)):
            return bool(np.isfinite(value).all())
        return True  # Named links, failure labels and None are not measurements.

    @staticmethod
    def _physical(env):
        def scalar(value):
            return float(value) if np.isfinite(value) else None
        return dict(object_bottom_m=scalar(env.metrics['object_bottom_m']),
                    slip_m=scalar(env.metrics['slip_m']),
                    max_force_n=scalar(env.metrics['max_force_n']),
                    max_mimic_error_rad=scalar(np.max(np.abs(
                        env.measurement.get('mimic_errors', np.zeros(10))))))

    def _checked_state(self, index, *, allow_terminal_mimic=False):
        """Check fresh state without advancing the task or weakening a guard."""
        env = self.envs[index]
        if not self._finite(env.measurement) or not self._finite(env.metrics):
            return 'nonfinite_state', None
        try:
            observation = env.observe()
        except ValueError:
            return 'nonfinite_state', None
        if not np.isfinite(observation).all():
            return 'nonfinite_state', None
        diagnostics = env.measurement.get('diagnostics', [])
        if not diagnostics or any('joint_vel_rad_s' not in h for h in diagnostics):
            return 'missing_joint_velocity_diagnostics', None
        velocities = np.concatenate([np.asarray(env.measurement['qd']).ravel()] +
            [np.asarray(h['joint_vel_rad_s']).ravel() for h in diagnostics])
        if np.max(np.abs(velocities)) > 3.:
            return 'joint_velocity', None
        if self.joint_guard is None:
            return 'missing_independent_joint_guard', None
        failures = list(self.joint_guard(index)) + [env.measurement['joint_failure']]
        for failure in failures:
            if failure and not (allow_terminal_mimic and failure == 'mimic_error'):
                return failure, None
        peak = float(np.max(np.abs(env.measurement.get('mimic_errors', np.zeros(10)))))
        if allow_terminal_mimic:
            if not .02 < peak < .03:
                return 'mimic_error', None
        elif peak > .02:
            return 'mimic_error', None
        # The normal task returns early once joint_failure has been assigned.
        # Re-run its unchanged geometry/load checks on a copy so a simultaneous
        # force/penetration/drop violation cannot hide behind mimic_error.
        task = deepcopy(env.task)
        task.failure = None
        task.success = False
        task.update(env.backend.dt, **env.metrics)
        if task.failure in PHYSICAL_FAILURES:
            return task.failure, None
        return None, observation.copy()

    def step_wait(self):
        if self.global_stop:
            raise RuntimeError('shared physics was globally stopped')
        if self._actions is None:
            raise RuntimeError('no vector step is pending')
        actions, self._actions = self._actions, None
        iterators = [env.step_iter(action) for env, action in zip(self.envs, actions)]
        results = [None] * self.num_envs
        requests = [None] * self.num_envs
        infos = [None] * self.num_envs
        recovered = [False] * self.num_envs
        reset_observations = [None] * self.num_envs

        def abort(index, failure, info):
            self.global_stop = True
            for iterator in iterators:
                iterator.close()
            if info.get('reset_validation'):
                info['reset_validation']['verified'] = False
                info['reset_validation']['passed'] = False
            raise UnrecoverablePhysicsError(failure, info, index)

        def fresh_check(index, *, allow_terminal_mimic=False):
            try:
                self.envs[index]._measure()
                return self._checked_state(index, allow_terminal_mimic=allow_terminal_mimic)
            except (ValueError, FloatingPointError) as error:
                return 'nonfinite_state', None

        def resume(index, substep):
            try:
                requests[index] = next(iterators[index])
                return
            except StopIteration as ended:
                result = ended.value
                results[index] = result
                requests[index] = None
            env = self.envs[index]
            obs, reward, terminated, truncated, original_info = result
            info = dict(original_info, physical=self._physical(env),
                        recovered_mimic=False, global_stop=False)
            info['TimeLimit.truncated'] = bool(truncated and not terminated)
            if terminated or truncated:
                info['terminal_observation'] = obs.copy()
            infos[index] = info
            failure = info.get('safety_failure')
            if not failure:
                return
            if failure != 'mimic_error':
                abort(index, failure, info)
            guard_failure, _ = self._checked_state(index, allow_terminal_mimic=True)
            if guard_failure:
                abort(index, guard_failure, info)
            validation = dict(version='validated_mimic_reset_v1', verified=False,
                passed=False, other_guards_safe=True, reset_guards_safe=False,
                post_reset_guards_safe=False,
                terminal_mimic_abs_rad=info['physical']['max_mimic_error_rad'],
                terminal_physics_substep=substep, checked_physics_substeps=0,
                remaining_physics_substeps=self.decimation-substep,
                validation_scope='immediate_reset_only')
            info['reset_validation'] = validation
            try:
                _, self.reset_infos[index] = env.reset()
            except Exception as error:
                validation['reset_error'] = str(error)
                abort(index, 'reset_failed', info)
            guard_failure, reset_obs = fresh_check(index)
            if guard_failure:
                abort(index, guard_failure, info)
            recovered[index] = True
            # backend.reset has written its seed joint targets and poses. None
            # requests hold those reset targets, never the old policy command.
            reset_observations[index] = reset_obs
            env.last_observation = reset_obs.copy()
            validation.update(reset_guards_safe=True, post_reset_guards_safe=True)

        for i in range(self.num_envs):
            resume(i, 0)
        for substep in range(1, self.decimation + 1):
            self.advance_all(requests)
            for i, pending in enumerate(self._pending_reset_info):
                if pending is not None:
                    # A reset on the previous tick's final substep has not yet
                    # seen physics. Even a bounded mimic recurrence here is
                    # failed reset validation, not another recoverable episode.
                    failure, _ = fresh_check(i)
                    if failure:
                        pending['first_post_reset_physics'] = dict(passed=False, failure=failure)
                        pending['reset_validation']['post_reset_guards_safe'] = False
                        abort(i, failure, pending)
                    self._pending_reset_info[i] = None
            for i in range(self.num_envs):
                if recovered[i]:
                    validation = infos[i]['reset_validation']
                    validation['checked_physics_substeps'] += 1
                    failure, obs = fresh_check(i)
                    if failure:
                        validation['post_reset_guards_safe'] = False
                        abort(i, failure, infos[i])
                    validation['validation_scope'] = 'reset_and_remaining_substeps'
                    reset_observations[i] = obs
                    self.envs[i].last_observation = obs.copy()
                elif results[i] is None:
                    resume(i, substep)
                else:
                    # Success/timeout slots also remain in the shared world.
                    # Check their evolving state, retaining the original terminal
                    # observation/metrics and never recursively recovering here.
                    failure, _ = fresh_check(i)
                    if failure:
                        abort(i, failure, infos[i])
        if any(result is None for result in results):
            raise RuntimeError('environment requested more than one control tick of physics')

        observations, rewards, dones = [], [], []
        for i, result in enumerate(results):
            obs, reward, terminated, truncated, _ = result
            done = bool(terminated or truncated)
            if recovered[i]:
                obs = reset_observations[i]
                infos[i]['recovered_mimic'] = True
                infos[i]['reset_validation'].update(verified=True, passed=True)
                if infos[i]['reset_validation']['checked_physics_substeps'] == 0:
                    # Keep independent evidence; later validation must never
                    # mutate a terminal transition already returned to SB3.
                    self._pending_reset_info[i] = deepcopy(infos[i])
            elif done:
                # Ordinary task termination keeps the existing SB3 reset rule.
                try:
                    obs, self.reset_infos[i] = self.envs[i].reset()
                except Exception as error:
                    infos[i]['reset_error'] = str(error)
                    abort(i, 'reset_failed', infos[i])
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
        return (np.stack(observations), np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=bool), infos)
