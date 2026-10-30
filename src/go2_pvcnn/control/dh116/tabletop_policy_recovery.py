"""Bounded collection recovery for the feedforward RSL-RL 3.1.3 PPO path.

Never repair policy weights or retain temporal credit across a quarantine tick.
Only a complete fresh rollout may reach compute_returns/update.
"""
import torch
from tensordict import TensorDict


CONTRACT = dict(version='discard_partial_policy_rollout_v1',
    invalid_policy_row='forward_nonfinite_action_or_nan_marker_to_env_isolated_reset',
    reset_validation='finite_observation_reward_and_done_recovered_ack_for_each_bad_row',
    storage='clear_entire_partial_rollout_and_pending_transition_before_reset',
    retry='one_per_slot_until_a_complete_fresh_finite_rollout_succeeds',
    invalid_packet='one_reset_then_validate_first_ordinary_result_per_slot',
    persistent_policy_fault='abort_without_repairing_weights_or_optimizer',
    temporal_credit='never_cross_discarded_collection_gap')


def finite_rows(value, n, device):
    if isinstance(value, (dict, TensorDict)):
        result = torch.ones(n, dtype=torch.bool, device=device)
        for item in value.values():
            result &= finite_rows(item, n, device)
        return result
    if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != n:
        raise RuntimeError('invalid policy transition shape')
    return torch.isfinite(value).reshape(n, -1).all(1)


class PolicyActionRecovery:
    def __init__(self, counts=None):
        self.counts = dict(env_control_steps=0, discarded_control_steps=0,
            discarded_transitions=0, discarded_rollouts=0, recoveries=0, recovered_rows=0)
        if counts is not None:
            for name in self.counts:
                value = counts.get(name, 0)
                if not isinstance(value, int) or value < 0:
                    raise ValueError('invalid policy recovery counter: '+name)
                self.counts[name] = value
        self._attempt_steps = 0

    @staticmethod
    def _clear(algorithm):
        algorithm.storage.clear()
        algorithm.transition.clear()
        # RSL 3.1.3 Transition.__init__ only annotates this field: clear() leaves
        # an existing log-prob tensor attached unless explicitly overwritten.
        algorithm.transition.actions_log_prob = None
        algorithm.policy.distribution = None

    def _discard(self, algorithm, n):
        if self._attempt_steps:
            self.counts['discarded_control_steps'] += self._attempt_steps
            self.counts['discarded_transitions'] += self._attempt_steps*n
            self.counts['discarded_rollouts'] += 1
        self._attempt_steps = 0
        self._clear(algorithm)

    def _advance(self, env, action, before_step, failure_counts):
        self.counts['env_control_steps'] += 1
        self._attempt_steps += 1
        if before_step is not None:
            before_step(self.counts['env_control_steps'])
        result = env.step(action)
        # Count actual episode failures once at the common return point, even
        # if this tick or its entire collection attempt is later discarded.
        _, _, done, extra = result
        failure = extra.get('failure')
        n = action.shape[0]
        if (isinstance(done, torch.Tensor) and done.shape == (n,)
                and isinstance(failure, torch.Tensor) and failure.shape == (n,)):
            valid = torch.isfinite(done) & (done == 1) & torch.isfinite(failure)
            valid &= (failure > 0) & (failure < len(failure_counts))
            codes = failure[valid]
            codes = codes[codes == codes.long()].long()
            failure_counts += torch.bincount(codes, minlength=len(failure_counts))
        return result

    def _accept_recovery(self, algorithm, result, bad):
        n, device = len(bad), bad.device
        self._clear(algorithm)
        try:
            obs = self._validate_result(result, n, device, bad)
        finally:
            self._discard(algorithm, n)
        self.counts['recoveries'] += 1
        self.counts['recovered_rows'] += int(bad.sum())
        return obs

    @staticmethod
    def _valid_result_rows(result, n, device):
        obs, reward, done, extra = result
        valid = finite_rows(obs, n, device) & finite_rows(reward, n, device)
        valid &= finite_rows(done, n, device)
        valid &= ((done == 0) | (done == 1)).reshape(n, -1).all(1)
        valid &= finite_rows(extra.get('terms', {}), n, device)
        if 'terminal_observation' in extra:
            valid &= finite_rows(extra['terminal_observation'], n, device)
        return valid

    @classmethod
    def _validate_result(cls, result, n, device, reset_rows=None):
        obs, _, done, extra = result
        if not bool(cls._valid_result_rows(result, n, device).all()):
            raise RuntimeError('invalid environment/reset result during policy collection')
        if reset_rows is not None:
            recovered = extra.get('recovered')
            if (not isinstance(recovered, torch.Tensor) or recovered.shape != (n,)
                    or done.shape != (n,) or not bool(finite_rows(recovered, n, device).all())
                    or not bool((done.bool() & (recovered == 1))[reset_rows].all())):
                raise RuntimeError('policy quarantine reset lacks done/recovered acknowledgement')
        return TensorDict({'policy': obs}, batch_size=[n], device=device)

    def _quarantine(self, algorithm, env, action, bad, attempted, before_step, failure_counts):
        n, device = action.shape[0], action.device
        self._clear(algorithm)
        if attempted is not None:
            if bool((attempted & bad).any()):
                raise RuntimeError('persistent policy fault: recurrence before a fresh full rollout')
            attempted |= bad
        # Preserve the ENTIRE original nonfinite action row for Env diagnosis.
        # Finite actions with corrupt value/log-prob/distribution need a marker.
        nonfinite_action = ~finite_rows(action, n, device)
        forwarded = torch.where(nonfinite_action[:, None], action, action.clamp(-1, 1)).clone()
        forwarded[bad & ~nonfinite_action, 0] = float('nan')
        result = self._advance(env, forwarded, before_step, failure_counts)
        return self._accept_recovery(algorithm, result, bad)

    @torch.inference_mode()
    def collect(self, algorithm, env, obs, steps, before_step=None):
        from .tabletop_tensor_task import FAILURE_CODES, FAILURE_NAMES
        n, device = len(obs), obs.device
        if not (env.continuous_motion and env.recovery) or algorithm.policy.is_recurrent:
            raise RuntimeError('policy recovery requires continuous-motion feedforward PPO with resets')
        if steps != algorithm.storage.num_transitions_per_env or algorithm.storage.step != 0:
            raise RuntimeError('policy collection requires an empty full-sized rollout storage')
        attempted = torch.zeros(n, dtype=torch.bool, device=device)
        packet_pending = torch.zeros_like(attempted)
        failure_counts = torch.zeros(max(FAILURE_NAMES)+1, dtype=torch.long, device=device)
        self._attempt_steps = 0
        try:
            while True:
                summary = dict(reward_sum=torch.zeros((), device=device), terms={},
                    occupancy=torch.zeros(4, device=device), relief_fraction=torch.zeros((), device=device),
                    clipped_fraction=torch.zeros((), device=device))
                for _ in range(steps):
                    bad = ~finite_rows(obs, n, device)
                    policy_attempts = attempted
                    if bool(bad.any()):
                        policy_attempts = None  # invalid input state belongs to Env recovery
                        action = torch.zeros((n, *algorithm.storage.actions_shape), device=device)
                    else:
                        try:
                            action = algorithm.act(obs)
                        except (RuntimeError, ValueError) as error:
                            raise RuntimeError('policy act is unrecoverable by environment reset') from error
                        transition = algorithm.transition
                        for value in (action, transition.observations, transition.actions,
                                      transition.values, transition.actions_log_prob,
                                      transition.action_mean, transition.action_sigma):
                            bad |= ~finite_rows(value, n, device)
                        bad |= ~(transition.action_sigma > 0).reshape(n, -1).all(1)
                    if bool(bad.any()):
                        if policy_attempts is None:
                            if bool((packet_pending & bad).any()):
                                raise RuntimeError('post_reset_invalid_packet recurrence: invalid input after recovery')
                            packet_pending |= bad
                        obs = self._quarantine(algorithm, env, action, bad, policy_attempts, before_step, failure_counts)
                        break
                    result = self._advance(env, action.clamp(-1, 1), before_step, failure_counts)
                    bad = ~self._valid_result_rows(result, n, device)
                    if bool((packet_pending & bad).any()):
                        ids=(packet_pending & bad).nonzero().flatten().tolist()
                        raise RuntimeError('post_reset_invalid_packet recurrence: env_ids='+str(ids))
                    # Quarantine's marker result cannot certify the ordinary
                    # return path. Only a subsequent valid ordinary tick can.
                    packet_pending &= bad
                    failure = result[3].get('failure')
                    recovered_invalid = torch.zeros(n, dtype=torch.bool, device=device)
                    if isinstance(failure, torch.Tensor) and failure.shape == (n,):
                        for name in ('invalid_action', 'nonfinite_state', 'nonfinite_joint_state'):
                            recovered_invalid |= failure == FAILURE_CODES[name]
                    if bool(recovered_invalid.any()):
                        # Env already reset these invalid rows. Require its
                        # finite result and done/recovered acknowledgements;
                        # another NaN marker would cause a duplicate reset.
                        # Env owns physical recurrence guards. Independent
                        # validated physical recoveries never spend the policy
                        # retry budget: a world reset cannot repair bad weights.
                        obs = self._accept_recovery(algorithm, result, recovered_invalid)
                        break
                    if bool(bad.any()):
                        packet_pending |= bad
                        obs = self._quarantine(algorithm, env, torch.zeros_like(action),
                                               bad, None, before_step, failure_counts)
                        break
                    obs = self._validate_result(result, n, device)
                    _, reward, done, extra = result
                    algorithm.process_env_step(obs, reward, done, {})
                    summary['reward_sum'] += reward.mean()
                    summary['occupancy'] += torch.bincount(extra['phase'], minlength=4)
                    summary['relief_fraction'] += extra['pressure_limited'].float().mean()
                    summary['clipped_fraction'] += (action.abs() > 1.).float().mean()
                    for name, value in extra.get('terms', {}).items():
                        summary['terms'][name] = summary['terms'].get(name, torch.zeros((), device=device))+value.mean()
                else:
                    # Validate bootstrap before RSL computes GAE over the buffer.
                    last_values = algorithm.policy.evaluate(obs).detach()
                    bad = ~finite_rows(last_values, n, device)
                    if bool(bad.any()):
                        action = torch.zeros((n, *algorithm.storage.actions_shape), device=device)
                        obs = self._quarantine(algorithm, env, action, bad, attempted, before_step, failure_counts)
                        continue
                    # Same RSL 3.1.3 computation as PPO.compute_returns, using
                    # the value tensor just validated rather than evaluating
                    # the critic a second time and trusting another result.
                    algorithm.storage.compute_returns(last_values, algorithm.gamma, algorithm.lam,
                        normalize_advantage=not algorithm.normalize_advantage_per_mini_batch)
                    if not all(bool(torch.isfinite(getattr(algorithm.storage, name)).all())
                               for name in ('returns', 'advantages')):
                        raise RuntimeError('nonfinite PPO returns/advantages; policy collection is unrecoverable')
                    self._attempt_steps = 0
                    return dict(summary, obs=obs, failures=failure_counts)
        except BaseException:
            self._discard(algorithm, n)
            raise
