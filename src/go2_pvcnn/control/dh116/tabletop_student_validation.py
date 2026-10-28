"""Deterministic closed-loop student qualification before any PPO optimization."""
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch


def _actor_hash(actor):
    digest = hashlib.sha256()
    for name, value in sorted(actor.state_dict().items()):
        value = value.detach().cpu().contiguous()
        if not bool(torch.isfinite(value).all()):
            raise ValueError('student actor contains nonfinite weights')
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _terminal_field(value, name, n, *, boolean=False):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (n,):
        raise ValueError(f'{name} must be a tensor with shape ({n},)')
    value = value.detach().cpu()
    if value.is_complex() or not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must contain finite real values')
    if boolean:
        if not bool(((value == 0) | (value == 1)).all()):
            raise ValueError(f'{name} must contain exact boolean values')
        return value.bool()
    return value


def validate_student(env, algorithm, args, contract, obs):
    """Validate from the caller's fresh reset using only production env steps.

    Every environment must finish three episodes within 2700 control steps,
    with at least three explicit, failure-free terminal successes overall.
    ``student_validation_steps`` can set a smaller diagnostic budget. No
    optimizer, rollout storage, normalization update or manual reset is used.
    Physical/other exceptions are recorded and re-raised without eligibility.
    """
    from .tabletop_tensor_training import observations
    from .tabletop_training import write_json

    budget = getattr(args, 'student_validation_steps', 2700)
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 2700:
        raise ValueError('student_validation_steps must be between 1 and 2700')
    n = env.n
    episodes = torch.zeros(n, dtype=torch.long)
    successes = torch.zeros(n, dtype=torch.long)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        'version': 'dh116-tabletop-student-validation-v1',
        'status': 'running', 'passed': False,
        'deterministic': True, 'optimization_performed': False,
        'required_episodes_per_env': 3, 'required_successes': 3,
        'control_step_budget': budget, 'control_steps': 0,
        'contract': contract, 'terminal_events': [],
    }
    policy = algorithm.policy
    was_training = policy.training

    def publish():
        total = int(episodes.sum())
        report.update(episode_counts=episodes.tolist(), success_counts=successes.tolist(),
            episodes=total, successes=int(successes.sum()),
            success_rate=float(successes.sum()) / total if total else 0.)
        write_json(output / 'student_validation.json', report)

    try:
        report['actor_hash'] = _actor_hash(policy.actor)
        current = observations(obs) if isinstance(obs, torch.Tensor) else obs
        policy.eval()
        with torch.inference_mode():
            for step in range(1, budget + 1):
                report['control_steps'] = step
                actions = policy.act_inference(current)
                next_obs, _, dones, extras = env.step(actions)
                if not isinstance(extras, Mapping):
                    raise ValueError('student validation requires terminal success and failure evidence')
                done = _terminal_field(dones, 'dones', n, boolean=True)
                success = _terminal_field(extras.get('success'), 'success', n, boolean=True)
                failure = _terminal_field(extras.get('failure'), 'failure', n)
                if (failure.dtype == torch.bool or bool((failure < 0).any())
                        or (failure.is_floating_point() and bool((failure != failure.floor()).any()))):
                    raise ValueError('failure must contain nonnegative integer codes')
                measured = {}
                evidence = extras.get('physical_evidence')
                if evidence is not None:
                    if not isinstance(evidence, Mapping):
                        raise ValueError('physical_evidence must be a mapping')
                    for name in ('object_bottom_m', 'normal_vertical_force_n', 'lift_vertical_force_n'):
                        if name in evidence:
                            measured[name] = _terminal_field(evidence[name], name, n)
                valid_success = done & success & (failure == 0)
                for index in done.nonzero().flatten().tolist():
                    event = dict(env_index=index, episode_index=int(episodes[index]),
                        control_step=step, success=bool(valid_success[index]),
                        task_success=bool(success[index]), failure=int(failure[index]))
                    event.update({name: float(value[index]) for name, value in measured.items()})
                    report['terminal_events'].append(event)
                episodes += done.long()
                successes += valid_success.long()
                # TensorEnv already performs isolated resets and returns fresh
                # observations. Its terminal snapshots must not drive the next action.
                current = observations(next_obs)
                if bool((episodes >= 3).all()):
                    break
        report['actor_unchanged'] = _actor_hash(policy.actor) == report['actor_hash']
        if not report['actor_unchanged']:
            raise RuntimeError('student actor changed during inference-only validation')
        enough_episodes = bool((episodes >= 3).all())
        report['passed'] = enough_episodes and int(successes.sum()) >= 3
        report['status'] = ('passed' if report['passed'] else
                            'insufficient_successes' if enough_episodes else 'budget_exhausted')
        publish()
        return report['passed'], report
    except Exception as error:
        report.update(passed=False, status='error', error=str(error))
        try:
            publish()
        except Exception as publication_error:
            error.add_note('Could not publish student validation report: ' + str(publication_error))
        raise
    finally:
        policy.train(was_training)
