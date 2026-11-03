"""Score every single-factor profile on the same full native Student episodes."""
import runpy
from pathlib import Path

import torch

from go2_pvcnn.control.dh116 import tabletop_surface_reward_ablation as rewards
from go2_pvcnn.control.dh116 import tabletop_surface_student_evaluation as evaluation
from go2_pvcnn.control.dh116.tabletop_surface_exploration import support_components


def evaluate(env, algorithm, args, contract, initial_obs):
    if (args.surface_reward_profile != 'baseline' or not args.surface_student_first_episode_only
            or args.surface_student_stop_on_success or args.surface_mpc):
        raise ValueError('use baseline, first-episode-only and deterministic Student actions')

    task = env.task
    baseline_names = set(task.terms)
    task.surface_reward_profile = 'support_deficit'
    rewards.enable(task)
    task.surface_reward_profile = 'baseline'
    task.surface_reward_ablation.terms = tuple(rewards.SINGLE_TERMS.values())
    task.surface_action_motion = rewards.ExecutedMotion(task.actions)
    for name in task.terms.keys() - baseline_names:
        del task.terms[name]

    zero = torch.zeros_like(task.reward)
    active = torch.ones_like(task.done)
    traces = []
    step_terms = {}
    max_lift, max_hold, max_drift = (zero.clone() for _ in range(3))
    contact_steps = torch.zeros_like(env.episodes)
    control_steps = torch.zeros_like(env.episodes)
    native_overcredit = zero.clone()
    legacy_overcredit = zero.clone()

    original_apply, original_step = rewards.apply_reward, env.step

    def score(t, pieces, valid, dt, hand_lift_force_n, weight, *, milestone_valid=None):
        shadow = {}
        original_apply(t, shadow, valid, dt, hand_lift_force_n, weight,
                       milestone_valid=milestone_valid)
        for name, value in shadow.items():
            step_terms[name] = step_terms.get(name, zero) + value
        measured = active & (valid | milestone_valid)
        max_lift.copy_(torch.maximum(max_lift, torch.where(measured,
            t.m['bottom'] - t.surface_reference_bottom_m, 0.)))
        max_hold.copy_(torch.maximum(max_hold, torch.where(measured,
            t.surface_lift_success.duration, 0.)))
        tracker = t.surface_early_hold
        max_drift.copy_(torch.maximum(max_drift, torch.where(
            measured & tracker.window_completed & tracker.slip_observable, tracker.reward_drift_m, 0.)))
        for legacy, peak in ((False, native_overcredit), (True, legacy_overcredit)):
            support = support_components(t.m, hand_lift_force_n, legacy=legacy)[1]
            excess = (support - hand_lift_force_n).clamp_min(0.).amax(-1)
            peak.copy_(torch.maximum(peak, torch.where(measured, excess, 0.)))

    def step(actions):
        step_terms.clear()
        obs, reward, done, extras = original_step(actions)
        conditions = extras['surface_diagnostics']['conditions']
        contact = (conditions['measurements_valid'] & conditions['opposed_pads']
                   & conditions['depth'] & conditions['ordinary_fraction'])
        control_steps.add_(active.long())
        contact_steps.add_((active & contact).long())
        traces.append(dict(active=active.cpu().clone(), done=(done & active).cpu(),
            success=(extras['success'] & active).cpu(), failure=extras['failure'].cpu(),
            baseline={name: torch.where(active, v, 0.).cpu() for name, v in extras['terms'].items()},
            baseline_total=torch.where(active, reward, 0.).cpu(),
            factors={name: torch.where(active, value, 0.).cpu() for name, value in step_terms.items()}))
        active.logical_and_(~done)
        return obs, reward, done, extras

    rewards.apply_reward, env.step = score, step
    try:
        result = native_evaluate(env, algorithm, args, contract, initial_obs)
    finally:
        rewards.apply_reward, env.step = original_apply, original_step

    baseline = {name: torch.stack([row['baseline'][name] for row in traces])
                for name in traces[0]['baseline']}
    baseline_total = torch.stack([row['baseline_total'] for row in traces])
    factors = {name: torch.stack([row['factors'][name] for row in traces])
               for name in rewards.SINGLE_TERMS.values()}
    profiles = {}
    for profile in ('baseline', 'v3_reference', *(f'v3_{name}_only' for name in rewards.SINGLE_TERMS)):
        scalars = dict(baseline)
        total = baseline_total.clone()
        for name in rewards.disabled_legacy_terms(profile):
            if name in scalars:
                total -= scalars.pop(name)
        for name in rewards.STAGE_TERMS[profile]:
            scalars[name] = factors[name]
            total += factors[name]
        profiles[profile] = dict(
            reward_term_totals={name: value.sum(0).tolist() for name, value in scalars.items()},
            total_reward=total.sum(0).tolist())
    output = Path(args.output)
    torch.save(dict(baseline=baseline, baseline_total=baseline_total, factors=factors,
        **{name: torch.stack([row[name] for row in traces]) for name in ('active','done','success','failure')}),
        output/'full_episode_reward_scalars.pt')
    terminals = [trace['failure'][trace['done']].tolist() for trace in traces if trace['done'].any()]
    audit = dict(result, profiles=profiles, scoring='independent_reward_terms_on_shared_native_substeps',
        first_episode_complete=(~active).cpu().tolist(), first_episode_control_steps=control_steps.cpu().tolist(),
        first_episode_terminal_codes=terminals, first_episode_native_successes=torch.stack(
            [row['success'] for row in traces]).sum(0).tolist(),
        max_net_lift_m=max_lift.cpu().tolist(), max_continuous_hold_s=max_hold.cpu().tolist(),
        max_measured_fixed_anchor_drift_m=max_drift.cpu().tolist(),
        valid_contact_rate=(contact_steps/control_steps.clamp_min(1)).cpu().tolist(),
        contact_rate_denominator='first_episode_control_steps',
        max_native_support_overcredit_n=native_overcredit.cpu().tolist(),
        max_legacy_support_overcredit_n=legacy_overcredit.cpu().tolist())
    evaluation.write_progress(output/'full_episode_reward_audit.json', audit)
    if active.any():
        raise RuntimeError('native evaluation budget ended before a complete episode in each environment')
    return result


if __name__ == '__main__':
    native_evaluate = evaluation.evaluate
    evaluation.evaluate = evaluate
    runpy.run_path(str(Path(__file__).with_name('train_dh116_tabletop_gpu_ppo.py')), run_name='__main__')
