"""Exclusive native micro-motion evidence, using the ordinary SDK action path."""
import json
from pathlib import Path

import torch


def perturbations(n, device):
    if not isinstance(n, int) or n <= 0:
        raise ValueError('positive environment count required')
    rows = [('zero', torch.zeros(n, 24, device=device))]
    for dim in range(24):
        for sign in (1, -1):
            action = torch.zeros(n, 24, device=device)
            action[:, dim] = sign * .03
            rows.append((f'dim{dim}_{sign:+d}', action))
    for hand in (0, 1):
        for sign in (1, -1):
            action = torch.zeros(n, 24, device=device)
            action[:, 6*hand:6*hand+6] = sign * .03
            action[:, 12+3*hand:15+3*hand] = sign * .03
            action[:, 18+3*hand:21+3*hand] = sign * .03
            rows.append((f'hand{hand}_joint_wrist_{sign:+d}', action))
    return rows


def run_mapping_probe(world, env, args):
    from .tabletop_bimanual_physical_action_adapter import audit_native_snapshot, HandPhysicalKinematics
    # USD files are opened read-only. Each side keeps its own joint frames.
    asset_root = Path(__file__).resolve().parents[4] / 'assets/fixed_dual_panda_dh116_v1/dependencies'
    sources = {side: next(asset_root.glob(f'*_DH116_{side}.usd')) for side in ('left', 'right')}
    models = {side: HandPhysicalKinematics.from_usd(sources[side], side)
              for side in ('left', 'right')}
    output = Path(args.output)
    samples, failures = [], []
    with (output / 'mapping_probe.jsonl').open('x') as stream:
        for case, action in perturbations(world.n, world.device):
            env.reset()
            baseline = world.roots.clone()
            start_q = world.q.clone()
            rows = audit_native_snapshot(world, models, list(range(world.n)))
            for row in rows:
                row.update(case=case, phase='reset',
                           episode_uid=f"mapping:{case}:{row['env_id']}:{int(env.diagnostic_episode_ids[row['env_id']])}",
                           physics_step=int(world._physics_steps), timestamp_s=float(world.w.sim.current_time))
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            for tick in range(4):
                _, _, done, _ = env.step(action.clone())
                if done.any():
                    failures.append(dict(case=case, tick=tick, env_ids=done.nonzero().flatten().tolist(),
                                         failure_codes=env.task.failure[done.bool()].tolist()))
                    break
            rows = audit_native_snapshot(world, models, list(range(world.n)))
            for row in rows:
                i = row['env_id']
                targets, target_sources = [], []
                for hi, hand in enumerate(world.hands):
                    actual = getattr(hand.data, 'joint_pos_target', None)
                    targets.append((actual[i,world.w.ids[hi]] if actual is not None
                                    else world.commands[i,hi].float()).tolist())
                    target_sources.append('native_articulation_data_joint_pos_target' if actual is not None
                                          else 'float32_submission_reconstructed')
                row.update(case=case, phase='after', physical_action24=action[i].tolist(),
                           episode_uid=f'mapping:{case}:{i}:{int(env.diagnostic_episode_ids[i])}',
                           sdk_active_target_rad=targets, sdk_active_target_source=target_sources,
                           sdk_root_target_w=world.root_targets[i].float().tolist(),
                           sdk_wrist_angles_rad=world.wrist_angles[i].tolist(),
                           active_q_rad=env.task.m['q'][i].reshape(2,6).tolist(),
                           actual_root_dxyz_m=(world.roots[i,:,:3]-baseline[i,:,:3]).tolist(),
                           actual_joint_delta_rad=(world.q[i]-start_q[i]).tolist(),
                           physics_step=int(world._physics_steps), timestamp_s=float(world.w.sim.current_time))
                samples.append(row)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
    result = dict(status='completed', cases=53, snapshots=len(samples), failures=failures,
                  action_amplitude=.03, policy_steps_per_case=4, source_assets={k:str(v) for k,v in sources.items()},
                  qualification='physical motion evidence; tracking and FK errors require separate acceptance')
    (output/'mapping_probe_summary.json').write_text(json.dumps(result, allow_nan=False, indent=2))
    return result
