"""Open-loop native fault replay. A diagnostic sample is taken before physical reset."""
import hashlib
import json
from pathlib import Path

import torch

from .tabletop_vision_mirror_ablation import m2_action_to_physics


def load_actions(path, count):
    steps = {}
    watched = set()
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            step, env = row['policy_step'], row['env_id']
            batch = steps.setdefault(step, {})
            if env in batch or not 0 <= env < count:
                raise ValueError('duplicate or unknown environment')
            batch[env] = row['action24']
            if row.get('done'):
                watched.add(env)
    result = {}
    for step in sorted(steps):
        if result and step != max(result) + 1:
            raise ValueError('policy steps must be consecutive')
        if len(steps[step]) != count:
            raise ValueError('incomplete environment batch')
        result[step] = torch.tensor([steps[step][i] for i in range(count)], dtype=torch.float32)
        if result[step].shape != (count, 24) or not result[step].isfinite().all():
            raise ValueError('invalid saved actions')
    if not result:
        raise ValueError('empty replay')
    return result, watched


def replay_action(raw, arm):
    bounded = raw.clamp(-1, 1)
    if arm == 'omit_inverse':
        return bounded
    issued = m2_action_to_physics(bounded)
    if arm not in ('recorded', 'right_rotation_after240', 'right_translation_after240'):
        part = {'neutral_fingers': slice(0, 12), 'neutral_translation': slice(12, 18),
                'neutral_rotation': slice(18, 24)}[arm]
        issued[:, part] = 0
    return issued


def late_replay_action(action, arm, completed_physics_steps):
    section = {'right_rotation_after240': slice(21,24),
               'right_translation_after240': slice(15,18)}[arm]
    actual = action.clone()
    if completed_physics_steps >= 240:
        actual[:,section] = 0
    return actual


def run_joint_replay(world, env, args):
    if getattr(args, 'bimanual_mapping_probe', False):
        from .tabletop_bimanual_native_probe import run_mapping_probe
        return run_mapping_probe(world, env, args)
    source = Path(args.vision_m2_replay)
    metadata = json.loads((source/'vision_mirror_preprocessing.json').read_text())
    if metadata.get('mode') != 'm2_full_mirror':
        raise ValueError('M2 replay requires recorded M2 actor actions')
    from .tabletop_vision_evaluation_contract import file_sha256
    policy_log = source/'policy_ppo_diagnostics.jsonl'
    actions, watched = load_actions(policy_log, world.n)
    output = Path(args.output)
    names = [list(hand.joint_names) for hand in world.hands]
    episodes = []
    audit = None
    if getattr(args, 'bimanual_joint_tracking_audit', False):
        from .tabletop_bimanual_joint_tracking_audit import JointTrackingAuditWriter
        audit = JointTrackingAuditWriter(output/'joint_tracking_480hz.jsonl')
    env.reset()
    fingerprint = hashlib.sha256()
    for value in (world.q, world.qd, world.roots, world.wp, world.wq, world.wv):
        fingerprint.update(value.detach().cpu().contiguous().numpy().tobytes())
    summary = dict(source=str(source), source_policy_sha256=file_sha256(policy_log),
                   arm=args.vision_replay_arm, joint_names=names,
                   active_joint_ids=[list(map(int, ids)) for ids in world.w.ids],
                   initial_state_sha256=fingerprint.hexdigest(), terminal_events=episodes)
    def emit(sample):
        stream.write(json.dumps(sample, allow_nan=False, separators=(',', ':'))+'\n')

    def observe(e, m, live, terminal):
        selected = sorted(watched.union(terminal.nonzero().flatten().tolist()))
        if audit is not None:
            from .tabletop_bimanual_joint_tracking_audit import capture_joint_tracking
            uids = [f'1:{i}:{int(e.diagnostic_episode_ids[i])}' for i in range(world.n)]
            rows = capture_joint_tracking(world, m, env_ids=selected, episode_uids=uids,
                policy_step=step, actor_raw24=raw, action_before_inverse24=raw.clamp(-1,1),
                action_after_inverse24=getattr(e, 'stage5_replay_issued', issued),
                live=live, terminal=terminal, failure_codes=e.task.failure)
            for row in rows:
                audit.append(row)
        for i in selected:
            uid = f'1:{i}:{int(e.diagnostic_episode_ids[i])}'
            row = dict(env_id=i, episode_uid=uid, policy_step=step, physics_step=world._physics_steps,
                       timestamp_s=float(world.w.sim.current_time), live=bool(live[i]),
                       terminal=bool(terminal[i]), failure=int(e.task.failure[i]),
                       actor_raw24=raw[i].tolist(), actor_bounded24=raw[i].clamp(-1,1).tolist(),
                       physical_action24=getattr(e, 'stage5_replay_issued', issued)[i].tolist(),
                       active_q_rad=m['q'][i].tolist(),
                       sdk_active_target_rad=world.commands[i].float().tolist(),
                       sdk_wrist_angles_rad=world.wrist_angles[i].tolist(),
                       q=world.q[i].tolist(), limits=world.limits[i].tolist(),
                       contact_loads_n=m['hand_object_loads'][i].tolist())
            emit(row)
            if row['terminal']:
                episodes.append({key:row[key] for key in ('env_id','episode_uid','physics_step',
                                                            'policy_step','timestamp_s','failure')})
                stream.flush()

    with (output/'joint_replay.jsonl').open('x') as stream:
        env.joint_replay_observer = observe
        late = args.vision_replay_arm in ('right_rotation_after240','right_translation_after240')
        if late:
            if getattr(env, 'bimanual_action_adapter', None) is not None:
                raise ValueError('late intervention requires an independent replay')
            def late_intervention(action):
                actual = late_replay_action(action, args.vision_replay_arm, world._physics_steps)
                env.stage5_replay_issued=actual
                return actual
            env.bimanual_action_adapter=late_intervention
        try:
            with torch.inference_mode():
                for step, raw in actions.items():
                    issued = replay_action(raw, args.vision_replay_arm)
                    env.step(issued.to(world.device).clone())
        finally:
            env.joint_replay_observer = None
            if late:
                env.bimanual_action_adapter=None
            if audit is not None:
                audit.close()
    with (output/'joint_replay_summary.json').open('x') as stream:
        json.dump(summary, stream, allow_nan=False, indent=2)
    return dict(status='completed', replay_policy_steps=len(actions), terminal_count=len(episodes))
