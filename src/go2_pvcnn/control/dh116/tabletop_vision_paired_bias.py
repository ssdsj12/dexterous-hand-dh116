"""Deterministic experiments on the native T2 cold-reset evaluation path."""
import json
from pathlib import Path

import numpy as np
import torch

from .tabletop_tensor_task import FAILURE_CODES
from .tabletop_vision_mirror_ablation import m2_action_to_physics, m2_observation_for_actor


def neutralize_right_z(action):
    if action.ndim != 2 or action.shape[1] != 24:
        raise ValueError('right Z intervention requires physical [env,24] actions')
    result = action.clone()
    result[:, 17] = 0
    return result


def paired_identity(checkpoint, manifest, seed, readback):
    fields=('trial_id','authored_state_sha256','readback_state_sha256')
    states=[[r[k] for k in fields] for r in readback['actual_states']]
    if not states or any(len(set(column)) != len(states) for column in zip(*states)):
        raise ValueError('paired identity must contain distinct authored and measured initial states')
    return dict(checkpoint_sha256=checkpoint, manifest_sha256=manifest, seed=seed, states=states)


def require_identical_pair(actual, expected):
    if actual != expected:
        raise ValueError('paired identity mismatch')


def summarize_trials(rows):
    def quantiles(values):
        values=[v for v in values if v is not None]
        return dict(count=len(values), **{f'p{n}':float(np.percentile(values,n)) if values else None
                    for n in (50,90,95)}, maximum=max(values, default=None))
    result={}
    for style in sorted({r['grasp_style'] for r in rows}):
        group=[r for r in rows if r['grasp_style']==style]
        done=[r for r in group if r['terminal'] and r['valid']]
        count=len(done)
        contact=sum(r['contact_valid_samples'] for r in done)
        result[style]=dict(expected=len(group),completed=count,
            lift_mm=quantiles([1000*r['lift_peak_m'] for r in done]),
            slip_mm=quantiles([1000*r['slip_max_m'] for r in done if r['slip_max_m'] is not None]),
            hold_s=quantiles([r['hold_max_s'] for r in done]),
            right_final_dz_mm=quantiles([1000*r['right_palm_final_dz_m'] for r in done]),
            joint_limit=sum(r['failure_reason']=='joint_limit' for r in done),
            native_success_rate=sum(r['first_terminal_success'] for r in done)/count if count else None,
            opposed_contact_fraction=sum(r['opposed_samples'] for r in done)/contact if contact else None,
            mean_episode_return=sum(r['task_return'] for r in done)/count if count else None)
    return result


def run_paired_arm(args, env, task, world, camera, history, obs, policy, readback, manifest, checkpoint, output):
    from .tabletop_vision_evaluation_contract import file_sha256
    from .tabletop_vision_training import _observations
    arm=args.paired_bias_arm
    identity=paired_identity(checkpoint,file_sha256(args.vision_evaluation_manifest),args.seed,readback)
    reference=None
    if args.paired_bias_reference:
        folder=Path(args.paired_bias_reference)
        original=json.loads((folder/'paired_results.json').read_text())
        require_identical_pair(identity,original['identity'])
        if original['arm']!='m0' or not original['complete']:
            raise ValueError('complete M0 reference required')
        if arm in ('replay','right_z_zero'):
            reference=np.load(folder/'issued_actions.npy',mmap_mode='r')
    n=world.n
    if world.hands[1].joint_names[world.w.ids[1][1]] != 'finger12':
        raise ValueError('native right finger12 SDK active mapping mismatch')
    finished=np.zeros(n,dtype=bool)
    physical=[]
    outputs=[]
    rewards=np.zeros(n)
    baseline=world.roots[:,:,2].clone()
    coordinated = arm in ('z2', 'z3')
    if coordinated:
        from .tabletop_bimanual_experiment_runtime import install_coordination
        install_coordination(env, world, arm, output)
    # One batched transfer per physical substep. The live flag belongs to the
    # physics engine; the first-episode mask is applied separately in reduction.
    fields=('live','done','native_success','measurement_valid','lift_m','qualified_hold_s',
            'slip_m','slip_valid','bilateral_opposed_contact','contact_valid','failure',
            'right_palm_dz_m','physics_step','timestamp_s','lift_valid','qualified_hold_valid',
            'slip_applicable','slip_reason_code','left_palm_dz_m','palm_sync_error_m',
            'left_tracking_error_rad','right_tracking_error_rad','left_load_n','right_load_n')
    fields += ('right_finger12_tracking_error_rad','right_finger12_q_rad',
               'right_finger12_qd_rad_s','right_finger12_sdk_target_rad','episode_id',
               'right_finger12_sdk_target_readback_valid')
    fields += ('policy_step',)
    if coordinated:
        fields += tuple(f'executed_action_{i}' for i in range(24))
        fields += tuple(f'objective_{term}' for term in ('lift','contact','slip','joint','sync','total'))
        fields += ('objective_valid','coordination_ready','cancel_upward_left','cancel_upward_right')
    def collect(packet):
        scalar = dict(failure=task.failure,right_palm_dz_m=world.roots[:,1,2]-baseline[:,1],
            physics_step=torch.full_like(task.failure,world._physics_steps),
            timestamp_s=torch.full_like(baseline[:,1],float(world.w.sim.current_time)))
        dz = world.roots[:,:,2]-baseline
        error = (world.commands.float().double()-task.m['q'].reshape(n,2,6)).abs().amax(-1)
        scalar.update(left_palm_dz_m=dz[:,0],palm_sync_error_m=(dz[:,0]-dz[:,1]).abs(),
            left_tracking_error_rad=error[:,0],right_tracking_error_rad=error[:,1],
            left_load_n=task.m['hand_object_loads'][:,0],right_load_n=task.m['hand_object_loads'][:,1])
        finger12 = world.hands[1].joint_names.index('finger12')
        native_target = getattr(world.hands[1].data, 'joint_pos_target', None)
        target12 = (native_target[:,finger12].double() if native_target is not None
                    else world.commands[:,1,1].float().double())
        scalar.update(right_finger12_tracking_error_rad=target12-world.q[:,1,finger12],
            right_finger12_q_rad=world.q[:,1,finger12],right_finger12_qd_rad_s=world.qd[:,1,finger12],
            right_finger12_sdk_target_rad=target12,episode_id=env.diagnostic_episode_ids,
            right_finger12_sdk_target_readback_valid=torch.full_like(task.failure,native_target is not None))
        scalar['policy_step'] = torch.full_like(task.failure, int(env._diagnostic_policy_step))
        if coordinated:
            scalar.update({f'executed_action_{i}':env.stage5_executed_action[:,i] for i in range(24)})
            info = env.stage5_coordination_diagnostics
            objective = info['objective']
            scalar.update({f'objective_{term}':objective.get(term, torch.full_like(baseline[:,0],float('nan')))
                           for term in ('lift','contact','slip','joint','sync','total')})
            scalar.update(objective_valid=torch.full_like(task.failure, bool(objective)),
                          coordination_ready=info['resume_ready'],
                          cancel_upward_left=info['cancel_upward_target'][:,0],
                          cancel_upward_right=info['cancel_upward_target'][:,1])
        physical.append(torch.stack([v.double() for v in
            [packet[k] if k in packet else scalar[k] for k in fields]],dim=1).cpu().numpy())
    prior=env.evaluation_telemetry.callback
    env.evaluation_telemetry.callback=collect
    try:
        with torch.inference_mode():
            for tick in range(args.vision_evaluation_max_control_steps):
                if arm in ('m0','z2','z3'): action=policy.act(obs)
                elif arm=='m2':
                    action=m2_action_to_physics(policy.act(m2_observation_for_actor(obs['policy'])))
                else:
                    # The reference trajectory has an explicit all-zero tail after
                    # its first terminal. Never copy actions from another episode.
                    raw=reference[tick].copy() if tick<len(reference) else np.zeros((n,24),dtype=np.float32)
                    action=torch.as_tensor(raw,device=world.device)
                    if arm=='right_z_zero':
                        action=neutralize_right_z(action)
                action[torch.as_tensor(finished,device=world.device)]=0
                outputs.append(action.cpu().numpy().copy())
                critic,reward,done,_=env.step(action.clone())
                rewards[~finished]+=reward.detach().cpu().numpy().reshape(n)[~finished]
                finished |= done.detach().cpu().numpy().reshape(n).astype(bool)
                if tick%60==0:
                    print('PAIRED_PROGRESS',arm,tick+1,int(finished.sum()),flush=True)
                if finished.all(): break
                if arm in ('m0','m2','z2','z3'):
                    previous=(env.stage5_executed_action if coordinated else action).clone()
                    previous[done.bool().reshape(n)]=0
                    obs=_observations(camera.sample(),world,previous,history,critic,reset_mask=done)
    except Exception as error:
        # A failed/incomplete trajectory can be audited but cannot serve as
        # the M0 reference or count as a completed first-terminal trial.
        if physical:
            np.savez_compressed(output/'incomplete_native_samples.npz',fields=fields,samples=np.stack(physical))
        if outputs:
            np.save(output/'incomplete_issued_actions.npy',np.stack(outputs),allow_pickle=False)
        with (output/'paired_failure.json').open('x',encoding='utf-8') as stream:
            json.dump(dict(arm=arm,identity=identity,error=str(error),complete=False,
                submitted_control_steps=len(outputs),observed_physics_steps=len(physical)),stream,allow_nan=False)
        raise
    finally:
        env.evaluation_telemetry.callback=prior
        if coordinated:
            env.bimanual_action_adapter=None
    np.save(output/'issued_actions.npy',np.stack(outputs),allow_pickle=False)
    data=np.stack(physical)
    np.savez_compressed(output/'native_samples.npz',fields=fields,samples=data)
    trials=[]
    failure_names={v:k for k,v in FAILURE_CODES.items()}
    for i,trial in enumerate(manifest['trials']):
        series=data[:,i,:]
        terminal=np.flatnonzero(series[:,1])
        if len(terminal):series=series[:terminal[0]+1]
        live=series[series[:,0].astype(bool)]
        final=live[-1]
        slips=live[:,6][live[:,7].astype(bool)]
        trials.append(dict(trial_id=trial['trial_id'],env_id=i,grasp_style=trial['grasp_style'],
            episode_uid=f"{output.name}:{i}:{int(live[0,fields.index('episode_id')])}",
            terminal=bool(len(terminal)),valid=bool(live[:,[3,14,15]].all()),
            slip_applicable_samples=int(live[:,16].sum()),slip_valid_samples=len(slips),
            first_terminal_success=bool(final[2]),
            failure_reason=failure_names[int(final[10])],physical_samples=len(live),
            terminal_step=int(final[12]) if len(terminal) else None,final_time_s=float(final[13]),
            right_palm_final_dz_m=float(final[11]),lift_peak_m=float(np.max(live[:,4])),
            hold_max_s=float(np.max(live[:,5])),slip_max_m=float(slips.max()) if len(slips) else None,
            opposed_samples=int((live[:,8]*live[:,9]).sum()),contact_valid_samples=int(live[:,9].sum()),
            task_return=float(rewards[i])))
    if file_sha256(args.vision_evaluation_checkpoint)!=checkpoint:
        raise RuntimeError('source checkpoint changed')
    result=dict(arm=arm,identity=identity,control_steps=len(outputs),complete=all(
        r['terminal'] and r['valid'] for r in trials),by_style=summarize_trials(trials),trials=trials)
    with (output/'paired_results.json').open('x') as stream:
        json.dump(result,stream,allow_nan=False)
    return dict(status='completed' if result['complete'] else 'incomplete',by_style=result['by_style'])
