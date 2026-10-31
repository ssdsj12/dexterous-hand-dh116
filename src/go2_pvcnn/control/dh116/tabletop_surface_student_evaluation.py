"""Native student and optional grip-MPC evaluation, with no optimizer updates."""
import copy,hashlib,json,os,time
from pathlib import Path
import torch


def evaluation_stop(episodes,initial_episodes,successes,initial_successes,*,
                    first_episode_only=False,stop_on_success=False):
    if stop_on_success and bool((successes>initial_successes).any()):
        return 'native_success_observed'
    if first_episode_only and bool((episodes>initial_episodes).all()):
        return 'all_envs_completed_first_episode'
    return None


def write_progress(path, payload):
    """Publish a complete compact JSON snapshot with a same-directory replace."""
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    try:
        temporary.write_text(json.dumps(payload,separators=(',',':')),encoding='utf-8')
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluation_identity(args, contract):
    """Bind the report to source bytes and an owned JSON contract record.

    Contract SHA256 uses UTF-8 json.dumps(sort_keys=True, separators=(',', ':')),
    including its default ASCII escaping. Keep the record for comparisons that
    intentionally canonicalize batch size or other declared family fields.
    """
    source=getattr(args,'surface_student_actor',None) or getattr(args,'resume',None)
    source_hash=None
    if source:
        digest=hashlib.sha256()
        with Path(source).open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                digest.update(chunk)
        source_hash=digest.hexdigest()
    encoded=json.dumps(contract,sort_keys=True,separators=(',',':')).encode('utf-8')
    return dict(actor_source_file_sha256=source_hash,
        contract_sha256=hashlib.sha256(encoded).hexdigest(),contract=json.loads(encoded))


def policy_weights_sha256(state):
    """Hash sorted state names, dtypes, shapes and contiguous CPU tensor bytes.

    Each JSON [name, dtype, shape] header and byte payload has an eight-byte
    big-endian length prefix. A uint8 view supports scalar/bfloat16 tensors
    without changing values or depending on torch.save archive metadata.
    """
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        header=json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode('utf-8')
        data=value.reshape(-1).view(torch.uint8).numpy().tobytes()
        for part in (header,data):
            digest.update(len(part).to_bytes(8,'big'));digest.update(part)
    return digest.hexdigest()


@torch.no_grad()
def diagnostic_snapshot(task):
    """Read owned diagnostics without advancing or resetting the native helper.

    Physical gates describe the current measured task state. Hold, pairing and
    drift describe SurfaceLiftSuccess's last completed native control window;
    neither snapshot certifies every raw substep or replaces episode success.
    """
    from .tabletop_surface_force_sharing import surface_targets
    m=task.m;success=task.surface_lift_success
    loads=m['effective_pad_loads_n'];vectors=m['effective_pad_vectors_w']
    depth=m['effective_pad_depth_m'];rejected=m['rejected_contact_loads_n']
    bottom=m['bottom'];velocity=m['object_velocity']
    reference=task.surface_reference_bottom_m
    contact_valid=(torch.isfinite(loads).all((1,2))&(loads>=0).all((1,2))
                   &torch.isfinite(vectors).all((1,2,3)))
    # surface_targets rejects an invalid whole batch. Clean only diagnostic
    # copies of bad rows so one invalid environment cannot hide its siblings.
    paired=surface_targets(torch.where(contact_valid[:,None,None],loads,0.),
        torch.where(contact_valid[:,None,None,None],vectors,0.))['paired']
    ordinary=rejected.sum(-1)
    fraction=ordinary/(ordinary+loads.sum((1,2))).clamp_min(1e-12)
    speed=velocity[:,:3].norm(dim=-1)
    measurements_valid=(contact_valid&torch.isfinite(bottom)&torch.isfinite(bottom-reference)
        &torch.isfinite(velocity).all(-1)&torch.isfinite(depth).all((1,2))
        &(depth>=0).all((1,2))&torch.isfinite(rejected).all(-1)&(rejected>=0).all(-1))
    conditions=dict(height=bottom>=reference+.03,velocity=speed<.03,
        opposed_pads=paired.all(-1),depth=(depth.amax((1,2))<=.0008)&(depth>=0).all((1,2)),
        ordinary_fraction=(fraction<=.05+1e-12)&(rejected>=0).all(-1),
        measurements_valid=measurements_valid)
    eligible=torch.stack(list(conditions.values())).all(0)
    conditions.update(drift=torch.isfinite(success._drift)&(success._drift<=.0005),
                      hold_window=success._window_good.clone())
    return dict(net_lift_m=bottom-reference,hold_s=success.duration.detach().clone(),
        drift_m=success._drift.detach().clone(),hold_paired=success._paired.detach().clone(),
        hold_window_good=success._window_good.detach().clone(),linear_speed_m_s=speed,
        max_pad_depth_m=depth.amax((1,2)),ordinary_fraction=fraction,
        physical_eligible=eligible,conditions=conditions)


@torch.no_grad()
def evaluate(env,algorithm,args,contract,initial_obs):
    from .tabletop_tensor_training import observations
    output=Path(args.output)
    mpc=None
    if args.surface_mpc:
        from .tabletop_surface_mpc_runtime import NativeSurfaceMPC
        mpc=NativeSurfaceMPC(env)
    policy=algorithm.policy;policy.eval()
    before={k:v.detach().clone() for k,v in policy.state_dict().items()}
    identity=evaluation_identity(args,contract)
    weights_before=policy_weights_sha256(before);weights_after=None
    first_episode_only=bool(getattr(args,'surface_student_first_episode_only',False))
    stop_on_success=bool(getattr(args,'surface_student_stop_on_success',False))
    actual_ticks=0;stop_reason='step_budget_exhausted'
    obs=initial_obs;rows=[];start=time.monotonic()
    max_lift=torch.zeros(env.n,device=env.device,dtype=torch.float64)
    max_hold=max_lift.clone()
    max_drift=max_lift.clone();failure_counts={}
    successes=env.successes.clone();episodes=env.episodes.clone()

    def report(steps, *, complete=False):
        episode_successes=(env.successes-successes).cpu().tolist()
        episode_counts=(env.episodes-episodes).cpu().tolist()
        return dict(status=('surface_student_evaluation_complete' if complete
                            else 'surface_student_evaluation_running'),optimization_performed=False,
            native_control_steps=steps,planned_native_control_steps=args.surface_student_eval_steps,
            first_episode_only=first_episode_only,stop_reason=stop_reason if complete else None,
            elapsed_wall_s=time.monotonic()-start,
            successes=episode_successes,episodes=episode_counts,
            episode_success_count=sum(episode_successes),episode_count=sum(episode_counts),
            max_observed_net_lift_m=max_lift.cpu().tolist(),max_nonterminal_hold_s=max_hold.cpu().tolist(),
            observed_hold_s=diag['hold_s'].cpu().tolist(),observed_drift_m=diag['drift_m'].cpu().tolist(),
            observed_net_lift_m=diag['net_lift_m'].cpu().tolist(),
            observed_linear_speed_m_s=diag['linear_speed_m_s'].cpu().tolist(),
            observed_max_pad_depth_m=diag['max_pad_depth_m'].cpu().tolist(),
            observed_ordinary_fraction=diag['ordinary_fraction'].cpu().tolist(),
            max_observed_drift_m=max_drift.cpu().tolist(),
            physical_eligible=diag['physical_eligible'].cpu().tolist(),
            hold_window_good=diag['hold_window_good'].cpu().tolist(),
            hold_paired=diag['hold_paired'].cpu().tolist(),
            condition_failure_counts={key:value.cpu().tolist() for key,value in failure_counts.items()},
            diagnostic_env_steps=steps*env.n,
            diagnostic_sample_source='post_env_step_task_state;terminal_rows_may_be_reset;not_raw_substep_coverage',
            condition_failure_count_unit='sampled_control_steps_per_env',
            actor_unchanged=True if complete else None,
            policy_weights_before_sha256=weights_before,policy_weights_after_sha256=weights_after,
            policy_weights_hash_equal=(weights_before==weights_after) if complete else None,
            policy_weights_hash_format='sha256_sorted_state_dict_name_dtype_shape_cpu_bytes_v1',
            execution_mode='deterministic_student_with_mpc' if mpc else 'deterministic_student',
            mpc_enabled=mpc is not None,mpc=(mpc.stats if mpc else None),
            success_source='ordinary_native_task;not_model_prediction',
            grasp_qualified=bool(sum(episode_successes)),**copy.deepcopy(identity))

    for tick in range(args.surface_student_eval_steps):
        nominal=policy.act_inference(obs).clamp(-1,1)
        actual=mpc.refine(nominal) if mpc else nominal
        previous=obs['policy'].clone()
        value,reward,done,extra=env.step(actual)
        if mpc:mpc.observe_result(done)
        diag=diagnostic_snapshot(env.task)
        max_lift=torch.maximum(max_lift,diag['net_lift_m'].clamp_min(0))
        max_hold=torch.maximum(max_hold,diag['hold_s'])
        max_drift=torch.maximum(max_drift,diag['drift_m'])
        for key,passed in diag['conditions'].items():
            if key not in failure_counts:failure_counts[key]=torch.zeros(env.n,device=env.device,dtype=torch.long)
            failure_counts[key]+=(~passed).long()
        rows.append(dict(observations=previous.cpu(),nominal_actions=nominal.cpu(),executed_actions=actual.cpu(),
            dones=done.cpu(),success=extra['success'].cpu(),failure=extra['failure'].cpu(),
            net_lift_m=(env.task.m['bottom']-env.task.surface_reference_bottom_m).cpu()))
        obs=observations(value)
        actual_ticks=tick+1
        observed_stop=evaluation_stop(env.episodes,episodes,env.successes,successes,
            first_episode_only=first_episode_only,stop_on_success=stop_on_success)
        if observed_stop:
            stop_reason=observed_stop
            break
        if actual_ticks%60==0 and actual_ticks<args.surface_student_eval_steps:
            write_progress(output/'student_evaluation_progress.json',report(actual_ticks))
    weights_after=policy_weights_sha256(policy.state_dict())
    unchanged=all(torch.equal(v,before[k]) for k,v in policy.state_dict().items())
    if not unchanged or weights_before!=weights_after:
        raise RuntimeError('student changed during inference-only evaluation')
    result=report(actual_ticks,complete=True)
    torch.save({key:torch.stack([row[key] for row in rows]) for key in rows[0]},output/'student_evaluation.pt')
    write_progress(output/'student_evaluation.json',result)
    write_progress(output/'student_evaluation_progress.json',result)
    return result
