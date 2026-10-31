"""Collect real pre-action observations by replaying qualified native actions.

Only first, reset-free episodes that actually succeed through the ordinary
TensorEnv can become demonstrations. Source physics evidence remains immutable.
"""
import hashlib
import json
from pathlib import Path
import time
import torch

VERSION='dh116-surface-demonstrations-v1'


def contact_style(control_mean_loads):
    """Classify the entire native hold window, using the real 60Hz means."""
    x=control_mean_loads
    if x.ndim!=3 or x.shape[1:]!=(2,5) or not len(x) or not torch.isfinite(x).all():
        raise ValueError('nonempty finite [control_period,2,5] loads required')
    names=('thumb','index','middle','ring','little')
    style='other_or_intermittent_contact'
    if bool((x[...,:2]>.05).all()&(x[...,2:]<=.05).all()):style='pinch_thumb_index'
    elif bool((x[...,:3]>.05).all()&(x[...,3:]<=.05).all()):style='tripod_thumb_index_middle'
    return dict(style=style,always_bearing_fingers_per_hand=[
        [names[d] for d in range(5) if bool((x[:,h,d]>.05).all())] for h in range(2)],
        mean_digit_load_n=x.mean(0).tolist())


def load_teacher_actions(path):
    path=Path(path)
    data=torch.load(path,map_location='cpu',weights_only=False)
    metadata=data.get('metadata',{})
    if metadata.get('qualified_lift') is not True or metadata.get('action_size')!=24:
        raise ValueError('qualified native 24D lift demonstration required')
    actions=data.get('control',{}).get('action')
    if not isinstance(actions,torch.Tensor) or actions.ndim!=3 or actions.shape[1:]!=(1,24):
        raise ValueError('one selected teacher trajectory with 24D actions required')
    if not torch.isfinite(actions).all() or (actions.abs()>1).any() or len(actions)<2:
        raise ValueError('finite bounded teacher actions required')
    identity=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_qualified=True,source_policy_observations_recorded=metadata.get('policy_observations_recorded',False),
        grasp_style=metadata.get('grasp_style','unknown'))
    samples=data.get('physics',{}).get('effective_pad_loads_n')
    if identity['grasp_style']=='unknown' and isinstance(samples,torch.Tensor) and samples.shape[1:]==(1,2,5) and len(samples)%8==0:
        identity['grasp_style']=contact_style(samples.reshape(-1,8,1,2,5).mean(1)[-300:,0])['style']
    return actions[:,0].float(),identity


@torch.no_grad()
def collect_replay(env,obs,actions,contract,output,*,source_identity):
    if not getattr(env.task,'surface_force_sharing',False):
        raise ValueError('surface task required for teacher replay')
    observation_size=contract['observation_size']
    if obs.ndim!=2 or obs.shape!=(env.n,observation_size):raise ValueError('versioned real observations required')
    if actions.ndim!=2 or actions.shape[1]!=24 or not torch.isfinite(actions).all() or (actions.abs()>1).any():
        raise ValueError('finite bounded 24D source actions required')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if hasattr(env.world,'surface_trace_limit'):
        # The source episode determines the finite evidence budget; capture its
        # final hold frames as well as its beginning (AK has 10,240 substeps).
        env.world.surface_trace_limit=max(env.world.surface_trace_limit,len(actions)*env.world.decimation)
    ended=torch.zeros(env.n,dtype=torch.bool,device=obs.device)
    tainted=ended.clone();accepted=ended.clone()
    ends=torch.zeros(env.n,dtype=torch.long,device=obs.device)
    before=[];after=[];commands=[];rewards=[];terminals=[]
    start=time.monotonic()
    for tick,source_action in enumerate(actions):
        action=source_action.to(device=obs.device).expand(env.n,-1).clone()
        before.append(obs.detach().cpu().clone());commands.append(action.cpu().clone())
        obs,reward,done,extra=env.step(action)
        after.append(torch.where(done[:,None],extra['terminal_observation'],obs).detach().cpu().clone())
        rewards.append(reward.detach().cpu().clone());terminals.append(done.detach().cpu().clone())
        tainted|=extra['recovered'].bool()
        first=done&~ended
        accepted|=first&extra['success'].bool()&(extra['failure']==0)&~tainted
        ends[first]=tick+1;ended|=done
        if (tick+1)%60==0 or tick+1==len(actions):
            progress=dict(status='teacher_replay',control_steps=tick+1,total_control_steps=len(actions),
                successful_replays=int(accepted.sum()),first_episode_done=ended.cpu().tolist(),
                elapsed_wall_s=time.monotonic()-start)
            (output/'surface_replay_progress.json').write_text(json.dumps(progress,indent=2))
            print('SURFACE_TEACHER_REPLAY',json.dumps(progress),flush=True)
        if bool(ended.all()):break
    indices=accepted.nonzero().flatten().cpu().tolist()
    report=dict(status='teacher_replay_validated' if indices else 'teacher_replay_failed',
        successful_replays=len(indices),accepted_environments=indices,
        first_episode_done=ended.cpu().tolist(),recovered_environments=tainted.cpu().tolist(),
        control_steps=len(before),elapsed_wall_s=time.monotonic()-start,source=source_identity,
        observation_size=observation_size,action_size=24,optimization_performed=False,trained_policy=False,
        success_source='ordinary_TensorEnv_real_native_success;no_script_target_override')
    traces=getattr(env.world,'surface_trace',[])
    styles={}
    if traces:
        keys=set.intersection(*(set(row) for row in traces))
        physics={k:torch.stack([row[k] for row in traces]) for k in keys}
        torch.save(physics,output/'surface_replay_physics.pt')
        for index in indices:
            hi=int(ends[index])*env.world.decimation
            lo=max(0,hi-300*env.world.decimation)
            x=physics['effective_pad_loads_n'][lo:hi,index]
            styles[index]=contact_style(x.reshape(-1,env.world.decimation,2,5).mean(1))
    if source_identity.get('grasp_style')=='pinch_thumb_index':
        indices=[i for i in indices if styles.get(i,{}).get('style')=='pinch_thumb_index']
    report.update(successful_replays=len(indices),accepted_environments=indices,grasp_style_audit=styles,
        status='teacher_replay_validated' if indices else 'teacher_replay_failed')
    (output/'surface_replay.json').write_text(json.dumps(report,indent=2))
    if indices:
        stacked=[torch.stack(values) for values in (before,after,commands,rewards,terminals)]
        columns=[[] for _ in stacked];episode_ids=[]
        for episode,index in enumerate(indices):
            length=int(ends[index])
            for target,source in zip(columns,stacked):target.append(source[:length,index])
            episode_ids.append(torch.full((length,),episode,dtype=torch.long))
        data=dict(version=VERSION,contract=contract,source=source_identity,
            qualification=dict(qualified=True,successful_replays=len(indices),
                proof_path=str(output/'surface_replay.json'),native_physics_path=str(output/'surface_replay_physics.pt')),
            episode_ids=torch.cat(episode_ids),
            episode_grasp_styles={episode:styles.get(index,{}).get('style','unclassified')
                for episode,index in enumerate(indices)})
        for name,values in zip(('observations','next_observations','actions','rewards','dones'),columns):
            data[name]=torch.cat(values)
        torch.save(data,output/'surface_demonstrations.pt')
    return report
