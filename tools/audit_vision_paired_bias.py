"""Audit first-episode native action tapes and aligned 480-Hz outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def distribution(values):
    values=[v for v in values if v is not None]
    names=('p50','p90','p95','max')
    points=np.percentile(values,[50,90,95,100]).tolist() if values else [None]*4
    return dict(count=len(values), mean=float(np.mean(values)) if values else None, **dict(zip(names,points)))


def check_replay(reference, observed, physical_end, *, zero_z):
    """Compare first-episode issued commands. M0 has a zero extension after its tape ends."""
    n=(physical_end+7)//8
    if len(observed)<n:
        raise ValueError('short action tape')
    expected=np.zeros((n,24),dtype=reference.dtype)
    available=min(n,len(reference))
    expected[:available]=reference[:available]
    if zero_z:
        expected[:,17]=0.
    if not np.isfinite(expected).all() or not np.array_equal(expected,observed[:n]):
        raise ValueError('different executed action in fixed-channel replay')
    return n


def mean_physical_z(actions, physics_steps):
    """One mean per hand and trial, with partial final 60-Hz tick weighted correctly."""
    steps=(physics_steps+7)//8
    weights=np.full(steps,8,dtype=np.float64)
    weights[-1]=physics_steps-8*(steps-1)
    return np.average(actions[:steps,[14,17]],axis=0,weights=weights).tolist()


class Arm:
    def __init__(self, folder):
        self.folder=Path(folder)
        self.result=json.loads((self.folder/'paired_results.json').read_text())
        if not self.result['complete']:
            raise ValueError('incomplete arm')
        self.name=self.result['arm']
        self.identity=self.result['identity']
        self.actions=np.load(self.folder/'issued_actions.npy',mmap_mode='r',allow_pickle=False)
        with np.load(self.folder/'native_samples.npz',allow_pickle=False) as data:
            self.c={name:i for i,name in enumerate(data['fields'].tolist())}
            self.ph=data['samples']
        steps,n=len(self.actions),len(self.result['trials'])
        if self.actions.shape!=(steps,n,24) or self.ph.shape!=(8*steps,n,len(self.c)):
            raise ValueError('invalid rate, action or physical shape')
        self.inputs={name:digest(self.folder/name) for name in
                     ('paired_results.json','issued_actions.npy','native_samples.npz','evaluation_reset_readback.json')}
        measured=json.loads((self.folder/'evaluation_reset_readback.json').read_text())
        states=[[r[k] for k in ('trial_id','authored_state_sha256','readback_state_sha256')]
                for r in measured['actual_states']]
        if states!=self.identity['states'] or any(len({r[i] for r in states})!=n for i in range(3)):
            raise ValueError('actual initial states failed uniqueness or identity')
        if not np.isfinite(self.actions).all() or np.max(np.abs(self.actions))>1.000001:
            raise ValueError('invalid action tape')

    def first(self,i):
        samples=self.ph[:,i]
        events=np.flatnonzero(samples[:,self.c['done']])
        if not len(events):
            raise ValueError('missing first terminal')
        end=int(events[0])+1
        first=samples[:end]
        if not (first[:,self.c['live']]==1).all() or (first[:,self.c['measurement_valid']]!=1).any():
            raise ValueError('invalid first-episode live measurement')
        if not np.all(np.diff(first[:,self.c['physics_step']])==1):
            raise ValueError('nonconsecutive 480 Hz steps')
        ts=first[:,self.c['timestamp_s']]
        if not np.all(np.isfinite(ts)) or not np.all(np.diff(ts)>0):
            raise ValueError('nonmonotonic simulation time')
        if self.actions[(end+7)//8:,i].any():
            raise ValueError('nonzero action after first terminal')
        row=self.result['trials'][i]
        if not (row['env_id']==i and row['physical_samples']==end and row['terminal']
                and row['terminal_step']==int(first[-1,self.c['physics_step']])
                and bool(first[-1,self.c['native_success']])==row['first_terminal_success']
                and self.identity['states'][i][0]==row['trial_id']):
            raise ValueError('terminal endpoint changed')
        return first

    def metrics(self,trace):
        def v(name):return trace[:,self.c[name]]
        valid=v('contact_valid').astype(bool);slip=v('slip_valid').astype(bool)
        for name in ('lift_m','qualified_hold_s','right_palm_dz_m'):
            if not np.isfinite(v(name)).all():
                raise ValueError('nonfinite valid native metric: '+name)
        if not np.isfinite(v('slip_m')[slip]).all():
            raise ValueError('invalid slip mask')
        return dict(lift_mm=float(1000*v('lift_m').max()),
            opposed_samples=int(np.sum(valid & v('bilateral_opposed_contact').astype(bool))),
            contact_valid_samples=int(np.sum(valid)),
            slip_valid_samples=int(np.sum(slip)),
            hold_s=float(v('qualified_hold_s').max()),
            slip_mm=float(1000*v('slip_m')[slip].max()) if slip.any() else None,
            right_final_dz_mm=float(1000*v('right_palm_dz_m')[-1]))


def compare(baseline, other, *, schedule=None):
    if baseline.identity!=other.identity or (schedule and schedule.identity!=baseline.identity):
        raise ValueError('checkpoint/seed/reset pair mismatch')
    if baseline.name!='m0':
        raise ValueError('M0 action source required')
    if schedule and (schedule.name!='replay' or other.name!='right_z_zero'):
        raise ValueError('identical-schedule control must be the 24D replay')
    control=schedule or baseline
    trials=[]
    replay=other.name in ('replay','right_z_zero')
    for i,label in enumerate(baseline.result['trials']):
        b=control.first(i); o=other.first(i)
        if schedule:
            check_replay(baseline.actions[:,i], control.actions[:,i], len(b),zero_z=False)
        if replay:
            check_replay(baseline.actions[:,i],other.actions[:,i],len(o),zero_z=other.name=='right_z_zero')
        ref_first=baseline.first(i) if schedule else b
        m=min(len(b),len(o))
        both=[arm.metrics(trace) for arm,trace in ((control,b[:m]),(other,o[:m]))]
        full=[arm.metrics(trace) for arm,trace in ((control,b),(other,o))]
        control_row=control.result['trials'][i];other_row=other.result['trials'][i]
        trials.append(dict(env_id=i,trial_id=label['trial_id'],grasp_style=label['grasp_style'],
            baseline_steps=len(b),alternative_steps=len(o),common_physics_steps=m,
            extension_beyond_m0_control_steps=max(0,(len(o)+7)//8-(len(ref_first)+7)//8),
            raw_value_exact_match_on_common_horizon={name:bool(np.array_equal(
                b[:m,control.c[name]],o[:m,other.c[name]],equal_nan=True))
                for name in ('lift_m','right_palm_dz_m','bilateral_opposed_contact','qualified_hold_s','slip_m')},
            full=full,common=both,baseline_return=control_row['task_return'],other_return=other_row['task_return'],
            common_mean_physical_z=[mean_physical_z(arm.actions[:,i],m) for arm in (control,other)],
            baseline_success=control_row['first_terminal_success'],other_success=other_row['first_terminal_success'],
            baseline_failure=control_row['failure_reason'],other_failure=other_row['failure_reason']))
    groups={}
    for style in sorted({r['grasp_style'] for r in trials}):
        g=[r for r in trials if r['grasp_style']==style]
        def summary(arm,scope):
            metrics=[r[scope][arm] for r in g];valid=sum(v['contact_valid_samples'] for v in metrics)
            return dict(lift_mm=distribution([v['lift_mm'] for v in metrics]),
                hold_s=distribution([v['hold_s'] for v in metrics]),
                slip_mm=distribution([v['slip_mm'] for v in metrics]),
                slip_valid_samples=sum(v['slip_valid_samples'] for v in metrics),
                contact_valid_samples=valid,
                opposed_fraction=sum(v['opposed_samples'] for v in metrics)/valid if valid else None,
                right_final_dz_mm=distribution([v['right_final_dz_mm'] for v in metrics]))
        groups[style]=dict(pairs=len(g),m0_zero_extension_trials=sum(r['extension_beyond_m0_control_steps']>0 for r in g),
            m0_zero_extension_control_steps=sum(r['extension_beyond_m0_control_steps'] for r in g),
            alternative_survived_longer=sum(r['baseline_steps']<r['alternative_steps'] for r in g),
            common_horizon_exact_lift_traces=sum(r['raw_value_exact_match_on_common_horizon']['lift_m'] for r in g),
            full=[summary(k,'full') for k in range(2)],common=[summary(k,'common') for k in range(2)],
            common_mean_physical_z_left_right=np.mean([r['common_mean_physical_z'] for r in g],axis=0).tolist(),
            paired_lift_delta_mm=distribution([r['full'][1]['lift_mm']-r['full'][0]['lift_mm'] for r in g]),
            full_episode_return_delta=distribution([r['other_return']-r['baseline_return'] for r in g]),
            success_count=[sum(r[k+'_success'] for r in g) for k in ('baseline','other')],
            joint_limit_count=[sum(r[k+'_failure']=='joint_limit' for r in g) for k in ('baseline','other')])
    return dict(arms=[control.name,other.name],groups=groups,trials=trials,input_hashes={
        arm.name:arm.inputs for arm in (baseline,control,other)},
        identical_23_channels_verified=replay, timestep_reward_recorded=False)


def check_full_cohort(result):
    styles={'pinch_thumb_index','tripod_thumb_index_middle'}
    if (set(result['groups'])!=styles or any(g['pairs']<100 for g in result['groups'].values())):
        raise ValueError('full paired report requires at least 100 distinct initial states per native grasp style')


def check_initial_visual_inputs(arms, checkpoint):
    """Independently reconstruct the first closed-loop control step on CPU."""
    import torch
    from go2_pvcnn.control.dh116.tabletop_vision_inference import load_vision_inference
    from go2_pvcnn.control.dh116.tabletop_vision_mirror_ablation import (
        m2_observation_for_actor, m2_action_to_physics)
    result={}; base=None
    for arm in arms:
        path=arm.folder/'evaluation_initial_visual_actor_inputs.pt'
        packet=torch.load(path,map_location='cpu',weights_only=True)
        raw=packet['policy']
        if base is None:base=raw
        diff=(raw-base).abs().reshape(len(raw),4,961)
        affected=torch.where(diff.flatten(1).any(1))[0].tolist()
        row=dict(sha256=digest(path),distinct_raw_rows_from_m0=len(affected),
                 differing_initial_env_ids=affected,raw_max_abs_diff=float(diff.max()),
                 field_max_abs_difference={name:float(diff[:,:,lo:hi].max()) for name,lo,hi in
                     (('paired_points_m',0,768),('point_validity_mask',768,896),
                      ('proprioception',896,958),('sensor_metadata',958,961))})
        if arm.name in ('m0','m2'):
            policy=load_vision_inference(checkpoint,packet['contract'],device='cpu')
            if policy.checkpoint_sha256!=arm.identity['checkpoint_sha256']:
                raise ValueError('incorrect checkpoint for initial action reconstruction')
            expected=(policy.act(raw) if arm.name=='m0'
                      else m2_action_to_physics(policy.act(m2_observation_for_actor(raw))))
            diff=(expected-torch.from_numpy(np.array(arm.actions[0],copy=True))).abs()
            row.update(initial_physical_action_reconstruction_max_abs_error=float(diff.max()),
                       initial_physical_action_reconstruction_mean_abs_error=float(diff.mean()))
            if float(diff.max())>1e-4:
                raise ValueError('initial physical action reconstruction mismatch')
        result[arm.name]=row
    return result


def identical_initial_input_subset(result, differing):
    omitted=set(differing)
    out={}
    for style in result['groups']:
        rows=[r for r in result['trials'] if r['grasp_style']==style and r['env_id'] not in omitted]
        out[style]=dict(pairs=len(rows),
            alternative_minus_control_lift_mm=distribution([r['full'][1]['lift_mm']-r['full'][0]['lift_mm'] for r in rows]),
            successes=[sum(r[k+'_success'] for r in rows) for k in ('baseline','other')],
            joint_limit=[sum(r[k+'_failure']=='joint_limit' for r in rows) for k in ('baseline','other')])
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('m0',type=Path)
    parser.add_argument('comparison',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--replay-control',type=Path)
    parser.add_argument('--checkpoint',type=Path,help='independently reconstruct the first actions on CPU')
    args=parser.parse_args()
    baseline=Arm(args.m0);other=Arm(args.comparison)
    schedule=Arm(args.replay_control) if args.replay_control else None
    result=compare(baseline,other,schedule=schedule)
    check_full_cohort(result)
    if args.checkpoint:
        result['initial_visual_audit']=check_initial_visual_inputs(
            [arm for arm in (baseline,other,schedule) if arm is not None],args.checkpoint)
        if other.name=='m2':
            ids=result['initial_visual_audit']['m2']['differing_initial_env_ids']
            result['matched_raw_initial_inputs']=identical_initial_input_subset(result,ids)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
