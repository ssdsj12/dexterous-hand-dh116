"""Default-off Student evaluation requests and pre-reset 480Hz evidence.

This writer consumes existing TensorEnv diagnostic callbacks and native read
payloads. It never controls, steps, resets, rewards, or labels expert actions.
Missing values stay NA with separate availability/finite flags. Backend DOF
targets, IsaacLab command cache, and reconstructed submissions remain distinct.
"""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


VERSION='dh116-student-evaluation-physics-audit-v1'
AUDIT_FIELDS={
    'raw_action24':(24,), 'requested_palm_target_m':(2,3),
    'pre_projection_palm_target_m':(2,3), 'projected_palm_target_m':(2,3),
    'final_sdk_palm_target_m':(2,3), 'control_frame_sdk_target_m':(2,3),
    'actual_palm_pose_m':(2,3), 'palm_displacement_m':(2,3),
    'actual_palm_velocity_m_s':(2,3), 'table_clearance_m':(2,),
    'safe_z_m':(2,), 'projection_reasons':(2,), 'contact_loads_n':(2,),
    'contact_valid':(2,), 'failure_code':(),
}
PAD_FIELDS={'effective_pad_loads_n':(2,5),'effective_pad_valid':(2,5),
    'effective_pad_vectors_w':(2,5,3),'effective_pad_points_w':(2,5,3),
    'effective_pad_depth_m':(2,5),'rejected_contact_loads_n':(2,)}


def _owned(value):
    return value.detach().cpu().numpy().copy() if isinstance(value,torch.Tensor) else np.array(value,copy=True)


def _plain(value):
    if isinstance(value,dict):return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(np.ndarray,torch.Tensor)):return _plain(_owned(value).tolist())
    if isinstance(value,(list,tuple)):return [_plain(v) for v in value]
    if isinstance(value,np.generic):return _plain(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def validate_evaluation_physics_diagnostics_options(args):
    enabled=bool(getattr(args,'vision_evaluation_physics_audit',False))
    if enabled and (not getattr(args,'vision_evaluation_checkpoint',None)
            or not getattr(args,'vision_student',False)
            or any(getattr(args,key,None) for key in ('paired_bias_arm','paired_bias_reference',
                'vision_teacher_source','surface_teacher_replay','vision_m2_replay',
                'bimanual_tripod_teacher_coordination_proof','bimanual_mapping_probe',
                'tripod_recovery_stage'))
            or getattr(args,'vision_mirror_preprocessing','m0_raw')!='m0_raw'):
        raise ValueError('Student physics audit requires independent plain M0 visual evaluation')
    return enabled


class EvaluationPhysicsDiagnostics:
    def __init__(self,env,world,manifest,output,*,chunk_batches=32,checkpoint_sha256=None,manifest_sha256=None):
        if type(chunk_batches) is not int or not 1<=chunk_batches<=128:
            raise ValueError('bounded diagnostic chunk_batches in1..128 required')
        self.env,self.world,self.n=env,world,world.n
        self.output=Path(output)/'evaluation_action_diagnostics'
        self.output.mkdir(exist_ok=False)
        self.prior=getattr(env,'physics_action_diagnostics',None)
        self.prior_policy_step=getattr(env,'_diagnostic_policy_step',0)
        self.chunk_batches=chunk_batches;self.buffer=[];self.paths=[];self.closed=False
        self.epoch=0;self.policy_step=-1;self.request=None;self.last_clock=None
        self.pending_measurement_sources=None
        self.initial_ids=_owned(env.diagnostic_episode_ids)
        self.first_closed=np.zeros(self.n,dtype=bool);self.terminal_keys=set()
        self.trials=[row['trial_id'] for row in manifest['trials']]
        self.styles=[row['grasp_style'] for row in manifest['trials']]
        if len(self.trials)!=self.n:raise ValueError('manifest/environment diagnostic identity mismatch')
        self.rows=0;self.policy_batches=0;self.backend_status={};self.events=[];self.coverage={}
        self.policy_path=self.output/'policy_requests.jsonl.gz'
        self.terminal_path=self.output/'terminals.jsonl.gz'
        self.policy_stream=gzip.open(self.policy_path,'xt',encoding='utf-8',compresslevel=1)
        try:self.terminal_stream=gzip.open(self.terminal_path,'xt',encoding='utf-8',compresslevel=1)
        except BaseException:
            self.policy_stream.close()
            raise
        self.checkpoint_sha256=checkpoint_sha256
        self.manifest_sha256=manifest_sha256
        env.physics_action_diagnostics=self

    def begin_reset(self):
        if self.prior is not None:self.prior.begin_reset()
        self.epoch+=1;self.last_clock=None;self.request=None;self.policy_step=-1
        self.pending_measurement_sources=None
        self.events.append(dict(kind='explicit_env_reset',reset_epoch=self.epoch,
            scope='new_diagnostic_identity;not_new_independent_manifest_trial'))

    def _uids(self,episodes):
        return np.array([f'{self.output.parent.name}:{self.epoch}:{i}:{int(ep)}' for i,ep in enumerate(episodes)])

    def submit(self,action,*,policy_step):
        request=_owned(action)
        if request.shape!=(self.n,24) or type(policy_step) is not int or policy_step<0:
            raise ValueError('actual Actor [N,24] request and explicit nonnegative policy step required')
        if policy_step<=self.policy_step:raise ValueError('policy submission clock must increase within reset epoch')
        self.policy_step=policy_step;self.request=request
        self.env._diagnostic_policy_step=policy_step
        episodes=_owned(self.env.diagnostic_episode_ids)
        step=getattr(self.world,'_physics_steps',None)
        step_available=isinstance(step,(int,np.integer)) and not isinstance(step,bool) and step>=0
        sim=getattr(getattr(self.world,'w',None),'sim',None)
        when=getattr(sim,'current_time',None)
        time_available=isinstance(when,(float,int,np.number)) and not isinstance(when,bool)
        self.policy_stream.write(json.dumps(_plain(dict(source='actual_Student_Actor_request24',
            reset_epoch=self.epoch,policy_step=policy_step,
            env_ids=np.arange(self.n),episode_ids=episodes,episode_uids=self._uids(episodes),
            trial_ids=self.trials,grasp_styles=self.styles,
            first_trial_eligible=(self.epoch==0)&~self.first_closed&(episodes==self.initial_ids),
            physics_step_before_submit=int(step) if step_available else None,
            timestamp_s_before_submit=float(when) if time_available else None,
            submit_physics_step_available=step_available,submit_timestamp_available=time_available,
            submit_timestamp_finite=bool(time_available and np.isfinite(when)),
            submit_clock_sources=dict(physics_step='TensorWorld._physics_steps before env.step' if step_available else None,
                timestamp='world.w.sim.current_time before env.step' if time_available else None),
            actor_request24=request,
            request_finite=np.isfinite(request).all(-1))),allow_nan=False,separators=(',',':'))+'\n')
        self.policy_batches+=1

    def _field(self,result,name,value,shape):
        available=value is not None
        if available:
            array=_owned(value)
            available=array.shape==(self.n,*shape)
        if not available:array=np.full((self.n,*shape),np.nan)
        result[name]=array
        result[name+'__available']=np.full(self.n,available,dtype=bool)
        result[name+'__finite']=np.isfinite(array).reshape(self.n,-1).all(-1) if array.size else np.zeros(self.n,dtype=bool)
        result[name+'__source_valid']=result[name+'__available']&result[name+'__finite']

    def capture_measurement_sources(self,measurement):
        """Keep only same-substep raw source presence/finite masks, once.

        TensorTask intentionally retains only its observation/reward keys, so
        its filtered ``m`` cannot establish whether raw wrist velocity existed.
        No measurement is refreshed or modified here.
        """
        prior_capture=getattr(self.prior,'capture_measurement_sources',None)
        if prior_capture is not None:prior_capture(measurement)
        sources={}
        for name in ('wrist_velocity','hand_object_loads','contact_valid'):
            value=measurement.get(name) if isinstance(measurement,dict) else None
            valid=np.zeros(self.n,dtype=bool)
            if value is not None and getattr(value,'ndim',0) and value.shape[0]==self.n:
                if isinstance(value,torch.Tensor):
                    valid=_owned(torch.isfinite(value).reshape(self.n,-1).all(-1))
                else:
                    valid=np.isfinite(value).reshape(self.n,-1).all(-1).copy()
            sources[name]=valid
        self.pending_measurement_sources=sources

    def record_batch(self,env_id,episode_id,policy_step,physics_step,timestamp_s,live,terminal,audit):
        measurement_sources=self.pending_measurement_sources
        self.pending_measurement_sources=None
        if self.prior is not None:self.prior.record_batch(env_id,episode_id,policy_step,physics_step,timestamp_s,live,terminal,audit)
        if self.closed or self.request is None:raise RuntimeError('Student diagnostic requires submit before physical sample')
        ids,episodes,steps,times,live,terminal=map(_owned,(env_id,episode_id,physics_step,timestamp_s,live,terminal))
        if (any(v.shape!=(self.n,) for v in (ids,episodes,steps,times,live,terminal))
                or not np.array_equal(ids,np.arange(self.n)) or not np.isfinite(times).all()
                or not all(np.issubdtype(value.dtype,np.integer) for value in (ids,episodes,steps))
                or not (steps==steps[0]).all() or not (times==times[0]).all()
                or int(policy_step)!=self.policy_step):
            raise ValueError('Student diagnostic physical identity/clock mismatch')
        clock=(int(steps.min()),float(times.min()))
        if self.last_clock is not None and (clock[0]<=self.last_clock[0] or clock[1]<self.last_clock[1]):
            raise ValueError('Student diagnostic physical clock must increase')
        self.last_clock=(int(steps.max()),float(times.max()))
        uid=self._uids(episodes)
        active_terminal=live.astype(bool)&terminal.astype(bool)
        if any(uid[i] in self.terminal_keys for i in np.flatnonzero(active_terminal)):
            raise ValueError('duplicate physical terminal identity')
        eligible=(self.epoch==0)&~self.first_closed&(episodes==self.initial_ids)
        result=dict(env_id=ids,episode_id=episodes,episode_uid=uid,reset_epoch=np.full(self.n,self.epoch),
            policy_step=np.full(self.n,policy_step),physics_step=steps,timestamp_s=times,
            live_before_substep=live,terminal=active_terminal,first_trial_eligible=eligible,
            post_terminal_noop=np.array([key in self.terminal_keys for key in uid]),
            after_first_trial_terminal=self.first_closed.copy(),trial_id=np.array(self.trials),
            grasp_style=np.array(self.styles))
        self._field(result,'actor_request24',self.request,(24,))
        for key,shape in AUDIT_FIELDS.items():
            self._field(result,'controller_raw_action24' if key=='raw_action24' else key,audit.get(key),shape)
        w=self.world
        self._field(result,'palm_pose_w',getattr(w,'roots',None),(2,7))
        self._field(result,'root_target_w_submission',getattr(w,'root_targets',None),(2,7))
        wp,wq=getattr(w,'wp',None),getattr(w,'wq',None)
        self._field(result,'object_pose_w',None if wp is None or wq is None else np.concatenate((_owned(wp),_owned(wq)),-1),(7,))
        pad=getattr(w,'pad_measurements',{})
        for key,shape in PAD_FIELDS.items():self._field(result,key,pad.get(key),shape)
        # The original audit intentionally contains some zero fallbacks. Keep
        # that payload, but never certify its fallback as an observed source.
        for name,source in (('actual_palm_velocity_m_s','wrist_velocity'),
                ('contact_loads_n','hand_object_loads'),('contact_valid','contact_valid')):
            result[name+'__source_valid']&=(measurement_sources or {}).get(source,False)
        requirements={
            'controller_raw_action24':getattr(self.env.task,'last_raw_action24',None),
            'actual_palm_pose_m':getattr(w,'episode_reset_root_pose_m',None),
            'palm_displacement_m':getattr(w,'episode_reset_root_pose_m',None),
            'final_sdk_palm_target_m':getattr(w,'episode_reset_root_pose_m',None),
        }
        for name,source_value in requirements.items():
            if source_value is None:
                result[name+'__source_valid'][:]=False
            else:
                original=_owned(source_value)
                valid_source=(np.isfinite(original).reshape(self.n,-1).all(-1)
                    if original.ndim and original.shape[0]==self.n else np.zeros(self.n,dtype=bool))
                result[name+'__source_valid']&=valid_source
        from .tabletop_joint_target_readback import read_joint_target_evidence
        hands=getattr(w,'hands',[])
        for hi,side in enumerate(('left','right')):
            names=list(hands[hi].joint_names) if len(hands)>hi else []
            d=len(names)
            for key,attr in (('q','q'),('qd','qd')):
                value=getattr(w,attr,None)
                self._field(result,side+'_'+key,None if value is None or not d else value[:,hi],(d,))
            evidence=read_joint_target_evidence(hands[hi]) if len(hands)>hi else {}
            status={k:v for k,v in evidence.items() if k not in ('backend','command_cache','dof_types')}
            types=evidence.get('dof_types',{})
            self._field(result,side+'_dof_types',types.get('values') if types.get('available') else None,(d,))
            status['dof_types']={k:v for k,v in types.items() if k not in ('values','finite_env')}
            for source,key in (('backend','backend_target'),('command_cache','command_cache')):
                packet=evidence.get(source,{})
                self._field(result,side+'_'+key,packet.get('values') if packet.get('available') else None,(d,))
                status[source]={k:v for k,v in packet.items() if k not in ('values','finite_env')}
            serial=json.dumps(_plain(status),sort_keys=True,allow_nan=False)
            if self.backend_status.get(side)!=serial:
                self.events.append(dict(kind='target_source_status',hand=side,reset_epoch=self.epoch,
                    physics_step=int(steps[0]),timestamp_s=float(times[0]),evidence=_plain(status)))
                self.backend_status[side]=serial
        for i in np.flatnonzero(active_terminal):
            row=dict(env_id=int(i),episode_id=int(episodes[i]),episode_uid=str(uid[i]),
                reset_epoch=self.epoch,policy_step=int(policy_step),physics_step=int(steps[i]),
                timestamp_s=float(times[i]),failure_code=_plain(result['failure_code'][i]),
                first_trial_eligible=bool(eligible[i]),
                values={k:_plain(v[i]) for k,v in result.items()})
            self.terminal_stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
            self.terminal_stream.flush();self.terminal_keys.add(str(uid[i]))
        for key in result:
            if key.endswith('__available'):
                name=key[:-len('__available')]
                counts=self.coverage.setdefault(name,dict(live_samples=0,available_live_samples=0,
                    finite_live_samples=0,source_valid_live_samples=0))
                active=live.astype(bool)
                counts['live_samples']+=int(active.sum())
                counts['available_live_samples']+=int((active&result[key]).sum())
                counts['finite_live_samples']+=int((active&result[name+'__finite']).sum())
                counts['source_valid_live_samples']+=int((active&result[name+'__source_valid']).sum())
        self.first_closed|=active_terminal&eligible
        self.buffer.append(result);self.rows+=self.n
        if len(self.buffer)>=self.chunk_batches:self._flush()

    def _flush(self):
        if not self.buffer:return
        path=self.output/f'physics_{len(self.paths):05d}.npz'
        values={key:np.stack([row[key] for row in self.buffer]) for key in self.buffer[0]}
        with path.open('xb') as stream:np.savez_compressed(stream,**values)
        self.paths.append(path);self.buffer.clear()

    def finish(self,*,error=None):
        if self.closed:return getattr(self,'summary',None)
        self.closed=True
        self.env.physics_action_diagnostics=self.prior
        self.env._diagnostic_policy_step=self.prior_policy_step
        try:self._flush()
        finally:
            self.policy_stream.close();self.terminal_stream.close()
        paths=self.paths+[self.policy_path,self.terminal_path]
        self.summary=dict(version=VERSION,status='error' if error else 'recorded',error=error,
            policy_batches=self.policy_batches,physics_rows=self.rows,terminal_count=len(self.terminal_keys),
            first_trial_terminal_count=int(self.first_closed.sum()),checkpoint_sha256=self.checkpoint_sha256,
            manifest_sha256=self.manifest_sha256,trial_identity_count=self.n,
            scope='diagnostic_repeat_of_same_manifest;not_additional_independent_initial_states',
            actor_source='actual_Student_Actor_request_before_begin_step;not_Teacher_or_latent',
            native_backend_target='root_physx_view.get_dof_position_targets;no_cache_fallback',
            command_cache='IsaacLab_user_command_cache;not_backend_or_actual_motion',
            final_sdk_palm_target_m='existing_TensorWorld_derived_submission_coordinate;not_backend_readback',
            numeric_absence='NaN plus __available=False;present_nonfinite preserved;upstream fallback has__source_valid=False',
            capture_timing='TensorEnv.record_physics_substep_after_consume_before_terminal_reset',
            terminal_policy='owned_first_terminal_per_episode_uid;later_noop_cannot_overwrite',
            reset_identity='run_name:explicit_Env_reset_epoch:env_id:diagnostic_episode_id',
            source_events=self.events,native_trials_certified_by_writer=0,
            new_independent_initial_states=0,student_success_claimed=False,
            source_coverage=self.coverage,
            writer_source_sha256=_sha(__file__),
            field_sources=dict(
                audit_fields='copied verbatim from existing TensorWorld.last_action_diagnostics at its original capture stage',
                raw_source_validity='presence/finite masks from the same TensorEnv raw measurement packet;consumed once;not filtered TensorTask.m',
                requested_palm_target_m='task.palm_targets snapshot in original advance;not an independently instrumented pre-control target',
                pre_projection_palm_target_m='original palms plus already bounded movement before original target projection',
                projected_palm_target_m='original projected world.palms target snapshot',
                palm_pose_w='current TensorWorld.roots actual base_link world pose',
                named_q_qd='current TensorWorld.refresh measured complete joint buffers;units by named DOF type',
                pad_fields='current original native effective-pad reducer;geometric reference points never establish contact'),
            physical_control_changed=False,rewards_changed=False,
            files=[dict(path=str(path),sha256=_sha(path)) for path in paths])
        path=self.output/'summary.json'
        with path.open('x',encoding='utf-8') as stream:json.dump(self.summary,stream,indent=2,allow_nan=False)
        self.paths.append(path)
        return self.summary


def install_evaluation_physics_diagnostics(env,world,manifest,output,*,enabled=False,**options):
    return EvaluationPhysicsDiagnostics(env,world,manifest,output,**options) if enabled else None
