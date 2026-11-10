"""Synchronized pre-action collection; labels never imply native qualification."""
import copy
from dataclasses import dataclass,field,fields
import math

import torch

from .tabletop_stereo_contract import require_aligned,validate_identity,require_sha
from .tabletop_stereo_observation import StereoHistoryBatch


def clone_tree(value,*,cpu=False):
    if isinstance(value,torch.Tensor):
        value=value.detach();return (value.cpu() if cpu else value).clone()
    if isinstance(value,dict):return {k:clone_tree(v,cpu=cpu) for k,v in value.items()}
    if isinstance(value,list):return [clone_tree(v,cpu=cpu) for v in value]
    if isinstance(value,tuple):return tuple(clone_tree(v,cpu=cpu) for v in value)
    return copy.deepcopy(value)


def _tensor(value,shape,name,*,boolean=False,finite=True):
    if (not isinstance(value,torch.Tensor) or value.shape!=shape
            or (value.dtype!=torch.bool if boolean else not value.is_floating_point())):
        raise ValueError(f'{name} must be '+('boolean' if boolean else 'floating')+f' tensor {shape}')
    if finite and not boolean and not bool(torch.isfinite(value).all()):raise ValueError(name+' must be finite')
    return value


def _actions(value,n,name):
    _tensor(value,(n,24),name)
    if bool((value.abs()>1).any()):raise ValueError(name+' must be canonical normalized24 in [-1,1]')
    return value


@dataclass
class PreActionState:
    sensor_history:StereoHistoryBatch
    privileged:torch.Tensor
    privileged_valid:torch.Tensor
    identities:list
    extra_labels:dict=field(default_factory=dict)
    label_validity:dict=field(default_factory=dict)
    state_sources:dict=field(default_factory=dict)
    privileged_identities:list|None=None
    force_qualification:dict|None=None

    def validate(self,*,max_image_age_s=1/60):
        if type(self.sensor_history) is not StereoHistoryBatch:raise TypeError('typed sensor history required')
        h=self.sensor_history.validate();n=len(h)
        _tensor(self.privileged,(n,371),'privileged')
        _tensor(self.privileged_valid,(n,371),'privileged_valid',boolean=True)
        if self.privileged.device!=h.proprio.device or self.privileged_valid.device!=h.proprio.device:
            raise ValueError('snapshot sensors and privileged labels must share a device')
        if not isinstance(self.identities,list) or len(self.identities)!=n:raise ValueError('one identity per environment required')
        seen=set()
        for i,row in enumerate(self.identities):
            validate_identity(row);key=(row['run_uuid'],row['env_id'])
            if key in seen:raise ValueError('duplicate environment identity')
            seen.add(key)
            if self.privileged_identities is not None:
                if len(self.privileged_identities)!=n:raise ValueError('privileged identity count mismatch')
                require_aligned(row,self.privileged_identities[i])
        if (not bool(h.history_valid[:,-1].all()) or not bool(h.image_valid[:,-1].all())
                or bool((h.image_age[:,-1]>max_image_age_s).any())):
            raise ValueError('stale/missing current stereo image or history frame')
        for name in ('extra_labels','label_validity','state_sources'):
            if not isinstance(getattr(self,name),dict):raise TypeError(name+' dictionary required')
        return self

    def clone(self):
        return type(self)(**{f.name:(self.sensor_history.clone() if f.name=='sensor_history' else clone_tree(getattr(self,f.name))) for f in fields(self)})


@dataclass
class StepResult:
    next_state:PreActionState
    reward:torch.Tensor
    done:torch.Tensor
    info:dict
    issued_actions:torch.Tensor
    applied_targets:dict|None=None


def require_teacher(qualification,source_sha256):
    if qualification is None:raise ValueError('qualified Teacher native proof is required before formal collection')
    require_sha(source_sha256,'Teacher checkpoint SHA256')
    from .tabletop_stereo_promotion import require_teacher_qualification
    return require_teacher_qualification(qualification,source_sha256)


def _model_device(model,fallback):
    try:return next(model.parameters()).device
    except (AttributeError,StopIteration):return fallback


def _validate_step(before,result,submitted,max_age):
    n=len(before.sensor_history)
    if type(result) is not StepResult:raise TypeError('env.step must return StepResult')
    _tensor(result.reward,(n,),'reward');_tensor(result.done,(n,),'done',boolean=True)
    _actions(result.issued_actions,n,'actually issued actions')
    if not torch.equal(result.issued_actions.to(submitted),submitted):raise ValueError('environment changed whole-action owner request without declared shielding')
    if not isinstance(result.info,dict):raise ValueError('step info dictionary required')
    execution=result.info.get('execution')
    if not isinstance(execution,list) or len(execution)!=n:raise ValueError('actual per-environment physics execution range required')
    after=result.next_state.validate(max_image_age_s=max_age)
    for i,(old,new,run) in enumerate(zip(before.identities,after.identities,execution)):
        if (not isinstance(run,dict) or run.get('physics_tick_start')!=old['physics_tick_end']
                or type(run.get('physics_tick_end')) is not int or run['physics_tick_end']<=run['physics_tick_start']
                or run.get('simulation_timestamp_start')!=old['simulation_timestamp']
                or not isinstance(run.get('simulation_timestamp_end'),(float,int))
                or not math.isfinite(run['simulation_timestamp_end']) or run['simulation_timestamp_end']<=old['simulation_timestamp']):
            raise ValueError('execution physics/time range must begin at the preserved pre-action state')
        if old['run_uuid']!=new['run_uuid'] or old['env_id']!=new['env_id'] or old['calibration_sha256']!=new['calibration_sha256']:
            raise ValueError('next-state run/environment/calibration identity mismatch')
        if new['physics_tick_start']<run['physics_tick_end'] or new['simulation_timestamp']<run['simulation_timestamp_end']:
            raise ValueError('next state precedes actual execution completion')
        h=after.sensor_history
        if bool(result.done[i]):
            if new['episode_uid']==old['episode_uid'] or new['reset_id']<=old['reset_id']:
                raise ValueError('terminal reset must change episode/reset identity')
            if int(h.history_valid[i].sum())!=1 or not bool((h.proprio[i,-1,38:]==0).all()):
                raise ValueError('reset history contains an earlier episode or previous action')
        else:
            if new['episode_uid']!=old['episode_uid'] or new['reset_id']!=old['reset_id']:
                raise ValueError('episode identity changed without a terminal reset')
            if new['policy_tick']<=old['policy_tick'] or new['frame_id']<=old['frame_id']:
                raise ValueError('next pre-action identity clocks must be increasing')
            if not torch.equal(h.proprio[i,-1,38:].to(submitted),submitted[i]):
                raise ValueError('history previous action differs from actual issued action')
    return execution


@torch.no_grad()
def collect_step(env,student,teacher,*,teacher_probability=1.,schema_only=False,
                 teacher_qualification=None,source_sha256=None,generator=None,
                 max_image_age_s=1/60,record_teacher_sample=False):
    if type(schema_only) is not bool or not math.isfinite(teacher_probability) or not 0<=teacher_probability<=1:
        raise ValueError('explicit schema mode and Teacher probability in [0,1] required')
    binding=None if schema_only else require_teacher(teacher_qualification,source_sha256)
    if not schema_only and getattr(env,'evidence_scope',None)!='native_runtime_capture_unqualified':
        raise ValueError('formal collection requires the native runtime path; CPU fixtures require schema_only=True')
    before=env.snapshot()
    if type(before) is not PreActionState:raise TypeError('env.snapshot must return PreActionState')
    before=before.clone().validate(max_image_age_s=max_image_age_s)
    n=len(before.sensor_history);device=before.privileged.device
    teacher_input=before.privileged.to(_model_device(teacher,device)).clone()
    teacher_mean=_actions(teacher.deterministic(teacher_input),n,'Teacher deterministic mean').to(device)
    teacher_mu=teacher.latent_mean(teacher_input).to(device) if hasattr(teacher,'latent_mean') else None
    teacher_std=teacher.log_std.exp().expand(n,24).to(device) if hasattr(teacher,'log_std') else None
    if teacher_mu is not None:_tensor(teacher_mu,(n,24),'Teacher latent mean')
    if teacher_std is not None:
        _tensor(teacher_std,(n,24),'Teacher latent std')
        if bool((teacher_std<=0).any()):raise ValueError('Teacher latent std must be positive')
    teacher_sample=None
    if record_teacher_sample:
        if not hasattr(teacher,'sample'):raise ValueError('Teacher does not expose a sampled policy action')
        teacher_sample=_actions(teacher.sample(teacher_input)['actions'],n,'Teacher sampled action').to(device)
    nominal=student_mu=None
    if student is not None:
        sensor=before.sensor_history.to(_model_device(student,device)).clone()
        nominal=_actions(student.deterministic(sensor),n,'Student nominal action').to(device)
        student_mu=student.latent_mean(sensor).to(device) if hasattr(student,'latent_mean') else None
        if student_mu is not None:_tensor(student_mu,(n,24),'Student latent mean')
    if teacher_probability<1 and nominal is None:raise ValueError('Student policy required for Student-owned execution')
    if teacher_probability in (0.,1.):owner=torch.full((n,),teacher_probability==1.,device=device,dtype=torch.bool)
    else:owner=(torch.rand(n,generator=generator,device='cpu')<teacher_probability).to(device)
    submitted=teacher_mean.clone() if nominal is None else torch.where(owner[:,None],teacher_mean,nominal)
    owner_names=['teacher' if value else 'student' for value in owner.tolist()]
    if hasattr(env,'set_action_owners'):
        env.set_action_owners(owner_names.copy())
    # Teacher labels were queried at exactly this preserved state. Only the
    # chosen complete action owns the following environment interval.
    result=env.step(submitted.clone())
    execution=_validate_step(before,result,submitted,max_image_age_s)
    if not schema_only and result.info.get('evidence_scope')!='native_runtime_capture_unqualified':
        raise ValueError('formal execution lacks the matching native runtime evidence scope')
    mode='teacher_reference' if teacher_probability==1 else 'dagger_student' if teacher_probability==0 else 'dagger_mixed'
    return clone_tree(dict(sensor_history=before.sensor_history.as_dict(),privileged=before.privileged,
        privileged_valid=before.privileged_valid,identities=before.identities,
        privileged_identities=before.identities if before.privileged_identities is None else before.privileged_identities,
        extra_labels=before.extra_labels,label_validity=before.label_validity,state_sources=before.state_sources,
        force_qualification=before.force_qualification,actor_raw=dict(teacher=teacher_mu,student=student_mu),
        teacher_mean=teacher_mean,teacher_sample=teacher_sample,teacher_std=teacher_std,student_nominal=nominal,
        issued_actions=result.issued_actions,applied_targets=result.applied_targets,execution=execution,
        reward=result.reward,done=result.done,info=result.info,action_owner=owner_names,
        mode=mode,teacher_probability=float(teacher_probability),dagger_student_samples=int((~owner).sum()),
        teacher_query_count=n,schema_only=schema_only,native_qualified=False,teacher_qualification=binding),cpu=True)


class TeacherCollector:
    def __init__(self,**options):self.options=options
    def run(self,env,student,teacher,steps,**options):
        if type(steps) is not int or steps<1:raise ValueError('positive bounded collection steps required')
        return [collect_step(env,student,teacher,**(self.options|options)) for _ in range(steps)]


def run(env,student,teacher,steps,**options):
    return TeacherCollector(**options).run(env,student,teacher,steps)
