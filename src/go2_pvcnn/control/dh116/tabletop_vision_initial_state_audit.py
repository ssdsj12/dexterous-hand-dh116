"""Opt-in native initial-state diagnostics; no Actor inputs or physics writes.

Pure sampling/comparison helpers do not attest native execution. Only the
concrete native training entry point can publish a native_verified sidecar.
Repeated authored templates are valid training states, not independent trials.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import torch

from .tabletop_evaluation_trials import _physical_state, state_sha256, _keys, _sha256
from .tabletop_vision_evaluation_contract import (
    PHYSICS_IDENTITY_VERSION, evaluation_physics_sha256, file_sha256)
from .tabletop_vision_experiment_checkpoint import _state_sha256

SCHEMA_VERSION='dh116-training-initial-physical-state-v1'
SIDECAR_NAME='training_initial_physical_state.json'
_STYLES=('pinch_thumb_index','tripod_thumb_index_middle')
_IDENTITY=('source_resume_sha256','initial_policy_sha256','scene_seed','num_envs',
    'groups','physics_identity','camera_sha256','source_checkpoint_version','source_derivation')


def _groups(groups,n):
    if type(n) is not int or n<2 or n%2:
        raise ValueError('initial-state audit requires a positive even native batch')
    expected=[dict(label=s,num_envs=n//2) for s in _STYLES]
    if groups!=expected or any(type(g.get('num_envs')) is not int for g in groups):
        raise ValueError('initial-state audit requires balanced contiguous native grasp groups')
    return [s for s in _STYLES for _ in range(n//2)]


def _clock(clock):
    _keys(clock,('simulation_time_s','simulation_step_index','tensor_world_physics_steps'),name='physics clock')
    t=clock['simulation_time_s']
    if type(t) not in (int,float) or not math.isfinite(t) or t<0:
        raise ValueError('finite native simulation clock required')
    for name in ('simulation_step_index','tensor_world_physics_steps'):
        if type(clock[name]) is not int or clock[name]<0:
            raise ValueError('nonnegative integer native physics clock required')
    return deepcopy(clock)


def sample_initial_physical_state(world,groups):
    """Read raw simulator tensors and clocks without refreshing or stepping.

    No native status is returned. Root/FK poses and object positions are local
    to clone origins; named articulation q/qd include every carriage joint.
    """
    labels=_groups(groups,world.n)
    sim=world.w.sim
    steps=world._physics_steps
    if type(steps) is not int or steps<0:
        raise ValueError('nonnegative integer TensorWorld physics clock required')
    clock=_clock(dict(simulation_time_s=float(sim.current_time),
        simulation_step_index=sim.current_time_step_index,tensor_world_physics_steps=steps))
    mounts=getattr(world.w,'mounts',None)
    roots=torch.stack([hand.data.root_pose_w if mounts is None else
        hand.data.body_link_pose_w[:,mounts[h].base_id] for h,hand in enumerate(world.hands)],1)
    q=torch.stack([h.data.joint_pos for h in world.hands],1)
    qd=torch.stack([h.data.joint_vel for h in world.hands],1)
    root_velocities=torch.stack([h.data.root_vel_w for h in world.hands],1)
    wire=world.wire.data
    rows=[]
    for i,label in enumerate(labels):
        local=roots[i].double().clone(); local[:,:3]-=world.origins[i]
        pose=torch.cat((wire.root_pos_w[i].double()-world.origins[i],wire.root_quat_w[i]))
        state=_physical_state(dict(joint_names=[list(h.joint_names) for h in world.hands],
            joint_positions=q[i].detach().cpu().tolist(),joint_velocities=qd[i].detach().cpu().tolist(),
            roots=local.detach().cpu().tolist(),root_velocities=root_velocities[i].detach().cpu().tolist(),
            wire_pose=pose.detach().cpu().tolist(),wire_velocity=wire.root_vel_w[i].detach().cpu().tolist()))
        rows.append(dict(env_id=i,grasp_style=label,physical_state=state,state_sha256=state_sha256(state)))
    return dict(num_envs=world.n,groups=deepcopy(groups),clock=clock,actual_states=rows)


def _snapshot(value):
    _keys(value,('num_envs','groups','clock','actual_states'),name='physical snapshot')
    labels=_groups(value['groups'],value['num_envs']); _clock(value['clock'])
    rows=value['actual_states']
    if not isinstance(rows,list) or len(rows)!=len(labels): raise ValueError('incomplete physical state coverage')
    result=[]
    for i,(row,label) in enumerate(zip(rows,labels)):
        _keys(row,('env_id','grasp_style','physical_state','state_sha256'),name='physical state row')
        if type(row['env_id']) is not int or row['env_id']!=i or row['grasp_style']!=label:
            raise ValueError('physical state environment/group ordering mismatch')
        state=_physical_state(row['physical_state'])
        if row['state_sha256']!=state_sha256(state): raise ValueError('physical state SHA256 mismatch')
        result.append(dict(row,physical_state=state))
    return dict(value,actual_states=result)


def compare_initial_physical_snapshots(before,after,*,tolerance=2e-5,require_same_clock=False):
    if type(tolerance) not in (int,float) or not math.isfinite(tolerance) or not 0<=tolerance<=2e-5:
        raise ValueError('physical state tolerance must be bounded by 2e-5')
    before,after=(_snapshot(s) for s in (before,after))
    if before['num_envs']!=after['num_envs'] or before['groups']!=after['groups']:
        raise ValueError('physical state groups/batch mismatch')
    if require_same_clock and before['clock']!=after['clock']:
        raise ValueError('native physics clock advanced during camera warmup')
    maximum=0.
    for left,right in zip(before['actual_states'],after['actual_states']):
        a,b=left['physical_state'],right['physical_state']
        if a['joint_names']!=b['joint_names']: raise ValueError('physical state joint names changed')
        for key in ('joint_positions','joint_velocities','roots','root_velocities','wire_pose','wire_velocity'):
            error=float((torch.tensor(a[key],dtype=torch.float64)-torch.tensor(b[key],dtype=torch.float64)).abs().max())
            maximum=max(maximum,error)
            if error>tolerance: raise ValueError('initial physical state mismatch: '+key)
    return dict(matched=True,max_abs_state_error=maximum,tolerance=tolerance)


def validate_camera_warmup(before,after):
    return compare_initial_physical_snapshots(before,after,require_same_clock=True)


def policy_state_sha256(policy):
    if not isinstance(policy,dict) or not policy or any(not isinstance(k,str)
        or not isinstance(v,torch.Tensor) or not bool(torch.isfinite(v).all()) for k,v in policy.items()):
        raise ValueError('complete finite policy tensor dictionary required')
    # Container class/order are not policy semantics (state_dict is OrderedDict).
    return _state_sha256(dict(policy))


def validate_training_initial_state_audit(value):
    _keys(value,('schema_version','validation_status','execution','diagnostic_only',
        'source_resume_sha256','initial_policy_sha256','source_checkpoint_version','source_derivation',
        'scene_seed','num_envs','groups','physics_identity','camera_sha256','actual_states',
        'after_camera_actual_states','warmup'),name='training initial-state audit')
    if (value['schema_version']!=SCHEMA_VERSION or value['validation_status']!='native_verified'
            or value['execution']!='native_isaac_tensor_world' or value['diagnostic_only'] is not True):
        raise ValueError('native_verified training initial-state evidence required')
    if value['source_resume_sha256'] is not None:
        _sha256(value['source_resume_sha256'],'source_resume_sha256')
    _sha256(value['initial_policy_sha256'],'initial_policy_sha256')
    _sha256(value['camera_sha256'],'camera_sha256')
    if type(value['scene_seed']) is not int or value['scene_seed']<0: raise ValueError('scene seed required')
    expected_version=(None if value['source_resume_sha256'] is None else
        'dh116-vision-student-ppo-v3-squashed-gaussian')
    if value['source_checkpoint_version']!=expected_version:
        raise ValueError('explicit v3 source version required')
    if not isinstance(value['source_derivation'],dict): raise ValueError('source derivation object required')
    identity=value['physics_identity']
    _keys(identity,('version','physics_sha256'),name='physics identity')
    if identity['version']!=PHYSICS_IDENTITY_VERSION: raise ValueError('known physics identity required')
    _sha256(identity['physics_sha256'],'physics_sha256')
    warmup=value['warmup']
    _keys(warmup,('before_clock','after_clock','physics_steps_advanced','max_abs_state_error'),name='camera warmup')
    before=dict(num_envs=value['num_envs'],groups=value['groups'],clock=warmup['before_clock'],actual_states=value['actual_states'])
    after=dict(before,clock=warmup['after_clock'],actual_states=value['after_camera_actual_states'])
    measured=validate_camera_warmup(before,after)
    if (type(warmup['physics_steps_advanced']) is not int or warmup['physics_steps_advanced']!=0
            or type(warmup['max_abs_state_error']) not in (int,float)
            or warmup['max_abs_state_error']!=measured['max_abs_state_error']):
        raise ValueError('camera warmup evidence does not match raw state/clock')
    return deepcopy(value)


def _load_audit(path):
    path=Path(path); digest=file_sha256(path)
    value=validate_training_initial_state_audit(json.loads(path.read_text()))
    if file_sha256(path)!=digest: raise ValueError('initial-state reference changed while loading')
    return value,digest


def _compare_audits(left,right,tolerance):
    left,right=(validate_training_initial_state_audit(v) for v in (left,right))
    if left['source_resume_sha256'] is None: raise ValueError('same explicit resume source required')
    for key in _IDENTITY:
        if type(left[key]) is not type(right[key]) or left[key]!=right[key]:
            raise ValueError('initial-state reference identity mismatch: '+key)
    def snapshot(v,key):
        return dict(num_envs=v['num_envs'],groups=v['groups'],clock=v['warmup']['before_clock'],actual_states=v[key])
    results=[compare_initial_physical_snapshots(snapshot(left,key),snapshot(right,key),tolerance=tolerance)
             for key in ('actual_states','after_camera_actual_states')]
    return dict(matched=True,max_abs_state_error=max(r['max_abs_state_error'] for r in results),tolerance=tolerance)


def compare_training_initial_state_audits(left_path,right_path,*,tolerance=2e-5):
    if Path(left_path).resolve()==Path(right_path).resolve(): raise ValueError('independent audit outputs required')
    left,a=_load_audit(left_path); right,b=_load_audit(right_path)
    result=_compare_audits(left,right,tolerance)
    if file_sha256(left_path)!=a or file_sha256(right_path)!=b:
        raise ValueError('initial-state evidence changed during comparison')
    return dict(result,reference_sha256=a,candidate_sha256=b)


def _require_native(world,device_world,env,native_camera):
    def concrete(obj,module,name):
        cls=getattr(sys.modules.get(module),name,None)
        return cls is not None and type(obj) is cls
    if not (native_camera and concrete(world,'go2_pvcnn.control.dh116.tabletop_gpu_isaac','GpuWorld')
            and concrete(device_world,'go2_pvcnn.control.dh116.tabletop_tensor_world','TensorWorld')
            and concrete(env,'go2_pvcnn.control.dh116.tabletop_tensor_env','TensorEnv')
            and device_world.w is world
            and concrete(world.sim,'isaaclab.sim.simulation_context','SimulationContext')):
        raise ValueError('initial-state audit requires concrete native training; fixtures cannot certify')


class TrainingInitialStateAudit:
    """Native-only lifecycle, bound to one independent training output."""
    def __init__(self,world,device_world,env,args,contract,*,native_camera,saved,source_resume_sha256):
        _require_native(world,device_world,env,native_camera)
        if contract['version']!='dh116-vision-student-ppo-v3-squashed-gaussian':
            raise ValueError('initial-state audit requires native v3 training')
        self.world=device_world; self.contract=deepcopy(contract); self.args=args
        self.saved=saved; self.source_sha=source_resume_sha256
        self.path=Path(args.output)/SIDECAR_NAME
        if self.path.exists() or self.path.is_symlink(): raise FileExistsError(self.path)
        self.reference_path=getattr(args,'vision_experiment_initial_state_reference',None)
        if self.reference_path and Path(self.reference_path).resolve()==self.path.resolve():
            raise ValueError('independent initial-state reference required')
        self.reference,self.reference_sha=_load_audit(self.reference_path) if self.reference_path else (None,None)
        self.before=self.after=None; self.policy_sha=None

    def before_camera(self):
        self.before=sample_initial_physical_state(self.world,self.contract['groups'])

    def after_camera(self):
        self.after=sample_initial_physical_state(self.world,self.contract['groups'])
        validate_camera_warmup(self.before,self.after)

    def finish(self,policy):
        self.policy_sha=policy_state_sha256(policy)
        warmup=validate_camera_warmup(self.before,self.after)
        source=self.saved or {}
        value=dict(schema_version=SCHEMA_VERSION,validation_status='native_verified',
            execution='native_isaac_tensor_world',diagnostic_only=True,
            source_resume_sha256=self.source_sha,initial_policy_sha256=self.policy_sha,
            source_checkpoint_version=source.get('version'),
            source_derivation={k:deepcopy(source[k]) for k in
                ('action_derivation','experiment_projection','reward_experiment_derivation','optimizer_reset_history') if k in source},
            scene_seed=self.args.seed,num_envs=self.world.n,groups=deepcopy(self.contract['groups']),
            physics_identity=dict(version=PHYSICS_IDENTITY_VERSION,
                physics_sha256=evaluation_physics_sha256(self.contract['privileged_critic_contract'])),
            camera_sha256=self.contract['camera_config_sha256'],actual_states=self.before['actual_states'],
            after_camera_actual_states=self.after['actual_states'],warmup=dict(
                before_clock=self.before['clock'],after_clock=self.after['clock'],physics_steps_advanced=0,
                max_abs_state_error=warmup['max_abs_state_error']))
        validate_training_initial_state_audit(value)
        if self.reference is not None:
            if file_sha256(self.reference_path)!=self.reference_sha:
                raise ValueError('initial-state reference changed after preparation')
            _compare_audits(self.reference,value,2e-5)
            if file_sha256(self.reference_path)!=self.reference_sha:
                raise ValueError('initial-state reference changed during comparison')
        with self.path.open('x',encoding='utf-8') as stream:
            json.dump(value,stream,indent=2,allow_nan=False)

    def verify_loaded_policy(self,policy):
        if self.policy_sha!=policy_state_sha256(policy):
            raise ValueError('restored policy differs from audited initial policy')
