"""Seeded authored cold states and actual TensorWorld reset/readback evidence.

No settled PhysX solver state is claimed. Writer and native readback identity
are verified separately. The small perturbations are fixed before any policy
execution and never selected using policy outcomes.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch

from .linear_coupling import ACTIVE, JOINTS, expand, active_limits, coupling_matrix
from .tabletop_evaluation_trials import (SCHEMA_VERSION, GRASP_STYLES,
    validate_manifest, state_sha256, _physical_state, _digest)

TASK_RESET_FIELDS=('reference','command','previous','old_command',
    'joint_requested_target','joint_governed_target','filtered_residual')
PERTURBATION=dict(active_joint_radius_rad=.0005, object_xy_radius_m=.00005,
    method='independent_uniform_local_rng_no_outcome_filtering',
    passive_joints='existing_linear_coupling_expand', velocities='template_unchanged')


def _validate_coupling(state):
    for names,q,qd in zip(state['joint_names'],state['joint_positions'],state['joint_velocities']):
        active=np.array([q[names.index(a)] for a in ACTIVE])
        # Float32 authored template endpoints may differ from the double limit
        # by one ulp; do not alter the writer state or accept larger violations.
        low,high=active_limits()
        if np.any(active<low-1e-7) or np.any(active>high+1e-7):
            raise ValueError('initial active state violates original coupling limits')
        expected=expand(np.maximum(low,np.minimum(high,active)))
        actual=np.array([q[names.index(a)] for a in JOINTS])
        speeds=np.array([qd[names.index(a)] for a in ACTIVE])
        actual_speed=np.array([qd[names.index(a)] for a in JOINTS])
        if (not np.allclose(actual,expected,atol=1e-7,rtol=0) or
                not np.allclose(actual_speed,coupling_matrix()@speeds,atol=1e-7,rtol=0)):
            raise ValueError('complete initial q/qd must satisfy existing affine coupling')


def template_from_initialization(layout, seed, *, wrist_joint_names=()):
    """Actual unshifted initialization layout/seed; include optional carriage.

    Raised-support callers must first use the existing support translation on
    both layout and seed, exactly as GpuWorld does. Carriage reset is zero.
    """
    names=list(JOINTS)+list(wrist_joint_names)
    roots=[]
    for side in ('left','right'):
        pose=list(layout['hand_roots'][side])
        pose[0]+=seed['root_xy'][0]; pose[1]+=seed['root_xy'][1]
        pose[2]+=seed['root_offset']; roots.append(pose)
    return _physical_state(dict(joint_names=[names,names.copy()],
        joint_positions=[list(row)+[0.]*len(wrist_joint_names) for row in seed['joint_positions']],
        joint_velocities=[list(row)+[0.]*len(wrist_joint_names) for row in seed['joint_velocities']],
        roots=roots,root_velocities=[[0.]*6 for _ in range(2)],
        wire_pose=list(seed['wire_pose']),wire_velocity=list(seed['wire_velocity'])))


def generate_cold_reset_manifest(templates, *, seed, split, provenance, count_per_style=100):
    if type(count_per_style) is not int or count_per_style<100:
        raise ValueError('at least 100 cold states per style required')
    if type(seed) is not int or seed<0: raise ValueError('nonnegative seed required')
    if set(templates)!=set(GRASP_STYLES): raise ValueError('both real grasp templates required')
    templates={style:_physical_state(value) for style,value in templates.items()}
    for state in templates.values(): _validate_coupling(state)
    low,high=active_limits(); rng=random.Random(seed); rows=[]
    for style in GRASP_STYLES:
        for index in range(count_per_style):
            state=deepcopy(templates[style])
            for h in range(2):
                names=state['joint_names'][h]; q=state['joint_positions'][h]
                active=np.array([q[names.index(name)] for name in ACTIVE])
                # Bounds intersect the existing coupled limits. No passive
                # degree of freedom is assigned an independent random angle.
                active=np.array([rng.uniform(max(lo,x-.0005),min(hi,x+.0005))
                                 for x,lo,hi in zip(active,low,high)])
                full=expand(active)
                for name,value in zip(JOINTS,full): q[names.index(name)]=float(value)
            for axis in (0,1): state['wire_pose'][axis]+=rng.uniform(-.00005,.00005)
            state=_physical_state(state)
            rows.append(dict(trial_id=f'{split}:{style}:{index:04d}',grasp_style=style,
                physical_state=state,state_sha256=state_sha256(state)))
    metadata=dict(reset_kind='cold_authored_reset',seed=seed,split=split,
        validation_status='authored_unverified',
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        template_sha256=_digest(dict(templates=templates,perturbation=PERTURBATION)),**provenance)
    return validate_manifest(dict(schema_version=SCHEMA_VERSION,metadata=metadata,trials=rows))


def freeze_cold_reset_manifest(checkpoint, primary_initialization, secondary_initialization,
                               output, *, seed, split, wrist_joint_names=(
                                   'wrist_x','wrist_y','wrist_z','wrist_roll','wrist_pitch','wrist_yaw')):
    """Exclusively freeze real initialization inputs plus a provenance sidecar.

    This CPU entrypoint checks the actual template file hashes against a trusted
    complete visual checkpoint, uses adaptive_initialization's coupling checks,
    and applies its recorded support translation. Native execution is deferred.
    """
    from .tabletop_vision_evaluation_contract import (file_sha256,
        evaluation_physics_sha256, PHYSICS_IDENTITY_VERSION)
    from .tabletop_vision_action_contract import LEGACY_VERSION, SQUASHED_VERSION, validate_squashed_checkpoint
    from .tabletop_vision_action_derivation import _validate_source
    from .tabletop_adaptive_initialization import adaptive_initialization
    from .tabletop_end_supported_scene import translate_end_support_seed
    inputs=[Path(p).resolve() for p in (checkpoint,primary_initialization,secondary_initialization)]
    output=Path(output).absolute(); sidecar=output.with_suffix('.provenance.json')
    for path in (output,sidecar):
        if path.resolve() in inputs: raise ValueError('manifest output cannot overwrite immutable input')
        if path.exists() or path.is_symlink(): raise FileExistsError(path)
    hashes=[file_sha256(p) for p in inputs]
    saved=torch.load(inputs[0],map_location='cpu',weights_only=False)
    if saved.get('version')==LEGACY_VERSION: _validate_source(saved)
    elif saved.get('version')==SQUASHED_VERSION: validate_squashed_checkpoint(saved)
    else: raise ValueError('unsupported visual template source checkpoint')
    contract=saved['contract']; critic=contract['privileged_critic_contract']
    if hashes[1:]!=[contract['primary_reset_sha256'],contract['secondary_reset_sha256']]:
        raise ValueError('real initialization file hash does not match checkpoint templates')
    scene=critic.get('end_support_scene'); templates={}
    for style,path in zip(GRASP_STYLES,inputs[1:]):
        payload=json.loads(path.read_text())
        layout=dict(hand_roots={side:list(pose) for side,pose in zip(('left','right'),payload['roots'])},
                    active_joints={},wire_center=payload['wire_pose'][:3])
        layout,reset=adaptive_initialization(payload,layout,payload['source_contract'],
            object_radius_m=critic.get('object_radius_m',.008))
        if scene is not None:
            reset=translate_end_support_seed(reset,scene)
            for pose in layout['hand_roots'].values(): pose[2]+=scene['dz_m']
        templates[style]=template_from_initialization(layout,reset,wrist_joint_names=wrist_joint_names)
    value=generate_cold_reset_manifest(templates,seed=seed,split=split,provenance=dict(
        source_sha256=hashes[0],physics_sha256=evaluation_physics_sha256(critic),camera_sha256=contract['camera_config_sha256']))
    provenance=dict(schema_version='dh116-cold-reset-generator-provenance-v1',
        validation_status='authored_unverified',seed=seed,split=split,
        immutable_inputs=[dict(path=str(p),sha256=h) for p,h in zip(inputs,hashes)],
        template_sha256=value['metadata']['template_sha256'],perturbation=PERTURBATION,
        physics_identity_version=PHYSICS_IDENTITY_VERSION,
        source_privileged_critic_contract_sha256=_digest(critic),
        wrist_joint_names=list(wrist_joint_names),native_execution_performed=False)
    output.parent.mkdir(parents=True,exist_ok=True); created=[]
    try:
        with output.open('x',encoding='utf-8') as stream:
            created.append(output); json.dump(value,stream,indent=2,allow_nan=False)
        with sidecar.open('x',encoding='utf-8') as stream:
            created.append(sidecar)
            provenance['manifest_file_sha256']=file_sha256(output)
            json.dump(provenance,stream,indent=2,allow_nan=False)
        if [file_sha256(p) for p in inputs]!=hashes:
            raise RuntimeError('immutable source changed during manifest generation')
    except BaseException:
        for path in reversed(created): path.unlink(missing_ok=True)
        raise
    return value


def install_evaluation_reset(world, task, manifest, *, min_trials_per_style=100):
    """Replace existing reset buffers; task reset retains its native semantics.

    Complete q/qd and roots/wire writer states are submitted by TensorWorld's
    original reset. Root velocities must match its documented zero convention.
    Nonzero carriage coordinates are rejected: native mount reset commands zero.
    """
    value=validate_manifest(manifest,min_trials_per_style=min_trials_per_style)
    trials=value['trials']
    if len(trials)!=world.n or task.n!=world.n: raise ValueError('manifest batch must equal native batch')
    if [r['grasp_style'] for r in trials] != [s for s in GRASP_STYLES for _ in range(world.n//2)]:
        raise ValueError('manifest must use balanced contiguous grasp groups')
    q=[x.clone() for x in world.seed_q]; qd=[x.clone() for x in world.seed_qd]
    roots=world.root_seed.clone(); wire=world.seed_wire.clone(); velocity=world.seed_velocity.clone()
    for i,trial in enumerate(trials):
        state=trial['physical_state']
        _validate_coupling(state)
        if any(x!=0 for row in state['root_velocities'] for x in row):
            raise ValueError('native reset convention requires zero root velocities')
        for h,hand in enumerate(world.hands):
            names=state['joint_names'][h]
            if set(names)!=set(hand.joint_names): raise ValueError('manifest must name the complete native articulation')
            indices=[names.index(name) for name in hand.joint_names]
            q[h][i]=torch.as_tensor([state['joint_positions'][h][j] for j in indices],device=q[h].device)
            qd[h][i]=torch.as_tensor([state['joint_velocities'][h][j] for j in indices],device=qd[h].device)
            for j,name in enumerate(hand.joint_names):
                if name.startswith('wrist_') and (q[h][i,j]!=0 or qd[h][i,j]!=0):
                    raise ValueError('native carriage reset requires zero joint coordinates and velocity')
            limits=world.limits[i,h]
            if bool(((q[h][i]<limits[:,0])|(q[h][i]>limits[:,1])).any()):
                raise ValueError('manifest joint positions violate original native limits')
            pose=torch.as_tensor(state['roots'][h],device=roots.device,dtype=roots.dtype)
            roots[i,h]=pose; roots[i,h,:3]+=world.origins[i]; roots[i,h,2]-=world.initial_offset
        wire[i]=torch.as_tensor(state['wire_pose'],device=wire.device,dtype=wire.dtype)
        wire[i,:3]+=world.origins[i]
        velocity[i]=torch.as_tensor(state['wire_velocity'],device=velocity.device,dtype=velocity.dtype)
    # Prepare every replacement before the first mutation.
    commands=torch.cat([q[h][:,world.w.ids[h]] for h in range(2)],1)
    replacements={}
    for name in TASK_RESET_FIELDS:
        current=getattr(task,name)
        if name=='reference':
            replacements[name]=getattr(task,'_per_env_reset_seed',{}).get(name,current).clone()
        else: replacements[name]=torch.zeros_like(current) if name=='filtered_residual' else commands.to(current).clone()
    for h in range(2): world.seed_q[h].copy_(q[h]); world.seed_qd[h].copy_(qd[h])
    world.root_seed.copy_(roots); world.seed_wire.copy_(wire); world.seed_velocity.copy_(velocity)
    task._per_env_reset_seed=replacements
    return value


def readback_evaluation_reset(world, manifest, *, min_trials_per_style=100, tolerance=2e-5):
    """Read actual native tensors, preserve full raw state, verify writer match.

    This function alone never certifies native execution, even with a fixture
    whose attributes resemble a TensorWorld. Certification belongs to runner.
    """
    value=validate_manifest(manifest,min_trials_per_style=min_trials_per_style)
    actual=[]; unique=set(); max_error=0.
    for i,trial in enumerate(value['trials']):
        roots=world.roots[i].clone(); roots[:,:3]-=world.origins[i]
        pose=torch.cat((world.wp[i]-world.origins[i],world.wq[i]))
        state=_physical_state(dict(joint_names=[list(h.joint_names) for h in world.hands],
            joint_positions=world.q[i].detach().cpu().tolist(),joint_velocities=world.qd[i].detach().cpu().tolist(),
            roots=roots.detach().cpu().tolist(),
            root_velocities=[h.data.root_vel_w[i].detach().cpu().tolist() for h in world.hands],
            wire_pose=pose.detach().cpu().tolist(),wire_velocity=world.wv[i].detach().cpu().tolist()))
        expected=trial['physical_state']
        for key in ('joint_positions','joint_velocities','roots','root_velocities','wire_pose','wire_velocity'):
            a=torch.tensor(state[key],dtype=torch.float64); b=torch.tensor(expected[key],dtype=torch.float64)
            error=float((a-b).abs().max()); max_error=max(max_error,error)
            if error>tolerance: raise RuntimeError(f'evaluation reset readback mismatch: {trial["trial_id"]} {key} {error}')
        digest=state_sha256(state)
        if digest in unique: raise RuntimeError('duplicate actual float32 native readback state')
        unique.add(digest); actual.append(dict(trial_id=trial['trial_id'],grasp_style=trial['grasp_style'],
            authored_state_sha256=trial['state_sha256'],readback_state_sha256=digest,physical_state=state))
    return dict(schema_version='dh116-evaluation-reset-readback-v1',native_validation_complete=False,
        actual_states=actual,unique_readback_count=len(unique),max_abs_writer_readback_error=max_error,
        writer_match_tolerance=tolerance,identity='actual_local_float32_state_not_placement_or_label',
        physics_steps_advanced=0)
