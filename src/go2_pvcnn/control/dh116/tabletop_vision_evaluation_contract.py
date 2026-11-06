"""Evaluation-only batch projection; never a relaxation of training resume."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .tabletop_vision_action_contract import LEGACY_VERSION, SQUASHED_VERSION, _ADDED, upgrade_action_contract
from .tabletop_vision_resume import _same, _differences, _validate_contract
from .tabletop_surface_contract import ACTION_CONTRACT, semantic_sha256

PHYSICS_IDENTITY_VERSION = 'dh116-evaluation-physical-semantics-v1'
_STYLES = ('pinch_thumb_index', 'tripod_thumb_index_middle')


def _known_training_reward(critic):
    """Only this separately versioned, pure reward extension is optional in eval."""
    if 'surface_lift_reward_experiment' not in critic:
        return None
    from .tabletop_lift_reward_experiment import experiment_contract, PROFILES
    value=critic['surface_lift_reward_experiment']
    profile=value.get('profile') if isinstance(value,dict) else None
    if profile not in PROFILES or profile=='R0' or not _same(value,experiment_contract(profile)):
        raise ValueError('evaluation requires a known exact pure training reward experiment')
    return value


def evaluation_physics_sha256(critic):
    """Bind all inherited semantics, independent of balanced batch repetition.

    The two known reset-height arrays must each be constant within its authored
    grasp group. Keep the constants keyed by actual grasp identity; do not erase
    nonuniform physical states or alter their precision. All original reward,
    control, material, geometry and native success fields remain in the digest.
    """
    if not isinstance(critic,dict): raise ValueError('evaluation critic contract required')
    value=deepcopy(critic)
    _known_training_reward(value)
    value.pop('surface_lift_reward_experiment',None)
    n=value.get('num_envs')
    for record,key in ((value,'success_bottom_m'),
            (value.get('surface_force_sharing',{}),'lift_reference_bottom_m')):
        if key not in record: continue
        heights=record[key]
        if (type(n) is not int or n<2 or n%2 or not isinstance(heights,list) or len(heights)!=n
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in heights)):
            raise ValueError('balanced finite native reference array required for evaluation physics identity')
        halves=(heights[:n//2],heights[n//2:])
        if any(any(type(v) is not type(part[0]) or v!=part[0] for v in part) for part in halves):
            raise ValueError('uniform native reference within each grasp required for evaluation physics identity')
        record[key]={'by_grasp':dict(zip(_STYLES,(part[0] for part in halves)))}
    return semantic_sha256(value)


def evaluation_execution_contract(projected):
    """R0 scene contract for actor evaluation, retaining full source elsewhere.

    Exact training resume never calls this helper. Check the original provenance
    fields before rebinding just the two hashes affected by the removed pure
    reward extension; other v3 action metadata is not regenerated or ignored.
    """
    value=deepcopy(projected)
    critic=value['privileged_critic_contract']
    experiment=_known_training_reward(critic)
    if experiment is None: return value
    provenance=value.get('physics_reward_provenance')
    if value.get('version')!=SQUASHED_VERSION or not isinstance(provenance,dict):
        raise ValueError('pure reward evaluation requires versioned v3 contract provenance')
    digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    authored={key:deepcopy(item) for key,item in critic.items() if 'sha256' in key}
    if (provenance.get('privileged_critic_contract_sha256')!=digest(critic) or
            not _same(provenance.get('critic_authored_configuration_sha256'),authored)):
        raise ValueError('evaluation reward source contract provenance mismatch')
    critic.pop('surface_lift_reward_experiment')
    if 'surface_contract_sha256' in critic:
        critic['surface_contract_sha256']=semantic_sha256(critic)
    provenance['privileged_critic_contract_sha256']=digest(critic)
    provenance['critic_authored_configuration_sha256']={key:deepcopy(item) for key,item in critic.items() if 'sha256' in key}
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def evaluation_contract(source, num_envs):
    """Project only batch dimensions and known repeated native references.

    Keep calibration, raster resolution, all physical/controller/action fields
    and network semantics exact. Source remains an owned immutable value.
    """
    if type(num_envs) is not int or num_envs < 2 or num_envs % 2:
        raise ValueError('evaluation contract requires a positive even batch')
    base = deepcopy(source)
    squashed = base.get('version') == SQUASHED_VERSION
    if squashed:
        base['version'] = LEGACY_VERSION
        for key in _ADDED: base.pop(key, None)
        base.get('network', {}).pop('actor_output', None)
    old_n = _validate_contract(base)
    for key in ('camera_config_sha256','primary_reset_sha256','secondary_reset_sha256'):
        digest=base.get(key)
        if (not isinstance(digest,str) or len(digest)!=64 or
                any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('visual source contract requires SHA256 provenance: '+key)
    raster=base.get('camera_resolution_wh')
    if (not isinstance(raster,list) or len(raster)!=2 or
            any(type(x) is not int or x<1 for x in raster)):
        raise ValueError('visual source contract requires real camera raster dimensions')
    expected_groups = [dict(label=s, num_envs=old_n//2) for s in
        ('pinch_thumb_index', 'tripod_thumb_index_middle')]
    if base.get('groups') != expected_groups:
        raise ValueError('evaluation contract requires contiguous balanced grasp groups')
    if base.get('network') != dict(type='masked_PointNet_two_hand_shared_encoder_plus_four_frame_MLP',
            point_mlp=[3,32,64,64], action_mlp=[772,128,128,24]):
        raise ValueError('unsupported visual network contract')
    if base.get('action_contract') != ACTION_CONTRACT:
        raise ValueError('evaluation requires the original physical action contract')
    base['num_envs'] = num_envs
    for group in base['groups']: group['num_envs'] = num_envs//2
    if base.get('camera_mode') == 'fixed_per_env': base['camera_batch_size'] = num_envs
    else: base['camera_batch_size'] = min(num_envs, base['camera_batch_size'])
    for field in ('minibatches',):
        if field in base.get('ppo', {}): base['ppo'][field] = min(8, num_envs)
    if 'logical_minibatches' in base.get('memory_contract', {}):
        base['memory_contract']['logical_minibatches'] = min(8, num_envs)
    critic = base['privileged_critic_contract']
    if 'num_envs' in critic: critic['num_envs'] = num_envs
    if 'minibatches' in critic: critic['minibatches'] = min(8, num_envs)
    for record, key in ((critic, 'success_bottom_m'),
            (critic.get('surface_force_sharing', {}), 'lift_reference_bottom_m')):
        if key not in record: continue
        values = record[key]
        if not isinstance(values, list) or len(values) != old_n:
            raise ValueError('invalid evaluation contract native references')
        halves = (values[:old_n//2], values[old_n//2:])
        if any(any(v != part[0] for v in part) for part in halves):
            raise ValueError('evaluation requires uniform source references within each grasp')
        record[key] = [part[0] for part in halves for _ in range(num_envs//2)]
    if 'surface_contract_sha256' in critic:
        critic['surface_contract_sha256'] = semantic_sha256(critic)
    return upgrade_action_contract(base) if squashed else base


def validate_evaluation_contract(source, current):
    expected = evaluation_contract(source, current.get('num_envs'))
    changes = list(_differences(evaluation_execution_contract(expected),
                               evaluation_execution_contract(current)))
    if changes:
        raise ValueError('unapproved evaluation contract changes: '+', '.join(r['path'] for r in changes))
    return deepcopy(current)


def validate_evaluation_output(output, checkpoint, manifest, *, precreated=None):
    if Path(output).is_symlink(): raise ValueError('evaluation output cannot be a symlink')
    output = Path(output).resolve()
    for source in (checkpoint, manifest):
        source = Path(source).resolve()
        if output == source or output in source.parents or source.parent == output:
            raise ValueError('evaluation output must be independent of immutable inputs')
    if output.is_symlink(): raise ValueError('evaluation output cannot be a symlink')
    if output.exists() and (precreated is None or output != Path(precreated).resolve()):
        raise FileExistsError(output)
    return output


def validate_vision_evaluation_options(args):
    """Read-only pre-Kit validation; no exact-training-resume N guard applies."""
    checkpoint=getattr(args,'vision_evaluation_checkpoint',None)
    manifest_path=getattr(args,'vision_evaluation_manifest',None)
    contact_flags=(getattr(args,'grasp_contact_diagnostics',False)
                   or getattr(args,'grasp_contact_probe',False))
    minimum=100
    if contact_flags:
        if not checkpoint:
            raise ValueError('contact diagnostics require exclusive visual evaluation; cannot start PPO')
        from .tabletop_grasp_contact_native import evaluation_minimum
        minimum=evaluation_minimum(args)
    if not checkpoint:
        if manifest_path: raise ValueError('evaluation manifest requires evaluation checkpoint')
        return
    if not manifest_path: raise ValueError('evaluation checkpoint requires frozen manifest')
    forbidden=('resume','initialize_from','play_checkpoint','control_probe_checkpoint',
        'teacher_search_cases','actor_initialization','surface_teacher_replay','surface_mpc',
        'surface_demonstrations','surface_student_actor','surface_student_eval_steps',
        'surface_dagger_collect','surface_corrections','manual_control','depth_coordinate_probe',
        'startup_only','probe_only','vision_resume_downsize')
    if any(getattr(args,key,None) for key in forbidden):
        raise ValueError('visual evaluation is exclusive of training/teacher/MPC/extra actor modes')
    if (getattr(args,'vision_warm_start_optimizer','preserve')!='preserve' or
            getattr(args,'vision_lift_reward_experiment','R0')!='R0'):
        raise ValueError('visual evaluation is exclusive of optimizer warm-start and training reward changes')
    if (not getattr(args,'vision_student',False) or not getattr(args,'tensor_pipeline',False)
            or not getattr(args,'headless',False)):
        raise ValueError('visual evaluation requires headless visual tensor mode')
    maximum=getattr(args,'vision_evaluation_max_control_steps',0)
    if type(maximum) is not int or not 1<=maximum<=100000:
        raise ValueError('bounded evaluation max control steps required')
    validate_evaluation_output(args.output,checkpoint,manifest_path)
    import json
    import torch
    from .tabletop_evaluation_trials import validate_manifest, _digest
    from .tabletop_vision_action_derivation import _validate_source
    from .tabletop_vision_action_contract import validate_squashed_checkpoint, _validate_static_resume_contract
    value=validate_manifest(json.loads(Path(manifest_path).read_text()),min_trials_per_style=minimum)
    if getattr(args,'seed',value['metadata']['seed'])!=value['metadata']['seed']:
        raise ValueError('native evaluation seed must match frozen manifest seed')
    if (getattr(args,'num_envs',None)!=len(value['trials'])
            or (not getattr(args,'grasp_contact_probe',False) and args.num_envs!=200)):
        raise ValueError('native evaluation requires the frozen balanced 200-environment batch')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved.get('version')==LEGACY_VERSION: _validate_source(saved)
    elif saved.get('version')==SQUASHED_VERSION: validate_squashed_checkpoint(saved)
    else: raise ValueError('unsupported visual evaluation version')
    source=saved['contract']; expected=evaluation_contract(source,args.num_envs)
    if (getattr(args,'vision_camera_mode','bounded')!='fixed' or
            [getattr(args,'vision_width',None),getattr(args,'vision_height',None)]!=[224,96]):
        raise ValueError('native evaluation requires fixed original calibrated 224x96 RGB-D')
    distribution='tanh_gaussian' if saved['version']==SQUASHED_VERSION else 'gaussian_clip'
    if getattr(args,'vision_action_distribution','gaussian_clip')!=distribution:
        raise ValueError('evaluation action distribution must match checkpoint version')
    _validate_static_resume_contract(args,evaluation_execution_contract(expected),{})
    if value['metadata']['camera_sha256']!=source['camera_config_sha256']:
        raise ValueError('evaluation manifest camera provenance mismatch')
    if value['metadata']['physics_sha256']!=evaluation_physics_sha256(source['privileged_critic_contract']):
        raise ValueError('evaluation manifest physics provenance mismatch')
