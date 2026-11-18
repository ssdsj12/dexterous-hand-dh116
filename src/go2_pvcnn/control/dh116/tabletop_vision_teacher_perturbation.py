"""Explicit Tier1 cold object-pose resets for fresh Teacher first episodes.

Authored states are not settled solver checkpoints or successful demonstrations.
The original native reset, geometry/contact/safety guards and first terminal
remain authoritative. Historical actions are references, never state queries.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

import torch

from .tabletop_evaluation_trials import (
    GRASP_STYLES, _physical_state, state_sha256, _digest, _keys, _sha256, _number)
from .tabletop_vision_evaluation_reset import (
    _validate_coupling, _install_reset_trials, _readback_reset_trials)


VERSION = 'dh116-teacher-perturbation-manifest-v1'
PERTURBATION = dict(object_xy_radius_m=.0005, world_z_yaw_radius_rad=math.pi / 180,
    method='independent_uniform_local_rng_no_outcome_filtering', other_state='base_template_unchanged')
PROVENANCE = ('source_checkpoint', 'teacher_reference', 'initializations', 'physics_sha256',
    'camera_sha256', 'source_privileged_critic_contract_sha256', 'parent_source_groups')


def file_identity(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _identity(value):
    _keys(value, ('path', 'sha256'), name='input identity')
    if not isinstance(value['path'], str) or not value['path']:
        raise ValueError('explicit input path required')
    _sha256(value['sha256'], 'input SHA256')
    return deepcopy(value)


def _family(state, group, physics):
    return _digest(dict(base_state_sha256=state_sha256(state), parent_source_group=group,
        physics_sha256=physics, scope='shared_parent_family_not_run_seed_or_env'))


def _perturb(base, xy, yaw):
    from .tabletop_free_wrist_grasp import multiply
    state = deepcopy(base)
    for axis in (0, 1): state['wire_pose'][axis] += xy[axis]
    z = torch.tensor([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)], dtype=torch.float64)
    state['wire_pose'][3:] = multiply(z, torch.tensor(base['wire_pose'][3:], dtype=torch.float64)).tolist()
    return _physical_state(state)


def validate_teacher_perturbation_manifest(payload):
    _keys(payload, ('schema_version', 'metadata', 'base_templates', 'trials'), name='Teacher manifest')
    if payload['schema_version'] != VERSION: raise ValueError('explicit Teacher perturbation schema required')
    meta = payload['metadata']
    _keys(meta, ('tier', 'reset_kind', 'validation_status', 'seed', 'perturbation',
        'generator_sha256', 'template_sha256') + PROVENANCE, name='Teacher metadata')
    if (type(meta['tier']) is not int or meta['tier'] != 1
            or meta['reset_kind'] != 'teacher_cold_object_pose_perturbation'
            or meta['validation_status'] != 'authored_unverified'
            or meta['perturbation'] != PERTURBATION):
        raise ValueError('only explicit authored/unverified Tier1 object-pose perturbation supported')
    if type(meta['seed']) is not int or meta['seed'] < 0: raise ValueError('nonnegative local seed required')
    for key in ('generator_sha256', 'template_sha256', 'physics_sha256', 'camera_sha256',
                'source_privileged_critic_contract_sha256'): _sha256(meta[key], key)
    for key in ('source_checkpoint', 'teacher_reference'): _identity(meta[key])
    for key in ('initializations', 'parent_source_groups', 'base_templates'):
        value = payload[key] if key == 'base_templates' else meta[key]
        if not isinstance(value, dict) or set(value) != set(GRASP_STYLES):
            raise ValueError('both original grasp families required: ' + key)
    for value in meta['initializations'].values(): _identity(value)
    for value in meta['parent_source_groups'].values(): _sha256(value, 'parent source group')
    bases = {style: _physical_state(value) for style, value in payload['base_templates'].items()}
    for state in bases.values(): _validate_coupling(state)
    if meta['template_sha256'] != _digest(bases): raise ValueError('base template SHA mismatch')
    trials = payload['trials']
    if not isinstance(trials, list) or not 2 <= len(trials) <= 16 or len(trials) % 2:
        raise ValueError('Teacher batch requires 2..16 even states')
    if [r.get('grasp_style') for r in trials] != [s for s in GRASP_STYLES for _ in range(len(trials)//2)]:
        raise ValueError('Teacher reset requires balanced contiguous original grasp groups')
    ids, states = set(), set()
    for row in trials:
        _keys(row, ('trial_id', 'grasp_style', 'base_state_sha256', 'delta', 'physical_state',
            'state_sha256', 'source_group_id', 'family_lineage_id'), name='Teacher trial')
        style = row['grasp_style']; base = bases[style]
        if not isinstance(row['trial_id'], str) or not row['trial_id'] or row['trial_id'] in ids:
            raise ValueError('unique Teacher trial ID required')
        _keys(row['delta'], ('object_xy_m', 'world_z_yaw_rad'), name='Tier1 delta')
        xy = row['delta']['object_xy_m']; yaw = _number(row['delta']['world_z_yaw_rad'], 'yaw')
        if not isinstance(xy, list) or len(xy) != 2: raise ValueError('two object XY deltas required')
        xy = [_number(v, 'XY') for v in xy]
        if any(abs(v) > .0005 for v in xy) or abs(yaw) > math.pi/180:
            raise ValueError('Tier1 XY/yaw bounds exceeded')
        state = _physical_state(row['physical_state']); digest = state_sha256(state)
        if (row['base_state_sha256'] != state_sha256(base) or state != _perturb(base, xy, yaw)
                or row['state_sha256'] != digest):
            raise ValueError('only original base plus declared object XY/world-Z yaw may change')
        if digest in states: raise ValueError('duplicate actual float32 writer state')
        group = meta['parent_source_groups'][style]
        if row['source_group_id'] != group or row['family_lineage_id'] != _family(base, group, meta['physics_sha256']):
            raise ValueError('conservative parent family/source group must be preserved')
        ids.add(row['trial_id']); states.add(digest)
    result = deepcopy(payload); result['base_templates'] = bases
    for row in result['trials']: row['physical_state'] = _physical_state(row['physical_state'])
    return result


def generate_teacher_perturbation_manifest(templates, *, seed, provenance, count_per_style=4):
    if type(seed) is not int or seed < 0: raise ValueError('nonnegative local seed required')
    if type(count_per_style) is not int or not 1 <= count_per_style <= 8:
        raise ValueError('1..8 Teacher states per style required')
    if not isinstance(templates, dict) or set(templates) != set(GRASP_STYLES):
        raise ValueError('both real base templates required')
    if set(provenance) != set(PROVENANCE): raise ValueError('complete original input provenance required')
    bases = {style: _physical_state(value) for style, value in templates.items()}
    rng = random.Random(seed); trials = []
    for style in GRASP_STYLES:
        base = bases[style]; group = provenance['parent_source_groups'][style]
        for index in range(count_per_style):
            xy = [rng.uniform(-.0005, .0005) for _ in range(2)]
            yaw = rng.uniform(-math.pi/180, math.pi/180); state = _perturb(base, xy, yaw)
            digest = state_sha256(state)
            trials.append(dict(trial_id=f'tier1:{style}:{digest}', grasp_style=style,
                base_state_sha256=state_sha256(base), delta=dict(object_xy_m=xy, world_z_yaw_rad=yaw),
                physical_state=state, state_sha256=digest, source_group_id=group,
                family_lineage_id=_family(base, group, provenance['physics_sha256'])))
    return validate_teacher_perturbation_manifest(dict(schema_version=VERSION, base_templates=bases,
        metadata=dict(tier=1, reset_kind='teacher_cold_object_pose_perturbation',
            validation_status='authored_unverified', seed=seed, perturbation=deepcopy(PERTURBATION),
            generator_sha256=file_identity(__file__)['sha256'], template_sha256=_digest(bases),
            **deepcopy(provenance)), trials=trials))


def _buffer_state(world, index):
    roots = world.root_seed[index].clone(); roots[:,2] += world.initial_offset
    roots[:,:3] -= world.origins[index]
    wire = world.seed_wire[index].clone(); wire[:3] -= world.origins[index]
    return _physical_state(dict(joint_names=[list(h.joint_names) for h in world.hands],
        joint_positions=[q[index].detach().cpu().tolist() for q in world.seed_q],
        joint_velocities=[q[index].detach().cpu().tolist() for q in world.seed_qd],
        roots=roots.detach().cpu().tolist(), root_velocities=[[0.]*6 for _ in range(2)],
        wire_pose=wire.detach().cpu().tolist(), wire_velocity=world.seed_velocity[index].detach().cpu().tolist()))


def install_teacher_perturbation_reset(world, task, manifest):
    value = validate_teacher_perturbation_manifest(manifest)
    if len(value['trials']) != world.n: raise ValueError('Teacher manifest/native batch mismatch')
    for index, row in enumerate(value['trials']):
        if _buffer_state(world, index) != value['base_templates'][row['grasp_style']]:
            raise ValueError('Teacher manifest base template differs from original runtime reset buffers')
    _install_reset_trials(world, task, value['trials'], object_pose_only=True)
    return value


def readback_teacher_perturbation_reset(world, manifest):
    value = validate_teacher_perturbation_manifest(manifest)
    result = _readback_reset_trials(world, value['trials'])
    result['schema_version'] = 'dh116-teacher-perturbation-reset-readback-v1'
    result['teacher_manifest_sha256'] = _digest(value)
    return result


def _write_json(path, value):
    serialized = json.dumps(value, indent=2, allow_nan=False)
    with Path(path).open('x', encoding='utf-8') as stream: stream.write(serialized + '\n')
    return file_identity(path)


def _verify_identity(identity):
    if file_identity(identity['path']) != identity:
        raise RuntimeError('Teacher perturbation immutable input path/SHA changed: ' + identity['path'])


def freeze_teacher_perturbation_manifest(checkpoint, primary_initialization, secondary_initialization,
        teacher_source, output, *, seed, count_per_style=4, wrist_joint_names=(
            'wrist_x','wrist_y','wrist_z','wrist_roll','wrist_pitch','wrist_yaw')):
    """CPU-only exclusive manifest from the real, hash-bound initialization path."""
    from .tabletop_bimanual_checkpoint_contract import VERSION as V4, unwrap_physical_adapter_checkpoint
    from .tabletop_vision_action_contract import LEGACY_VERSION, SQUASHED_VERSION, validate_squashed_checkpoint
    from .tabletop_vision_action_derivation import _validate_source
    from .tabletop_vision_evaluation_contract import evaluation_physics_sha256
    from .tabletop_adaptive_initialization import adaptive_initialization
    from .tabletop_end_supported_scene import translate_end_support_seed
    from .tabletop_vision_evaluation_reset import template_from_initialization
    from .tabletop_vision_supervision_dataset import load_teacher_reference_dataset
    inputs = [file_identity(p) for p in (checkpoint, primary_initialization, secondary_initialization, teacher_source)]
    output = Path(output).absolute(); sidecar = output.with_suffix('.provenance.json')
    for path in (output, sidecar):
        if path.resolve() in [Path(item['path']) for item in inputs]: raise ValueError('cannot overwrite immutable input')
        if path.exists() or path.is_symlink(): raise FileExistsError(path)
    saved = torch.load(inputs[0]['path'], map_location='cpu', weights_only=False)
    if saved.get('version') == V4: saved = unwrap_physical_adapter_checkpoint(saved, allow_explicit=True)
    if saved.get('version') == LEGACY_VERSION: _validate_source(saved)
    elif saved.get('version') == SQUASHED_VERSION: validate_squashed_checkpoint(saved)
    else: raise ValueError('unsupported visual source checkpoint')
    contract = saved['contract']; critic = contract['privileged_critic_contract']
    if [item['sha256'] for item in inputs[1:3]] != [contract['primary_reset_sha256'], contract['secondary_reset_sha256']]:
        raise ValueError('real initialization file hash does not match source checkpoint')
    reference = load_teacher_reference_dataset(inputs[3]['path'], critic)
    bases = {}; scene = critic.get('end_support_scene')
    for style, identity in zip(GRASP_STYLES, inputs[1:3]):
        payload = json.loads(Path(identity['path']).read_text())
        layout = dict(hand_roots={side:list(pose) for side,pose in zip(('left','right'),payload['roots'])},
            active_joints={}, wire_center=payload['wire_pose'][:3])
        layout, reset = adaptive_initialization(payload,layout,payload['source_contract'],
            object_radius_m=critic.get('object_radius_m',.008))
        if scene is not None:
            reset = translate_end_support_seed(reset, scene)
            for pose in layout['hand_roots'].values(): pose[2] += scene['dz_m']
        bases[style] = template_from_initialization(layout, reset, wrist_joint_names=wrist_joint_names)
    value = generate_teacher_perturbation_manifest(bases,seed=seed,count_per_style=count_per_style,
        provenance=dict(source_checkpoint=inputs[0],teacher_reference=inputs[3],
            initializations=dict(zip(GRASP_STYLES, inputs[1:3])),physics_sha256=evaluation_physics_sha256(critic),
            camera_sha256=contract['camera_config_sha256'],source_privileged_critic_contract_sha256=_digest(critic),
            parent_source_groups={s:reference['references'][s]['source_trajectory_id'] for s in GRASP_STYLES}))
    for item in inputs: _verify_identity(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_id = _write_json(output, value)
    _write_json(sidecar, dict(version=VERSION,manifest=manifest_id,immutable_inputs=inputs,
        generator=file_identity(__file__),tier=1,reset_kind='original_native_reset_buffers',
        native_execution_performed=False,native_success_claimed=False,state_query_oracle=False,
        source_group_policy='parent_reset_family_preserved;seed_or_env_does_not_create_independence'))
    return value


class TeacherPerturbationSession:
    """Opt-in evidence around the existing final reset and first camera packet."""
    def __init__(self, path, world, task, source, references, contract, output, *, native_runtime_verified,
                 camera_path):
        from .tabletop_vision_evaluation_contract import evaluation_physics_sha256
        self.input_identity = file_identity(path); self.output = Path(output); self.world = world
        self.manifest = validate_teacher_perturbation_manifest(json.loads(Path(path).read_text()))
        meta = self.manifest['metadata']; self.native = bool(native_runtime_verified)
        self.first_packet_verified = False; self.reset_readback = None; self.evidence = {}
        self.radius = float(contract['privileged_critic_contract'].get('object_radius_m', .008))
        self.inputs = [meta['source_checkpoint'],meta['teacher_reference'],*meta['initializations'].values()]
        camera = file_identity(camera_path)
        if camera['sha256'] != meta['camera_sha256']: raise ValueError('actual runtime camera file SHA differs')
        self.inputs.append(camera); self.evidence['runtime_camera'] = camera
        if (source['source_sha256'] != meta['teacher_reference']['sha256']
                or str(Path(source['source_path']).resolve()) != meta['teacher_reference']['path']):
            raise ValueError('Teacher reference differs from perturbation input source')
        if ([meta['initializations'][s]['sha256'] for s in GRASP_STYLES] !=
                [contract['primary_reset_sha256'],contract['secondary_reset_sha256']]
                or meta['camera_sha256'] != contract['camera_config_sha256']
                or meta['physics_sha256'] != evaluation_physics_sha256(contract['privileged_critic_contract'])
                or meta['source_privileged_critic_contract_sha256'] != _digest(contract['privileged_critic_contract'])):
            raise ValueError('Teacher perturbation changes original initialization/camera/physics contract')
        trials = self.manifest['trials']
        if len(references) != len(trials) or any(ref['grasp_style'] != row['grasp_style']
                or ref['source_trajectory_id'] != row['source_group_id'] for ref,row in zip(references,trials)):
            raise ValueError('Teacher perturbation reference parent family differs')
        self.verify_inputs()
        frozen = self.output/'teacher_perturbation_manifest.json'
        with frozen.open('xb') as stream: stream.write(Path(path).read_bytes())
        self.evidence['manifest'] = file_identity(frozen)
        if self.evidence['manifest']['sha256'] != self.input_identity['sha256']:
            raise RuntimeError('Teacher perturbation manifest changed while copying')
        install_teacher_perturbation_reset(world,task,self.manifest)

    def verify_inputs(self):
        for identity in [self.input_identity,*self.inputs]: _verify_identity(identity)

    def after_reset(self, clock):
        self.reset_clock = deepcopy(clock)
        self.reset_readback = readback_teacher_perturbation_reset(self.world,self.manifest)
        self.evidence['reset_readback'] = _write_json(self.output/'teacher_perturbation_reset_readback.json',
            dict(clock=self.reset_clock,readback=self.reset_readback,native_runtime_verified=self.native,
                source='original_env_reset_then_full_named_native_tensor_readback',success_qualified=False))

    def validate_first_packet(self, packet, clock):
        if self.reset_readback is None: raise RuntimeError('actual final reset readback required before first camera packet')
        after = readback_teacher_perturbation_reset(self.world,self.manifest)
        if clock != self.reset_clock: raise RuntimeError('Teacher perturbation camera warmup changed SDK/world clock')
        if after != self.reset_readback: raise RuntimeError('Teacher perturbation camera warmup changed physical reset')
        if not bool(packet['image_valid'].all() & packet['object_valid'].all()):
            raise RuntimeError('Teacher perturbation requires actually visible valid RGB-D before SensorHistory')
        from .tabletop_vision_audit import audit_initial_points
        geometry = audit_initial_points(packet,self.world,radius=self.radius,length=.5)
        if not geometry.get('passed'): raise RuntimeError('original initial camera geometry failed')
        self.evidence['camera_geometry'] = _write_json(self.output/'initial_camera_geometry_validation.json',geometry)
        self.evidence['warmup_readback'] = _write_json(self.output/'teacher_perturbation_camera_readback.json',
            dict(clock=deepcopy(clock),readback=after,source='after_real_first_camera_before_SensorHistory'))
        self.first_packet_verified = True; self.verify_inputs()

    def sample_metadata(self, index):
        if not self.first_packet_verified: raise RuntimeError('first actual sensor packet/reset proof not verified')
        trial = self.manifest['trials'][index]
        actual = self.reset_readback['actual_states'][index]
        return dict(source_group_id=trial['source_group_id'],teacher_perturbation_tier=1,
            teacher_perturbation_manifest_sha256=self.input_identity['sha256'],
            teacher_perturbation_trial_id=trial['trial_id'],
            teacher_base_state_sha256=trial['base_state_sha256'],teacher_reset_authored_state_sha256=trial['state_sha256'],
            teacher_reset_readback_state_sha256=actual['readback_state_sha256'],
            teacher_family_lineage_id=trial['family_lineage_id'],teacher_object_pose_delta=deepcopy(trial['delta']))

    def summary(self):
        return dict(version=VERSION,tier=1,input_manifest=self.input_identity,evidence=deepcopy(self.evidence),
            native_runtime_verified=self.native,first_packet_verified=self.first_packet_verified,
            state_query_oracle=False,counts_as_student_success=False,
            source_group_policy='parent_reset_family_preserved;not_run_seed_env_or_frame_count',
            new_metadata_sources=dict(source_group_id='manifest_verified_parent_reference_family',
                tier_and_delta='explicit_authored_manifest',reset_authored_state_sha256='canonical_float32_writer_state',
                reset_readback_state_sha256='actual_original_reset_before_camera_unchanged_after_camera'),
            implementation={name:file_identity(Path(__file__).with_name(name+'.py')) for name in (
                'tabletop_vision_teacher_perturbation','tabletop_vision_evaluation_reset','tabletop_vision_teacher_collection')})


def save_failed_or_incomplete_pool(data, successful_env_ids, output):
    """Separate unqualified executed rows; never synthesize missing observations."""
    from .tabletop_vision_supervision_dataset import build_dataset
    output = Path(output); successful_env_ids = set(successful_env_ids)
    indices = [i for i,row in enumerate(data['sample_metadata']) if row['env_id'] not in successful_env_ids]
    result = dict(version='dh116-teacher-failed-incomplete-pool-v1',samples=len(indices),dataset=None,
        excluded_successful_env_ids=sorted(successful_env_ids),native_success_claimed=False,
        use='audit_only_not_BC_labels',selection='all_executed_rows_from_unqualified_or_incomplete_first_episodes')
    if indices:
        metadata = [data['sample_metadata'][i] for i in indices]; ids = {str(row['episode_id']) for row in metadata}
        failed = build_dataset(data['observations'][indices],data['actions'][indices],data['reference_actions'][indices],
            metadata,{key:value for key,value in data['episodes'].items() if key in ids},
            visual_contract=data['visual_contract'],teacher_source_sha256=data['teacher_source_sha256'],require_successful=False)
        path = output/'failed_or_incomplete_supervision.pt'
        with path.open('xb') as stream: torch.save(failed,stream)
        result['dataset'] = dict(file_identity(path),samples=len(indices))
    result['manifest'] = _write_json(output/'failed_or_incomplete_pool.json',result)
    return result


def tier1_pinch_success_mask(observed_qualified, references):
    """Tier1 augments pinch only; other observed outcomes remain independent."""
    return observed_qualified & torch.tensor([r['grasp_style']==GRASP_STYLES[0] for r in references],
        dtype=torch.bool,device=observed_qualified.device)


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','primary-initialization','secondary-initialization','teacher-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--count-per-style',type=int,default=4)
    args=parser.parse_args(argv)
    value=freeze_teacher_perturbation_manifest(args.checkpoint,args.primary_initialization,
        args.secondary_initialization,args.teacher_source,args.output,seed=args.seed,count_per_style=args.count_per_style)
    print(json.dumps(dict(manifest=file_identity(args.output),trials=len(value['trials']),
        native_execution_performed=False,native_success_claimed=False)))


if __name__=='__main__': main()
