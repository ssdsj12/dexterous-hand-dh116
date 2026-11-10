"""Tier1 authored-state/collection CPU fixtures never certify native success."""
import copy
import importlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_vision_evaluation_reset import templates, world_stub
from go2_pvcnn.control.dh116.tabletop_evaluation_trials import GRASP_STYLES, state_sha256


def api():
    name = 'go2_pvcnn.control.dh116.tabletop_vision_teacher_perturbation'
    assert importlib.util.find_spec(name), 'explicit Tier1 Teacher perturbation entrypoint is missing'
    return importlib.import_module(name)


def provenance():
    identity = lambda name: dict(path='/immutable/' + name, sha256='a' * 64)
    return dict(source_checkpoint=identity('actor.pt'), teacher_reference=identity('teacher.pt'),
        initializations={style: identity(style + '.json') for style in GRASP_STYLES},
        physics_sha256='b' * 64, camera_sha256='c' * 64,
        source_privileged_critic_contract_sha256='d' * 64,
        parent_source_groups={style: str(i + 1) * 64 for i, style in enumerate(GRASP_STYLES)})


def manifest(count=1, seed=116603):
    return api().generate_teacher_perturbation_manifest(templates(), seed=seed,
        provenance=provenance(), count_per_style=count)


def test_tier1_changes_only_wire_xy_and_world_z_yaw_and_preserves_rng():
    m = api(); base = templates(); before = copy.deepcopy(base); rng = torch.get_rng_state().clone()
    value = m.generate_teacher_perturbation_manifest(base, seed=116603,
        provenance=provenance(), count_per_style=4)
    assert base == before and torch.equal(rng, torch.get_rng_state())
    assert value == m.generate_teacher_perturbation_manifest(base, seed=116603,
        provenance=provenance(), count_per_style=4)
    assert value['metadata']['tier'] == 1 and len(value['trials']) == 8
    assert value['metadata']['validation_status'] == 'authored_unverified'
    assert len({row['state_sha256'] for row in value['trials']}) == 8
    for row in value['trials']:
        initial = value['base_templates'][row['grasp_style']]; state = row['physical_state']
        assert row['source_group_id'] == provenance()['parent_source_groups'][row['grasp_style']]
        for name in initial:
            if name != 'wire_pose': assert state[name] == initial[name]
        assert state['wire_pose'][2] == initial['wire_pose'][2]
        assert all(abs(x) <= .0005 for x in row['delta']['object_xy_m'])
        yaw = row['delta']['world_z_yaw_rad']; assert abs(yaw) <= math.pi / 180
        # Independent wxyz world-Z left multiplication, not intrinsic yaw.
        w, x, y, z = initial['wire_pose'][3:]; c, s = math.cos(yaw / 2), math.sin(yaw / 2)
        expected = torch.tensor([c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w], dtype=torch.float32)
        actual = torch.tensor(state['wire_pose'][3:])
        assert torch.equal(actual, expected) or torch.equal(actual, -expected)


@pytest.mark.parametrize('kind', ['joint', 'root', 'wire_z', 'velocity', 'xy', 'yaw', 'tier', 'group', 'duplicate'])
def test_tier1_rejects_unauthorized_changes_even_if_writer_sha_is_recomputed(kind):
    m = api(); value = manifest(); row = value['trials'][0]
    if kind == 'joint': row['physical_state']['joint_positions'][0][0] += .001
    elif kind == 'root': row['physical_state']['roots'][0][0] += .001
    elif kind == 'wire_z': row['physical_state']['wire_pose'][2] += .001
    elif kind == 'velocity': row['physical_state']['wire_velocity'][0] = .001
    elif kind == 'xy': row['delta']['object_xy_m'][0] = .000501
    elif kind == 'yaw': row['delta']['world_z_yaw_rad'] = math.pi / 180 + 1e-6
    elif kind == 'tier': value['metadata']['tier'] = 2
    elif kind == 'group': row['source_group_id'] = 'f' * 64
    elif kind == 'duplicate': value['trials'].append(copy.deepcopy(row))
    row['state_sha256'] = state_sha256(row['physical_state'])
    with pytest.raises(ValueError): m.validate_teacher_perturbation_manifest(value)


def test_tier1_group_is_conservative_parent_family_not_run_or_random_seed():
    a, b = manifest(seed=1), manifest(seed=2)
    assert {row['state_sha256'] for row in a['trials']}.isdisjoint(row['state_sha256'] for row in b['trials'])
    assert [row['source_group_id'] for row in a['trials']] == [row['source_group_id'] for row in b['trials']]
    assert [row['family_lineage_id'] for row in a['trials']] == [row['family_lineage_id'] for row in b['trials']]


def test_tier1_pool_filters_pinch_without_changing_observed_native_success_flags():
    flags=torch.tensor([True,False,True,True]);before=flags.clone()
    refs=[dict(grasp_style=s) for s in [GRASP_STYLES[0]]*2+[GRASP_STYLES[1]]*2]
    selected=api().tier1_pinch_success_mask(flags,refs)
    assert selected.tolist()==[True,False,False,False] and torch.equal(flags,before)


def task_stub():
    from go2_pvcnn.control.dh116.tabletop_vision_evaluation_reset import TASK_RESET_FIELDS
    return SimpleNamespace(n=2, **{name: torch.zeros(2, 12) for name in TASK_RESET_FIELDS})


def base_reset_buffers(dw, value):
    """Fixture sets the original unperturbed reset family before opt-in install."""
    from go2_pvcnn.control.dh116.tabletop_vision_evaluation_reset import _install_reset_trials
    trials=[dict(trial_id=str(i), grasp_style=s, physical_state=value['base_templates'][s])
        for i,s in enumerate(GRASP_STYLES)]
    _install_reset_trials(dw, task_stub(), trials)


def execute_original_buffer_reset(dw):
    dw.q=torch.stack(dw.seed_q,1); dw.qd=torch.stack(dw.seed_qd,1)
    dw.roots=dw.root_seed.clone(); dw.roots[...,2]+=dw.initial_offset
    dw.wp=dw.seed_wire[:,:3].clone(); dw.wq=dw.seed_wire[:,3:].clone(); dw.wv=dw.seed_velocity.clone()


def test_tier1_uses_original_reset_buffers_and_readback_without_eval_fixture_manifest():
    m=api(); value=manifest(); dw=world_stub(); task=task_stub(); base_reset_buffers(dw,value)
    before=[q.clone() for q in dw.seed_q]
    m.install_teacher_perturbation_reset(dw,task,value)
    assert all(torch.equal(a,b) for a,b in zip(before,dw.seed_q))
    execute_original_buffer_reset(dw)
    evidence=m.readback_teacher_perturbation_reset(dw,value)
    assert evidence['native_validation_complete'] is False and evidence['unique_readback_count']==2
    dw.wp[0,0]+=.001
    with pytest.raises(RuntimeError,match='mismatch'):m.readback_teacher_perturbation_reset(dw,value)


def test_tier1_preserves_original_double_command_seed_and_all_nonobject_buffers_bitwise():
    m=api();value=manifest();dw=world_stub();task=task_stub();base_reset_buffers(dw,value)
    # Original grasp_command is double; native q is its float32 encoding.
    # Installing an evaluation-style q seed would silently round the Teacher start.
    task._per_env_reset_seed=dict(command=torch.full((2,12),.200000003,dtype=torch.float64),
        reference=task.reference.clone())
    original_seed=copy.deepcopy(task._per_env_reset_seed)
    before={name:copy.deepcopy(getattr(dw,name)) for name in
        ('seed_q','seed_qd','root_seed','seed_velocity')}
    z=dw.seed_wire[:,2].clone()
    m.install_teacher_perturbation_reset(dw,task,value)
    for key,a in before.items():
        b=getattr(dw,key)
        if isinstance(a,list):assert all(torch.equal(x,y) for x,y in zip(a,b))
        else:assert torch.equal(a,b)
    assert torch.equal(z,dw.seed_wire[:,2])
    assert all(torch.equal(v,task._per_env_reset_seed[k]) for k,v in original_seed.items())


def test_tier1_refuses_wrong_base_family_before_mutating_any_buffer():
    m=api(); value=manifest(); dw=world_stub(); task=task_stub(); base_reset_buffers(dw,value)
    dw.seed_wire[0,2]+=.001; before=dw.seed_wire.clone()
    with pytest.raises(ValueError,match='base|template'):m.install_teacher_perturbation_reset(dw,task,value)
    assert torch.equal(before,dw.seed_wire)


def test_options_keep_old_default_but_perturb_requires_actual_mpc_and_sidecar():
    from test_vision_teacher_collection import options
    from go2_pvcnn.control.dh116.tabletop_vision_teacher_collection import validate_options
    kw=dict(vision_teacher_perturbation_manifest='tier1.json', vision_teacher_mpc=True,
        vision_teacher_mpc_profile='pressure_geometry_constrained', solid_cylinder_radius_m=.008,
        object_kind='solid_cylinder', vision_sequence_context_sidecar=True, vision_width=224,vision_height=96)
    assert validate_options(options(**kw)) is None
    for changes in ({'vision_teacher_mpc':False},{'vision_teacher_source':None},
                    {'vision_sequence_context_sidecar':False},{'vision_width':160},
                    {'vision_teacher_mpc_profile':'local_pad'}):
        with pytest.raises(ValueError):validate_options(options(**dict(kw,**changes)))


def test_real_input_freezer_checks_source_hashes_and_exclusive_output(tmp_path, monkeypatch):
    from test_vision_action_derivation import source_checkpoint
    from go2_pvcnn.control.dh116.tabletop_vision_evaluation_contract import file_sha256
    from go2_pvcnn.control.dh116 import tabletop_vision_supervision_dataset as dataset
    checkpoint,saved=source_checkpoint(tmp_path); paths=[]
    for i,style in enumerate(GRASP_STYLES):
        state=templates()[style]; path=tmp_path/f'initial{i}.json'
        path.write_text(json.dumps(dict(format='dh116-adaptive-grasp-initialization-v1',source_contract={},
            roots=state['roots'],joint_positions=state['joint_positions'],grasp_command=[.25]*12,
            wire_pose=state['wire_pose'])))
        paths.append(path)
    saved['contract']['primary_reset_sha256']=file_sha256(paths[0])
    saved['contract']['secondary_reset_sha256']=file_sha256(paths[1]);torch.save(saved,checkpoint)
    teacher=tmp_path/'real_reference.pt';teacher.write_bytes(b'canonical-reference-fixture')
    monkeypatch.setattr(dataset,'load_teacher_reference_dataset',lambda *args:dict(
        source_sha256=file_sha256(teacher),source_path=str(teacher),references={
            style:dict(source_trajectory_id=str(i+1)*64) for i,style in enumerate(GRASP_STYLES)}))
    m=api();output=tmp_path/'tier1.json';before=file_sha256(checkpoint)
    value=m.freeze_teacher_perturbation_manifest(checkpoint,*paths,teacher,output,
        seed=11,count_per_style=1,wrist_joint_names=())
    assert len(value['trials'])==2 and output.exists() and output.with_suffix('.provenance.json').exists()
    assert file_sha256(checkpoint)==before
    assert value['metadata']['teacher_reference']['sha256']==file_sha256(teacher)
    with pytest.raises(FileExistsError):
        m.freeze_teacher_perturbation_manifest(checkpoint,*paths,teacher,output,seed=11,wrist_joint_names=())
    paths[0].write_text('{}')
    with pytest.raises(ValueError,match='hash|SHA'):
        m.freeze_teacher_perturbation_manifest(checkpoint,*paths,teacher,tmp_path/'refused.json',seed=12,wrist_joint_names=())


def session_fixture(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_vision_evaluation_contract import evaluation_physics_sha256
    from go2_pvcnn.control.dh116.tabletop_evaluation_trials import _digest
    m=api(); value=manifest();contract=dict(privileged_critic_contract={'observation_size':371},
        primary_reset_sha256='a'*64,secondary_reset_sha256='a'*64,camera_config_sha256='c'*64)
    value['metadata']['physics_sha256']=evaluation_physics_sha256(contract['privileged_critic_contract'])
    value['metadata']['source_privileged_critic_contract_sha256']=_digest(contract['privileged_critic_contract'])
    # Re-generate rather than editing family IDs after changing physical provenance.
    prov=provenance();prov.update({k:value['metadata'][k] for k in
        ('physics_sha256','source_privileged_critic_contract_sha256')})
    for key in ('source_checkpoint','teacher_reference'):
        path=tmp_path/(key+'.pt');path.write_bytes(key.encode());prov[key]=m.file_identity(path)
    for style in GRASP_STYLES:
        path=tmp_path/(style+'.json');path.write_text('{}');prov['initializations'][style]=m.file_identity(path)
    contract.update(primary_reset_sha256=prov['initializations'][GRASP_STYLES[0]]['sha256'],
        secondary_reset_sha256=prov['initializations'][GRASP_STYLES[1]]['sha256'])
    camera=tmp_path/'camera.json';camera.write_text('{"fixed":true}')
    contract['camera_config_sha256']=m.file_identity(camera)['sha256'];prov['camera_sha256']=contract['camera_config_sha256']
    value=m.generate_teacher_perturbation_manifest(templates(),seed=11,provenance=prov,count_per_style=1)
    path=tmp_path/'tier1.json';path.write_text(json.dumps(value));output=tmp_path/'out';output.mkdir()
    dw=world_stub();base_reset_buffers(dw,value);task=task_stub()
    refs=[dict(grasp_style=s,source_trajectory_id=prov['parent_source_groups'][s]) for s in GRASP_STYLES]
    source=dict(source_sha256=prov['teacher_reference']['sha256'],source_path=prov['teacher_reference']['path'])
    session=m.TeacherPerturbationSession(path,dw,task,source,refs,contract,output,
        native_runtime_verified=False,camera_path=camera)
    execute_original_buffer_reset(dw)
    return session,dw,task,output


def test_session_requires_exact_warmup_state_clock_before_first_sensor_history(tmp_path,monkeypatch):
    m=api();session,dw,task,out=session_fixture(tmp_path)
    clock=dict(physics_step=0,sdk_physics_step=18,timestamp_s=.0375)
    session.after_reset(clock)
    packet=dict(image_valid=torch.ones(2,dtype=torch.bool),object_valid=torch.ones(2,dtype=torch.bool))
    from go2_pvcnn.control.dh116 import tabletop_vision_audit as audit
    calls=[];monkeypatch.setattr(audit,'audit_initial_points',lambda *a,**kw:(calls.append(1) or {'passed':True}))
    with pytest.raises(RuntimeError,match='clock'):session.validate_first_packet(packet,dict(clock,physics_step=1))
    session.validate_first_packet(packet,clock)
    assert calls==[1] and not session.summary()['native_runtime_verified']
    meta=session.sample_metadata(0)
    assert meta['source_group_id']==provenance()['parent_source_groups'][GRASP_STYLES[0]]
    assert meta['teacher_perturbation_tier']==1 and meta['teacher_reset_readback_state_sha256']
    assert (out/'teacher_perturbation_reset_readback.json').exists()
    assert (out/'initial_camera_geometry_validation.json').exists()
    # An unsafe later mutation cannot be papered over by matching the old clock.
    dw.wp[0,0]+=.001
    with pytest.raises(RuntimeError,match='mismatch'):session.validate_first_packet(packet,clock)


def test_session_rechecks_all_input_file_sha_and_cannot_count_fixture_native(tmp_path,monkeypatch):
    session,dw,task,out=session_fixture(tmp_path)
    source=Path(session.manifest['metadata']['teacher_reference']['path'])
    source.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='SHA|changed'):session.verify_inputs()


def test_session_rechecks_actual_runtime_camera_bytes(tmp_path):
    session,dw,task,out=session_fixture(tmp_path)
    (tmp_path/'camera.json').write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match='SHA|changed'):session.verify_inputs()


def test_failed_pool_keeps_failed_incomplete_rows_without_successful_episode(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_vision_supervision_dataset import build_dataset
    from go2_pvcnn.control.dh116.tabletop_surface_contract import ACTION_CONTRACT
    m=api();metadata=[];episodes={}
    for i in range(3):
        episodes[str(i)]=dict(env_id=i,first_episode=True,success=i==0,recovered=False,native_physics=True,
            grasp_style=GRASP_STYLES[0],source_episode_id=i,source_trajectory_id='a'*64,
            proof_path='/actual/report.json')
        metadata.append(dict(env_id=i,episode_id=i,tick=0,frame_id=1,simulation_time_s=0.,physics_step=0,
            grasp_style=GRASP_STYLES[0],source_episode_id=i,source_trajectory_id='a'*64,previous_action=[0.]*24))
    data=build_dataset(torch.ones(3,3844),torch.zeros(3,24),torch.zeros(3,24),metadata,episodes,
        visual_contract={'actor_observation_size':3844,'critic_observation_size':371,'action_size':24,
            'actor_uses_privileged_state':False,'action_contract':ACTION_CONTRACT,
            'camera_config_sha256':'c'*64,'primary_reset_sha256':'d'*64,'secondary_reset_sha256':'e'*64},
        teacher_source_sha256='b'*64,require_successful=False)
    report=m.save_failed_or_incomplete_pool(data,{0},tmp_path)
    failed=torch.load(report['dataset']['path'],weights_only=False)
    assert len(failed['actions'])==2 and set(failed['episodes'])=={'1','2'}
    assert report['excluded_successful_env_ids']==[0] and report['native_success_claimed'] is False
    assert report['dataset']['sha256']==m.file_identity(report['dataset']['path'])['sha256']


def test_collection_optin_orders_reset_camera_proof_before_observation_and_keeps_real_labels(tmp_path,monkeypatch):
    from test_vision_training import fixture,Camera
    from test_vision_teacher_collection import options
    from go2_pvcnn.control.dh116 import tabletop_vision_teacher_collection as collection
    from go2_pvcnn.control.dh116 import tabletop_vision_supervision_dataset as dataset
    from go2_pvcnn.control.dh116 import tabletop_vision_training as training
    from go2_pvcnn.control.dh116 import tabletop_vision_sequence_capture as capture
    from go2_pvcnn.control.dh116 import tabletop_vision_command_shadow as shadow
    m=api();dw,task,env,args=fixture(tmp_path);events=[]
    for key,value in vars(options(num_envs=2,vision_teacher_perturbation_manifest='explicit.json',
        vision_teacher_mpc=True,vision_teacher_mpc_profile='pressure_geometry_constrained',
        solid_cylinder_radius_m=.008,object_kind='solid_cylinder',vision_sequence_context_sidecar=True,
        vision_width=224,vision_height=96)).items():setattr(args,key,value)
    refs=[dict(source_episode_id=i,grasp_style=s,source_trajectory_id=s,actions=torch.full((3,24),.1*(i+1)))
        for i,s in enumerate(GRASP_STYLES)]
    monkeypatch.setattr(dataset,'load_teacher_reference_dataset',lambda *a:dict(source_sha256='a'*64,episodes=refs))
    original_observations=training._observations
    def observations(*a,**kw):
        assert events[-1]=='packet_proof' if env.tick==0 else True
        events.append('history');return original_observations(*a,**kw)
    monkeypatch.setattr(training,'_observations',observations)
    class Session:
        def __init__(self,*a,**kw):events.append('install');assert kw['native_runtime_verified'] is False
        def after_reset(self,clock):events.append('reset_proof')
        def validate_first_packet(self,packet,clock):events.append('packet_proof')
        def sample_metadata(self,i):return dict(source_group_id='parent',teacher_perturbation_tier=1,teacher_reset_readback_state_sha256='b'*64)
        def verify_inputs(self):pass
        def summary(self):return dict(tier=1,native_runtime_verified=False)
    monkeypatch.setattr(m,'TeacherPerturbationSession',Session)
    class Capture:
        def __init__(self,*a,**kw):self.records=[]
        def begin_tick(self,*a,**kw):self.records.append(1)
        def finish_tick(self,*a):pass
        def close(self):pass
        def write_unqualified(self,path,**kw):return {'path':str(path)}
    monkeypatch.setattr(capture,'SequenceContextCapture',Capture)
    monkeypatch.setattr(shadow,'command_state',lambda *a:{})
    class Optimizer:
        def refine(self,ref,**kw):return ref+.05,{'per_env':[dict(active=True,changed=True,env=i) for i in range(2)]}
        def observe_result(self,*a):pass
    monkeypatch.setattr(collection,'_teacher_optimizer',lambda *a:Optimizer())
    sim=SimpleNamespace(current_time=0.,current_time_step_index=0)
    def reset():events.append('reset');return torch.zeros(2,371)
    env.reset=reset;received=[]
    def step(action):
        received.append(action.clone());env.tick+=1;sim.current_time+=1/60;sim.current_time_step_index+=8
        done=torch.tensor([env.tick==1,env.tick==3]);return torch.zeros(2,371),torch.zeros(2),done,dict(
            success=done,failure=torch.zeros(2,dtype=torch.long),recovered=torch.zeros(2,dtype=torch.bool))
    env.step=step
    class RealFrameFixture(Camera):
        def set_raw_frame_sink(self,sink):self.sink=sink
        def sample(self):
            events.append('camera');frame=sim.current_time_step_index+1
            self.sink(dict(frame_id=frame,simulation_time=sim.current_time,rendered_simulation_time=sim.current_time))
            return dict(super().sample(),frame_id=frame,simulation_time=sim.current_time)
    result=collection.run_teacher_collection(SimpleNamespace(sim=sim),dw,task,env,args,
        {'observation_size':371},camera=RealFrameFixture(2))
    assert events[:6]==['install','reset','reset_proof','camera','packet_proof','history']
    data=torch.load(args.output/'all_executed_supervision.pt',weights_only=False)
    assert len(data['actions'])==4 and result['successful_episodes']==0
    assert all(row['source_group_id']=='parent' and row['teacher_perturbation_tier']==1 for row in data['sample_metadata'])
    assert all(ep['source_group_id']=='parent' for ep in data['episodes'].values())
    for row,action in zip(data['sample_metadata'],data['actions']):
        assert torch.equal(action,received[row['tick']][row['env_id']])
    assert result['teacher_perturbation']['tier']==1
    assert result['failed_or_incomplete_pool']['samples']==4
    assert (args.output/'failed_or_incomplete_supervision.pt').exists()
