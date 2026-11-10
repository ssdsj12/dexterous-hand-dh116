"""CPU sampling/schema oracles. Fixtures never certify native execution."""
import ast
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_vision_evaluation_reset import world_stub
from go2_pvcnn.control.dh116.tabletop_evaluation_trials import state_sha256


def api():
    name='go2_pvcnn.control.dh116.tabletop_vision_initial_state_audit'
    assert importlib.util.find_spec(name), 'training initial-state diagnostics missing'
    return importlib.import_module(name)


GROUPS=[dict(label='pinch_thumb_index',num_envs=1),dict(label='tripod_thumb_index_middle',num_envs=1)]


def raw_world():
    dw=world_stub(); dw._physics_steps=48
    dw.w.sim=SimpleNamespace(current_time=.1,current_time_step_index=48)
    dw.w.mounts=None
    for h,hand in enumerate(dw.hands):
        hand.data.joint_pos=dw.q[:,h].clone()
        hand.data.joint_vel=dw.qd[:,h].clone()
        hand.data.root_pose_w=dw.roots[:,h].clone()
        hand.data.root_pose_w[:,:3]+=dw.origins
    dw.wire=SimpleNamespace(data=SimpleNamespace(
        root_pos_w=dw.origins+torch.tensor([0.,0.,.308]),
        root_quat_w=torch.tensor([[1.,0.,0.,0.]]).repeat(2,1),root_vel_w=torch.zeros(2,6)))
    return dw


def shaped_sidecar():
    # A valid-shaped schema fixture for comparison tests, not native evidence.
    snap=api().sample_initial_physical_state(raw_world(),GROUPS)
    return dict(schema_version='dh116-training-initial-physical-state-v1',
        validation_status='native_verified',execution='native_isaac_tensor_world',diagnostic_only=True,
        source_resume_sha256='a'*64,initial_policy_sha256='b'*64,
        source_checkpoint_version='dh116-vision-student-ppo-v3-squashed-gaussian',source_derivation={},
        scene_seed=11061103,num_envs=2,groups=deepcopy(GROUPS),
        physics_identity=dict(version='dh116-evaluation-physical-semantics-v1',physics_sha256='c'*64),
        camera_sha256='d'*64,actual_states=snap['actual_states'],
        after_camera_actual_states=deepcopy(snap['actual_states']),
        warmup=dict(before_clock=snap['clock'],after_clock=deepcopy(snap['clock']),
            physics_steps_advanced=0,max_abs_state_error=0.))


def write_sidecar(path,value):
    path.write_text(json.dumps(value)); return path


def test_sampling_reads_full_raw_native_fields_localizes_origins_and_allows_templates():
    dw=raw_world(); dw.q.fill_(99.) # Actor/world cache cannot replace raw simulator readback.
    before=torch.get_rng_state().clone()
    snap=api().sample_initial_physical_state(dw,GROUPS)
    assert torch.equal(before,torch.get_rng_state())
    assert 'validation_status' not in snap and 'native_verified' not in snap.values()
    rows=snap['actual_states']; assert len(rows)==2
    assert rows[0]['physical_state']==rows[1]['physical_state']
    assert rows[0]['state_sha256']==rows[1]['state_sha256']
    assert rows[0]['physical_state']['joint_positions'][0][0]==0.
    assert set(rows[0]['physical_state'])=={'joint_names','joint_positions','joint_velocities',
        'roots','root_velocities','wire_pose','wire_velocity'}
    dw.hands[0].data.joint_pos[0,0]=.2
    assert rows[0]['physical_state']['joint_positions'][0][0]==0.


@pytest.mark.parametrize('change',['q','qd','root','root_velocity','wire','wire_velocity','clock','simulation_time','tensor_world_clock'])
def test_render_warmup_rejects_state_or_physics_clock_changes(change):
    dw=raw_world(); m=api(); before=m.sample_initial_physical_state(dw,GROUPS)
    if change=='q': dw.hands[0].data.joint_pos[1,0]+=.001
    if change=='qd': dw.hands[0].data.joint_vel[0,0]+=.001
    if change=='root': dw.hands[0].data.root_pose_w[0,0]+=.001
    if change=='root_velocity': dw.hands[1].data.root_vel_w[0,1]+=.001
    if change=='wire': dw.wire.data.root_pos_w[0,1]+=.001
    if change=='wire_velocity': dw.wire.data.root_vel_w[0,3]+=.001
    if change=='clock': dw.w.sim.current_time_step_index+=1
    if change=='simulation_time': dw.w.sim.current_time+=1/480
    if change=='tensor_world_clock': dw._physics_steps+=1
    after=m.sample_initial_physical_state(dw,GROUPS)
    with pytest.raises(ValueError,match='state|clock'):
        m.validate_camera_warmup(before,after)


def test_policy_digest_is_stable_over_order_clone_device_and_sensitive_to_values():
    from collections import OrderedDict
    m=api(); a={'actor.w':torch.ones(2,3),'critic.b':torch.zeros(1)}
    digest=m.policy_state_sha256(a)
    assert digest==m.policy_state_sha256({k:a[k].clone() for k in reversed(a)})
    assert digest==m.policy_state_sha256(OrderedDict(reversed(list(a.items()))))
    b=deepcopy(a); b['critic.b'][0]=1.
    assert digest!=m.policy_state_sha256(b)
    b=deepcopy(a); b['actor.w']=b['actor.w'].double()
    assert digest!=m.policy_state_sha256(b)


def test_two_sidecars_accept_same_source_and_physics_with_float_tolerance(tmp_path):
    m=api(); left=shaped_sidecar(); right=deepcopy(left)
    right['actual_states'][0]['physical_state']['joint_positions'][0][0]=.00001
    right['actual_states'][0]['state_sha256']=state_sha256(right['actual_states'][0]['physical_state'])
    right['after_camera_actual_states']=deepcopy(right['actual_states'])
    result=m.compare_training_initial_state_audits(write_sidecar(tmp_path/'a.json',left),
        write_sidecar(tmp_path/'b.json',right),tolerance=2e-5)
    assert result['matched'] and 0<result['max_abs_state_error']<=2e-5


@pytest.mark.parametrize('field,value',[
    ('validation_status','authored_unverified'),('source_resume_sha256','e'*64),
    ('initial_policy_sha256','e'*64),('scene_seed',0),('camera_sha256','e'*64),
    ('source_derivation',{'reward_experiment_derivation':{'profile':'R1'}}),
    ('physics_identity',{'version':'dh116-evaluation-physical-semantics-v1','physics_sha256':'e'*64}),
    ('num_envs',4),('groups',list(reversed(GROUPS)))])
def test_reference_rejects_fixture_status_or_identity_mismatch(tmp_path,field,value):
    m=api(); left=shaped_sidecar(); right=deepcopy(left); right[field]=value
    with pytest.raises(ValueError):
        m.compare_training_initial_state_audits(write_sidecar(tmp_path/'a.json',left),
            write_sidecar(tmp_path/'b.json',right))


def test_native_publisher_rejects_fixture_before_reset_or_output(tmp_path):
    from test_vision_squashed_training import setup,run
    dw,task,env,args=setup(tmp_path)
    args.vision_experiment_initial_state_audit=True
    args.vision_student=True
    env.reset=lambda:pytest.fail('fixture must be rejected before reset')
    with pytest.raises(ValueError,match='native'):
        run(dw,task,env,args)
    assert not list(args.output.iterdir())


def test_opt_in_flags_exist_in_both_launchers_and_default_false():
    root=Path(__file__).resolve().parents[1]
    for script in ('train_dh116_tabletop_gpu_ppo.py','run_dh116_surface_student.py'):
        tree=ast.parse((root/'scripts'/script).read_text())
        args={n.args[0].value:n for n in ast.walk(tree) if isinstance(n,ast.Call)
            and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and n.args
            and isinstance(n.args[0],ast.Constant)}
        flag=args.get('--vision-experiment-initial-state-audit')
        assert flag is not None, 'optional native initial-state CLI missing'
        assert {k.arg:ast.literal_eval(k.value) for k in flag.keywords}=={'action':'store_true'}
        assert '--vision-experiment-initial-state-reference' in args


def test_reference_option_requires_opt_in_and_never_enters_legacy_or_evaluation(tmp_path):
    from test_vision_squashed_training import setup
    from go2_pvcnn.control.dh116.tabletop_vision_action_contract import validate_vision_action_options
    _,_,_,args=setup(tmp_path)
    args.vision_experiment_initial_state_reference=tmp_path/'prior.json'
    with pytest.raises(ValueError,match='audit'):
        validate_vision_action_options(args)
    args.vision_experiment_initial_state_reference=None; args.vision_experiment_initial_state_audit=True
    args.vision_action_distribution='gaussian_clip'
    with pytest.raises(ValueError,match='v3'):
        validate_vision_action_options(args)


def test_reference_load_detects_file_replacement(tmp_path,monkeypatch):
    m=api(); value=shaped_sidecar(); path=write_sidecar(tmp_path/'reference.json',value)
    original=m.file_sha256; calls=0
    def replace_on_recheck(candidate):
        nonlocal calls
        calls+=1
        if calls==2:
            changed=deepcopy(value); changed['scene_seed']+=1
            write_sidecar(path,changed)
        return original(candidate)
    monkeypatch.setattr(m,'file_sha256',replace_on_recheck)
    with pytest.raises(ValueError,match='changed'):
        m._load_audit(path)


def test_disabled_audit_performs_no_sampling_or_publishing(tmp_path,monkeypatch):
    from test_vision_squashed_training import setup,run
    m=api()
    def forbidden(*a,**kw): pytest.fail('disabled diagnostics performed work')
    monkeypatch.setattr(m,'sample_initial_physical_state',forbidden)
    monkeypatch.setattr(m,'TrainingInitialStateAudit',forbidden)
    dw,task,env,args=setup(tmp_path)
    run(dw,task,env,args)
    assert not (args.output/'training_initial_physical_state.json').exists()


def test_reference_mismatch_hook_precedes_ppo_construction(tmp_path,monkeypatch):
    from test_vision_warm_start import source,resumed
    from test_vision_squashed_training import run
    from go2_pvcnn.control.dh116 import tabletop_vision_training as training
    m=api(); path,_=source(tmp_path); dw,task,env,args=resumed(tmp_path,path,'reset')
    args.vision_student=True; args.vision_experiment_initial_state_audit=True
    seen=[]
    class UncertifiedAuditSpy:
        def __init__(self,*a,**kw): seen.append('prepare')
        def before_camera(self): seen.append('before_camera')
        def after_camera(self): seen.append('after_camera')
        def finish(self,policy):
            seen.append('reference_compare')
            raise ValueError('initial physical state mismatch')
    monkeypatch.setattr(m,'TrainingInitialStateAudit',UncertifiedAuditSpy)
    monkeypatch.setattr(training,'make_squashed_vision_algorithm',lambda *a,**kw:pytest.fail('PPO built before reference check'))
    with pytest.raises(ValueError,match='state mismatch'):
        run(dw,task,env,args)
    assert seen==['prepare','before_camera','after_camera','reference_compare']
    assert env.tick==0 and not (args.output/'training_initial_physical_state.json').exists()
