"""D0 CPU fixtures only: never Native, a solver, or a candidate evaluator."""
from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from go2_pvcnn.control.dh116.linear_coupling import JOINTS
from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_state import SampleClock
from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_articulation_measurements import ArticulationMeasurementProvider


WRIST = ('wrist_x','wrist_y','wrist_z','wrist_roll','wrist_pitch','wrist_yaw')


def api():
    path = Path(__file__).parents[1]/'src/go2_pvcnn/control/dh116/tabletop_bimanual_mpc_model_descriptor.py'
    assert path.is_file(), 'mounted descriptor module is missing'
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_bimanual_mpc_model_descriptor')


def clock(env=0, step=480, **kw):
    d=dict(run_uuid='fixture-only',env_id=env,episode_uid=f'ep{env}',reset_generation=0,
           world_step=step,sdk_step=step+1000,world_time_s=step/480,sdk_time_s=step/480+10,policy_step=step//8)
    d.update(kw)
    return SampleClock(**d)


class View:
    def __init__(self, side):
        self.count,self.max_dofs,self.max_links=2,17,23
        self.is_homogeneous=True
        names=list(reversed(JOINTS+WRIST))
        self.prim_paths=[f'/World/envs/env_{i}/{side}Hand/wrist_fixed' for i in (1,0)]
        roots=[p.rsplit('/',1)[0] for p in self.prim_paths]
        links=['wrist_anchor']+[f'body_{i}' for i in range(22)]
        joints=names+[f'fixed_{i}' for i in range(5)]
        self.dof_paths=[r+'/'+n for r in roots for n in names]
        self.link_paths=[r+'/'+n for r in roots for n in links]
        types=[1 if n in WRIST[:3] else 0 for n in names]
        self.shared_metatype=NS(fixed_base=True,link_count=23,dof_count=17,joint_count=22,
            link_names=links,link_indices={n:i for i,n in enumerate(links)},
            dof_names=names,dof_indices={n:i for i,n in enumerate(names)},dof_types=types,
            joint_names=joints,joint_indices={n:i for i,n in enumerate(joints)},joint_types=[1]*17+[0]*5,
            joint_dof_offsets=list(range(17))+[-1]*5,joint_dof_counts=[1]*17+[0]*5)
        self.calls=[]
        vec=np.stack([np.arange(17),np.arange(17)+100.]).astype(np.float32)
        self.arrays={n:vec.copy() for n in ('get_dof_positions','get_dof_velocities','get_dof_position_targets',
            'get_dof_velocity_targets','get_dof_actuation_forces','get_dof_stiffnesses','get_dof_dampings',
            'get_dof_max_forces','get_dof_max_velocities','get_dof_armatures','get_dof_friction_coefficients',
            'get_dof_projected_joint_forces')}
        self.arrays.update(get_dof_types=np.tile(types,(2,1)),get_drive_types=np.ones((2,17),np.uint8),
            get_dof_limits=np.tile([-10.,10.],(2,17,1)),get_generalized_mass_matrices=np.tile(np.eye(17),(2,1,1)),
            get_coriolis_and_centrifugal_compensation_forces=np.ones((2,17)),
            get_gravity_compensation_forces=np.ones((2,17)),get_link_incoming_joint_force=np.ones((2,23,6)),
            get_masses=np.ones((2,23))*.1,get_coms=np.tile([0,0,0,0,0,0,1.],(2,23,1)),
            get_inertias=np.tile(np.eye(3).reshape(9),(2,23,1)),get_disable_gravities=np.zeros((2,23),np.uint8),
            get_link_transforms=np.tile([0,0,0,0,0,0,1.],(2,23,1)),get_jacobians=np.ones((2,22,6,17)))

    def get_metatype(self,i): return deepcopy(self.shared_metatype)

    def __getattr__(self,n):
        if n in self.arrays:
            def get(): self.calls.append(n); return self.arrays[n]
            return get
        raise AttributeError(n)


def attr(value, authored=True):
    return dict(available=True,value=value,authored=authored,property_stack=[dict(layer_identifier='fixture.usda',path='/property')],error=None)


class Facade:
    def __init__(self): self.calls=[]; self.mutate=None

    def capture(self,stage,binding):
        self.calls.append(binding['articulation_path'])
        root=binding['articulation_path'].rsplit('/',1)[0]
        links=[dict(path=p,attributes={'physics:mass':attr(0.,False)},
            xform=dict(matrix=np.eye(4).tolist(),reset_stack=False),applied_schemas=['PhysicsRigidBodyAPI']) for p in binding['link_paths']]
        js=[]
        meta=binding['metatype']
        for i,n in enumerate(meta['joint_names']):
            k=meta['joint_dof_counts'][i]
            typ='PhysicsFixedJoint' if k==0 else ('PhysicsPrismaticJoint' if n in WRIST[:3] else 'PhysicsRevoluteJoint')
            aa={n:attr(v) for n,v in {'physics:jointEnabled':True,'physics:excludeFromArticulation':False,
                'physics:localPos0':[0.,0.,0.],'physics:localPos1':[0.,0.,0.],
                'physics:localRot0':[1.,0.,0.,0.],'physics:localRot1':[1.,0.,0.,0.],
                'physics:axis':'Y' if '/rightHand' in root else 'X'}.items()}
            js.append(dict(path=root+'/'+n,type_name=typ,body0=[binding['link_paths'][i]],body1=[binding['link_paths'][i+1]],
                active=True,defined=True,attributes=aa,applied_schemas=[],mimic_instances=[]))
        result=dict(units=dict(meters_per_unit=1.,kilograms_per_unit=1.,up_axis='Z'),links=links,joints=js,
            world_attachments=[dict(path=root+'/wrist_fixed',type_name='PhysicsFixedJoint',body0=[],
                body1=[binding['link_paths'][0]],active=True,defined=True,
                attributes={'physics:jointEnabled':attr(True)},applied_schemas=['PhysicsArticulationRootAPI'])],
            inactive_joints=[],stage_layers=[dict(identifier='fixture.usda',sha256='a'*64)])
        if self.mutate: self.mutate(result)
        return result


def fixture():
    m=api(); stage=object(); now=[clock(0),clock(1)]
    hands={s:NS(root_physx_view=View(s)) for s in ('left','right')}
    for h in hands.values():
        h.joint_names=list(reversed(h.root_physx_view.shared_metatype.dof_names))
        h.body_names=list(reversed(h.root_physx_view.shared_metatype.link_names))
    source=dict(stage_identity='actual-fixture-stage',meters_per_unit=1.,kilograms_per_unit=1.,
                sdk_sources=[dict(path='/fixture/api.py',sha256='a'*64)])
    def reader(side,path):
        return dict(deepcopy(source),articulation_path=path,root_view_python_identity=id(hands[side].root_physx_view))
    old=ArticulationMeasurementProvider(hands=hands,clock_reader=lambda i:now[i],source_reader=reader,enabled=True)
    samples=old.capture_batch(clocks=now)
    for h in hands.values(): h.root_physx_view.calls.clear()
    scene=NS(get_gravity=lambda:np.array([0.,0.,-9.81],dtype=np.float32))
    facade=Facade()
    p=m.MountedModelDescriptorProvider(stage=stage,stage_reader=lambda:stage,hands=hands,scene_view=scene,
        clock_reader=lambda i:now[i],source_reader=reader,stage_facade=facade,enabled=True)
    return NS(m=m,p=p,stage=stage,now=now,hands=hands,samples=samples,facade=facade,source=source,scene=scene,old=old)


def test_disabled_is_zero_access_even_with_bomb_handles():
    m=api()
    class Bomb:
        def __getattribute__(self,n): raise AssertionError(n)
    p=m.MountedModelDescriptorProvider(stage=Bomb(),stage_reader=Bomb(),hands=Bomb(),scene_view=Bomb(),
        clock_reader=Bomb(),source_reader=Bomb())
    s=p.capture_batch(clocks=(clock(),),articulation_samples=(None,))[0]
    assert not s.collection_valid and not s.structure_valid
    assert 'disabled' in s.errors[0]
    assert not any(s.qualification.values())


def test_same_transaction_partial_owns_getter_before_post_clock_failure():
    from dataclasses import replace
    f=fixture();view=f.hands['left'].root_physx_view;original=view.get_link_transforms
    def torn():
        value=original();f.now[0]=replace(f.now[0],sdk_step=f.now[0].sdk_step+1);return value
    view.get_link_transforms=torn
    with pytest.raises(RuntimeError,match='clock/episode'):f.p.capture_batch(clocks=[clock(0),clock(1)],articulation_samples=f.samples)
    partial=f.p.last_partial
    assert partial['complete_measurement'] is False and partial['clocks'][0]['sdk_step']==1480
    assert partial['raw']['owned_reads']['left.get_link_transforms']['value'][0][0][6]==1.
    view.arrays['get_link_transforms'][:]=99.
    assert partial['raw']['owned_reads']['left.get_link_transforms']['value'][0][0][6]==1.
    assert partial['raw']['read_brackets'][-1]['after'] is None
    assert partial['raw']['read_brackets'][-1]['after_error']
    assert f.p.last_samples is None and not any(partial['qualification'].values())
    with pytest.raises(TypeError):partial['raw']['owned_reads']['new']=1


def test_partial_never_falls_back_to_a_prior_transaction_and_na_stays_in_returned_raw():
    f=fixture();view=f.hands['left'].root_physx_view
    view.get_jacobians=lambda:(_ for _ in ()).throw(RuntimeError('CPU unsupported Jacobian'))
    out=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)
    assert not out[0].collection_valid and out[0].raw['hands']['left']['readbacks']['get_jacobians']['available'] is False
    assert 'unsupported Jacobian' in out[0].raw['hands']['left']['readbacks']['get_jacobians']['missing_reason']
    assert f.p.last_partial is None
    f.p.last_partial={'old':'must be cleared'}
    with pytest.raises(ValueError):f.p.capture_batch(clocks=(),articulation_samples=())
    assert f.p.last_partial is None and f.p.last_samples is None


def test_unencodable_composed_read_cannot_mask_later_source_failure_or_discard_owned_numeric_partial():
    f=fixture();f.facade.mutate=lambda g:g.update(unencodable=object())
    original=f.p.source_reader;count=[0]
    def source(side,path):
        count[0]+=1;record=original(side,path)
        if count[0]>=5:record['stage_identity']='source changed in current transaction'
        return record
    f.p.source_reader=source
    with pytest.raises(RuntimeError,match='source identity changed'):f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)
    partial=f.p.last_partial
    assert partial['raw']['owned_reads']['left.get_link_transforms']['value'][0][0][6]==1.
    bad=partial['raw']['owned_reads']['left.composed_before']
    assert bad['available'] is False and bad['missing_reason']
    assert partial['complete_measurement'] is False


@pytest.mark.parametrize('channel',['get_jacobians','gravity'])
def test_unencodable_raw_getter_partial_cannot_replace_the_original_acquisition_error(channel):
    f=fixture();failure=RuntimeError('actual source failed after malformed getter')
    malformed=np.empty((2,22,6,17) if channel=='get_jacobians' else (3,),dtype=object);malformed.fill(object())
    if channel=='gravity':f.scene.get_gravity=lambda:malformed
    else:f.hands['left'].root_physx_view.get_jacobians=lambda:malformed
    original=f.p.source_reader;count=[0]
    def source(side,path):
        count[0]+=1
        if count[0]>=5:raise failure
        return original(side,path)
    f.p.source_reader=source
    with pytest.raises(RuntimeError) as caught:f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)
    assert caught.value is failure
    partial=f.p.last_partial
    assert partial['raw']['owned_reads']['left.get_link_transforms']['value'][0][0][6]==1.
    bad=partial['raw']['gravity'] if channel=='gravity' else partial['raw']['hands']['left']['readbacks'][channel]
    assert bad['available'] is False and bad['missing_reason']


def test_reuses_owned_full_d_rows_and_only_reads_four_new_getter_kinds():
    f=fixture(); out=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)
    assert all(s.collection_valid and s.structure_valid for s in out)
    for side in ('left','right'):
        hand=out[0].descriptor['hands'][side]
        assert hand['binding']['selected_view_index']==1
        assert hand['measurements']['q'][0]==100.
        assert hand['graph']['dof_joint_rows']==tuple(range(17))
        assert hand['graph']['root_link_row']==0
        assert f.hands[side].root_physx_view.calls==['get_disable_gravities','get_link_transforms','get_jacobians']
    assert not any(out[0].qualification.values())
    assert out[0].descriptor['hands']['left']['graph']['joints'][0]['attributes']['physics:axis']['value']=='X'
    assert out[0].descriptor['hands']['right']['graph']['joints'][0]['attributes']['physics:axis']['value']=='Y'


def test_raw_owned_before_aliasing_and_deeply_immutable():
    f=fixture(); v=f.hands['left'].root_physx_view; orig=v.get_link_transforms
    def alias(): v.arrays['get_disable_gravities'][:]=1; return orig()
    v.get_link_transforms=alias
    out=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)
    s=out[0]
    assert s.descriptor['hands']['left']['readbacks']['get_disable_gravities']['selected'][0]==0
    f.samples[0].raw['hands']['left']['binding']['dof_names'][0]='mutated'
    assert s.descriptor['hands']['left']['binding']['dof_names'][0]=='wrist_yaw'
    with pytest.raises(TypeError): s.raw['x']=1
    with pytest.raises(TypeError): s.descriptor['hands']['left']['binding']['dof_names'][0]='x'


@pytest.mark.parametrize('fault',['axis','cycle','duplicate_joint','missing_link','non_si','quaternion','scale'])
def test_bad_composed_structure_keeps_raw_and_is_not_qualified(fault):
    f=fixture()
    def mutate(g):
        if fault=='axis': g['joints'][0]['attributes']['physics:axis']=attr('UNKNOWN')
        elif fault=='cycle': g['joints'][0]['body0']=g['joints'][-1]['body1']
        elif fault=='duplicate_joint': g['joints'][1]['path']=g['joints'][0]['path']
        elif fault=='missing_link': g['links'].pop()
        elif fault=='non_si': g['units']['meters_per_unit']=.01
        elif fault=='quaternion': g['joints'][0]['attributes']['physics:localRot0']=attr([0,0,0,0])
        else: g['links'][0]['xform']['matrix'][0][0]=2.
    f.facade.mutate=mutate
    s=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]
    assert not s.structure_valid and not s.collection_valid and s.errors
    assert s.raw['hands']['left']['composed_before']['value']
    assert not any(s.qualification.values())


def test_authored_mass_sentinel_is_not_invented_inertia_or_zero_force():
    f=fixture(); s=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]
    assert s.structure_valid
    hand=s.descriptor['hands']['left']
    assert hand['graph']['links'][0]['attributes']['physics:mass']['value']==0.
    assert hand['measurements']['link_mass_raw'][0]==.1
    assert hand['semantics']['inertia_frame_qualified'] is False
    assert hand['semantics']['mimic_dynamics_qualified'] is False


@pytest.mark.parametrize('fault',['stale_sample','view_changed','source_changed','graph_torn','clock_torn','wrong_episode'])
def test_torn_or_replayed_identity_is_rejected_without_history_commit(fault):
    f=fixture()
    if fault=='stale_sample': f.now[:]=[clock(0,481),clock(1,481)]
    elif fault=='view_changed': f.hands['left'].root_physx_view=View('left')
    elif fault=='source_changed': f.source['sdk_sources'][0]['sha256']='b'*64
    elif fault=='graph_torn':
        orig=f.facade.capture
        def torn(stage,binding):
            g=orig(stage,binding)
            if len(f.facade.calls)>4: g['joints'][0]['attributes']['physics:localPos0']=attr([.01,0,0])
            return g
        f.facade.capture=torn
    elif fault=='clock_torn':
        def torn(): f.now[1]=clock(1,481); return np.array([0.,0.,-9.81])
        f.scene.get_gravity=torn
    else: f.now[0]=clock(0,episode_uid='other')
    with pytest.raises((RuntimeError,ValueError),match='clock|sample|source|view|composed|identity'):
        f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)
    assert f.p.last_samples is None and f.p._previous=={}


@pytest.mark.parametrize('name',['get_disable_gravities','get_link_transforms','get_jacobians'])
def test_getter_error_preserves_na_and_does_not_fill_missing_data(name):
    f=fixture()
    def bad(): raise RuntimeError('unsupported fixture getter')
    setattr(f.hands['left'].root_physx_view,name,bad)
    s=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]
    assert s.structure_valid and not s.collection_valid
    assert s.raw['hands']['left']['readbacks'][name]['value'] is None
    assert 'unsupported' in s.raw['hands']['left']['readbacks'][name]['missing_reason']


def test_terminal_is_saved_and_cannot_be_reused_after_noop():
    f=fixture(); f.now[:]=[clock(0,terminal=True),clock(1,terminal=True)]
    f.samples=f.old.__class__(hands=f.hands,clock_reader=lambda i:f.now[i],source_reader=f.p.source_reader,enabled=True).capture_batch(clocks=f.now)
    s=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]
    assert s.clock.terminal and s.collection_valid
    previous=f.p._previous.copy()
    f.now[:]=[clock(0,481),clock(1,481)]
    with pytest.raises(ValueError,match='terminal'):
        f.p.capture_batch(clocks=f.now,articulation_samples=(None,None))
    assert f.p._previous==previous


def test_inactive_and_reset_do_not_read_or_commit_live_state():
    f=fixture(); f.now[1]=clock(1,reset=True)
    out=f.p.capture_batch(clocks=f.now,articulation_samples=(None,None),active_mask=(False,True))
    assert not any(s.collection_valid for s in out)
    assert not f.facade.calls and f.p._previous=={}
    assert all(not h.root_physx_view.calls for h in f.hands.values())


def test_composed_link_and_joint_traversal_permutation_cannot_change_sdk_rows():
    f=fixture()
    def reorder(g): g['links'].reverse(); g['joints'].reverse()
    f.facade.mutate=reorder
    s=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]
    assert s.collection_valid
    assert s.descriptor['hands']['left']['graph']['root_link_row']==0
    assert [r['path'] for r in s.descriptor['hands']['left']['graph']['links']]==f.samples[0].raw['hands']['left']['binding']['link_paths']


def test_live_layout_mutation_during_new_getter_rolls_back_transaction():
    f=fixture(); v=f.hands['left'].root_physx_view; orig=v.get_disable_gravities
    def mutate():
        a=orig(); v.prim_paths.reverse(); return a
    v.get_disable_gravities=mutate
    with pytest.raises(RuntimeError,match='layout|metadata|binding'):
        f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)
    assert f.p._previous=={} and f.p.last_samples is None


def test_bad_inactive_numeric_row_is_raw_evidence_not_active_row_failure():
    f=fixture()
    for h in f.hands.values(): h.root_physx_view.arrays['get_link_transforms'][0,0,0]=np.nan
    s=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples,active_mask=(True,False))[0]
    assert s.collection_valid
    assert np.isnan(s.raw['hands']['left']['readbacks']['get_link_transforms']['value'][0][0][0])


def test_composed_reader_runtime_error_is_explicit_raw_na_not_zero_graph():
    f=fixture()
    def bad(stage,binding): raise RuntimeError('composed accessor unsupported')
    f.facade.capture=bad
    s=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]
    assert not s.collection_valid and not s.structure_valid
    assert s.raw['hands']['left']['composed_before']['value'] is None
    assert 'unsupported' in s.raw['hands']['left']['composed_before']['missing_reason']


def test_missing_active_sdk_source_identity_is_rejected():
    f=fixture(); f.source['sdk_sources']=[]
    f.samples=f.old.__class__(hands=f.hands,clock_reader=lambda i:f.now[i],source_reader=f.p.source_reader,enabled=True).capture_batch(clocks=f.now)
    with pytest.raises(RuntimeError,match='source'):
        f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)


def test_valid_reset_opens_new_episode_but_repeated_clock_is_rejected():
    f=fixture(); f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)
    f.now[:]=[clock(i,481,episode_uid=f'new{i}',reset_generation=1) for i in range(2)]
    f.samples=f.old.capture_batch(clocks=f.now)
    s=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]
    assert s.collection_valid and s.clock.episode_uid=='new0'
    with pytest.raises(ValueError,match='stale|repeated'):
        f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)


class FakeLayer:
    identifier='fixture.usda'; realPath='fixture.usda'
    def ExportToString(self): return 'CPU-owned USD-shaped fixture only'


class FakeAttr:
    def __init__(self,name,record,stage=None): self.name,self.record,self.stage=name,record,stage
    def GetName(self): return self.name
    def Get(self): return deepcopy(self.record['value'])
    def HasAuthoredValueOpinion(self): return self.record.get('authored',True)
    def GetPropertyStack(self,time):
        if self.stage is not None:
            self.stage.stack_calls.append(('attribute',self.name,time))
            assert time is self.stage.default_time,'the same explicit Default TimeCode is required'
        return [NS(layer=FakeLayer(),path='/source.'+self.name)]
    def GetResolveInfo(self): return NS(GetSource=lambda:'Default')


class FakeRel:
    def __init__(self,name,targets,stage=None): self.name,self.targets,self.stage=name,targets,stage
    def GetName(self): return self.name
    def GetTargets(self): return self.targets[:]
    def HasAuthoredTargets(self): return True
    def GetPropertyStack(self,time):
        if self.stage is not None:
            self.stage.stack_calls.append(('relationship',self.name,time))
            assert time is self.stage.default_time,'the same explicit Default TimeCode is required'
        return [NS(layer=FakeLayer(),path='/source.'+self.name)]


class FakePrim:
    def __init__(self,stage,record): self.stage,self.record=stage,record
    def GetPath(self): return self.record['path']
    def GetTypeName(self): return self.record.get('type_name','Xform')
    def IsActive(self): return self.record.get('active',True)
    def IsDefined(self): return self.record.get('defined',True)
    def IsValid(self): return True
    def IsInstanceProxy(self): return False
    def GetAppliedSchemas(self): return self.record.get('applied_schemas',[])[:]
    def GetAttributes(self): return [FakeAttr(k,v,self.stage) for k,v in self.record.get('attributes',{}).items()]
    def GetRelationships(self): return [FakeRel(k,v,self.stage) for k,v in self.record.get('relationships',{}).items()]
    def GetRelationship(self,name):
        return next((r for r in self.GetRelationships() if r.GetName()==name),None)
    def IsA(self,schema):
        return self.GetTypeName().endswith('Joint') if schema is FakeJoint else self.GetTypeName()=='PhysicsScene'
    def GetParent(self): return self.stage.GetPrimAtPath(self.GetPath().rsplit('/',1)[0])


class FakeJoint:
    def __init__(self,prim): self.prim=prim
    def GetBody0Rel(self): return FakeRel('physics:body0',self.prim.record.get('body0',[]))
    def GetBody1Rel(self): return FakeRel('physics:body1',self.prim.record.get('body1',[]))


class FakeStage:
    def __init__(self,g):
        self.default_time=object();self.default_calls=0;self.stack_calls=[]
        self.nodes={}
        for row in g['links']+g['joints']+g['world_attachments']+g['inactive_joints']:
            row=deepcopy(row)
            if 'body0' in row:
                row.setdefault('relationships',{}).update({'physics:body0':row['body0'],'physics:body1':row['body1']})
            self.nodes[row['path']]=FakePrim(self,row)
        root=g['links'][0]['path'].rsplit('/',1)[0]
        self.nodes[root]=FakePrim(self,dict(path=root,attributes={'physics:mass':attr(5.)}))
        self.nodes['/']=FakePrim(self,dict(path='/'))
    def GetPrimAtPath(self,path): return self.nodes.get(str(path))
    def TraverseAll(self): return iter(self.nodes.values())
    def GetLayerStack(self,includeSessionLayers): return [FakeLayer()]


def fake_usd_facade():
    m=api(); f=fixture(); b=f.samples[0].raw['hands']['left']['binding']; g=f.facade.capture(f.stage,b)
    g['inactive_joints']=[dict(g['world_attachments'][0],path=b['articulation_path'].rsplit('/',1)[0]+'/root_joint',active=False)]
    stage=FakeStage(g)
    geom=NS(GetStageMetersPerUnit=lambda s:1.,GetStageUpAxis=lambda s:'Z',
        XformCache=lambda:NS(GetLocalToWorldTransform=lambda p:np.eye(4)),
        Xformable=lambda p:NS(GetLocalTransformation=lambda:np.eye(4),GetResetXformStack=lambda:False))
    physics=NS(Joint=FakeJoint,Scene='SceneSchema',GetStageKilogramsPerUnit=lambda s:1.)
    def default():stage.default_calls+=1;return stage.default_time
    core=NS(TimeCode=NS(Default=default))
    facade=m.ComposedUSDStageFacade(usd_geom=geom,usd_physics=physics,usd_core=core)
    return NS(m=m,b=b,g=g,stage=stage,facade=facade,core=core)


def test_property_stacks_use_one_shared_explicit_default_timecode_for_both_property_types():
    f=fake_usd_facade();g=f.facade.capture(f.stage,f.b)
    assert len(g['joints'])==22,[(r['path'],r['attributes'].get('physics:jointEnabled')) for r in g['inactive_joints']]
    assert f.stage.default_calls==1
    assert {kind for kind,_,_ in f.stage.stack_calls}=={'attribute','relationship'}
    assert all(time is f.stage.default_time for _,_,time in f.stage.stack_calls)
    assert all(r['attributes']['physics:jointEnabled']['available'] is True for r in g['joints'])
    assert all(r['relationships']['physics:body0']['available'] is True for r in g['joints'])
    graph=f.m.assemble_mounted_graph(g,f.b)
    assert len(graph['links'])==23 and len(graph['joints'])==22 and len(graph['inactive_joints'])==1


@pytest.mark.parametrize('kind',['attribute','relationship'])
def test_property_stack_failure_remains_raw_na_after_explicit_default_read(monkeypatch,kind):
    f=fake_usd_facade();cls=FakeAttr if kind=='attribute' else FakeRel;original=cls.GetPropertyStack
    target='physics:jointEnabled' if kind=='attribute' else 'physics:body0'
    def failed(self,time):
        assert time is f.stage.default_time
        if self.name==target:raise RuntimeError('synthetic unresolved property stack provenance')
        return original(self,time)
    monkeypatch.setattr(cls,'GetPropertyStack',failed);g=f.facade.capture(f.stage,f.b)
    row=(g['inactive_joints'] if kind=='attribute' else g['joints'])[0]
    record=row['attributes'][target] if kind=='attribute' else row['relationships'][target]
    assert record['available'] is False and record['property_stack']==[]
    assert record['error']=='RuntimeError: synthetic unresolved property stack provenance'
    assert record['value' if kind=='attribute' else 'targets'] is None
    if kind=='attribute':
        assert not g['joints']
        with pytest.raises(ValueError):f.m.assemble_mounted_graph(g,f.b)
    else:
        # Preserve the existing coverage boundary: actual flat targets survive
        # an unavailable provenance record, without inventing a stack/default.
        assert row['body0'] and len(g['joints'])==22


def test_usd_default_timecode_is_lazy_and_failure_is_not_replaced_with_numeric_time():
    m=api()
    class Bomb:
        def __getattribute__(self,name):raise AssertionError('constructor touched USD core: '+name)
    facade=m.ComposedUSDStageFacade(usd_core=Bomb())
    assert facade.usd_geom is None and facade.usd_physics is None
    f=fake_usd_facade();error=RuntimeError('synthetic Default TimeCode unavailable')
    def failed():raise error
    f.facade.usd_core=NS(TimeCode=NS(Default=failed))
    with pytest.raises(RuntimeError) as caught:f.facade.capture(f.stage,f.b)
    assert caught.value is error and f.stage.stack_calls==[]


def test_injected_usd_facade_collects_full_graph_defaults_and_inactive_root():
    f=fake_usd_facade(); payload=f.facade.capture(f.stage,f.b)
    graph=f.m.assemble_mounted_graph(payload,f.b)
    assert len(graph['links'])==23 and len(graph['joints'])==22
    assert graph['inactive_joints'][0]['path'].endswith('/root_joint')
    assert graph['links'][0]['attributes']['physics:mass']['authored'] is False
    assert graph['stage_layers'][0]['sha256']


def test_composed_attribute_retains_resolve_source_not_only_authored_flag():
    f=fake_usd_facade(); g=f.facade.capture(f.stage,f.b)
    assert g['links'][0]['attributes']['physics:mass']['resolve_source']=='Default'


def test_composed_joint_relationship_provenance_and_parent_mass_are_retained():
    f=fake_usd_facade(); g=f.facade.capture(f.stage,f.b)
    joint=g['joints'][0]
    assert joint['relationships']['physics:body0']['targets']==joint['body0']
    assert joint['relationships']['physics:body0']['property_stack'][0]['layer']['sha256']
    ancestors=g['links'][0]['ancestor_physics_attributes']
    assert any(r['attributes'].get('physics:mass',{}).get('value')==5. for r in ancestors)


def test_mimic_unknown_compliance_is_raw_not_perfect_affine_constraint():
    f=fake_usd_facade(); joint=f.stage.GetPrimAtPath(f.b['dof_paths'][0]); ref=f.b['dof_paths'][1]
    prefix='physxMimicJoint:rotX:'
    joint.record['applied_schemas']=['PhysxMimicJointAPI:rotX']
    joint.record['relationships'][prefix+'referenceJoint']=[ref]
    joint.record['attributes'].update({prefix+'gearing':attr(-.88),prefix+'offset':attr(.4),
        prefix+'referenceJointAxis':attr('rotX'),prefix+'naturalFrequency':attr(20.)})
    g=f.facade.capture(f.stage,f.b); graph=f.m.assemble_mounted_graph(g,f.b)
    mimic=graph['joints'][0]['mimic_instances'][0]
    assert mimic['reference_joint']==[ref]
    assert mimic['attributes'][prefix+'naturalFrequency']['value']==20.
    assert mimic['compliance_backend_qualified'] is False


def test_bool_disable_flags_are_preserved_without_casting_to_physical_forces():
    f=fixture()
    for h in f.hands.values(): h.root_physx_view.arrays['get_disable_gravities']=np.zeros((2,23),dtype=bool)
    s=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]
    assert s.collection_valid
    assert s.raw['hands']['left']['readbacks']['get_disable_gravities']['raw_dtype']=='bool'


def test_raw_owned_articulation_numeric_mutation_cannot_silently_change_provenance():
    f=fixture(); f.samples[0].raw['hands']['left']['getters']['get_dof_positions']['value'][1][0]=-999.
    s=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]
    assert not s.collection_valid
    assert any('raw' in e and 'q' in e for e in s.errors)


def test_composed_and_live_source_units_must_agree():
    f=fixture(); f.source['meters_per_unit']=.01
    f.samples=f.old.__class__(hands=f.hands,clock_reader=lambda i:f.now[i],source_reader=f.p.source_reader,enabled=True).capture_batch(clocks=f.now)
    s=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]
    assert not s.collection_valid and not s.structure_valid
    assert any('source units' in e for e in s.errors)


def test_layout_changed_by_final_composed_read_is_not_committed():
    f=fixture(); orig=f.facade.capture
    def mutate(stage,b):
        payload=orig(stage,b)
        if len(f.facade.calls)==8: f.hands['left'].root_physx_view.prim_paths.reverse()
        return payload
    f.facade.capture=mutate
    with pytest.raises(RuntimeError,match='layout|binding'):
        f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)
    assert f.p._previous=={}


def test_reused_field_keeps_measurement_source_unit_frame_and_clock():
    f=fixture(); s=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]
    receipt=s.descriptor['hands']['left']['measurement_receipts']['link_com_pose_xyzw_raw']
    assert receipt['clock']['episode_uid']=='ep0'
    assert receipt['source'].endswith('get_coms()')
    assert 'quaternion_xyzw' in receipt['unit']


@pytest.mark.parametrize('fault',['shape','nonfinite','flag_enum','zero_quaternion'])
def test_malformed_new_readback_keeps_raw_and_refuses_collection(fault):
    f=fixture(); v=f.hands['left'].root_physx_view
    if fault=='shape': v.arrays['get_jacobians']=np.ones((2,17,22,6))
    elif fault=='nonfinite': v.arrays['get_link_transforms'][1,0,0]=np.nan
    elif fault=='flag_enum': v.arrays['get_disable_gravities'][1,0]=2
    else: v.arrays['get_link_transforms'][1,0,3:]=0.
    s=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]
    assert s.structure_valid and not s.collection_valid
    assert s.raw['hands']['left']['readbacks']


def test_episode_uid_cannot_be_reused_after_an_intervening_reset():
    f=fixture(); f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)
    for step,uid,generation in [(481,'second',1),(482,'ep',2)]:
        f.now[:]=[clock(i,step,episode_uid=(f'ep{i}' if uid=='ep' else f'{uid}{i}'),reset_generation=generation) for i in range(2)]
        f.samples=f.old.capture_batch(clocks=f.now)
        if generation==1: f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)
        else:
            with pytest.raises(ValueError,match='episode.*reus|episode.*generation'):
                f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)


def test_scene_view_swap_during_getter_is_rejected_and_gravity_has_source_identity():
    f=fixture()
    s=f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)[0]
    assert s.raw['gravity']['source']=='existing_scene_view.get_gravity()'
    assert s.raw['handle_identity']['scene_view_python_identity']==id(f.scene)
    f=fixture()
    def changed():
        f.p.scene_view=NS(get_gravity=lambda:np.array([0,0,-9.81])); return np.array([0.,0.,-9.81])
    f.scene.get_gravity=changed
    with pytest.raises(RuntimeError,match='scene.*identity'):
        f.p.capture_batch(clocks=f.now,articulation_samples=f.samples)
    assert f.p._previous=={}
