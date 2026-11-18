from fixture_paths import ARCHIVE, ASSET, BACKEND, CONTACT, CONTACT_SHA, SURFACE_REFS
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('pxr.Usd')
from pxr import Usd
from go2_pvcnn.control.dh116.collision_scene import SourceCollisionScene

ASSET=ASSET


@pytest.fixture(scope='module')
def scene():
    if not ASSET.exists():pytest.skip('mounted asset unavailable')
    return SourceCollisionScene(Usd.Stage.Open(str(ASSET)))


def home():
    return {s:np.r_[[0,-.6,0,-2.2,0,1.8,.7],np.full(6,.2)] for s in ('left','right')}


def test_measured_certificate_reuses_only_remaining_margin_and_rejects_new_collision():
    import json
    obj=SourceCollisionScene(Usd.Stage.Open(str(ASSET)),cache_certificates=True)
    q=json.loads((ARCHIVE/'grasp_nominal_touch_pose_v1.json').read_text())['states']
    poses={p:t.copy() for s,m in obj.models.items() for p,t in m.fk(q[s])[0].items() if p in obj.geometry}
    wire=np.c_[np.full(41,.43),np.linspace(-.25,.25,41),np.full(41,.308)]
    delta={s:np.r_[np.full(7,.151e-4),np.full(11,.31e-4)] for s in q}
    kwargs=dict(permit_pad_contact=True,measured_body_poses=poses,physical_motion_delta=delta)
    first=obj.check(q,wire,**kwargs)
    assert first['source_clearance_passed'] and not first['certificate_reused']
    second=obj.check(q,wire,**kwargs)
    assert second['certificate_reused'] and second['minimum_lower_bound_m']==first['minimum_lower_bound_m']
    poses['/FixedDualPandaDH116/left_arm/dh116/base_link'][0,3]+=.000001
    third=obj.check(q,wire,**kwargs)
    assert third['certificate_reused']
    assert third['minimum_lower_bound_m']<=first['minimum_lower_bound_m']-1.99e-6
    poses['/FixedDualPandaDH116/left_arm/dh116/base_link'][2,3]=.3
    collision=obj.check(q,wire,**kwargs)
    assert not collision['certificate_reused'] and not collision['source_clearance_passed']
    # Changing permitted contacts or interval envelopes must invalidate reuse.
    for s,m in obj.models.items():
        poses.update({p:t.copy() for p,t in m.fk(q[s])[0].items() if p in obj.geometry})
    assert not obj.check(q,wire,measured_body_poses=poses,physical_motion_delta=delta)['certificate_reused']
    delta['left'][0]*=2
    assert not obj.check(q,wire,**kwargs)['certificate_reused']


def test_cached_axis_is_only_a_hint_and_rechecks_current_vertices(monkeypatch):
    import go2_pvcnn.control.dh116.collision_scene as module
    raw=module.separation_lower_bound;calls=[]
    def counted(*a,**kw):calls.append(1);return raw(*a,**kw)
    monkeypatch.setattr(module,'separation_lower_bound',counted)
    obj=SourceCollisionScene.__new__(SourceCollisionScene);obj._separation_axes={}
    a=np.array([[0,0,0],[1,1,1]]);b=a+[3,0,0];axes=np.eye(3)
    assert obj._separation('pair',a,b,axes,.001)==2
    assert obj._separation('pair',a,b,axes,.001)==2 and len(calls)==1
    assert obj._separation('pair',a,b-[2.5,0,0],axes,.001)<0 and len(calls)==2


def test_all_authored_collision_bodies_covered(scene):
    assert len(scene.geometry)==50
    assert len(scene.empty_fixed_flanges)==2
    assert len(scene.pad_bodies)==4
    assert len(scene.layer_pins)>=5


def test_palm_wire_contact_is_rejected_even_if_pad_contact_permitted(scene):
    q=home();body='/FixedDualPandaDH116/left_arm/dh116/base_link'
    poses=scene.models['left'].fk(q['left'])[0];t=poses[body]
    center=scene.geometry[body]['vertices'].mean(0)@t[:3,:3].T+t[:3,3]
    wire=np.array([center-[0,.001,0],center+[0,.001,0]])
    r=scene.check(q,wire,permit_pad_contact=True)
    assert not r['source_clearance_passed']
    assert any(x['kind']=='wire' and x['body']==body for x in r['failures'])
    assert not r['motion_qualified']


def test_missing_side_or_out_of_limit_arm_rejected(scene):
    with pytest.raises(ValueError):scene.check({'left':home()['left']},[[0,0,0],[0,1,0]])
    q=home();q['right'][0]=100
    with pytest.raises(ValueError):scene.check(q,[[0,0,0],[0,1,0]])


def test_zero_sweep_matches_static_and_invalid_sweep_rejected(scene):
    q=home();wire=[[.43,-.25,.308],[.43,.25,.308]]
    static=scene.check(q,wire);delta={s:np.zeros(13) for s in q}
    swept=scene.check(q,wire,active_half_delta=delta)
    assert static['minimum_lower_bound_m']==swept['minimum_lower_bound_m']
    assert swept['joint_box_covered'] and not swept['motion_qualified']
    delta['left'][0]=-.1
    with pytest.raises(ValueError):scene.check(q,wire,active_half_delta=delta)


def test_measured_body_pose_overrides_nominal_fk_for_collision(scene):
    q=home();poses={}
    for side,model in scene.models.items():poses.update(model.fk(q[side])[0])
    poses={p:poses[p].copy() for p in scene.geometry}
    wire=np.array([[2.,-.25,1.],[2.,.25,1.]])
    nominal=scene.check(q,wire)
    measured=scene.check(q,wire,measured_body_poses=poses)
    assert nominal['minimum_lower_bound_m']==measured['minimum_lower_bound_m']
    body='/FixedDualPandaDH116/left_arm/dh116/base_link'
    poses[body][:3,3]=[2.,0.,1.]
    r=scene.check(q,wire,measured_body_poses=poses)
    assert r['measured_body_poses_used'] and not r['motion_qualified']
    assert any(f['body']==body and f['kind']=='wire' for f in r['failures'])


def test_measured_pose_missing_or_invalid_is_rejected(scene):
    q=home();wire=[[2,0,1],[2,1,1]]
    with pytest.raises(ValueError,match='measured'):scene.check(q,wire,measured_body_poses={})
    poses={}
    for side,model in scene.models.items():poses.update(model.fk(q[side])[0])
    poses={p:poses[p].copy() for p in scene.geometry}
    next(iter(poses.values()))[0,0]=2.
    with pytest.raises(ValueError,match='measured'):scene.check(q,wire,measured_body_poses=poses)


def test_measured_interval_accounts_for_all_physical_joint_motion_and_wire(scene):
    q=home();poses={}
    for side,model in scene.models.items():poses.update(model.fk(q[side])[0])
    poses={p:poses[p] for p in scene.geometry};delta={s:np.zeros(18) for s in q}
    body='/FixedDualPandaDH116/left_arm/dh116/base_link';t=poses[body]
    center=scene.geometry[body]['vertices'].mean(0)@t[:3,:3].T+t[:3,3]
    wire=np.array([center-[0,.001,0],center+[0,.001,0]])
    a=scene.check(q,wire,measured_body_poses=poses)
    b=scene.check(q,wire,measured_body_poses=poses,physical_motion_delta=delta,wire_motion_bound=.005)
    bound=lambda r:next(f['lower_bound_m'] for f in r['failures'] if f['body']==body and f['kind']=='wire')
    assert bound(b)==pytest.approx(bound(a)-.005)
    assert b['measured_interval_bound_used'] and not b['joint_box_covered']
    with pytest.raises(ValueError):scene.check(q,wire,physical_motion_delta=delta)
    delta['left'][0]=-.1
    with pytest.raises(ValueError):scene.check(q,wire,measured_body_poses=poses,physical_motion_delta=delta)


@pytest.mark.parametrize('side',['left','right'])
def test_actual_usd_sweep_bounds_dominate_interior_point_motion(scene,side):
    from go2_pvcnn.control.dh116.linear_coupling import coupling_matrix
    from go2_pvcnn.control.dh116.sweep_bounds import relative_motion_bound
    model=scene.models[side];q=home()[side];half=np.full(13,.01)
    full=np.r_[half[:7],np.abs(coupling_matrix())@half[7:]]
    paths=[model.bodies[b] for b in ('base_link','finger12_link')]
    records=[scene.geometry[p] for p in paths];local=[r['vertices'][0] for r in records]
    def points(state):
        pose=model.fk(state)[0]
        return [pose[p][:3,:3]@v+pose[p][:3,3] for p,v in zip(paths,local)]
    initial=points(q);distance=np.linalg.norm(initial[1]-initial[0])
    bound=relative_motion_bound(records[0]['motion_radii'],records[1]['motion_radii'],full,same_tree=True)
    for d in np.random.default_rng(42).uniform(-half,half,size=(20,13)):
        moved=points(q+d)
        for a,b,r in zip(initial,moved,records):
            assert np.linalg.norm(b-a)<=r['motion_radii']@full+1e-12
        assert abs(np.linalg.norm(moved[1]-moved[0])-distance)<=bound+1e-12
