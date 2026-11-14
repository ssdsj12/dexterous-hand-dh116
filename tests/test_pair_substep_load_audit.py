import copy
import numpy as np
import pytest
from go2_pvcnn.control.dh116.tabletop_pair_substep_load_audit import audit_capture
def pair(side,pad,step,force,normal,node=0):
    n=np.array(normal,dtype=float)
    return dict(mesh_name=side+'.'+pad,substep=step,node_index=node,
        outward_normal_world=n.tolist(),normal_force_on_rod_n=force,
        force_on_rod_world_n=(force*n).tolist(),
        tangential_force_on_rod_world_n=[0.,0.,0.],impulse_on_rod_ns=(force*n/5760).tolist())
def capture(pairs):
    return dict(pairs=pairs,substep_dt_s=1/5760,physics_step=20,rod_tick=216)
def test_time_disjoint_contacts_never_become_simultaneous_opposition():
    c=capture([pair('left','finger14_link',0,.1,[1,0,0]),
               pair('left','finger23_link',1,.1,[-1,0,0])])
    r=audit_capture(c,solver_substeps=12,complete_capture=True)
    assert not any(s['hands']['left']['raw_surface_paired'] for s in r['substeps'])
    assert r['positive_effective_contact_qualified'] is False
def test_threshold_is_strict_and_original_point_zero_five():
    for force,expected in [(.05,False),(.05001,True)]:
        r=audit_capture(capture([pair('left','finger14_link',0,force,[1,0,0]),
            pair('left','finger23_link',0,force,[-1,0,0])]),solver_substeps=12,complete_capture=True)
        assert r['substeps'][0]['hands']['left']['raw_surface_paired'] is expected
        assert r['native_qualification'] is False
def test_multiple_nodes_per_pad_sum_without_fake_digit_count():
    p=[pair('left','finger14_link',0,.04,[1,0,0],node=n) for n in (0,1)]
    r=audit_capture(capture(p),solver_substeps=12,complete_capture=True)
    h=r['substeps'][0]['hands']['left']
    assert h['bearing_digit_count']==1 and h['normal_load_n'][0]==pytest.approx(.08)
    assert not h['raw_surface_paired']
def test_duplicate_pair_cannot_double_load():
    p=pair('left','finger14_link',0,.04,[1,0,0])
    with pytest.raises(ValueError):audit_capture(capture([p,p]),solver_substeps=12,complete_capture=True)
def test_force_and_impulse_consistency_are_checked():
    for field,value in [('normal_force_on_rod_n',-.1),('impulse_on_rod_ns',[1,0,0]),
                        ('outward_normal_world',[2,0,0]),('force_on_rod_world_n',[float('nan'),0,0])]:
        p=pair('right','finger14_link',0,.1,[1,0,0]);p[field]=value
        with pytest.raises(ValueError):audit_capture(capture([p]),solver_substeps=12,complete_capture=True)
def test_missing_capture_is_unknown_not_empty_measured_zero():
    with pytest.raises(ValueError):audit_capture(capture([]),solver_substeps=12,complete_capture=False)
def test_known_empty_complete_capture_is_zero_loaded_not_grasp():
    r=audit_capture(capture([]),solver_substeps=12,complete_capture=True)
    assert len(r['substeps'])==12
    assert all(not h['raw_surface_paired'] for s in r['substeps'] for h in s['hands'].values())
@pytest.mark.parametrize('substep',[-1,12,1.5,True])
def test_out_of_clock_pair_rejected(substep):
    with pytest.raises(ValueError):
        audit_capture(capture([pair('left','finger14_link',substep,.1,[1,0,0])]),
            solver_substeps=12,complete_capture=True)
def test_normals_and_force_vectors_reported_separately():
    # Strong tangent load must not silently become an observed normal direction.
    p=pair('left','finger14_link',0,.1,[1,0,0])
    p['tangential_force_on_rod_world_n']=[0.,.2,0.]
    p['force_on_rod_world_n']=[.1,.2,0.]
    p['impulse_on_rod_ns']=(np.array([.1,.2,0.])/5760).tolist()
    r=audit_capture(capture([p]),solver_substeps=12,complete_capture=True)
    h=r['substeps'][0]['hands']['left']
    assert h['normal_vectors_world'][0]==pytest.approx([.1,0,0])
    assert h['force_on_rod_world_n'][0]==pytest.approx([.1,.2,0])

def test_binding_rejects_foreign_clock_and_preserves_nominal_time_scope():
    from go2_pvcnn.control.dh116.tabletop_pair_substep_load_audit import bind_clock
    c=dict(physics_step=3,rod_tick=12,substep_dt_s=1/5760)
    clock=dict(run_uuid='run',env_id=0,episode_uid='ep',reset_id=1,rod_uid='rod',
        physics_step=3,rod_tick=12,timestamp_s=3/480,clock_valid=True)
    rod=dict(clock)
    bind_clock(c,rod,clock,expected_physics_step=3)
    for field,value in [('physics_step',4),('rod_tick',13),('episode_uid','other'),
                        ('timestamp_s',float('nan')),('clock_valid',False)]:
        bad=dict(clock);bad[field]=value
        with pytest.raises(ValueError):bind_clock(c,rod,bad,expected_physics_step=3)
