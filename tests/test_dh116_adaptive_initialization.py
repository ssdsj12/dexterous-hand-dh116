from copy import deepcopy

import numpy as np
import pytest


def fixture():
    from go2_pvcnn.control.dh116.linear_coupling import expand
    q = np.array([.5, .5, .5, .5, .5, .5])
    provenance = dict(pose_sha256='p', source_sha256=dict(left='l', right='r'),
                      physics_hz=480, control_hz=60, solver_type=0, position_iterations=16)
    payload = dict(format='dh116-adaptive-grasp-initialization-v1',
        source_contract=provenance, roots=[[.5,.2,.4,1,0,0,0],[.5,-.2,.4,1,0,0,0]],
        joint_positions=[expand(q).tolist(), expand(q).tolist()],
        grasp_command=np.tile(q,2).tolist(), wire_pose=[.43,0,.308,1,0,0,0])
    layout = dict(hand_roots=dict(left=np.zeros(7),right=np.zeros(7)),
                  active_joints=dict(left=q,right=q),wire_center=np.zeros(3))
    return payload, layout, provenance


def test_authored_seed_preserves_exact_roots_without_mutating_inputs():
    from go2_pvcnn.control.dh116.tabletop_adaptive_initialization import adaptive_initialization
    payload, layout, provenance = fixture()
    before = deepcopy(payload)
    configured, state = adaptive_initialization(payload, layout, provenance)
    for i,side in enumerate(('left','right')):
        actual = configured['hand_roots'][side].copy()
        actual[2] += state['root_offset']
        np.testing.assert_allclose(actual, payload['roots'][i])
    assert payload == before
    assert np.all(layout['wire_center'] == 0)
    assert np.all(state['joint_velocities'] == 0)
    assert np.all(state['wire_velocity'] == 0)
    np.testing.assert_allclose(state['initial_command'], payload['grasp_command'])


@pytest.mark.parametrize('key,value', [('pose_sha256','wrong'),('physics_hz',240),
                                     ('source_sha256',dict(left='wrong',right='r'))])
def test_authored_geometry_requires_exact_provenance(key,value):
    from go2_pvcnn.control.dh116.tabletop_adaptive_initialization import adaptive_initialization
    payload, layout, provenance = fixture()
    payload['source_contract'] = dict(provenance, **{key:value})
    with pytest.raises(ValueError,match='provenance'):
        adaptive_initialization(payload, layout, provenance)


@pytest.mark.parametrize('field,index,value', [('roots',(0,3),0),
    ('joint_positions',(0,2),.9),('grasp_command',(0,),float('nan')),
    ('wire_pose',(2,),.2),('wire_pose',(3,),0)])
def test_bad_geometry_is_rejected(field,index,value):
    from go2_pvcnn.control.dh116.tabletop_adaptive_initialization import adaptive_initialization
    payload, layout, provenance = fixture()
    a=np.array(payload[field]);a[index]=value;payload[field]=a.tolist()
    with pytest.raises(ValueError):
        adaptive_initialization(payload,layout,provenance)


def test_adaptive_cli_requires_fresh_geometry_and_rejects_old_policy_transfer():
    from go2_pvcnn.control.dh116.tabletop_adaptive_initialization import validate_adaptive_options
    validate_adaptive_options(adaptive_grasp=True, grasp_initialization='new.json',
                              close_reset_snapshot=None, initialize_from=None, physics_hz=480)
    for change in (dict(grasp_initialization=None),dict(initialize_from='old.zip'),
                   dict(close_reset_snapshot='old.json'),dict(physics_hz=240)):
        args=dict(adaptive_grasp=True,grasp_initialization='new.json',
                  close_reset_snapshot=None,initialize_from=None,physics_hz=480)
        args.update(change)
        with pytest.raises(ValueError):
            validate_adaptive_options(**args)


def test_legacy_cli_needs_measured_snapshot_and_forbids_unused_geometry():
    from go2_pvcnn.control.dh116.tabletop_adaptive_initialization import validate_adaptive_options
    args=dict(adaptive_grasp=False,grasp_initialization=None,
              close_reset_snapshot='old.json',initialize_from=None,physics_hz=240)
    validate_adaptive_options(**args)
    for change in (dict(close_reset_snapshot=None),dict(grasp_initialization='ignored.json')):
        with pytest.raises(ValueError):
            validate_adaptive_options(**dict(args,**change))


def test_hollow_object_contract_checks_asset_hash_and_outer_geometry(tmp_path):
    import hashlib
    import json
    from go2_pvcnn.control.dh116.tabletop_adaptive_initialization import object_spec
    directory=tmp_path/'assets/dh116_hollow_straw_v1';directory.mkdir(parents=True)
    usd=directory/'straw.usda';usd.write_text('test asset bytes')
    mass=900*np.pi*(.008**2-.0075**2)*.5
    manifest=dict(mass_kg=mass,outer_radius_m=.008,inner_radius_m=.0075,
        length_m=.5,wall_thickness_m=.0005,density_kg_m3=900.,axis='Y',collision_segments=32,
        usd_sha256=hashlib.sha256(usd.read_bytes()).hexdigest())
    path=directory/'manifest.json';path.write_text(json.dumps(manifest))
    actual_usd,actual_mass,contract=object_spec(tmp_path,'hollow_straw')
    assert actual_usd == str(usd)
    assert actual_mass == pytest.approx(mass)
    assert contract['object']['usd_sha256'] == manifest['usd_sha256']
    usd.write_text('changed collision shape')
    with pytest.raises(ValueError,match='hash'):
        object_spec(tmp_path,'hollow_straw')
    manifest['usd_sha256']=hashlib.sha256(usd.read_bytes()).hexdigest()
    manifest['outer_radius_m']=.01;path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='geometry'):
        object_spec(tmp_path,'hollow_straw')


def test_original_solid_object_leaves_legacy_contract_unchanged(tmp_path):
    from go2_pvcnn.control.dh116.tabletop_adaptive_initialization import object_spec
    assert object_spec(tmp_path,'solid_cylinder') == (None,.12,{})
    with pytest.raises(ValueError):
        object_spec(tmp_path,'not_an_object')
