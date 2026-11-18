import pytest

from go2_pvcnn.control.dh116.tabletop_teacher_task_contact import route_object_reports


def make_report(pairs, points, anchors):
    return dict(headers=pairs, contacts=points, friction_anchors=anchors,
                friction_source_available=True)


def header(a, b, offset=0, friction=0):
    return dict(actor0=a, actor1=b, collider0=a+'/shape', collider1=b+'/shape',
                stage_id=1, contact_data_offset=offset, num_contact_data=1,
                friction_anchors_offset=friction, num_friction_anchors_data=1)


def test_contact_poll_preserves_separate_sources_and_actual_partner_side():
    report=make_report([header('/hand/finger14', '/rod')],
        [dict(impulse=[0.,0.,.02], normal=[0.,0.,1.])],
        [dict(impulse=[.01,0.,0.])])
    result=route_object_reports(report, objects={'/rod':0},
        digits={'/hand/finger14':(0,'left','thumb')})
    pair=result[0][0]
    assert pair['object_actor_index']==1
    assert pair['digit']==('left','thumb')
    assert pair['source_contact_impulse_world_n_s']==[[0.,0.,.02]]
    assert pair['source_friction_impulse_world_n_s']==[[.01,0.,0.]]
    assert pair['object_force_world_n'] is None


def test_external_and_cross_environment_sources_remain_unresolved():
    report=make_report([header('/rod','/table'),header('/rod','/other/finger14',1,1)],
        [dict(impulse=[0.,0.,.02])]*2,[dict(impulse=[.01,0.,0.])]*2)
    result=route_object_reports(report, objects={'/rod':0, '/other/rod':1},
        digits={'/other/finger14':(1,'right','thumb')})
    assert [p['partner_scope'] for p in result[0]]==['unmapped_partner','cross_environment_digit']
    assert result[1]==[]


def test_orphan_anchor_cannot_be_silently_dropped():
    with pytest.raises(ValueError, match='unowned'):
        route_object_reports(make_report([], [], [dict(impulse=[0.,0.,.1])]),
            objects={'/rod':0},digits={})


def test_readback_is_configured_before_world_and_restored_after_probe():
    from contextlib import ExitStack
    from types import SimpleNamespace

    from go2_pvcnn.control.dh116.tabletop_teacher_task_contact import install_probe_world

    events = []

    class World:
        def __init__(self, value):
            events.append(('world',value))

    module = SimpleNamespace(GpuWorld=World)
    with ExitStack() as stack:
        install_probe_world(stack, module, lambda: events.append(('readback',False)) or {'enabled':True})
        instance = module.GpuWorld('original arguments')
        assert instance.contact_probe_before_reset == {'enabled':True}
    assert module.GpuWorld is World
    assert events == [('readback',False),('world','original arguments')]


def test_report_hypotheses_use_object_actor_order_without_merging_channels():
    from go2_pvcnn.control.dh116.tabletop_teacher_task_contact import momentum_comparison

    report=make_report([header('/hand/finger14','/rod')],
        [dict(impulse=[0.,0.,-.01])],[dict(impulse=[-.003,0.,0.])])
    pairs=route_object_reports(report, objects={'/rod':0},
        digits={'/hand/finger14':(0,'left','thumb')})[0]
    result=momentum_comparison(pairs, before=[0.,0.,0.], after=[.15,0.,.4],
        mass_kg=.02, gravity_world_m_s2=[0.,0.,-10.], dt_s=.01)
    assert result['non_gravity_impulse_world_n_s']==pytest.approx([.003,0.,.01])
    assert result['actor0_contact_impulse_world_n_s']==pytest.approx([0.,0.,.01])
    assert result['actor0_friction_impulse_world_n_s']==pytest.approx([.003,0.,0.])
    assert result['raw_on_actor0_residual_n_s']==pytest.approx(0.)
    assert result['raw_on_actor1_residual_n_s']>.02
    assert result['source_qualified'] is False


def test_empty_report_keeps_momentum_residual_visible():
    from go2_pvcnn.control.dh116.tabletop_teacher_task_contact import momentum_comparison

    result=momentum_comparison([],before=[0.,0.,0.],after=[0.,0.,.1],
        mass_kg=.02,gravity_world_m_s2=[0.,0.,-10.],dt_s=.01)
    assert result['raw_on_actor0_residual_n_s']==pytest.approx(.004)
    assert result['raw_on_actor1_residual_n_s']==pytest.approx(.004)
    assert result['source_qualified'] is False


def test_task_probe_pre_reset_readback_is_never_implicit(tmp_path, capsys, monkeypatch):
    from scripts import collect_dh116_teacher_task_contact_probe as cli

    checkpoint=tmp_path/'checkpoint.pt'
    checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(cli, 'load_config', lambda path: ({}, None))
    import hashlib
    import json
    base=['--config',str(tmp_path/'config.json'),'--checkpoint',str(checkpoint),
          '--checkpoint-sha256',hashlib.sha256(b'fixture').hexdigest(),'--output',str(tmp_path/'probe')]
    cli.main(base)
    assert json.loads(capsys.readouterr().out)['readback_timing']=='post-reset'
    cli.main([*base,'--readback-timing','pre-reset'])
    assert json.loads(capsys.readouterr().out)['readback_timing']=='pre-reset'
