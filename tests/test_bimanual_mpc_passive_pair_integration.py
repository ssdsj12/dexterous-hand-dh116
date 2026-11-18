"""CPU-only taps preserve the original Scene path and getter sequence."""
from dataclasses import asdict

from test_bimanual_mpc_passive_pair_values import fixture
from go2_pvcnn.control.dh116 import tabletop_bimanual_mpc_scene_measurements as S


def provider(f,observer=None):
    return S.SceneMeasurementProvider(scene=f.scene,rigid_body_view=f.rigid,
        clock_reader=lambda e:f.clocks[e],contact_views=f.bundle,enabled=True,
        passive_pair_observer=observer)


def test_scene_opt_in_taps_each_existing_numeric_getter_once_without_changing_fields():
    old=fixture();base=provider(old).capture_batch(clocks=old.clocks)
    f=fixture();p=provider(f,f.observer);actual=p.capture_batch(clocks=f.clocks)
    snapshot=p.last_passive_pair_snapshot
    assert snapshot.capture_valid,snapshot.errors
    assert len(snapshot.pairs)==6 and len(f.sim.created)==3
    assert tuple(dict(x.fields) for x in actual)==tuple(dict(x.fields) for x in base)
    assert all(not x.fields['support.table_reaction'].valid for x in actual)
    assert all(v.calls==[('matrix',1/480),('normal',1/480),('friction',1/480)] for v in f.bundle.views.values())
    assert all(tuple(r['before'])==tuple(asdict(c) for c in f.clocks) for r in snapshot.raw['read_brackets'])


def test_default_off_preserves_original_scene_raw_and_numeric_call_order():
    f=fixture();a=provider(f).capture_batch(clocks=f.clocks)
    g=fixture();b=provider(g).capture_batch(clocks=g.clocks)
    assert tuple(dict(s.fields) for s in a)==tuple(dict(s.fields) for s in b)
    assert [v.calls for v in f.bundle.views.values()]==[v.calls for v in g.bundle.views.values()]
    assert all('metadata_after' not in [r['name'] for r in s.raw['read_brackets']] for s in a)


def test_scene_does_not_return_previous_passive_snapshot_after_begin_failure():
    f=fixture();p=provider(f,f.observer);p.capture_batch(clocks=f.clocks)
    assert p.last_passive_pair_snapshot is not None
    f.source['stage_identity']='new-stage'
    try:p.capture_batch(clocks=f.clocks)
    except ValueError:pass
    else:raise AssertionError('source drift accepted')
    assert p.last_passive_pair_snapshot is None
