"""CPU fixtures validate contact ledger arithmetic and source ownership only."""
import pytest

from go2_pvcnn.control.dh116 import tabletop_stereo_contact_calibration as audit


OBJECT = '/World/Object'
PARTNER = '/World/Finger'
FINGER_COLLIDER = PARTNER + '/collision'
INVENTORY = dict(object_collider=OBJECT, finger_collider=FINGER_COLLIDER)


def header(body_a=OBJECT, body_b=PARTNER, collider_a=OBJECT, collider_b=FINGER_COLLIDER, **changes):
    result = dict(actor0=body_a, actor1=body_b, collider0=collider_a, collider1=collider_b,
                  stage_id=7, contact_data_offset=0, num_contact_data=1,
                  friction_anchors_offset=0, num_friction_anchors_data=0)
    result.update(changes)
    return result


def row(headers=(), points=(), anchors=(), *, step=1):
    dt = 1 / 480
    return dict(physics_step=step, sdk_step=step + 2, timestamp=(step+2)*dt, dt=dt,
                velocity_before=[0., 0., 0.], velocity_after=[.002, 0., 0.],
                mass_kg=1., gravity_m_s2=[0., 0., 0.],
                tensor=dict(dt_scaling_max_error_n=0., normal=dict(force_n=[], normal=[]),
                            friction=dict(force_n=[])),
                full_report=dict(headers=list(headers), contacts=list(points),
                                 friction_anchors=list(anchors)))


def point(sign=1):
    return dict(impulse=[sign*.002, 0., 0.], normal=[sign*1., 0., 0.],
                position=[0., 0., 0.], separation=0.)


def summarize(rows):
    return audit.summarize_pair_calibration(rows, object_path=OBJECT,
                                           expected_partner=PARTNER, inventory=INVENTORY)


def test_wrong_collider_never_contributes_to_residuals():
    report = summarize([row([header(collider_b=PARTNER+'/wrong')], [point()])])
    assert report['nonzero_report_steps'] == 0
    assert report['report_impulse_hypotheses'] is None
    strict = report['polled_source_audit']
    assert strict['identity_matched_header_count'] == 0
    assert strict['object_unclassified_contact_header_count'] == 1
    assert strict['external_support_free'] is None


def test_mixed_body_orders_preserve_separate_unsigned_hypotheses():
    forward = row([header()], [point()])
    reverse = row([header(PARTNER, OBJECT, FINGER_COLLIDER, OBJECT)], [point(-1)], step=2)
    report = summarize([forward, reverse])
    strict = report['polled_source_audit']
    assert report['report_impulse_hypotheses'] is None
    assert strict['identity_matched_header_count'] == 2
    assert strict['sign_hypotheses_by_actor_order']['object_actor0']['contact_only']['impulse_on_actor0']['max'] < 1e-12
    assert strict['sign_hypotheses_by_actor_order']['object_actor1']['contact_only']['impulse_on_actor0']['max'] < 1e-12
    steps = strict['steps']
    assert steps[0]['pairs'][0]['body_a'] == dict(actor=OBJECT, collider=OBJECT)
    assert steps[0]['pairs'][0]['force_on_object_world_n_if_raw_on_actor0']['contact_only'] == pytest.approx([.96, 0., 0.])
    assert steps[1]['pairs'][0]['force_on_object_world_n_if_raw_on_actor0']['contact_only'] == pytest.approx([.96, 0., 0.])
    assert all(pair['force_on_object_verified_world_n'] is None for step in steps for pair in step['pairs'])
    assert report['R1_qualified'] is False and report['R2_qualified'] is False


@pytest.mark.parametrize('bad,error', [
    (row([header(), header()], [point()]), 'overlap'),
    (row([header()], [point(), point()]), 'unowned'),
    (row([header(contact_data_offset=1)], [point()]), 'outside'),
    (row([header(num_contact_data=1.5)], [point()]), 'integer'),
])
def test_raw_polled_ownership_rejects_invalid_ranges(bad, error):
    with pytest.raises(ValueError, match=error):
        summarize([bad])


def test_other_object_partner_prevents_momentum_attribution_but_is_not_support_proof():
    table = '/World/Table'
    step = row([header(), header(body_b=table, collider_b=table, contact_data_offset=1)],
               [point(), point()])
    report = summarize([step])
    strict = report['polled_source_audit']
    assert report['report_impulse_hypotheses'] is None
    assert strict['identity_matched_header_count'] == 1
    assert strict['object_unclassified_contact_header_count'] == 1
    assert strict['steps'][0]['unclassified_object_partners'] == [table]
    assert strict['sign_hypotheses_by_actor_order'] == {}
    assert strict['external_support_free'] is None and strict['attachment_free'] is None


def test_unknown_inventory_does_not_turn_legacy_or_empty_poll_into_zero_force():
    legacy = row()
    del legacy['full_report']
    legacy['pair_reports'] = []
    report = summarize([legacy, row(step=2)])['polled_source_audit']
    assert report['legacy_flattened_steps'] == 1
    assert report['empty_structured_poll_steps'] == 1
    assert report['identity_matched_header_count'] == 0
    assert report['steps'] == []
    assert report['force_on_object_verified_world_n'] is None


def test_file_audit_binds_exact_inputs_and_never_rewrites_archive(tmp_path):
    import hashlib
    import json
    ledger = tmp_path/'raw.jsonl'
    inventory = tmp_path/'inventory.json'
    stage = tmp_path/'calibration.usda'
    stage.write_bytes(b'#usda 1.0\n')
    inventory.write_text(json.dumps(dict(INVENTORY, stage_sha256=hashlib.sha256(stage.read_bytes()).hexdigest())))
    ledger.write_text(json.dumps(row())+'\n')
    sha = hashlib.sha256(ledger.read_bytes()).hexdigest()
    inventory_sha = hashlib.sha256(inventory.read_bytes()).hexdigest()
    evidence = audit.audit_archived_contact_sources(ledger, inventory, stage,
        raw_sha256=sha, inventory_sha256=inventory_sha, object_path=OBJECT, expected_partner=PARTNER)
    assert evidence['source_artifact']['raw_sha256'] == sha
    assert evidence['source_artifact']['inventory_sha256'] == inventory_sha
    assert evidence['summary']['polled_source_audit']['empty_structured_poll_steps'] == 1
    with pytest.raises(ValueError, match='SHA256'):
        audit.audit_archived_contact_sources(ledger, inventory, stage, raw_sha256='0'*64,
            inventory_sha256=inventory_sha, object_path=OBJECT, expected_partner=PARTNER)
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == sha
