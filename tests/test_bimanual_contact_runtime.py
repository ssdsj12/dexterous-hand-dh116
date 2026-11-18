"""CPU-only measured contact labels around the unchanged conserved ledger."""
from dataclasses import replace
import importlib
import importlib.util

import pytest
import torch

from go2_pvcnn.control.dh116.tabletop_bimanual_mpc_state import SampleClock

MODULE = 'go2_pvcnn.control.dh116.tabletop_bimanual_contact_runtime'
DT = 1 / 480
SHA = 'a' * 64


def clock(env=0, step=0, generation=0, **changes):
    values = dict(run_uuid='contact-fixture', env_id=env, episode_uid=f'ep{env}-{generation}',
        reset_generation=generation, world_step=step, sdk_step=500 + step,
        world_time_s=step * DT, sdk_time_s=10 + step * DT, policy_step=step // 8,
        terminal=False, reset=False)
    values.update(changes)
    return SampleClock(**values)


def api(styles=('pinch_thumb_index',), **kw):
    assert importlib.util.find_spec(MODULE) is not None, 'new bimanual contact runtime is missing'
    m = importlib.import_module(MODULE)
    r = m.BimanualContactRuntime(len(styles), 'cpu', grasp_styles=styles, **kw)
    r.reset(torch.arange(r.n), clocks=tuple(clock(i, reset=True) for i in range(r.n)), source_sha=SHA)
    return r


def packet(r, step, *, contact=False, generation=None, **changes):
    shape = (r.n, 2, 5)
    loads = torch.zeros(shape, dtype=torch.float64)
    if contact:
        loads[r.required_pad_mask] = .1
    vectors = torch.zeros((*shape, 3), dtype=torch.float64)
    vectors[..., 0] = loads
    generations = [0] * r.n if generation is None else generation
    values = dict(clocks=tuple(clock(e, step, generations[e]) for e in range(r.n)), source_sha=SHA,
        live=torch.ones(r.n, dtype=torch.bool), measurement_basis='actual_object_filtered_source_patch',
        measurement_valid=torch.ones(r.n, dtype=torch.bool), pad_valid=loads > 0,
        loads_n=loads, vectors_w=vectors, depth_m=torch.zeros(shape, dtype=torch.float64),
        rejected_loads_n=torch.zeros(r.n, 2, dtype=torch.float64),
        measured_gap_m=torch.full(shape, .002, dtype=torch.float64), geometry_valid=torch.ones(shape, dtype=torch.bool),
        geometry_alignment_cos=torch.ones(shape, dtype=torch.float64))
    values.update(changes)
    return values


def observe(r, step, **kw):
    return r.observe_substep(**packet(r, step, **kw))


def retry(r, clocks):
    return r.request_retry(clocks=clocks, source_sha=SHA,
        geometry_eligible=torch.ones((r.n, 2, 5), dtype=torch.bool))


def reserve(r, clocks):
    return r.reserve_approach(clocks=clocks, source_sha=SHA,
        requested_approach_m=torch.full((r.n, 2, 5), .001, dtype=torch.float64))


def commit(r, clocks, ticket, identity='execution-one'):
    amount = ticket['allowed_distance_m']
    return r.commit_execution(clocks=clocks, source_sha=SHA,
        reservation_id=ticket['reservation_id'], execution_id=identity,
        physics_step=max(c.world_step for c in clocks) + 1,
        approved_approach_m=amount, executed_mask=amount > 0,
        requested_action24=torch.zeros(r.n, 24), approved_action24=torch.zeros(r.n, 24),
        final_target_rad=torch.zeros(r.n, 12, dtype=torch.float64))


def test_explicit_both_hand_required_masks_keep_pinch_and_tripod_independent():
    r = api(('pinch_thumb_index', 'tripod_thumb_index_middle'))
    assert r.required_pad_mask[0].tolist() == [[True, True, False, False, False]] * 2
    assert r.required_pad_mask[1].tolist() == [[True, True, True, False, False]] * 2
    s = observe(r, 1)
    assert s['rows'][0]['states'][0] == ('NO_CONTACT', 'NO_CONTACT', None, None, None)
    assert s['rows'][1]['states'][1][:3] == ('NO_CONTACT',) * 3
    assert not any(s['qualification'].values())


def test_geometry_is_only_candidate_and_cannot_emit_real_contact_events():
    r = api(); p = packet(r, 1)
    p['measured_gap_m'][:] = .001; p['geometry_alignment_cos'][:] = .8
    s = r.observe_substep(**p)
    assert s['rows'][0]['states'][0][:2] == ('CONTACT_CANDIDATE',) * 2
    assert not s['ledger']['measured_contact'].any()
    assert not any(e['kind'] in ('contact_observed', 'contact_acquired') for e in s['events'])
    assert s['rows'][0]['opposition']['value'] is None


def test_actual_contact_and_ten_samples_recovery_stable_are_not_task_success():
    r = api()
    for step in range(1, 10): s = observe(r, step, contact=True)
    assert s['rows'][0]['states'][0][:2] == ('VALID_CONTACT',) * 2
    s = observe(r, 10, contact=True)
    assert s['rows'][0]['states'][0][:2] == ('STABLE',) * 2
    assert s['ledger']['stable_time_s'][0, 0, 0] >= .02
    assert s['stability_scope'] == 'original_20ms_recovery_pressure_only_not_bilateral_200ms_or_task_hold'
    assert s['rows'][0]['slip']['classification_valid'].sum() == 0
    assert s['rows'][0]['opposition']['valid'] is False
    assert not any(s['qualification'].values())


def test_unknown_measurement_is_na_without_a_fabricated_loss_then_known_absence_is_lost():
    r = api(); observe(r, 1, contact=True)
    health = torch.ones((1, 2, 5), dtype=torch.bool); health[0, 0, 0] = False
    s = observe(r, 2, contact=True, measurement_valid=health)
    assert s['rows'][0]['states'][0][0] is None and not s['rows'][0]['valid'][0, 0]
    assert not any(e['kind'] == 'contact_lost' and e['clock'].world_step == 2 for e in s['events'])
    s = observe(r, 3)
    assert s['rows'][0]['states'][0][0] == 'LOST'
    assert s['rows'][0]['states'][1][:2] == ('LOST',) * 2


def test_slip_requires_explicit_threshold_units_and_measured_source():
    r = api(); p = packet(r, 1, contact=True)
    p.update(slip_speed_m_s=torch.full((1, 2, 5), .003, dtype=torch.float64),
             slip_valid=torch.ones((1, 2, 5), dtype=torch.bool), slip_source='actual_relative_point_speed')
    with pytest.raises(ValueError, match='threshold'):r.observe_substep(**p)
    m = importlib.import_module(MODULE)
    threshold = m.SlipThreshold(.002, 'm/s', 'declared configuration', SHA)
    r = api(slip_threshold=threshold); p = packet(r, 1, contact=True)
    p.update(slip_speed_m_s=torch.full((1, 2, 5), .003, dtype=torch.float64),
             slip_valid=torch.ones((1, 2, 5), dtype=torch.bool), slip_source='actual_relative_point_speed')
    s = r.observe_substep(**p)
    assert s['rows'][0]['states'][0][:2] == ('SLIPPING',) * 2
    assert s['rows'][0]['slip']['threshold']['value'] == .002
    with pytest.raises(ValueError):m.SlipThreshold(.002, 'mm/s', 'declared', SHA)


def test_opposition_requires_actual_contact_provenance_and_both_required_hands():
    r = api(); p = packet(r, 1, contact=True)
    p.update(opposition_confirmed=torch.tensor([True]), opposition_valid=torch.tensor([True]),
             opposition_source='actual_contact_pair_observation', opposition_basis='geometry_only')
    with pytest.raises(ValueError, match='opposition'):r.observe_substep(**p)
    p['opposition_basis'] = 'actual_contact_observation'
    s = r.observe_substep(**p)
    assert s['rows'][0]['opposition']['valid'] is True and s['rows'][0]['opposition']['value'] is True
    p = packet(r, 2); p.update(opposition_confirmed=torch.tensor([True]), opposition_valid=torch.tensor([True]),
        opposition_source='actual_contact_pair_observation', opposition_basis='actual_contact_observation')
    s = r.observe_substep(**p)
    assert s['rows'][0]['opposition']['value'] is None and not s['rows'][0]['opposition']['valid']


@pytest.mark.parametrize('fault', ['episode', 'source', 'sdk_clock', 'world_time', 'measurement_basis', 'repeated', 'policy_jump'])
def test_bad_identity_clock_or_basis_rejects_atomically(fault):
    r = api(('pinch_thumb_index', 'tripod_thumb_index_middle')); observe(r, 1)
    before = r.snapshot()['ledger']; p = packet(r, 2)
    if fault == 'episode':p['clocks'] = (p['clocks'][0], replace(p['clocks'][1], episode_uid='wrong'))
    elif fault == 'source':p['source_sha'] = (SHA, 'b' * 64)
    elif fault == 'sdk_clock':p['clocks'] = tuple(replace(c, sdk_step=c.sdk_step + 1) for c in p['clocks'])
    elif fault == 'world_time':p['clocks'] = tuple(replace(c, world_time_s=c.world_time_s + .01) for c in p['clocks'])
    elif fault == 'measurement_basis':p['measurement_basis'] = 'geometry_only'
    elif fault == 'repeated':p['clocks'] = tuple(clock(i, 1) for i in range(r.n))
    else:p['clocks'] = tuple(replace(c, policy_step=c.policy_step + 5) for c in p['clocks'])
    with pytest.raises(ValueError):r.observe_substep(**p)
    after = r.snapshot()['ledger']
    for name in ('last_physics_step', 'attempt_time_s', 'executed_approach_m', 'stable_time_s'):
        assert torch.equal(before[name], after[name])


def test_budget_reservation_commit_and_contact_flicker_delegate_without_replenishment():
    r = api(); p = packet(r, 1); r.observe_substep(**p)
    retry(r, p['clocks']); assert r.snapshot()['rows'][0]['states'][0][:2] == ('REACQUIRING',) * 2
    ticket = reserve(r, p['clocks']); amount = ticket['allowed_distance_m'].clone()
    assert amount[0, 0, 0] == pytest.approx(.001 / 480)
    assert commit(r, p['clocks'], ticket)['duplicate'] is False
    assert commit(r, p['clocks'], ticket)['duplicate'] is True
    spent = r.snapshot()['ledger']['executed_approach_m'].clone()
    observe(r, 2, contact=True); observe(r, 3)
    after = r.snapshot()['ledger']
    assert torch.equal(after['executed_approach_m'], spent)
    assert after['attempt_time_s'][0, 0, 0] >= DT
    assert after['limits'] == dict(speed_m_s=.001, distance_m=.001, attempt_time_s=1.,
        no_progress_s=.5, progress_resolution_m=1e-6, recovery_stable_s=.02)
    assert torch.equal(after['executed_approach_m'], amount)


def test_selected_real_reset_clears_only_selected_state_source_and_budget():
    r = api(('pinch_thumb_index', 'tripod_thumb_index_middle')); p = packet(r, 1)
    r.observe_substep(**p); retry(r, p['clocks']); commit(r, p['clocks'], reserve(r, p['clocks']))
    before = r.snapshot()['ledger']
    reset_clocks = (clock(0, 1, 1, reset=True), clock(1, 1))
    r.reset(torch.tensor([0]), clocks=reset_clocks, source_sha='b' * 64)
    s = r.snapshot(); after = s['ledger']
    assert after['executed_approach_m'][0].sum() == 0
    assert torch.equal(before['executed_approach_m'][1], after['executed_approach_m'][1])
    assert s['rows'][0]['states'] == ((None,) * 5,) * 2
    assert s['rows'][0]['source_sha'] == 'b' * 64 and s['rows'][1]['source_sha'] == SHA
    with pytest.raises(ValueError):r.reset(torch.tensor([0]), clocks=reset_clocks, source_sha='b' * 64)
    # The old sibling's already committed interval remains owed after reset.
    s = observe(r, 2, generation=[1, 0], source_sha=('b' * 64, SHA))
    assert s['ledger']['attempt_time_s'][1, 0, 0] >= DT
    assert s['ledger']['attempt_time_s'][0].sum() == 0
    assert torch.equal(s['ledger']['executed_approach_m'][1], before['executed_approach_m'][1])


@pytest.mark.parametrize('fault', ['world_frame', 'sdk_frame', 'time_frame', 'policy_frame'])
def test_selected_reset_batch_requires_one_physical_frame_before_any_mutation(fault):
    m = importlib.import_module(MODULE)
    r = m.BimanualContactRuntime(2, 'cpu', grasp_styles=('pinch_thumb_index', 'tripod_thumb_index_middle'))
    first, second = clock(0, reset=True), clock(1, reset=True)
    if fault == 'world_frame':second = clock(1, 1, reset=True)
    elif fault == 'sdk_frame':second = replace(second, sdk_step=second.sdk_step + 1)
    elif fault == 'time_frame':second = replace(second, sdk_time_s=second.sdk_time_s + DT)
    else:second = replace(second, policy_step=1)
    before = r.snapshot()['ledger']
    with pytest.raises(ValueError, match='frame'):
        r.reset(torch.arange(2), clocks=(first, second), source_sha=SHA)
    after = r.snapshot()
    assert all(row['clock'] is None for row in after['rows']) and after['events'] == []
    for key in ('episode_ids', 'last_physics_step', 'executed_approach_m', 'attempt_time_s'):
        assert torch.equal(before[key], after['ledger'][key])


def test_single_selected_reset_does_not_require_unselected_slot_at_the_same_tick():
    r = api(('pinch_thumb_index', 'tripod_thumb_index_middle'));observe(r, 1)
    before = r.snapshot()['rows'][1]
    r.reset(torch.tensor([0]), clocks=(clock(0, 3, 1, reset=True), clock(1, 1)), source_sha='b' * 64)
    after = r.snapshot()['rows'][1]
    assert after['clock'] == before['clock'] and after['states'] == before['states']
    assert torch.equal(after['valid'], before['valid']) and after['source_sha'] == before['source_sha']


def test_first_terminal_is_pinned_and_later_noop_cannot_retag_it():
    r = api(); p = packet(r, 1, contact=True)
    p['clocks'] = (replace(p['clocks'][0], terminal=True),)
    first = r.observe_substep(**p)['terminal_samples'][0]
    assert first['clock'].terminal and first['states'][0][0] == 'VALID_CONTACT'
    p = packet(r, 2, generation=[1]); p['source_sha'] = 'b' * 64
    s = r.observe_substep(**p)
    assert s['terminal_samples'][0]['clock'] == first['clock']
    assert s['rows'][0]['clock'] == first['clock']
    assert s['ledger']['last_physics_step'].tolist() == [1]


def test_first_terminal_requires_actual_live_sample_without_mutating_prior_state():
    r = api(); observe(r, 1)
    p = packet(r, 2, live=torch.tensor([False]));p['clocks'] = (replace(p['clocks'][0], terminal=True),)
    with pytest.raises(ValueError, match='terminal'):r.observe_substep(**p)
    assert r.snapshot()['rows'][0]['clock'].world_step == 1
    assert not r.snapshot()['terminal_samples']


def test_known_contact_loss_stays_lost_when_geometry_is_still_a_candidate():
    r = api(); observe(r, 1, contact=True)
    p = packet(r, 2);p['measured_gap_m'][:] = .0005
    s = r.observe_substep(**p)
    assert s['rows'][0]['geometry_candidate'][0, 0]
    assert s['rows'][0]['states'][0][:2] == ('LOST',) * 2


def test_configuration_cannot_retag_already_observed_states():
    r = api()
    with pytest.raises(AttributeError):r.grasp_styles = ('tripod_thumb_index_middle',)
    with pytest.raises(AttributeError):r.slip_threshold = object()


def test_original_half_second_watchdog_and_one_second_budget_survive_retry_without_replenishment():
    r = api(); p = packet(r, 1);r.observe_substep(**p);retry(r, p['clocks'])
    for step in range(2, 242):s = observe(r, step)
    assert s['ledger']['attempt_time_s'][0, 0, 0] == pytest.approx(.5)
    assert s['ledger']['round_stopped'][0, 0, 0]
    refused = retry(r, packet(r, 241)['clocks'])
    assert not refused['retry_allowed'].any()
    assert refused['remaining_distance_m'][0, 0, 0] == .001
    p = packet(r, 242);p['measured_gap_m'][:] -= .000002
    r.observe_substep(**p);opened = retry(r, p['clocks'])
    assert opened['retry_allowed'][0, 0, 0] and opened['remaining_time_s'][0, 0, 0] == pytest.approx(.5)
    for step in range(243, 483):
        p = packet(r, step);p['measured_gap_m'][:] -= .000002;s = r.observe_substep(**p)
    assert s['ledger']['attempt_time_s'][0, 0, 0] == pytest.approx(1.)
    assert s['ledger']['time_exhausted'][0, 0, 0]
    assert s['ledger']['executed_approach_m'].sum() == 0
    assert not retry(r, p['clocks'])['retry_allowed'].any()


def test_recovery_and_other_hand_known_loss_is_concurrent_diagnostic_not_causation():
    r = api(); p = packet(r, 1, contact=True)
    p['loads_n'][0, 1] = 0.; p['vectors_w'][0, 1] = 0.; p['pad_valid'][0, 1] = False
    r.observe_substep(**p); retry(r, p['clocks'])
    s = observe(r, 2)
    rows = [e for e in s['events'] if e['kind'] == 'concurrent_coordination_failure']
    assert rows and rows[0]['recovering_hand'] == 'right' and rows[0]['lost_hand'] == 'left'
    assert rows[0]['causal_evidence'] is False and rows[0]['clock'].world_step == 2
    assert not any(s['qualification'].values())


def test_outputs_are_owned_and_context_bound_without_actor_or_target_methods():
    r = api(); s = observe(r, 1, contact=True)
    s['ledger']['executed_approach_m'][:] = 99.
    s['rows'][0]['valid'][:] = False
    assert r.snapshot()['ledger']['executed_approach_m'].sum() == 0
    assert r.snapshot()['rows'][0]['valid'][0, 0]
    assert all(e['clock'].run_uuid == 'contact-fixture' and e['source_sha'] == SHA for e in r.snapshot()['events'])
    assert not hasattr(r, 'write_target') and not hasattr(r, 'actor_observation')
