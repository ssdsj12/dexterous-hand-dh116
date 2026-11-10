"""CPU contract for measured contact recovery; no simulator is imported."""
import importlib
import importlib.util

import pytest
import torch


MODULE = 'go2_pvcnn.control.dh116.tabletop_tripod_contact_recovery'
DT = 1 / 480
SHA = 'a' * 64


def api(n=1):
    assert importlib.util.find_spec(MODULE) is not None, 'measured tripod recovery module is missing'
    m = importlib.import_module(MODULE)
    f = m.TripodContactRecovery(n, 'cpu')
    f.reset(torch.arange(n), episode_ids=torch.arange(n) + 10, source_sha=SHA)
    return f


def packet(f, contact=False, gap=.001, **changes):
    shape = (f.n, 2, 5)
    loads = torch.zeros(shape, dtype=torch.float64)
    if contact: loads[:, 1, :3] = .1
    vectors = torch.zeros((*shape, 3), dtype=torch.float64)
    vectors[..., 0] = loads
    p = dict(episode_ids=f.episode_ids.clone(), physics_step=int(f.last_physics_step.max()) + 1,
        dt_s=DT, live=torch.ones(f.n, dtype=torch.bool), terminal=torch.zeros(f.n, dtype=torch.bool),
        measurement_valid=torch.ones(f.n, dtype=torch.bool), pad_valid=loads > 0,
        loads_n=loads, vectors_w=vectors, depth_m=torch.zeros(shape, dtype=torch.float64),
        rejected_loads_n=torch.zeros(f.n, 2, dtype=torch.float64),
        measured_gap_m=torch.full(shape, gap, dtype=torch.float64),
        geometry_valid=torch.ones(shape, dtype=torch.bool))
    p.update(changes)
    return p


def observe(f, **kw):
    f.observe_substep(**packet(f, **kw))
    return f.snapshot()


def retry(f, eligible=True, reason='none'):
    shape = (f.n, 2, 5)
    return f.request_retry(episode_ids=f.episode_ids.clone(), control_tick=0,
        observation_revision=int(f.last_physics_step.max()),
        geometry_eligible=torch.full(shape, eligible, dtype=torch.bool), blocked_reason=reason)


def reserve(f, distance=.000001, dt_s=DT):
    return f.reserve_approach(episode_ids=f.episode_ids.clone(), control_tick=0, dt_s=dt_s,
        requested_approach_m=torch.full((f.n, 2, 5), distance, dtype=torch.float64))


def commit(f, reservation, execution_id='execution-0', amount=None, **changes):
    amount = reservation['allowed_distance_m'] if amount is None else amount
    kw = dict(episode_ids=f.episode_ids.clone(), reservation_id=reservation['reservation_id'],
        execution_id=execution_id, physics_step=int(f.last_physics_step.max()) + 1,
        approved_approach_m=amount.clone(), executed_mask=amount > 0,
        requested_action24=torch.zeros(f.n, 24), final_target_rad=torch.zeros(f.n, 12, dtype=torch.float64),
        approved_action24=torch.zeros(f.n, 24))
    kw.update(changes)
    return f.commit_execution(**kw)


def test_real_contact_nine_samples_do_not_meet_twenty_ms_ten_do():
    f = api()
    for _ in range(9): s = observe(f, contact=True)
    assert (s['state'][0, 1, :3] == f.CONTACT).all()
    assert not s['stable'][0, 1, :3].any()
    s = observe(f, contact=True)
    assert (s['state'][0, 1, :3] == f.STABLE).all()
    assert s['stable'][0, 1, :3].all()
    assert s['stable_time_s'][0, 1, 0] >= .02


def test_geometry_cannot_manufacture_contact_or_mutate_measurement():
    f = api(); p = packet(f, gap=0.)
    before = {k: v.clone() for k, v in p.items() if isinstance(v, torch.Tensor)}
    f.observe_substep(**p); s = f.snapshot()
    assert not s['measured_contact'].any()
    assert (s['state'][0, 1, :3] == f.LOST).all()
    for k, v in before.items(): assert torch.equal(v, p[k])


@pytest.mark.parametrize('bad', ['normal', 'depth', 'rejected'])
def test_raw_contact_does_not_become_stable_without_original_control_gates(bad):
    f = api()
    for _ in range(12):
        p = packet(f, contact=True)
        if bad == 'normal': p['vectors_w'][:, 1] = 0
        elif bad == 'depth': p['depth_m'][:, 1, 0] = .0007
        else: p['rejected_loads_n'][:, 1] = .1
        f.observe_substep(**p)
    s = f.snapshot()
    assert s['measured_contact'][0, 1, :3].all()
    assert not s['stable'][0, 1, 0]


def test_independent_required_digits_unknown_is_not_a_measured_loss_event():
    f = api()
    for _ in range(10): observe(f, contact=True)
    p = packet(f, contact=True); p['loads_n'][0, 1, 1] = 0; p['pad_valid'][0, 1, 1] = False
    f.observe_substep(**p); s = f.snapshot()
    assert s['state'][0, 1].tolist()[:3] == [f.STABLE, f.LOST, f.STABLE]
    before = sum(e['kind'] == 'contact_lost' for e in s['events'])
    observe(f, contact=True, measurement_valid=torch.zeros(1, dtype=torch.bool))
    s = f.snapshot()
    assert not s['contact_known'].any()
    assert sum(e['kind'] == 'contact_lost' for e in s['events']) == before


def test_transient_model_failure_is_retryable_without_budget_replenishment():
    f = api(); observe(f); retry(f)
    r = reserve(f); commit(f, r)
    retry(f, False, 'model_infeasible')
    s = observe(f)
    assert (s['state'][0, 1, :3] == f.REACQUIRING).all()
    assert s['attempt_time_s'][0, 1, 0] == pytest.approx(DT)
    spent = s['executed_approach_m'].clone()
    retry(f)
    assert reserve(f)['allowed_distance_m'][0, 1, 0] > 0
    assert torch.equal(f.snapshot()['executed_approach_m'], spent)


def test_candidates_and_reservations_do_not_spend_and_denied_execution_spends_zero():
    f = api(); observe(f); retry(f)
    r = reserve(f)
    assert not f.snapshot()['executed_approach_m'].any()
    commit(f, r, amount=torch.zeros_like(r['allowed_distance_m']))
    assert not f.snapshot()['executed_approach_m'].any()


def test_commit_is_guard_approved_distance_not_object_progress_and_is_idempotent():
    f = api(); observe(f); retry(f)
    r = reserve(f); result = commit(f, r)
    first = f.snapshot()
    again = commit(f, r)
    assert again['duplicate'] is True and result['duplicate'] is False
    assert torch.equal(first['executed_approach_m'], f.snapshot()['executed_approach_m'])
    observe(f, gap=.0005)
    assert torch.equal(first['executed_approach_m'], f.snapshot()['executed_approach_m'])
    e = next(e for e in first['events'] if e['kind'] == 'execution_committed')
    assert e['execution_id'] == 'execution-0' and e['source_sha'] == SHA
    assert torch.equal(e['requested_action24'], torch.zeros(1, 24))
    assert e['episode_ids'].tolist() == [10]


def test_overcommit_and_conflicting_duplicate_reject_atomically():
    f = api(); observe(f); retry(f); r = reserve(f)
    with pytest.raises(ValueError, match='reservation'):
        commit(f, r, amount=r['allowed_distance_m'] + .001)
    assert not f.snapshot()['executed_approach_m'].any()
    commit(f, r)
    with pytest.raises(ValueError, match='duplicate'):
        commit(f, r, approved_action24=torch.ones(1, 24))


def test_outstanding_reservations_cannot_oversubscribe_remaining_distance():
    f = api(); observe(f); retry(f)
    rs = [reserve(f, .001) for _ in range(3)]
    total = sum(r['allowed_distance_m'] for r in rs)
    assert total.max() <= .001 * DT + 1e-15
    for r in rs: f.cancel_reservation(r['reservation_id'])
    for i in range(480):
        r = reserve(f, .001); commit(f, r, execution_id=f'execution-{i}')
        observe(f, gap=.001 - (i+1)*.0000015)
    s = f.snapshot()
    assert s['distance_exhausted'][0, 1, :3].all()
    assert s['attempt_time_s'][0, 1, 0] == pytest.approx(1., abs=1e-14)
    assert s['time_exhausted'][0, 1, :3].all()
    retry(f)
    assert not reserve(f)['allowed_distance_m'].any()


def test_reservation_speed_cap_and_cancellation_release_without_spending():
    f = api(); observe(f); retry(f); r = reserve(f, .001)
    assert r['allowed_distance_m'].max() == pytest.approx(.001 * DT)
    f.cancel_reservation(r['reservation_id'])
    assert not f.snapshot()['reserved_approach_m'].any()
    assert not f.snapshot()['executed_approach_m'].any()


def test_expired_reservation_and_future_execution_are_rejected_without_charging():
    f = api(); observe(f); retry(f); r = reserve(f)
    with pytest.raises(ValueError, match='step'):
        commit(f, r, physics_step=999)
    observe(f)
    with pytest.raises(ValueError, match='reservation'):
        commit(f, r)
    assert not f.snapshot()['executed_approach_m'].any()


def test_same_physics_step_cannot_commit_another_execution_identity():
    f = api(); observe(f); retry(f); r = reserve(f); commit(f, r)
    second = reserve(f)
    with pytest.raises(ValueError, match='step'):
        commit(f, second, execution_id='second', amount=torch.zeros_like(second['allowed_distance_m']))


def test_retry_geometry_signature_requires_real_changed_measurement_not_new_revision():
    f = api(); signature = torch.zeros(1, 2, 5, 6, dtype=torch.float64)
    observe(f, measured_geometry_signature=signature); retry(f)
    for _ in range(240): observe(f, measured_geometry_signature=signature)
    retry(f)
    assert not f.snapshot()['round_active'].any()
    signature[..., 3] = .1
    observe(f, measured_geometry_signature=signature); retry(f)
    assert f.snapshot()['round_active'][0, 1, :3].all()


def test_no_contact_pad_valid_false_remains_known_but_sensor_fault_is_unknown():
    f = api(2)
    health = torch.ones(2, 2, 5, dtype=torch.bool); health[1, 1, 0] = False
    observe(f, sensor_healthy=health)
    s = f.snapshot()
    assert s['contact_known'][0, 1, :3].all()
    assert s['contact_known'][1, 1, 1:3].all()
    assert not s['contact_known'][1, 1, 0]
    assert not s['measured_contact'].any()


def test_contact_flicker_preserves_distance_time_and_watchdog():
    f = api(); observe(f); retry(f); r = reserve(f); commit(f, r)
    for _ in range(10): observe(f)
    before = f.snapshot()
    observe(f, contact=True)
    mid = f.snapshot()
    observe(f)
    after = f.snapshot()
    assert after['executed_approach_m'][0, 1, 0] == before['executed_approach_m'][0, 1, 0]
    assert after['attempt_time_s'][0, 1, 0] >= before['attempt_time_s'][0, 1, 0]
    assert after['no_progress_s'][0, 1, 0] >= before['no_progress_s'][0, 1, 0]
    assert mid['state'][0, 1, 0] == f.CONTACT


def test_stable_contact_pauses_acquisition_time_and_later_loss_keeps_budget():
    f = api(); observe(f); retry(f); observe(f)
    for _ in range(10): observe(f, contact=True)
    before = f.snapshot()
    for _ in range(15): observe(f, contact=True)
    assert torch.equal(f.snapshot()['attempt_time_s'], before['attempt_time_s'])
    observe(f)
    assert f.snapshot()['state'][0, 1, 0] == f.LOST
    retry(f)
    assert torch.equal(f.snapshot()['attempt_time_s'], before['attempt_time_s'])


def test_half_second_no_real_progress_stops_round_and_tick_alone_cannot_restart():
    f = api(); observe(f); retry(f)
    for _ in range(240): observe(f)
    s = f.snapshot()
    assert s['round_stopped'][0, 1, :3].all()
    assert not s['round_active'][0, 1, :3].any()
    before = s['attempt_time_s'].clone()
    for _ in range(3): observe(f); retry(f)
    assert not f.snapshot()['round_active'].any()
    assert torch.equal(before, f.snapshot()['attempt_time_s'])
    observe(f, gap=.00099); retry(f)
    assert f.snapshot()['round_active'][0, 1, :3].all()
    assert torch.equal(before, f.snapshot()['attempt_time_s'])


def test_real_measured_progress_avoids_watchdog_but_one_second_locks_until_reset():
    f = api(); observe(f, gap=.002); retry(f)
    for i in range(480): observe(f, gap=.002 - (i + 1) * .000002)
    s = f.snapshot()
    assert s['time_exhausted'][0, 1, :3].all()
    assert s['attempt_time_s'].max() == pytest.approx(1.)
    assert not s['executed_approach_m'].any()
    observe(f, gap=.0008); retry(f)
    assert not reserve(f)['allowed_distance_m'].any()


def test_invalid_geometry_never_counts_as_progress():
    f = api(); observe(f); retry(f)
    for _ in range(240):
        observe(f, gap=0., geometry_valid=torch.zeros(1, 2, 5, dtype=torch.bool))
    assert f.snapshot()['round_stopped'][0, 1, 0]


def test_corrupt_sensor_cannot_clear_progress_watchdog_using_finite_geometry():
    f = api(); observe(f); retry(f)
    for i in range(240):
        p = packet(f, gap=.001 - (i+1)*.000002)
        p['loads_n'][0, 1, :3] = float('nan')
        f.observe_substep(**p)
    s = f.snapshot()
    assert not s['contact_known'][0, 1, :3].any()
    assert s['round_stopped'][0, 1, :3].all()


def test_disabled_environment_and_pads_do_not_accumulate_recovery_state():
    module = importlib.import_module(MODULE)
    mask = torch.zeros(2, 2, 5, dtype=torch.bool); mask[1, 1, :3] = True
    f = module.TripodContactRecovery(2, 'cpu', enabled_pad_mask=mask)
    f.reset(torch.arange(2), episode_ids=torch.tensor([10, 11]), source_sha=SHA)
    observe(f); retry(f); r = reserve(f); commit(f, r)
    s = f.snapshot()
    assert not s['round_active'][0].any()
    assert not s['executed_approach_m'][0].any()
    assert not any(e['env'] == 0 for e in s['events'])


@pytest.mark.parametrize('bad', ['time', 'action', 'target', 'mask'])
def test_bad_execution_packet_is_atomic(bad):
    f = api(); observe(f); retry(f); r = reserve(f)
    changes = {}
    if bad == 'time': changes['physics_step'] = -1
    elif bad == 'action': changes['approved_action24'] = torch.full((1, 24), 1.01)
    elif bad == 'target': changes['final_target_rad'] = torch.full((1, 12), float('nan'))
    else: changes['executed_mask'] = torch.zeros(1, 2, 5, dtype=torch.bool)
    before = f.snapshot()
    with pytest.raises(ValueError): commit(f, r, **changes)
    after = f.snapshot()
    assert torch.equal(before['executed_approach_m'], after['executed_approach_m'])
    assert torch.equal(before['reserved_approach_m'], after['reserved_approach_m'])


def test_reset_clears_only_selected_episode_and_rejects_stale_execution():
    f = api(2); observe(f); retry(f); r = reserve(f); commit(f, r)
    old_ids = f.episode_ids.clone(); before = f.snapshot()
    f.reset(torch.tensor([0]), episode_ids=torch.tensor([99]), source_sha='b' * 64)
    s = f.snapshot()
    assert not s['executed_approach_m'][0].any()
    assert torch.equal(s['executed_approach_m'][1], before['executed_approach_m'][1])
    assert not s['reserved_approach_m'][0].any()
    assert s['last_physics_step'][0] == -1
    with pytest.raises(ValueError, match='episode'):
        commit(f, r, episode_ids=old_ids, execution_id='stale')


@pytest.mark.parametrize('replayed', [10, 11])
def test_physical_episode_identity_must_advance_without_reset_replay(replayed):
    f = api(); observe(f); retry(f); commit(f, reserve(f))
    f.reset(torch.tensor([0]), episode_ids=torch.tensor([11]), source_sha=SHA)
    observe(f); retry(f); commit(f, reserve(f), execution_id='next-episode')
    before = f.snapshot()
    with pytest.raises(ValueError, match='episode'):
        f.reset(torch.tensor([0]), episode_ids=torch.tensor([replayed]), source_sha=SHA)
    after = f.snapshot()
    for name in ('episode_ids', 'last_execution_step', 'executed_approach_m', 'attempt_time_s'):
        assert torch.equal(before[name], after[name])


def test_nonmonotonic_partial_reset_rejects_whole_packet_atomically():
    f = api(2); observe(f); retry(f); commit(f, reserve(f))
    before = f.snapshot()
    with pytest.raises(ValueError, match='episode'):
        f.reset(torch.tensor([0, 1]), episode_ids=torch.tensor([9, 12]), source_sha=SHA)
    after = f.snapshot()
    for name in ('episode_ids', 'last_execution_step', 'executed_approach_m', 'attempt_time_s'):
        assert torch.equal(before[name], after[name])


def test_terminal_cannot_restart_or_accept_late_execution():
    f = api(); observe(f); retry(f); r = reserve(f)
    observe(f, terminal=torch.ones(1, dtype=torch.bool), contact=True)
    retry(f)
    assert not f.snapshot()['round_active'].any()
    assert not f.snapshot()['stable'].any()
    with pytest.raises(ValueError, match='terminal'):
        commit(f, r)


def test_snapshots_events_and_reservations_own_their_tensors():
    f = api(); observe(f); retry(f); r = reserve(f)
    original = r['allowed_distance_m'].clone()
    r['allowed_distance_m'].zero_()
    r['allowed_distance_m'] = original
    commit(f, r)
    s = f.snapshot(); s['executed_approach_m'].zero_(); s['events'][-1]['approved_approach_m'].zero_()
    assert f.snapshot()['executed_approach_m'][0, 1, 0] > 0
    assert f.snapshot()['events'][-1]['approved_approach_m'][0, 1, 0] > 0


@pytest.mark.parametrize('bad', ['nan', 'shape', 'step', 'episode'])
def test_invalid_packets_fail_before_any_mutation(bad):
    f = api(); observe(f); before = f.snapshot(); p = packet(f)
    if bad == 'nan': p['loads_n'][0, 1, 0] = float('nan')
    elif bad == 'shape': p['pad_valid'] = torch.ones(1, 2, 4, dtype=torch.bool)
    elif bad == 'step': p['physics_step'] -= 1
    else: p['episode_ids'] += 1
    if bad == 'nan':
        f.observe_substep(**p)
        assert not f.snapshot()['contact_known'][0, 1, 0]
    else:
        with pytest.raises(ValueError): f.observe_substep(**p)
        assert torch.equal(f.snapshot()['attempt_time_s'], before['attempt_time_s'])
        assert torch.equal(f.snapshot()['last_physics_step'], before['last_physics_step'])
