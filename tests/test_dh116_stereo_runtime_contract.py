"""Independent stereo pipeline identity/output guards; CPU only."""
import importlib
from pathlib import Path

import pytest


def module():
    return importlib.import_module('go2_pvcnn.control.dh116.tabletop_stereo_contract')


def identity(**changes):
    value = dict(run_uuid='run-a', env_id=0, episode_uid='run-a:0:0', reset_id=0,
        policy_tick=0, physics_tick_start=0, physics_tick_end=0,
        simulation_timestamp=0., frame_id=1, calibration_sha256='a'*64)
    return dict(value, **changes)


def test_canonical_actions_and_frozen_state_are_shared():
    m = module()
    assert m.action_sha256() == '498cebedfe520cc2f6593b32776581c3da826ba95e4db441b7938d29176c55d9'
    assert len(m.state_contract()['dimensions']) == 371
    c = m.make_contract('teacher', {'reward_profile': 'teacher_r0_existing'})
    assert c['privileged_size'] == 371 and c['action_size'] == 24
    assert m.validate_contract(c, kind='teacher') == c
    with pytest.raises(ValueError, match='action'):
        m.validate_contract(dict(c, action_contract_sha256='f'*64))
    with pytest.raises(ValueError, match='kind'):
        m.validate_contract(c, kind='student')


@pytest.mark.parametrize('changes', [dict(episode_uid='other'), dict(env_id=1),
    dict(reset_id=1), dict(policy_tick=1), dict(simulation_timestamp=.1),
    dict(frame_id=2), dict(calibration_sha256='b'*64)])
def test_exact_pre_action_alignment_rejects_every_identity_mismatch(changes):
    with pytest.raises(ValueError, match='aligned'):
        module().require_aligned(identity(), identity(**changes))


def test_invalid_time_and_boolean_integer_rejected():
    for changes in (dict(env_id=True), dict(simulation_timestamp=float('nan')),
                    dict(physics_tick_start=9, physics_tick_end=8)):
        with pytest.raises(ValueError):
            module().validate_identity(identity(**changes))


def test_exclusive_output_and_json_never_overwrite(tmp_path):
    m = module()
    run = m.new_run(tmp_path / 'new')
    m.write_json_new(run / 'contract.json', {'valid': True})
    before = (run / 'contract.json').read_bytes()
    with pytest.raises(FileExistsError):
        m.new_run(run)
    with pytest.raises(FileExistsError):
        m.write_json_new(run / 'contract.json', {'valid': False})
    assert before == (run / 'contract.json').read_bytes()
    with pytest.raises(ValueError):
        m.write_json_new(run / 'invalid.json', {'value': float('nan')})
    assert not (run / 'invalid.json').exists()


def test_episode_uid_is_unique_even_for_identical_reset_indices():
    m = module()
    a = m.EpisodeIdentities(2, run_uuid='a')
    b = m.EpisodeIdentities(2, run_uuid='b')
    assert a.uid(0) != a.uid(1) != b.uid(1)
    prior = a.uid(0)
    a.reset([0])
    assert a.uid(0) != prior and a.reset_ids == [1, 0]
    with pytest.raises(ValueError):
        a.reset([0, 0])
