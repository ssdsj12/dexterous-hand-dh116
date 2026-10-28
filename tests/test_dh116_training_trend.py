from test_dh116_training_observer import health


def rows(n):
    return [dict(iteration=i+1,vertical_contact_force={'hands_total_mean_n':-.5},
                 load_ready_env_fraction=.1,max_object_bottom_m=.3,successes=0) for i in range(n)]


def test_baseline_plus_three_complete_windows_required():
    check=health('training_trend')
    assert check(rows(49))['status']=='collecting_baseline'
    assert check(rows(50))['no_improvement_windows']==0
    assert check(rows(199))['diagnosis_required'] is False
    r=check(rows(200));assert r['diagnosis_required'] and r['no_improvement_windows']==3
    assert r['status']=='diagnosis_required' and r['last_complete_iteration']==200


def test_real_improvement_resets_streak_but_reward_alone_does_not():
    check=health('training_trend');data=rows(200)
    for x in data[150:]:
        x['mean_step_reward']=1000.;x['reward_terms']={'force_progress':1000.}
    assert check(data)['diagnosis_required']
    for x in data[150:]:x['vertical_contact_force']['hands_total_mean_n']=-.45
    r=check(data);assert not r['diagnosis_required'] and r['no_improvement_windows']==0
    assert r['windows'][-1]['improved_metrics']==['signed_force_mean_n']


def test_height_jitter_ignored_and_stability_proxy_or_real_height_count():
    check=health('training_trend');data=rows(200)
    for x in data[150:]:x['max_object_bottom_m']=.30002
    assert check(data)['diagnosis_required']
    for x in data[150:]:x['load_ready_env_fraction']=.11
    assert not check(data)['diagnosis_required']
    data=rows(200);data[-1]['max_object_bottom_m']=.303
    assert not check(data)['diagnosis_required']


def test_missing_nonfinite_or_discontinuous_data_cannot_claim_stagnation():
    check=health('training_trend');data=rows(200);data[10]['iteration']=99
    assert check(data)['status']=='invalid_data'
    data=rows(200);data[10]['vertical_contact_force']['hands_total_mean_n']=float('nan')
    assert check(data)['status']=='invalid_data'
    data=rows(200);del data[10]['load_ready_env_fraction']
    assert check(data)['status']=='invalid_data'


def test_resume_uses_current_run_windows_and_new_success_counts_as_progress():
    check=health('training_trend');data=rows(200)
    for x in data:x['iteration']+=403
    data[-1]['successes']=1
    r=check(data);assert not r['diagnosis_required'] and r['last_complete_iteration']==603
    assert 'successful_grasps' in r['windows'][-1]['improved_metrics']


def test_reader_ignores_incomplete_line_and_newer_than_monitor(tmp_path):
    import json
    path=tmp_path/'iterations.jsonl'
    data=rows(2)
    path.write_text(''.join(json.dumps(x)+'\n' for x in data)+'{"iteration":')
    read=health('complete_iteration_rows')
    assert read(path,2)==data
    assert read(path,1)==data[:1]
