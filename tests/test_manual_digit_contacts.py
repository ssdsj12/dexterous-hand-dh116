from go2_pvcnn.control.dh116.tabletop_manual_digit_contacts import summarize


def points():
    return [dict(hand=h,digit=d,position_w=[.007 if d=='thumb' else -.007,0.,-.002],separation_m=-.0001)
            for h in ('left','right') for d in ('thumb','index')]


def loads():return {h:{d:1. for d in ('thumb','index')} for h in ('left','right')}


def test_both_digits_on_both_hands_are_required():
    p=points();r=summarize(p,loads(),[0,0,0],[0,1,0])
    assert r['all_four_digits_below_axis']
    p[1]['position_w'][2]=.0002
    r=summarize(p,loads(),[0,0,0],[0,1,0])
    assert not r['all_four_digits_below_axis'] and not r['digits']['left_index']['all_points_below_axis']


def test_average_below_does_not_hide_one_upper_contact():
    p=points();p.append(dict(hand='left',digit='thumb',position_w=[.007,0.,.001],separation_m=0.))
    r=summarize(p,loads(),[0,0,0],[0,1,0])
    assert r['digits']['left_thumb']['mean_radial_z_mm']<0
    assert not r['digits']['left_thumb']['all_points_below_axis']
    assert not r['all_four_digits_below_axis']


def test_unloaded_or_missing_finger_is_not_a_valid_grasp():
    l=loads();l['right']['index']=0.
    assert not summarize(points(),l,[0,0,0],[0,1,0])['all_four_digits_below_axis']
    assert not summarize(points()[:-1],loads(),[0,0,0],[0,1,0])['all_four_digits_below_axis']
