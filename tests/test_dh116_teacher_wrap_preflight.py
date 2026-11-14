import pytest

from go2_pvcnn.control.dh116.tabletop_teacher_wrap_preflight import screen_wrap_hand


def hand():
    return dict(distal_order=['thumb','index','middle','ring','little'],
        patch_distance_m=[.0002]*5, patch_normal_dot=[-.95]*5,
        radial_directions=[[1.,0.,0.]]+[[-1.,0.,0.]]*4,
        hand_normals=[[-1.,0.,0.]]+[[1.,0.,0.]]*4,
        contact_digits=['thumb','index','middle','ring','little'],
        penetration=.01, object_in_hand=.01, self=.01, table=0.,
        minimum_clearance_m=.001, collision_units='mm', distance_units='m')


def test_wrap_requires_every_named_distal_even_if_old_labels_claim_all_five():
    row = hand()
    row['patch_distance_m'][4] = .01
    result = screen_wrap_hand(row)
    assert result['geometry_screen_passed'] is False
    assert result['missing_aligned_digits'] == ['little']


def test_opposition_must_involve_thumb_not_just_any_two_distals():
    row = hand()
    row['radial_directions'] = [[0.,0.,1.],[1.,0.,0.],[-1.,0.,0.],[1.,0.,0.],[1.,0.,0.]]
    row['hand_normals'] = [[0.,0.,-1.],[-1.,0.,0.],[1.,0.,0.],[-1.,0.,0.],[-1.,0.,0.]]
    assert screen_wrap_hand(row)['opposed_thumb_partners'] == []
    assert screen_wrap_hand(row)['geometry_screen_passed'] is False


def test_valid_geometry_is_only_a_geometry_screen():
    result = screen_wrap_hand(hand())
    assert result['geometry_screen_passed'] is True
    assert result['opposed_thumb_partners'] == ['index','middle','ring','little']
    assert result['native_contact_verified'] is False


@pytest.mark.parametrize('key,value', [
    ('patch_normal_dot',[float('nan')]*5),
    ('radial_directions',[[0.,0.,0.]]*5),
    ('distal_order',['thumb','thumb','middle','ring','little']),
])
def test_malformed_geometry_cannot_be_screened(key,value):
    row=hand()
    row[key]=value
    with pytest.raises(ValueError):
        screen_wrap_hand(row)
