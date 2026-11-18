from go2_pvcnn.control.dh116.tabletop_manual_thumb_stability import ThumbStability


def test_steady_total_does_not_hide_thumb_oscillation():
    tracker=ThumbStability()
    for i in range(30):
        thumb=1. if i%2 else 3.
        row=tracker.update([[thumb,5-thumb],[2.,3.]],1/60)
    assert row['thumb_force_stable']==[False,True]


def test_both_thumbs_need_nonzero_load_and_a_full_stable_window():
    tracker=ThumbStability()
    for _ in range(10):row=tracker.update([[2.,3.],[2.2,2.8]],1/60)
    assert row['thumb_force_stable']==[False,False]
    for _ in range(15):row=tracker.update([[2.,3.],[2.2,2.8]],1/60)
    assert row['thumb_force_stable']==[True,True]
    tracker.reset()
    for _ in range(30):row=tracker.update([[0.,0.],[.1,.1]],1/60)
    assert row['thumb_force_stable']==[False,False]


def test_reset_removes_previous_stability_proof():
    tracker=ThumbStability()
    for _ in range(30):tracker.update([[2.,3.],[2.,3.]],1/60)
    tracker.reset()
    assert tracker.update([[2.,3.],[2.,3.]],1/60)['thumb_force_stable']==[False,False]
