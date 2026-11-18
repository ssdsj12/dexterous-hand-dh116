import numpy as np
import pytest
from go2_pvcnn.control.dh116.path_certificate import certify_joint_path


def waypoints():
    return [{'left':np.zeros(13),'right':np.zeros(13)},
            {'left':np.ones(13),'right':np.ones(13)}]


def test_subdivision_covers_whole_joint_path_not_only_endpoints():
    def check(mid,delta,leg):
        b=.01-.1*delta['left'][0]
        return dict(source_clearance_passed=b>=.001,joint_box_covered=True,minimum_lower_bound_m=b)
    r=certify_joint_path(waypoints(),check,max_depth=8)
    assert r['static_wire_source_sweep_passed'] and not r['motion_qualified']
    spans=r['covered_intervals']
    assert len(spans)>1 and spans[0]['start']==0 and spans[-1]['end']==1
    assert all(a['end']==b['start'] for a,b in zip(spans,spans[1:]))


def test_failed_interval_cannot_be_counted_as_passed():
    def check(mid,delta,leg):
        return dict(source_clearance_passed=False,joint_box_covered=True,minimum_lower_bound_m=-.01)
    r=certify_joint_path(waypoints(),check,max_depth=3)
    assert not r['static_wire_source_sweep_passed'] and r['failed_interval']['leg']==0


def test_static_only_checker_cannot_certify_sweep():
    with pytest.raises(ValueError):
        certify_joint_path(waypoints(),lambda *a:dict(source_clearance_passed=True,joint_box_covered=False))
