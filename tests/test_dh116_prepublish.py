from pathlib import Path
import numpy as np
import pytest
from go2_pvcnn.control.dh116.prepublish import prepublish_contact_gates, CHANNELS


class Mesh:
    def query_closest_points(self, points):
        return points, np.full(len(points), .0075)


def test_standalone_contact_check_retains_depth_and_speed_limits():
    state=dict(rod_position=np.zeros((3,3)),rod_velocity=np.zeros((3,3)))
    meshes={c:Mesh() for c in CHANNELS}
    impulses={c:dict(linear_w=[0.]*3,angular_com_w=[0.]*3,
                    peak_force_n=0.,contact_duration_s=0.) for c in CHANNELS}
    def check():
        return prepublish_contact_gates(state,meshes,meshes,impulses=impulses,
            mode='no-contact',selected_channel=None,max_speed_m_s=2.,
            max_pad_penetration_m=.001,radius_m=.008)
    failures,metrics=check()
    assert not failures
    assert metrics['max_pad_penetration_m']==pytest.approx(.0005)
    state['rod_velocity'][0,0]=3.
    assert 'speed gate exceeded before publish' in check()[0]
    state['rod_velocity'][:]=0.
    impulses[CHANNELS[0]]['peak_force_n']=1.
    assert 'unexpected channel reaction before publish' in check()[0]


def test_runner_uses_standalone_contact_check():
    text=(Path(__file__).parents[1]/'scripts/verify_fixed_dual_panda_dh116.py').read_text()
    assert 'from go2_pvcnn.control.dh116.prepublish import prepublish_contact_gates' in text
    assert 'from scripts.verify_fixed_dual_panda_o6_moving_contact import' not in text
