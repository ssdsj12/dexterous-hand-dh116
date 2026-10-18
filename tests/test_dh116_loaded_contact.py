import numpy as np
import pytest
from go2_pvcnn.control.dh116.loaded_contact import normal_contact_sample


def sample(**changes):
    fields=dict(force_on_mesh=[-1.,0,0],normal=[1.,0,0],gaps=[-.00001,.002],
        relative_normal_speeds=[0.,0.],contact_duration_s=.0001,dt=.0001)
    fields.update(changes);return normal_contact_sample(**fields)


def test_positive_load_and_full_duration_required_not_just_distance():
    assert sample()['valid']
    assert not sample(force_on_mesh=[0,0,0])['valid']
    assert not sample(contact_duration_s=0.)['valid']
    assert not sample(gaps=[.001,.002])['valid']


def test_separation_and_wrong_force_direction_rejected():
    assert not sample(relative_normal_speeds=[.002,0.])['valid']
    assert not sample(force_on_mesh=[1,0,0])['valid']


def test_invalid_normal_or_time_rejected():
    with pytest.raises(ValueError):sample(normal=[2,0,0])
    with pytest.raises(ValueError):sample(contact_duration_s=.0002)
    with pytest.raises(ValueError):sample(gaps=[np.nan,.002])


def test_contact_sample_reports_parent_element_without_confusing_quadrature_index():
    result=sample(element_ids=np.array([3,4]))
    assert result['closest_element']==3
    assert result['closest_sample']==0
    with pytest.raises(ValueError):sample(element_ids=np.array([1]))
